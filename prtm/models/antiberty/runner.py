import torch
import transformers
from prtm.models.antiberty.model import AntiBERTy
from prtm.models.igfold.utils.general import exists


LABEL_TO_SPECIES = {
    0: "Camel",
    1: "Human",
    2: "Mouse",
    3: "Rabbit",
    4: "Rat",
    5: "Rhesus",
}
LABEL_TO_CHAIN = {0: "Heavy", 1: "Light"}

SPECIES_TO_LABEL = {v: k for k, v in LABEL_TO_SPECIES.items()}
CHAIN_TO_LABEL = {v: k for k, v in LABEL_TO_CHAIN.items()}


class AntiBERTyRunner:
    def __init__(self, checkpoint_path: str, vocab_file: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AntiBERTy.from_pretrained(checkpoint_path).to(self.device)
        self.model.eval()

        self.tokenizer = transformers.BertTokenizer(
            vocab_file=vocab_file, do_lower_case=False
        )

    def embed(self, sequences, hidden_layer=-1, return_attention=False):
        """
        Embed a list of sequences.

        Args:
            sequences (list): list of sequences
            hidden_layer (int): which hidden layer to use (0 to 8)
            return_attention (bool): whether to return attention matrices

        Returns:
            list(torch.Tensor): list of embeddings (one tensor per sequence)

        """
        sequences = [list(s) for s in sequences]
        for s in sequences:
            for i, c in enumerate(s):
                if c == "_":
                    s[i] = "[MASK]"

        sequences = [" ".join(s) for s in sequences]
        tokenizer_out = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        )
        tokens = tokenizer_out["input_ids"].to(self.device)
        attention_mask = tokenizer_out["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=return_attention,
            )

        # gather embeddings
        embeddings = outputs.hidden_states
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = list(embeddings.detach())

        for i, a in enumerate(attention_mask):
            embeddings[i] = embeddings[i][:, a == 1]

        if exists(hidden_layer):
            for i in range(len(embeddings)):
                embeddings[i] = embeddings[i][hidden_layer]

        # gather attention matrices
        if return_attention:
            attentions = outputs.attentions
            attentions = torch.stack(attentions, dim=1)
            attentions = list(attentions.detach())

            for i, a in enumerate(attention_mask):
                attentions[i] = attentions[i][:, :, a == 1]
                attentions[i] = attentions[i][:, :, :, a == 1]

            return embeddings, attentions

        return embeddings

    def fill_masks(self, sequences):
        """
        Fill in the missing residues in a list of sequences. Each missing token is
        represented by an underscore character.

        Args:
            sequences (list): list of sequences with _ (underscore) tokens

        Returns:
            list: list of sequences with missing residues filled in
        """
        sequences = [list(s) for s in sequences]
        for s in sequences:
            for i, c in enumerate(s):
                if c == "_":
                    s[i] = "[MASK]"

        sequences = [" ".join(s) for s in sequences]
        tokenizer_out = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        )
        tokens = tokenizer_out["input_ids"].to(self.device)
        attention_mask = tokenizer_out["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
            )
            logits = outputs.prediction_logits
            logits[:, :, self.tokenizer.all_special_ids] = -float("inf")

        predicted_tokens = torch.argmax(logits, dim=-1)
        tokens[tokens == self.tokenizer.mask_token_id] = predicted_tokens[
            tokens == self.tokenizer.mask_token_id
        ]

        predicted_seqs = self.tokenizer.batch_decode(
            tokens,
            skip_special_tokens=True,
        )
        predicted_seqs = [s.replace(" ", "") for s in predicted_seqs]

        return predicted_seqs

    def classify(
        self, sequences, species_label=None, chain_label=None, graft_label=None
    ):
        """
        Classify a list of sequences by species and chain type. Sequences may contain
        missing residues, which are represented by an underscore character.

        Args:
            sequences (list): list of sequences

        Returns:
            list: list of species predictions
            list: list of chain type predictions
        """

        sequences = [list(s) for s in sequences]
        for s in sequences:
            for i, c in enumerate(s):
                if c == "_":
                    s[i] = "[MASK]"

        sequences = [" ".join(s) for s in sequences]
        tokenizer_out = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        )
        tokens = tokenizer_out["input_ids"].to(self.device)
        attention_mask = tokenizer_out["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
            )
            species_logits = outputs.species_logits
            chain_logits = outputs.chain_logits
            graft_logits = outputs.graft_logits

        if exists(species_label) or exists(chain_label) or exists(graft_label):
            out_dict = {}
            if exists(species_label):
                species_label = torch.tensor(
                    [SPECIES_TO_LABEL[species_label]],
                ).to(self.device)
                species_ll = -1 * torch.nn.functional.cross_entropy(
                    species_logits, species_label
                )
                out_dict["species_ll"] = species_ll.item()
            if exists(chain_label):
                chain_label = torch.tensor(
                    [CHAIN_TO_LABEL[chain_label]],
                ).to(self.device)
                chain_ll = -1 * torch.nn.functional.cross_entropy(
                    chain_logits, chain_label
                )
                out_dict["chain_ll"] = chain_ll.item()
            if exists(graft_label):
                graft_label = torch.tensor(
                    [int(graft_label)],
                ).to(self.device)
                graft_ll = -1 * torch.nn.functional.cross_entropy(
                    graft_logits, graft_label
                )
                out_dict["graft_ll"] = graft_ll.item()

            return out_dict

        species_preds = torch.argmax(species_logits, dim=-1)
        chain_preds = torch.argmax(chain_logits, dim=-1)

        species_preds = [LABEL_TO_SPECIES[p.item()] for p in species_preds]
        chain_preds = [LABEL_TO_CHAIN[p.item()] for p in chain_preds]

        return species_preds, chain_preds

    def pseudo_log_likelihood(self, sequences, batch_size=None):
        plls = []
        for s in sequences:
            masked_sequences = []
            for i in range(len(s)):
                masked_sequence = list(s[:i]) + ["[MASK]"] + list(s[i + 1 :])
                masked_sequences.append(" ".join(masked_sequence))

            tokenizer_out = self.tokenizer(
                masked_sequences,
                return_tensors="pt",
                padding=True,
            )
            tokens = tokenizer_out["input_ids"].to(self.device)
            attention_mask = tokenizer_out["attention_mask"].to(self.device)

            logits = []
            with torch.no_grad():
                if not exists(batch_size):
                    batch_size_ = len(masked_sequences)
                else:
                    batch_size_ = batch_size

                from tqdm import tqdm

                for i in tqdm(range(0, len(masked_sequences), batch_size_)):
                    batch_end = min(i + batch_size_, len(masked_sequences))
                    tokens_ = tokens[i:batch_end]
                    attention_mask_ = attention_mask[i:batch_end]

                    outputs = self.model(
                        input_ids=tokens_,
                        attention_mask=attention_mask_,
                    )

                    logits.append(outputs.prediction_logits)

            logits = torch.cat(logits, dim=0)
            logits[:, :, self.tokenizer.all_special_ids] = -float("inf")
            logits = logits[:, 1:-1]  # remove CLS and SEP tokens

            # get masked token logits
            logits = torch.diagonal(logits, dim1=0, dim2=1).unsqueeze(0)
            labels = self.tokenizer.encode(
                " ".join(list(s)),
                return_tensors="pt",
            )[:, 1:-1].to(logits.device)
            nll = torch.nn.functional.cross_entropy(
                logits,
                labels,
                reduction="mean",
            )
            pll = -nll

            plls.append(pll)

        plls = torch.stack(plls, dim=0)

        return plls
