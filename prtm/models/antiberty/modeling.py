import os
import random
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import transformers
from prtm.models.antiberty import config
from prtm.models.antiberty.model import AntiBERTy
from prtm.utils import hub_utils

__all__ = [
    "AntiBERTyForSequenceEmbedding",
]

ANTIBERTY_MODEL_URLS = {
    "base": "https://files.pythonhosted.org/packages/a1/3b/2cf48ec21956252fdc5c5dd1b7f8bb8b12f5208bd3eaaad412ced3ed0ff5/antiberty-0.1.3.tar.gz",  # noqa: E501
}
ANTIBERTY_MODEL_CONFIGS = {
    "base": config.AntiBERTyConfig(),
}


def _get_model_config(model_name: str) -> config.AntiBERTyConfig:
    """Get the model config for a given model name."""
    return ANTIBERTY_MODEL_CONFIGS[model_name]


class _AntiBERTyBase:
    def __init__(
        self,
        model_name: str = "base",
        random_seed: Optional[int] = None,
    ):
        self.model_name = model_name
        self.cfg = _get_model_config(model_name)
        self.model = AntiBERTy(transformers.PretrainedConfig(**asdict(self.cfg)))

        self.load_weights(ANTIBERTY_MODEL_URLS[model_name])
        self.model.eval()

        vocab_dir = os.path.dirname(os.path.abspath(__file__))
        self.tokenizer = transformers.BertTokenizer(
            vocab_file=os.path.join(vocab_dir, "vocab.txt"), do_lower_case=False
        )

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        if random_seed is not None:
            torch.manual_seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

    @classmethod
    @property
    def available_models(cls):
        return list(ANTIBERTY_MODEL_URLS.keys())

    def load_weights(self, weights_url: str):
        """Load weights from a weights url."""
        state_dict = hub_utils.load_state_dict_from_tar_gz_url(
            weights_url,
            extract_member="antiberty-0.1.3/antiberty/trained_models/AntiBERTy_md_smooth/pytorch_model.bin",
            model_name=f"antiberty_{self.model_name}.pth",
            map_location="cpu",
        )
        self.model.load_state_dict(state_dict)

    def _prepare_sequence(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Takes an amino acid sequence and returns a tokenized tensor along
        with an attention mask for the sequence.
        """
        sequence = list(sequence)
        # Replace masked residues with [MASK]
        for i, res in enumerate(sequence):
            if res == "_":
                sequence[i] = "[MASK]"

        # Tokenize the sequence
        sequence = " ".join(sequence)
        tokenizer_out = self.tokenizer(
            [sequence],
            return_tensors="pt",
            padding=True,
        )
        tokens = tokenizer_out["input_ids"].to(self.device)
        attention_mask = tokenizer_out["attention_mask"].to(self.device)

        return tokens, attention_mask

    def __call__(self):
        raise NotImplementedError


class AntiBERTyForSequenceEmbedding(_AntiBERTyBase):
    @torch.no_grad()
    def __call__(
        self, sequence: str, hidden_layer: Optional[int] = -1
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Embed a single sequence."""
        tokens, attention_mask = self._prepare_sequence(sequence)
        outputs = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        # gather embeddings
        embeddings = outputs.hidden_states
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = list(embeddings.detach())

        for i, a in enumerate(attention_mask):
            embeddings[i] = embeddings[i][:, a == 1]

        if hidden_layer is not None:
            for i in range(len(embeddings)):
                embeddings[i] = embeddings[i][hidden_layer]

        # gather attention matrices
        attentions = outputs.attentions
        attentions = torch.stack(attentions, dim=1)
        attentions = list(attentions.detach())

        for i, a in enumerate(attention_mask):
            attentions[i] = attentions[i][:, :, a == 1]
            attentions[i] = attentions[i][:, :, :, a == 1]

        return embeddings[0], {"attention": attentions[0]}


class AntiBERTyForSequenceInpainting(_AntiBERTyBase):
    @torch.no_grad()
    def __call__(self, sequence: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Inpaint a single sequence."""
        tokens, attention_mask = self._prepare_sequence(sequence)
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

        return predicted_seqs[0], {}


class AntiBERTyForAntibodySequenceClassification(_AntiBERTyBase):
    LABEL_TO_SPECIES = {
        0: "Camel",
        1: "Human",
        2: "Mouse",
        3: "Rabbit",
        4: "Rat",
        5: "Rhesus",
    }
    LABEL_TO_CHAIN = {0: "Heavy", 1: "Light"}

    @torch.no_grad()
    def __call__(self, sequence: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Classify an antibody sequence."""
        tokens, attention_mask = self._prepare_sequence(sequence)
        outputs = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
        )
        species_logits = outputs.species_logits
        chain_logits = outputs.chain_logits

        species_preds = torch.argmax(species_logits, dim=-1)
        chain_preds = torch.argmax(chain_logits, dim=-1)

        species_preds = [self.LABEL_TO_SPECIES[p.item()] for p in species_preds]
        chain_preds = [self.LABEL_TO_CHAIN[p.item()] for p in chain_preds]

        return {"species": species_preds, "chain_type": chain_preds}, {}
