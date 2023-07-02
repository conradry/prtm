# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


from proteome.models.folding.esm.inverse_folding.gvp_transformer_encoder import GVPTransformerEncoder
from proteome.models.folding.esm.inverse_folding.transformer_decoder import TransformerDecoder
from proteome.models.folding.esm.inverse_folding.util import rotate, CoordBatchConverter
from proteome.models.folding.esm import config


class GVPTransformerModel(nn.Module):
    """
    GVP-Transformer inverse folding model.

    Architecture: Geometric GVP-GNN as initial layers, followed by
    sequence-to-sequence Transformer encoder and decoder.
    """

    def __init__(self, cfg: config.ESMIFConfig):
        super().__init__()
        encoder_embed_tokens = self.build_embedding(cfg)
        decoder_embed_tokens = self.build_embedding(cfg)
        encoder = self.build_encoder(cfg, encoder_embed_tokens)
        decoder = self.build_decoder(cfg, decoder_embed_tokens)
        self.cfg = cfg
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def build_encoder(cls, cfg: config.ESMIFConfig, encoder_embed_tokens: nn.Embedding):
        encoder = GVPTransformerEncoder(cfg, encoder_embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, cfg: config.ESMIFConfig, embed_tokens: nn.Embedding):
        decoder = TransformerDecoder(cfg, embed_tokens)
        return decoder

    @classmethod
    def build_embedding(cls, cfg: config.ESMIFConfig):
        num_embeddings = len(cfg.alphabet)
        padding_idx = cfg.alphabet.padding_idx
        emb = nn.Embedding(num_embeddings, cfg.encoder_embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=(cfg.encoder_embed_dim ** -0.5))
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(
        self,
        coords,
        padding_mask,
        confidence,
        prev_output_tokens,
        return_all_hiddens: bool = False,
        features_only: bool = False,
    ):
        encoder_out = self.encoder(coords, padding_mask, confidence,
            return_all_hiddens=return_all_hiddens)
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra
    
    def sample(self, coords, partial_seq=None, temperature=1.0, confidence=None, device=None):
        """
        Samples sequences based on multinomial sampling (no beam search).

        Args:
            coords: L x 3 x 3 list representing one backbone
            partial_seq: Optional, partial sequence with mask tokens if part of
                the sequence is known
            temperature: sampling temperature, use low temperature for higher
                sequence recovery and high temperature for higher diversity
            confidence: optional length L list of confidence scores for coordinates
        """
        L = len(coords)
        # Convert to batch format
        batch_converter = CoordBatchConverter(self.decoder.alphabet)
        batch_coords, confidence, _, _, padding_mask = (
            batch_converter([(coords, confidence, None)], device=device)
        )
        
        # Start with prepend token
        mask_idx = self.decoder.alphabet.get_idx('<mask>')
        sampled_tokens = torch.full((1, 1+L), mask_idx, dtype=int)
        sampled_tokens[0, 0] = self.decoder.alphabet.get_idx('<cath>')
        if partial_seq is not None:
            for i, c in enumerate(partial_seq):
                sampled_tokens[0, i+1] = self.decoder.alphabet.get_idx(c)
            
        # Save incremental states for faster sampling
        incremental_state = dict()
        
        # Run encoder only once
        encoder_out = self.encoder(batch_coords, padding_mask, confidence)
        
        # Make sure all tensors are on the same device if a GPU is present
        if device:
            sampled_tokens = sampled_tokens.to(device)
        
        # Decode one token at a time
        for i in range(1, L+1):
            logits, _ = self.decoder(
                sampled_tokens[:, :i], 
                encoder_out,
                incremental_state=incremental_state,
            )
            logits = logits[0].transpose(0, 1)
            logits /= temperature
            probs = F.softmax(logits, dim=-1)
            if sampled_tokens[0, i] == mask_idx:
                sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
        sampled_seq = sampled_tokens[0, 1:]
        
        # Convert back to string via lookup
        return ''.join([self.decoder.alphabet.get_tok(a) for a in sampled_seq])
