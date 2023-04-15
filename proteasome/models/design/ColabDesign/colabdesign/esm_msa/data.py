# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence, Tuple, Union
import re
import numpy as np
import jax.numpy as jnp

from .constants import proteinseq_toks

RawMSA = Sequence[Tuple[str, str]]


class FastaBatchedDataset(object):
  def __init__(self, sequence_labels, sequence_strs):
    self.sequence_labels = list(sequence_labels)
    self.sequence_strs = list(sequence_strs)

  @classmethod
  def from_file(cls, fasta_file):
    sequence_labels, sequence_strs = [], []
    cur_seq_label = None
    buf = []

    def _flush_current_seq():
      nonlocal cur_seq_label, buf
      if cur_seq_label is None:
        return
      sequence_labels.append(cur_seq_label)
      sequence_strs.append("".join(buf))
      cur_seq_label = None
      buf = []

    with open(fasta_file, "r") as infile:
      for line_idx, line in enumerate(infile):
        if line.startswith(">"):  # label line
          _flush_current_seq()
          line = line[1:].strip()
          if len(line) > 0:
            cur_seq_label = line
          else:
            cur_seq_label = f"seqnum{line_idx:09d}"
        else:  # sequence line
          buf.append(line.strip())

    _flush_current_seq()

    assert len(set(sequence_labels)) == len(sequence_labels), "Found duplicate sequence labels"

    return cls(sequence_labels, sequence_strs)

  def __len__(self):
    return len(self.sequence_labels)

  def __getitem__(self, idx):
    return self.sequence_labels[idx], self.sequence_strs[idx]

  def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
    sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
    sizes.sort()
    batches = []
    buf = []
    max_len = 0

    def _flush_current_buf():
      nonlocal max_len, buf
      if len(buf) == 0:
        return
      batches.append(buf)
      buf = []
      max_len = 0

    for sz, i in sizes:
      sz += extra_toks_per_seq
      if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
        _flush_current_buf()
      max_len = max(max_len, sz)
      buf.append(i)

    _flush_current_buf()
    return batches


class Alphabet(object):
  def __init__(
    self,
    standard_toks: Sequence[str],
    prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
    append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
    prepend_bos: bool = True,
    append_eos: bool = False,
    use_msa: bool = False,
  ):
    self.standard_toks = list(standard_toks)
    self.prepend_toks = list(prepend_toks)
    self.append_toks = list(append_toks)
    self.prepend_bos = prepend_bos
    self.append_eos = append_eos
    self.use_msa = use_msa

    self.all_toks = list(self.prepend_toks)
    self.all_toks.extend(self.standard_toks)
    for i in range((8 - (len(self.all_toks) % 8)) % 8):
      self.all_toks.append(f"<null_{i  + 1}>")
    self.all_toks.extend(self.append_toks)

    self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

    self.unk_idx = self.tok_to_idx["<unk>"]
    self.padding_idx = self.get_idx("<pad>")
    self.cls_idx = self.get_idx("<cls>")
    self.mask_idx = self.get_idx("<mask>")
    self.eos_idx = self.get_idx("<eos>")

  def __len__(self):
    return len(self.all_toks)

  def get_idx(self, tok):
    return self.tok_to_idx.get(tok, self.unk_idx)

  def get_tok(self, ind):
    return self.all_toks[ind]

  def to_dict(self):
    return {"toks": self.toks}

  def get_batch_converter(self):
    if self.use_msa:
      return MSABatchConverter(self)
    else:
      return BatchConverter(self)

  @classmethod
  def from_dict(cls, d, **kwargs):
    return cls(standard_toks=d["toks"], **kwargs)

  @classmethod
  def from_architecture(cls, name: str) -> "Alphabet":
    if name in ("ESM-1", "protein_bert_base"):
      standard_toks = proteinseq_toks["toks"]
      prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
      append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
      prepend_bos = True
      append_eos = False
      use_msa = False
    elif name in ("ESM-1b", "roberta_large"):
      standard_toks = proteinseq_toks["toks"]
      prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
      append_toks = ("<mask>",)
      prepend_bos = True
      append_eos = True
      use_msa = False
    elif name in ("MSA Transformer", "msa_transformer"):
      standard_toks = proteinseq_toks["toks"]
      prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
      append_toks = ("<mask>",)
      prepend_bos = True
      append_eos = False
      use_msa = True
    else:
      raise ValueError("Unknown architecture selected")
    return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)


class BatchConverter(object):
  """Callable to convert an unprocessed (labels + strings) batch to a
  processed (labels + tensor) batch.
  """

  def __init__(self, alphabet):
    self.alphabet = alphabet

  def __call__(self, raw_batch: Sequence[Tuple[str, str]], return_j=True):
    # RoBERTa uses an eos token, while ESM-1 does not.
    batch_size = len(raw_batch)
    max_len = max(len(seq_str) for _, seq_str in raw_batch)
    tokens_np = np.ones(
      [
        batch_size,
        max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)
      ],
      dtype=np.int64
    ) * self.alphabet.padding_idx

    labels = []
    strs = []

    for i, (label, seq_str) in enumerate(raw_batch):
      labels.append(label)
      strs.append(seq_str)
      if self.alphabet.prepend_bos:
        tokens_np[i, 0] = self.alphabet.cls_idx
      seq = np.array([self.alphabet.get_idx(s) for s in seq_str], dtype=np.int64)
      tokens_np[
        i,
        int(self.alphabet.prepend_bos): len(seq_str) + int(self.alphabet.prepend_bos),
      ] = seq
      if self.alphabet.append_eos:
        tokens_np[i, len(seq_str) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

    if return_j:
      tokens = jnp.array(tokens_np)
    else:
      tokens = tokens_np

    return labels, strs, tokens


class MSABatchConverter(BatchConverter):
  def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA], return_j=True):
    if isinstance(inputs[0][0], str):
      # Input is a single MSA
      raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
    else:
      raw_batch = inputs  # type: ignore

    batch_size = len(raw_batch)
    max_alignments = max(len(msa) for msa in raw_batch)
    max_seqlen = max(len(msa[0][1]) for msa in raw_batch)

    tokens_np = np.ones(
      [
        batch_size,
        max_alignments,
        max_seqlen + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
      ],
      dtype=np.int64,
    ) * self.alphabet.padding_idx

    labels = []
    strs = []

    for i, msa in enumerate(raw_batch):
      msa_seqlens = set(len(seq) for _, seq in msa)
      if not len(msa_seqlens) == 1:
        raise RuntimeError(
          "Received unaligned sequences for input to MSA, all sequence "
          "lengths must be equal."
        )
      msa_labels, msa_strs, msa_tokens = super().__call__(msa, return_j=False)
      labels.append(msa_labels)
      strs.append(msa_strs)
      tokens_np[i, :msa_tokens.shape[0], :msa_tokens.shape[1]] = msa_tokens

    if return_j:
      tokens = jnp.array(tokens_np)
    else:
      tokens = tokens_np

    return labels, strs, tokens


def read_fasta(
  path,
  keep_gaps=True,
  keep_insertions=True,
  to_upper=False,
):
  with open(path, "r") as f:
    for result in read_alignment_lines(
      f, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
    ):
      yield result


def read_alignment_lines(
  lines,
  keep_gaps=True,
  keep_insertions=True,
  to_upper=False,
):
  seq = desc = None

  def parse(s):
    if not keep_gaps:
      s = re.sub("-", "", s)
    if not keep_insertions:
      s = re.sub("[a-z]", "", s)
    return s.upper() if to_upper else s

  for line in lines:
    # Line may be empty if seq % file_line_width == 0
    if len(line) > 0 and line[0] == ">":
      if seq is not None:
        yield desc, parse(seq)
      desc = line.strip()
      seq = ""
    else:
      assert isinstance(seq, str)
      seq += line.strip()
  assert isinstance(seq, str) and isinstance(desc, str)
  yield desc, parse(seq)
