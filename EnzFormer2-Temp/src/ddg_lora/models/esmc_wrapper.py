"""ESMC masked-site readout with explicit tokenizer invariants.

Residue position ``i`` maps to encoded position ``i + 1`` because ESMC adds
one BOS token. Logits are sliced to the tokenizer vocabulary before softmax:
the model output dimension may include padding entries. These invariants define
the masked-log-odds feature used by EnzFormer2_temp v2.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from ddg_lora.constants import AMINO_ACIDS_20

log = logging.getLogger(__name__)


def _resolve_device(spec: str) -> torch.device:
    """Resolve a device spec ("auto"|"cuda"|"cpu"|"cuda:N") to torch.device."""
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"requested device={spec!r}, but this PyTorch build cannot use CUDA "
            f"(torch={torch.__version__}). On DGX Spark, use a CUDA-enabled "
            "NVIDIA/NGC PyTorch environment instead of a CPU-only wheel."
        )
    return torch.device(spec)


@dataclass(frozen=True)
class TokenizerInfo:
    mask_token: str
    mask_token_id: int
    cls_token_id: int
    eos_token_id: int
    pad_token_id: int
    vocab_size: int
    aa_to_token_id: dict[str, int]
    # Encoded length for input of length L is L + sequence_offset_pre + sequence_offset_post.
    sequence_offset_pre: int = 1   # BOS
    sequence_offset_post: int = 1  # EOS

    def encoded_index(self, position_0idx: int) -> int:
        return position_0idx + self.sequence_offset_pre


class ESMCMaskedLM:
    """ESMC 600M wrapper exposing masked-site log-probabilities."""

    def __init__(self, model: Any, tokenizer: Any, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        aa_to_id = {aa: int(tokenizer.convert_tokens_to_ids(aa)) for aa in AMINO_ACIDS_20}
        self.tok_info = TokenizerInfo(
            mask_token=tokenizer.mask_token,
            mask_token_id=int(tokenizer.mask_token_id),
            cls_token_id=int(tokenizer.cls_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            pad_token_id=int(tokenizer.pad_token_id),
            vocab_size=int(tokenizer.vocab_size),
            aa_to_token_id=aa_to_id,
        )
        # Sanity: every standard amino acid must have a unique id within vocab.
        ids = list(aa_to_id.values())
        if len(set(ids)) != 20:
            raise RuntimeError(
                f"Standard amino acids do not map to 20 unique token ids: {aa_to_id}"
            )
        if any(i >= self.tok_info.vocab_size for i in ids):
            raise RuntimeError(
                f"Some amino-acid token ids exceed tokenizer vocab_size={self.tok_info.vocab_size}: {aa_to_id}"
            )

    # ----- construction -------------------------------------------------------

    @classmethod
    def load(cls, model_name: str = "esmc_600m", device: str = "auto") -> "ESMCMaskedLM":
        """Load ESMC and its tokenizer onto the resolved device.

        Raises a clear error if the esm SDK isn't available or the loader fails.
        """
        try:
            from esm.models.esmc import ESMC
            from esm.tokenization import get_esmc_model_tokenizers
        except ImportError as e:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "esm SDK not available; install it before running ESMC code "
                f"(import error: {e})"
            ) from e
        dev = _resolve_device(device)
        try:
            model = ESMC.from_pretrained(model_name, device=dev)
        except Exception as e:  # pragma: no cover - depends on env / cache
            raise RuntimeError(
                f"failed to load ESMC weights {model_name!r} on {dev}: {e}. "
                "Check network access, HF_TOKEN, and disk space in HF cache."
            ) from e
        model.eval()
        tokenizer = get_esmc_model_tokenizers()
        return cls(model, tokenizer, dev)

    # ----- masking helpers ----------------------------------------------------

    def get_masked_sequence(self, sequence: str, position_0idx: int) -> str:
        """Return the input sequence with the residue at `position_0idx` replaced by mask."""
        if not sequence:
            raise ValueError("sequence must be non-empty")
        if not (0 <= position_0idx < len(sequence)):
            raise IndexError(
                f"position_0idx {position_0idx} out of range for sequence of length {len(sequence)}"
            )
        return sequence[:position_0idx] + self.tok_info.mask_token + sequence[position_0idx + 1:]

    def amino_acid_to_token_id(self, aa: str) -> int:
        if aa not in self.tok_info.aa_to_token_id:
            raise KeyError(f"unknown amino acid: {aa!r}")
        return self.tok_info.aa_to_token_id[aa]

    # ----- forward / readout --------------------------------------------------

    @torch.inference_mode()
    def forward_logits(self, masked_sequence: str) -> torch.Tensor:
        """Run ESMC on a single (already-masked) sequence and return raw logits.

        Returns a tensor of shape (encoded_len, V) where encoded_len = L + 2 and
        V is the model's logits last dim (V >= tokenizer.vocab_size). The caller
        is responsible for slicing to `tok_info.vocab_size` and softmaxing.
        """
        from esm.sdk.api import ESMProtein, LogitsConfig

        prot = ESMProtein(sequence=masked_sequence)
        tensor = self.model.encode(prot)
        out = self.model.logits(tensor, LogitsConfig(sequence=True))
        # logits.sequence: (1, encoded_len, V)
        seq_logits = out.logits.sequence[0]  # drop batch dim
        if not torch.isfinite(seq_logits).all():
            raise RuntimeError("ESMC logits contained non-finite values")
        return seq_logits

    def get_position_logits(
        self, masked_sequence: str, position_0idx: int
    ) -> torch.Tensor:
        """Return the logits row at the residue position (1-D, length tok_info.vocab_size)."""
        seq_logits = self.forward_logits(masked_sequence)
        idx = self.tok_info.encoded_index(position_0idx)
        if not (0 <= idx < seq_logits.shape[0]):
            raise IndexError(
                f"resolved encoded index {idx} out of range; logits length={seq_logits.shape[0]}"
            )
        return seq_logits[idx, : self.tok_info.vocab_size]

    def masked_log_probs_at_position(
        self, sequence: str, position_0idx: int
    ) -> torch.Tensor:
        """Mask one residue and return log probabilities over the tokenizer vocab."""
        masked = self.get_masked_sequence(sequence, position_0idx)
        row_logits = self.get_position_logits(masked, position_0idx)
        return F.log_softmax(row_logits.float(), dim=-1)

    def masked_logodds_forward(
        self,
        sequence: str,
        position_0idx: int,
        *,
        wt_aa: str,
        mut_aa: str,
    ) -> float:
        """Return log p(mut | masked context) - log p(wt | masked context)."""
        log_probs = self.masked_log_probs_at_position(sequence, position_0idx)
        wt_id = self.amino_acid_to_token_id(wt_aa)
        mut_id = self.amino_acid_to_token_id(mut_aa)
        return float((log_probs[mut_id] - log_probs[wt_id]).item())

    def masked_logodds_pair(
        self,
        wt_sequence: str,
        mut_sequence: str,
        position_0idx: int,
        *,
        wt_aa: str,
        mut_aa: str,
    ) -> tuple[float, float]:
        """Return forward and reverse masked log-odds for a single mutation."""
        forward = self.masked_logodds_forward(
            wt_sequence, position_0idx, wt_aa=wt_aa, mut_aa=mut_aa
        )
        reverse = self.masked_logodds_forward(
            mut_sequence, position_0idx, wt_aa=mut_aa, mut_aa=wt_aa
        )
        return forward, reverse

    # ----- hidden-state readout (regression path) ----------------------------

    def forward_residue_hidden_states(self, sequence: str) -> torch.Tensor:
        """Return last-layer residue hidden states with shape ``[L, D]``.

        Unlike :meth:`forward_logits` / :meth:`get_position_logits`, this method
        is **not** wrapped in ``torch.inference_mode``: the regression path
        backpropagates into the LoRA parameters, so the forward must remain
        autograd-aware. Callers that only need inference should wrap the call
        in ``torch.inference_mode`` themselves.

        BOS and EOS positions are stripped using ``tok_info.sequence_offset_pre``
        / ``...post``. The returned tensor's dtype matches the backbone's
        autocast/precision policy.
        """
        from esm.sdk.api import ESMProtein

        if not sequence:
            raise ValueError("sequence must be non-empty")

        prot = ESMProtein(sequence=sequence)
        tensor = self.model.encode(prot)
        tokens = tensor.sequence.unsqueeze(0)  # [L+2] -> [1, L+2]
        output = self.model.forward(sequence_tokens=tokens)
        embeddings = output.embeddings  # [1, L+2, D]
        if embeddings.dim() != 3 or embeddings.shape[0] != 1:
            raise RuntimeError(
                f"unexpected embeddings shape {tuple(embeddings.shape)}; "
                "expected [1, L+2, D]"
            )
        pre = self.tok_info.sequence_offset_pre
        post = self.tok_info.sequence_offset_post
        encoded_len = embeddings.shape[1]
        residue_len = encoded_len - pre - post
        if residue_len != len(sequence):
            raise RuntimeError(
                f"residue length mismatch: encoded={encoded_len}, "
                f"pre={pre}, post={post}, expected residue_len={len(sequence)} "
                f"got {residue_len}"
            )
        # Strip BOS/EOS, drop batch dim -> [L, D].
        return embeddings[0, pre : encoded_len - post, :]

    def forward_residue_hidden_states_batch(
        self, sequences: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return padded last-layer residue hidden states for a sequence batch.

        Returns ``(hidden, lengths)`` where ``hidden`` has shape
        ``[B, max_L, D]`` and ``lengths`` has shape ``[B]``. Rows shorter than
        ``max_L`` contain ESMC pad/EOS-derived embeddings beyond their true
        length; callers must use ``lengths`` when averaging over residues.

        This method intentionally uses ESMC's private ``_tokenize`` helper
        because the public SDK ``encode`` path only accepts one
        :class:`ESMProtein` at a time, while ``ESMC.forward`` already supports
        batched token tensors.
        """
        if not sequences:
            raise ValueError("sequences must be non-empty")
        seqs = [str(s) for s in sequences]
        if any(not s for s in seqs):
            raise ValueError("all sequences must be non-empty")

        if not hasattr(self.model, "_tokenize"):
            raise RuntimeError(
                "ESMC model does not expose _tokenize; cannot run batched hidden-state forward"
            )

        tokens = self.model._tokenize(seqs)  # [B, max_L + BOS/EOS]
        output = self.model.forward(sequence_tokens=tokens)
        embeddings = output.embeddings  # [B, encoded_max_len, D]
        if embeddings.dim() != 3 or embeddings.shape[0] != len(seqs):
            raise RuntimeError(
                f"unexpected batched embeddings shape {tuple(embeddings.shape)}; "
                f"expected [{len(seqs)}, L+2, D]"
            )

        lengths = torch.tensor([len(s) for s in seqs], device=self.device, dtype=torch.long)
        pre = self.tok_info.sequence_offset_pre
        max_len = int(lengths.max().item())
        residue_embeddings = embeddings[:, pre : pre + max_len, :]
        if residue_embeddings.shape[1] != max_len:
            raise RuntimeError(
                f"batched residue length mismatch: got {residue_embeddings.shape[1]}, "
                f"expected {max_len}"
            )
        return residue_embeddings, lengths

    # ----- adapter integration helpers ---------------------------------------

    def parameters(self):
        """Convenience wrapper so callers can iterate without going through self.model."""
        return self.model.parameters()

    def named_modules(self):
        return self.model.named_modules()

    def named_parameters(self):
        return self.model.named_parameters()
