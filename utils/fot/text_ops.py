from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def basic_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def stable_hash_to_bucket(token: str, *, num_buckets: int) -> int:
    if num_buckets <= 0:
        raise ValueError(f"num_buckets must be > 0, got {num_buckets}")
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(h, "little", signed=False)
    return int(val % int(num_buckets))


@dataclass(frozen=True)
class TokenBatch:
    input_ids: torch.Tensor  # (B, L) long
    attention_mask: torch.Tensor  # (B, L) float32


def tokenize_to_buckets(
    texts: Sequence[str],
    *,
    num_buckets: int,
    max_len: int,
    device: torch.device,
) -> TokenBatch:
    if max_len <= 0:
        raise ValueError(f"max_len must be > 0, got {max_len}")

    token_ids: List[List[int]] = []
    for t in texts:
        toks = basic_tokenize(t)[:max_len]
        ids = [stable_hash_to_bucket(tok, num_buckets=num_buckets) for tok in toks]
        if not ids:
            ids = [0]
        token_ids.append(ids)

    l = max(len(ids) for ids in token_ids)
    l = min(l, max_len)
    b = len(token_ids)

    input_ids = torch.zeros((b, l), dtype=torch.long, device=device)
    attn = torch.zeros((b, l), dtype=torch.float32, device=device)
    for i, ids in enumerate(token_ids):
        ids = ids[:l]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, : len(ids)] = 1.0

    return TokenBatch(input_ids=input_ids, attention_mask=attn)


def tokenize_choices_to_buckets(
    choices: Sequence[Sequence[str]],
    *,
    num_buckets: int,
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a batch of multiple-choice options.

    Returns:
      - input_ids: (B, N, L)
      - attention_mask: (B, N, L)
    """
    b = len(choices)
    n = max(len(row) for row in choices) if b > 0 else 0
    if n == 0:
        raise ValueError("Empty choices batch.")

    flat: List[str] = []
    for row in choices:
        row = list(row)
        if len(row) < n:
            row = row + [""] * (n - len(row))
        flat.extend(row[:n])

    tb = tokenize_to_buckets(flat, num_buckets=num_buckets, max_len=max_len, device=device)
    input_ids = tb.input_ids.view(b, n, -1)
    attn = tb.attention_mask.view(b, n, -1)
    return input_ids, attn

