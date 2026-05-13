"""
Fast pre-tokenized dataset loader for Kauldron training.

Loads pre-tokenized OpenWebText from HuggingFace Hub (created by precompute_openwebtext.py).
Variable-length token sequences are padded to max_length on-the-fly — no tokenizer needed!

Usage in training notebook:
    from fast_dataset_loader import get_fast_openwebtext

    train_ds = get_fast_openwebtext(
        repo_id="YOUR_USERNAME/openwebtext-gemma3-tokenized-1024",
        batch_size=32,
        max_length=1024,
    )
"""

import os

import numpy as np
import grain.python as grain
from kauldron import kd


class _PadToLength(grain.MapTransform):
    """Pad variable-length input_ids to fixed max_length and create input_mask."""

    def __init__(self, max_length: int = 1024):
        self.max_length = max_length

    def map(self, element: dict) -> dict:
        tokens = element["input_ids"]
        if isinstance(tokens, list):
            tokens = np.array(tokens, dtype=np.int32)
        original_len = min(len(tokens), self.max_length)
        tokens = tokens[: self.max_length]
        pad_len = self.max_length - len(tokens)
        if pad_len > 0:
            tokens = np.pad(tokens, (0, pad_len), constant_values=0)
        # Маска: 1 для реальных токенов, 0 для паддинга
        input_mask = np.zeros(self.max_length, dtype=np.int32)
        input_mask[:original_len] = 1
        return {"tokens": tokens, "input_mask": input_mask}


class _KeepTokens(grain.MapTransform):
    """Keep only the 'tokens' field."""

    def map(self, element: dict) -> dict:
        return {"tokens": element["tokens"]}


def get_fast_openwebtext(
    repo_id: str = "veriga/openwebtext-gemma3-tokenized-1024",
    batch_size: int = 32,
    max_length: int = 1024,
    split: str = "train",
    shuffle: bool = True,
    seed: int = 42,
    num_epochs: int | None = None,
    token: str | None = None,
    cache_dir: str | None = None
    
) -> kd.data.Pipeline:
    """
    Create a Kauldron dataset pipeline from pre-tokenized HuggingFace dataset.

    Args:
        repo_id: HuggingFace dataset repo ID with pre-tokenized data.
        batch_size: Number of examples per batch.
        max_length: Pad/truncate to this many tokens.
        split: Dataset split (default: 'train').
        shuffle: Whether to shuffle (default: True).
        seed: Random seed for shuffling.
        num_epochs: Number of epochs (None = infinite).
        token: HuggingFace API token (default: $HF_TOKEN env var).

    Returns:
        Kauldron dataset pipeline ready for training.
    """
    from datasets import load_dataset

    hf_token = token or os.environ.get("HF_TOKEN")

    print(f"⚡ Loading pre-tokenized dataset from {repo_id} (split='{split}')...")
    # hf_ds = load_dataset(repo_id, split=split, token=hf_token)
    # print(f"   {len(hf_ds):,} documents loaded")

    return kd.data.py.HuggingFace(
        path=repo_id, 
        split="train", 
        shuffle=True, 
        num_workers=1,
        num_epochs=None,
        batch_size=batch_size,
        transforms=[
            _PadToLength(max_length=max_length),
            kd.data.py.Elements(keep=["tokens", "input_mask"]),
        ],
        cache_dir = cache_dir
    )