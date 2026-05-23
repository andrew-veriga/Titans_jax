"""
Training DataLoader for precomputed activations.

Loads .npy shards produced by precompute_activations.py and yields
(hidden_state, tokens, mask) tuples ready for TitansBlock training.

Supports:
  - Streaming from disk (no need to load everything into RAM)
  - Shuffling within a buffer
  - Train/val split
  - Variable-length padding (respects original mask)

Usage in training:
    from activations_dataloader import ActivationDataLoader

    loader = ActivationDataLoader(
        activation_dir="./activations_layer23",
        batch_size=4,
        shuffle=True,
    )

    for batch in loader:
        hidden = batch["hidden"]       # (B, L, 1152) — input to layer 23
        tokens = batch["tokens"]       # (B, L) — original token ids (for CE loss targets)
        mask   = batch["mask"]         # (B, L) — 1 for real tokens, 0 for padding
        break
"""

import json
import os
from glob import glob
from typing import Optional, Iterator

import numpy as np


class ActivationDataLoader:
    """Streams precomputed activation shards from disk."""

    def __init__(
        self,
        activation_dir: str,
        token_dataset_repo: Optional[str] = None,
        token_dataset_local: Optional[str] = None,
        batch_size: int = 4,
        seq_len: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        buffer_size: int = 1000,
        val_fraction: float = 0.0,
        split: str = "train",
        hf_token: Optional[str] = None,
    ):
        """
        Args:
            activation_dir: Directory with .npy shards from precompute_activations.py.
            token_dataset_repo: HF repo with original tokens (needed for CE loss targets).
            token_dataset_local: Local path with original tokens.
            batch_size: Examples per batch.
            seq_len: Sequence length (must match precomputed activations).
            shuffle: Shuffle examples.
            seed: Random seed.
            buffer_size: Number of examples in shuffle buffer.
            val_fraction: Fraction of data for validation.
            split: "train" or "val".
            hf_token: HuggingFace token for loading token dataset.
        """
        self.activation_dir = activation_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size
        self.val_fraction = val_fraction
        self.split = split

        # Load metadata
        meta_path = os.path.join(activation_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.embed_dim = self.metadata.get("embed_dim", 1152)
            self.num_shards = self.metadata.get("num_shards", 0)
            self.total_examples = self.metadata.get("total_examples", 0)
        else:
            raise FileNotFoundError(f"No metadata.json found in {activation_dir}")

        # Discover shards
        self.shard_paths = sorted(
            glob(os.path.join(activation_dir, "shard_*.npy"))
        )
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.npy files in {activation_dir}")
        
        print(f"📦 ActivationDataLoader: {len(self.shard_paths)} shards, "
              f"~{self.total_examples:,} examples, embed_dim={self.embed_dim}")

        # Token source (for CE loss — we need the original token IDs)
        self.token_source = token_dataset_repo or token_dataset_local
        self._token_ds = None

    def _load_tokens(self):
        """Lazy-load token dataset for CE loss targets."""
        if self._token_ds is not None:
            return self._token_ds
        
        if self.token_source is None:
            print("⚠️  No token source provided — tokens will not be available in batches")
            return None

        from datasets import load_dataset, load_from_disk
        import os
        
        if os.path.isdir(self.token_source):
            self._token_ds = load_from_disk(self.token_source)
        else:
            token = self.hf_token or os.environ.get("HF_TOKEN")
            self._token_ds = load_dataset(self.token_source, split="train", token=token)
        
        print(f"📄 Token dataset loaded: {len(self._token_ds):,} examples")
        return self._token_ds

    def _shard_indices(self):
        """Return shard indices for current split."""
        n = len(self.shard_paths)
        if self.val_fraction > 0:
            n_val = max(1, int(n * self.val_fraction))
            if self.split == "val":
                return list(range(n_val))
            else:
                return list(range(n_val, n))
        return list(range(n))

    def _iter_shards(self) -> Iterator[np.ndarray]:
        """Iterate over shards, optionally shuffled."""
        indices = self._shard_indices()
        rng = np.random.default_rng(self.seed)
        
        if self.shuffle:
            rng.shuffle(indices)
        
        for idx in indices:
            shard = np.load(self.shard_paths[idx])
            # shard shape: (batch_per_shard, seq_len, embed_dim)
            yield shard

    def _iter_examples(self) -> Iterator[dict]:
        """Iterate over individual examples from all shards."""
        token_ds = self._load_tokens()
        example_idx = 0
        
        for shard in self._iter_shards():
            for i in range(shard.shape[0]):
                hidden = shard[i]  # (seq_len, embed_dim)
                
                # Determine mask from hidden (zeroed-out positions are padding)
                mask = (np.abs(hidden).sum(axis=-1) > 1e-8).astype(np.int32)
                
                result = {
                    "hidden": hidden,
                    "mask": mask,
                    "index": example_idx,
                }
                
                # Get corresponding tokens if available
                if token_ds is not None and example_idx < len(token_ds):
                    ex = token_ds[example_idx]
                    if "tokens" in ex:
                        tokens = np.array(ex["tokens"], dtype=np.int32)
                    elif "input_ids" in ex:
                        tokens = np.array(ex["input_ids"], dtype=np.int32)
                    else:
                        tokens = None
                    
                    if tokens is not None:
                        # Pad/truncate to seq_len
                        tokens = tokens[:self.seq_len]
                        pad_len = self.seq_len - len(tokens)
                        if pad_len > 0:
                            tokens = np.pad(tokens, (0, pad_len), constant_values=0)
                        result["tokens"] = tokens
                
                example_idx += 1
                yield result

    def _shuffle_buffer(self, examples_iter) -> Iterator[dict]:
        """Buffer-based shuffling."""
        rng = np.random.default_rng(self.seed + (1 if self.split == "val" else 0))
        buffer = []
        
        for ex in examples_iter:
            buffer.append(ex)
            if len(buffer) >= self.buffer_size:
                idx = rng.integers(0, len(buffer))
                yield buffer.pop(idx)
        
        # Drain buffer
        while buffer:
            idx = rng.integers(0, len(buffer))
            yield buffer.pop(idx)

    def __iter__(self) -> Iterator[dict]:
        """Yield batches of (hidden, tokens, mask)."""
        examples = self._iter_examples()
        
        if self.shuffle:
            examples = self._shuffle_buffer(examples)
        
        batch = []
        for ex in examples:
            batch.append(ex)
            if len(batch) == self.batch_size:
                yield {
                    "hidden": np.stack([b["hidden"] for b in batch]),
                    "tokens": np.stack([b["tokens"] for b in batch]) if "tokens" in batch[0] else None,
                    "mask": np.stack([b["mask"] for b in batch]),
                }
                batch = []
        
        # Yield remaining
        if batch:
            yield {
                "hidden": np.stack([b["hidden"] for b in batch]),
                "tokens": np.stack([b["tokens"] for b in batch]) if "tokens" in batch[0] else None,
                "mask": np.stack([b["mask"] for b in batch]),
            }

    def __len__(self):
        return self.total_examples // self.batch_size


# ---------------------------------------------------------------------------
# JAX-compatible tf.data version for Kauldron integration
# ---------------------------------------------------------------------------

def make_kauldron_pipeline(
    activation_dir: str,
    token_dataset_repo: Optional[str] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    seed: int = 42,
):
    """Create a Kauldron-compatible tf.data pipeline from precomputed activations.
    
    Usage:
        from activations_dataloader import make_kauldron_pipeline
        
        pipeline = make_kauldron_pipeline(
            activation_dir="./activations_layer23",
            token_dataset_repo="veriga/openwebtext-gemma3-tokenized-1024",
            batch_size=4,
        )
        # Use with Kauldron trainer
    """
    import tensorflow as tf
    
    # Load metadata
    meta_path = os.path.join(activation_dir, "metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)
    
    embed_dim = metadata["embed_dim"]
    seq_len = metadata["max_seq_len"]
    
    # Collect all shard paths
    shard_paths = sorted(glob(os.path.join(activation_dir, "shard_*.npy")))
    
    def example_generator():
        rng = np.random.default_rng(seed)
        indices = list(range(len(shard_paths)))
        if shuffle:
            rng.shuffle(indices)
        
        for idx in indices:
            shard = np.load(shard_paths[idx])
            example_indices = list(range(shard.shape[0]))
            if shuffle:
                rng.shuffle(example_indices)
            for i in example_indices:
                yield shard[i]
    
    ds = tf.data.Dataset.from_generator(
        example_generator,
        output_signature=tf.TensorSpec(shape=(seq_len, embed_dim), dtype=tf.float32),
    )
    
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=seed)
    
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds
