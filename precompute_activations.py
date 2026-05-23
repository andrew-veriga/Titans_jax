"""
Precompute layer-23 input activations from Gemma-3-1B.

Runs a forward pass through layers 0–22 with stop_gradient, saving the hidden
state tensor that feeds into layer 23. The output is a directory of .npy shards
(streamable for training) or a single large .npz file.

Why: Training TitansBlock on layer 23 with end-to-end backward through all 22
preceding Gemma layers exceeds TPU v6 HBM during XLA compilation. By
precomputing the frozen-layer activations offline, the training graph becomes
tiny — only TitansBlock + head — and fits easily in HBM.

Usage:
    # Basic — precompute for dataset of 1024-token sequences
    python precompute_activations.py \
        --gemma_ckpt /path/to/gemma-3-1b-pt \
        --dataset_repo veriga/openwebtext-gemma3-tokenized-1024 \
        --output_dir ./activations_layer23 \
        --target_layer 23 \
        --batch_size 8 \
        --max_seq_len 1024

    # Resume interrupted run
    python precompute_activations.py \
        --gemma_ckpt /path/to/gemma-3-1b-pt \
        --dataset_repo veriga/openwebtext-gemma3-tokenized-1024 \
        --output_dir ./activations_layer23 \
        --resume

    # Use local tokenized dataset
    python precompute_activations.py \
        --gemma_ckpt /path/to/gemma-3-1b-pt \
        --local_dataset ./tokenized_openwebtext \
        --output_dir ./activations_layer23

    # Limit number of examples (for testing)
    python precompute_activations.py \
        ... \
        --max_examples 1000

Output structure:
    activations_layer23/
    ├── shard_000000.npy   # shape (N, seq_len, 1152), dtype float32
    ├── shard_000001.npy
    ├── ...
    └── metadata.json      # config used, shard sizes, etc.

Each .npy shard contains `batch_size` examples.
Shard size = batch_size × seq_len × 1152 × 4 bytes ≈ 36 MB per shard (bs=8, len=1024).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute frozen-layer activations for Titans training"
    )
    # --- Gemma checkpoint ---
    p.add_argument(
        "--gemma_ckpt",
        type=str,
        required=True,
        help="Path to official Gemma-3-1B checkpoint directory",
    )

    # --- Dataset source (mutually exclusive) ---
    ds = p.add_mutually_exclusive_group(required=True)
    ds.add_argument(
        "--dataset_repo",
        type=str,
        default=None,
        help="HuggingFace repo with pre-tokenized dataset (e.g. veriga/openwebtext-gemma3-tokenized-1024)",
    )
    ds.add_argument(
        "--local_dataset",
        type=str,
        default=None,
        help="Local directory with pre-tokenized dataset (saved via datasets.save_to_disk)",
    )

    # --- Output ---
    p.add_argument(
        "--output_dir",
        type=str,
        default="./activations_layer23",
        help="Output directory for .npy shards (default: ./activations_layer23)",
    )

    # --- Model config ---
    p.add_argument(
        "--target_layer",
        type=int,
        default=23,
        help="Target Titans layer index; activations are the input to this layer (default: 23)",
    )
    p.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Pad/truncate sequences to this length (must match training) (default: 1024)",
    )

    # --- Performance ---
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for forward pass (default: 8)",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Computation dtype for forward pass (default: bfloat16)",
    )
    p.add_argument(
        "--save_dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Dtype for saved activations (default: float32 — safer for training)",
    )

    # --- Limits ---
    p.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit total examples processed (default: all)",
    )

    # --- Resume ---
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed shard in output_dir",
    )

    # --- HF token ---
    p.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token")

    return p.parse_args()


# ---------------------------------------------------------------------------
# JAX forward pass: runs through layers 0..(target_layer-1), returns hidden
# state at the input of target_layer. No gradients, no backward graph.
# ---------------------------------------------------------------------------

def make_forward_to_layer(target_layer: int):
    """
    Returns a jitted function that runs Gemma-3-1B forward pass from token
    embeddings up to (but not including) `target_layer`, then applies
    final_norm ONLY if target_layer == num_layers (i.e. output of last layer).

    For target_layer=23 in a 26-layer model, this runs layers 0–22 and returns
    the hidden state that would be the input to layer 23.

    The returned function is:
        fn(params, tokens) -> hidden_state  # shape (B, L, embed_dim)
    """
    from gemma import gm

    # Load model architecture (no weights yet)
    model = gm.nn.Gemma3_1B()

    @jax.jit
    def forward_fn(params, tokens):
        """Forward pass through layers 0 to target_layer-1.
        
        Uses model.__call__ with return_hidden_states=True to get all
        intermediate hidden states, then extracts the one at target_layer input.
        """
        # We use the model's internal _forward to get embeddings + run blocks
        # But we need access to intermediate layer outputs.
        # Strategy: use model.apply with modified config to stop at target_layer.
        
        # Actually, the simplest approach with gemma.gm is to run the full model
        # with return_hidden_states and extract what we need.
        # The hidden_states output contains [embedding_output, layer_0_out, ..., layer_25_out, final_norm_out]
        
        output = model.apply(
            {'params': params},
            tokens,
            return_hidden_states=True,
        )
        # output.hidden_states is a tuple of length num_layers+2:
        #   [embeddings, layer_0_output, layer_1_output, ..., layer_25_output, final_norm_output]
        # For target_layer=23, we want the INPUT to layer 23 = OUTPUT of layer 22
        # which is hidden_states[23] (index: 1 for embeddings + 22 for layer 22 = 23)
        return output.hidden_states[target_layer]

    return forward_fn


# ---------------------------------------------------------------------------
# Alternative: manual forward pass with explicit layer iteration
# More control, works even if return_hidden_states API changes
# ---------------------------------------------------------------------------

def make_manual_forward_to_layer(target_layer: int):
    """
    Manual forward: embed → layers 0..target_layer-1.
    Uses gemma.gm internals but stops early.
    """
    from gemma import gm

    model = gm.nn.Gemma3_1B()
    
    @jax.jit
    def forward_fn(params, tokens):
        # Access embedder and blocks directly through model.apply
        # We need to do this inside a custom forward function
        
        # Use the model's __call__ but intercept hidden states
        output = model.apply(
            {'params': params},
            tokens,
            return_hidden_states=True,
        )
        
        # hidden_states[0] = embeddings
        # hidden_states[i+1] = output of layer i
        # We want output of layer (target_layer - 1) = input to target_layer
        return output.hidden_states[target_layer]
    
    return forward_fn


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_tokenized_dataset(
    repo_id: Optional[str] = None,
    local_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    max_seq_len: int = 1024,
):
    """Load pre-tokenized dataset and yield batches of token arrays."""

    if local_path is not None:
        from datasets import load_from_disk
        ds = load_from_disk(local_path)
        print(f"📂 Loaded local dataset: {len(ds):,} examples from {local_path}")
    else:
        from datasets import load_dataset
        token = hf_token or os.environ.get("HF_TOKEN")
        ds = load_dataset(repo_id, split="train", token=token)
        print(f"📂 Loaded HF dataset: {len(ds):,} examples from {repo_id}")

    return ds


def dataset_to_batches(ds, batch_size: int, max_seq_len: int, max_examples: Optional[int] = None):
    """Convert dataset to batched numpy arrays of tokens."""
    
    total = len(ds)
    if max_examples is not None:
        total = min(total, max_examples)
    
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_examples = ds[start:end]
        
        # Extract tokens (handle both "input_ids" and "tokens" column names)
        if "tokens" in batch_examples:
            raw_tokens = batch_examples["tokens"]
        elif "input_ids" in batch_examples:
            raw_tokens = batch_examples["input_ids"]
        else:
            raise ValueError(f"Dataset has neither 'tokens' nor 'input_ids' column. "
                           f"Available: {list(batch_examples.keys())}")
        
        # Pad/truncate each example
        batch_tokens = []
        batch_masks = []
        for tokens in raw_tokens:
            if isinstance(tokens, list):
                tokens = np.array(tokens, dtype=np.int32)
            original_len = min(len(tokens), max_seq_len)
            tokens = tokens[:max_seq_len]
            pad_len = max_seq_len - len(tokens)
            if pad_len > 0:
                tokens = np.pad(tokens, (0, pad_len), constant_values=0)
            mask = np.zeros(max_seq_len, dtype=np.int32)
            mask[:original_len] = 1
            batch_tokens.append(tokens)
            batch_masks.append(mask)
        
        yield np.stack(batch_tokens), np.stack(batch_masks)


# ---------------------------------------------------------------------------
# Main precomputation loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    dtype_map = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}
    compute_dtype = dtype_map[args.dtype]
    save_dtype = np.float32 if args.save_dtype == "float32" else (
        np.float16 if args.save_dtype == "float16" else jnp.bfloat16
    )

    # --- Load Gemma ---
    print("🔧 Loading Gemma-3-1B model and parameters...")
    from gemma import gm
    
    params = gm.ckpts.load_params(args.gemma_ckpt)
    # Convert to compute dtype
    params = jax.tree_util.tree_map(
        lambda x: x.astype(compute_dtype) if hasattr(x, 'astype') else x,
        params
    )
    print(f"   Parameters loaded, dtype={args.dtype}")

    # --- Build forward function ---
    forward_fn = make_forward_to_layer(args.target_layer)
    
    # Warmup: compile with dummy input
    print("🔥 Compiling forward pass (JIT warmup)...")
    dummy_tokens = jnp.zeros((1, args.max_seq_len), dtype=jnp.int32)
    t0 = time.time()
    _ = forward_fn(params, dummy_tokens)
    compile_time = time.time() - t0
    print(f"   Compiled in {compile_time:.1f}s")

    # Verify output shape
    test_out = forward_fn(params, jnp.ones((1, 4), dtype=jnp.int32))
    embed_dim = test_out.shape[-1]
    print(f"   Output shape: (B, L, {embed_dim})")

    # --- Load dataset ---
    ds = load_tokenized_dataset(
        repo_id=args.dataset_repo,
        local_path=args.local_dataset,
        hf_token=args.hf_token,
        max_seq_len=args.max_seq_len,
    )

    # --- Resume logic ---
    start_shard = 0
    if args.resume:
        metadata_path = os.path.join(args.output_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            start_shard = meta.get("next_shard", 0)
            print(f"📋 Resuming from shard {start_shard}")
        else:
            print("📋 No metadata found, starting from scratch")

    # --- Process batches ---
    total_examples = 0
    shard_idx = start_shard
    t_start = time.time()

    print(f"\n🚀 Processing batches (batch_size={args.batch_size})...")
    batch_gen = dataset_to_batches(ds, args.batch_size, args.max_seq_len, args.max_examples)

    # Skip already-processed batches if resuming
    for _ in range(start_shard):
        next(batch_gen, None)

    for batch_tokens, batch_masks in batch_gen:
        # Forward pass
        hidden = forward_fn(params, jnp.array(batch_tokens))
        
        # Convert to numpy and save
        hidden_np = np.array(hidden.astype(jnp.float32))
        
        # Apply mask: zero out padding positions
        mask_expanded = batch_masks[:, :, None].astype(np.float32)  # (B, L, 1)
        hidden_np = hidden_np * mask_expanded
        
        # Save shard
        shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:06d}.npy")
        np.save(shard_path, hidden_np.astype(save_dtype))
        
        shard_idx += 1
        total_examples += len(batch_tokens)

        # Progress
        elapsed = time.time() - t_start
        examples_per_sec = total_examples / max(elapsed, 0.001)
        shard_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        
        print(
            f"  Shard {shard_idx:5d} | "
            f"{total_examples:7d} examples | "
            f"{examples_per_sec:6.1f} ex/s | "
            f"{shard_size_mb:5.1f} MB/shard | "
            f"{elapsed:6.1f}s elapsed",
            flush=True,
        )

        # Update metadata (for resume)
        metadata = {
            "target_layer": args.target_layer,
            "max_seq_len": args.max_seq_len,
            "embed_dim": embed_dim,
            "batch_size": args.batch_size,
            "save_dtype": args.save_dtype,
            "compute_dtype": args.dtype,
            "total_examples": total_examples,
            "num_shards": shard_idx,
            "next_shard": shard_idx,
            "source_repo": args.dataset_repo,
            "source_local": args.local_dataset,
            "gemma_ckpt": args.gemma_ckpt,
        }
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    # --- Final summary ---
    elapsed = time.time() - t_start
    total_size = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in os.listdir(args.output_dir)
        if f.endswith(".npy")
    )
    
    print(f"\n✅ Done!")
    print(f"   Total examples:  {total_examples:,}")
    print(f"   Total shards:    {shard_idx}")
    print(f"   Total size:      {total_size / (1024**3):.2f} GB")
    print(f"   Time:            {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Throughput:      {total_examples / max(elapsed, 0.001):.1f} examples/s")
    print(f"   Output:          {args.output_dir}/")


if __name__ == "__main__":
    main()
