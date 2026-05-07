"""
Pre-compute tokenized OpenWebText dataset and upload to HuggingFace Hub.

This script tokenizes the Skylion007/openwebtext dataset using Gemma3Tokenizer,
truncates to max_length tokens (no padding), and uploads the result as a HuggingFace
dataset for fast loading during training.

Usage (Colab):
    !pip install gemma datasets huggingface_hub
    huggingface-cli login
    python precompute_openwebtext.py --repo_id YOUR_USERNAME/openwebtext-gemma3-tokenized-1024

Usage (local test):
    python precompute_openwebtext.py --repo_id test/dataset --max_samples 1000 --push false
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Tokenize OpenWebText and upload to HF Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="veriga/openwebtext-gemma3-tokenized-1024",
        help="HuggingFace dataset repo ID, e.g. 'username/openwebtext-gemma3-tokenized-1024'",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token for upload (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Truncate sequences to this many tokens (default: 1024)",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Filter out documents with fewer tokens (default: 10)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing). None = all.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of parallel workers for tokenization (default: 8)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (default: 'train')",
    )
    parser.add_argument(
        "--push",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to push to HF Hub (default: true)",
    )
    parser.add_argument(
        "--private",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Make HF dataset private (default: true)",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="500MB",
        help="Max shard size for HF upload (default: '500MB')",
    )
    args = parser.parse_args()

    # --- Imports (after argparse so --help is fast) ---
    from datasets import load_dataset, Dataset
    import numpy as np

    # Gemma tokenizer
    try:
        from gemma import gm
        tokenizer = gm.text.Gemma3Tokenizer()
    except ImportError:
        print("ERROR: gemma package not found. Install with: pip install gemma")
        sys.exit(1)

    print(f"📦 Loading OpenWebText (split='{args.split}')...")
    ds = load_dataset("Skylion007/openwebtext", split=args.split)

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"   Limited to {len(ds)} samples for testing")

    print(f"🔤 Tokenizing {len(ds)} documents (max_length={args.max_length}, num_proc={args.num_proc})...")

    def tokenize_fn(example):
        """Tokenize a single document. Returns variable-length token list."""
        text = example["text"]
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        # Tokenize with BOS token, truncate to max_length
        tokens = tokenizer.encode(text, add_bos=True)[:args.max_length]
        return {"input_ids": tokens, "length": len(tokens)}

    # Tokenize (removes 'text' column to save space)
    tokenized = ds.map(
        tokenize_fn,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=args.num_proc,
        # input_ids is variable-length list, so we don't specify features here
        # Arrow will infer the correct type
    )

    # Filter out very short documents
    before = len(tokenized)
    tokenized = tokenized.filter(
        lambda x: x["length"] >= args.min_length,
        desc=f"Filtering (min_length={args.min_length})",
        num_proc=args.num_proc,
    )
    after = len(tokenized)
    print(f"   Filtered: {before} → {after} documents (removed {before - after} short docs)")

    # Stats
    lengths = tokenized["length"]
    print(f"📊 Token length statistics:")
    print(f"   Mean:   {np.mean(lengths):.1f}")
    print(f"   Median: {np.median(lengths):.1f}")
    print(f"   P95:    {np.percentile(lengths, 95):.1f}")
    print(f"   P99:    {np.percentile(lengths, 99):.1f}")
    print(f"   Max:    {np.max(lengths)}")
    print(f"   Min:    {np.min(lengths)}")

    # Estimate size
    total_tokens = sum(lengths)
    size_gb = total_tokens * 4 / (1024 ** 3)  # int32 = 4 bytes
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Estimated size (int32, no padding): {size_gb:.1f} GB")

    if args.push == "true":
        print(f"⬆️  Uploading to https://huggingface.co/datasets/{args.repo_id} ...")
        tokenized.push_to_hub(
            args.repo_id,
            token=args.hf_token,
            private=(args.private == "true"),
            max_shard_size=args.max_shard_size,
        )
        print(f"✅ Done! Dataset available at: https://huggingface.co/datasets/{args.repo_id}")
    else:
        # Save locally for inspection
        local_path = "./tokenized_openwebtext_test"
        tokenized.save_to_disk(local_path)
        print(f"✅ Saved locally to {local_path}/")
        print(f"   Load with: datasets.load_from_disk('{local_path}')")


if __name__ == "__main__":
    main()