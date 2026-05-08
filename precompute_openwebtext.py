"""
Pre-compute tokenized OpenWebText dataset and upload to HuggingFace Hub.

This script tokenizes the Skylion007/openwebtext dataset using Gemma3Tokenizer,
truncates to max_length tokens (no padding), and uploads the result as a HuggingFace
dataset for fast loading during training.

Key features:
  - Saves tokenized dataset to disk BEFORE uploading (safe against upload failures)
  - Retry logic for upload (handles network interruptions)
  - Can skip tokenization and upload a previously saved dataset (--load_from_disk)
  - Reads HF_TOKEN from .env file automatically (no need to pass --hf_token)

Usage (Colab — full run):
    !pip install gemma datasets huggingface_hub
    huggingface-cli login
    python precompute_openwebtext.py --repo_id YOUR_USERNAME/openwebtext-gemma3-tokenized-1024

Usage (Colab — retry upload only, after a failed upload):
    python precompute_openwebtext.py --repo_id YOUR_USERNAME/openwebtext-gemma3-tokenized-1024 --load_from_disk ./tokenized_openwebtext

Usage (local test):
    python precompute_openwebtext.py --repo_id test/dataset --max_samples 1000 --push false
"""

import argparse
import os
import sys
import time


def load_env_token():
    """Read HF_TOKEN from .env file in the script's directory.

    Returns the token string, or None if not found.
    Handles quotes around the value: HF_TOKEN='...' or HF_TOKEN="..." or HF_TOKEN=...
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return None
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("HF_TOKEN="):
                value = line[len("HF_TOKEN="):].strip()
                # Remove surrounding quotes
                if (value.startswith("'") and value.endswith("'")) or \
                   (value.startswith('"') and value.endswith('"')):
                    value = value[1:-1]
                return value if value else None
    return None


def upload_with_retry(dataset, repo_id, token=None, private=True,
                      max_shard_size="500MB", max_retries=5, retry_delay=30):
    """Upload dataset to HF Hub with retry logic.

    Retries on any exception (network errors, timeouts, etc.).
    Uses exponential backoff: retry_delay * 2^attempt.
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"⬆️  Upload attempt {attempt}/{max_retries} → "
                  f"https://huggingface.co/datasets/{repo_id} ...")
            dataset.push_to_hub(
                repo_id,
                token=token,
                private=private,
                max_shard_size=max_shard_size,
            )
            print(f"✅ Upload succeeded on attempt {attempt}!")
            return True
        except KeyboardInterrupt:
            print("\n⛔ Upload interrupted by user.")
            raise
        except Exception as e:
            wait = retry_delay * (2 ** (attempt - 1))
            print(f"❌ Upload failed (attempt {attempt}/{max_retries}): "
                  f"{type(e).__name__}: {e}")
            if attempt < max_retries:
                print(f"   Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                print(f"💀 All {max_retries} upload attempts failed.")
                print(f"   Dataset is saved locally — you can retry later with --load_from_disk")
                return False
    return False


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
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./tokenized_openwebtext",
        help="Local directory to save tokenized dataset (default: './tokenized_openwebtext')",
    )
    parser.add_argument(
        "--load_from_disk",
        type=str,
        default=None,
        help="Skip tokenization and load a previously saved dataset from this path. "
             "Useful for retrying upload without re-tokenizing.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Max number of upload retry attempts (default: 5)",
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=30,
        help="Initial delay in seconds between retries, doubles each attempt (default: 30)",
    )
    args = parser.parse_args()

    # ── Resolve HF token: CLI arg > .env file > env var ──
    if args.hf_token is None:
        args.hf_token = load_env_token()
    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")
    if args.hf_token:
        # Set env var so ALL HF operations (load_dataset, push_to_hub, etc.) use it
        os.environ["HF_TOKEN"] = args.hf_token
        print(f"🔑 HF token loaded ({len(args.hf_token)} chars)")
    else:
        print("⚠️  No HF token found (set --hf_token, .env, or HF_TOKEN env var)")

    # --- Imports (after argparse so --help is fast) ---
    from datasets import load_dataset, load_from_disk, Dataset
    import numpy as np

    # ──────────────────────────────────────────────────
    # Mode 1: Load previously saved dataset (skip tokenization)
    # ──────────────────────────────────────────────────
    if args.load_from_disk is not None:
        print(f"📂 Loading previously saved dataset from: {args.load_from_disk}")
        tokenized = load_from_disk(args.load_from_disk)
        print(f"   Loaded {len(tokenized):,} examples")

        # Filter by min_length if needed
        if "length" in tokenized.column_names and args.min_length > 1:
            before = len(tokenized)
            tokenized = tokenized.filter(
                lambda x: x["length"] >= args.min_length,
                desc=f"Filtering (min_length={args.min_length})",
                num_proc=args.num_proc,
            )
            after = len(tokenized)
            print(f"   Filtered: {before:,} → {after:,} documents "
                  f"(removed {before - after:,} short docs, min_length={args.min_length})")

        # Print stats
        if "length" in tokenized.column_names:
            lengths = tokenized["length"]
            print(f"📊 Token length statistics:")
            print(f"   Count:  {len(lengths):,}")
            print(f"   Mean:   {np.mean(lengths):.1f}")
            print(f"   Median: {np.median(lengths):.1f}")
            print(f"   Total:  {sum(lengths):,} tokens")

        if args.push == "true":
            success = upload_with_retry(
                tokenized,
                args.repo_id,
                token=args.hf_token,
                private=(args.private == "true"),
                max_shard_size=args.max_shard_size,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )
            if success:
                print(f"✅ Done! Dataset available at: "
                      f"https://huggingface.co/datasets/{args.repo_id}")
            else:
                print(f"⚠️  Upload failed. Dataset is still saved at: {args.load_from_disk}")
                sys.exit(1)
        else:
            print(f"⚠️  --push is false, skipping upload.")
            print(f"   Dataset loaded from: {args.load_from_disk}")
        return

    # ──────────────────────────────────────────────────
    # Mode 2: Full tokenization pipeline
    # ──────────────────────────────────────────────────

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

    print(f"🔤 Tokenizing {len(ds)} documents (max_length={args.max_length}, "
          f"num_proc={args.num_proc})...")

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

    # ── Save to disk FIRST (before upload, so we don't lose work) ──
    save_dir = args.save_dir
    print(f"💾 Saving tokenized dataset to: {save_dir} ...")
    tokenized.save_to_disk(save_dir)
    print(f"✅ Saved successfully! Size on disk:")
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(save_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    if total_size > 1024 ** 3:
        print(f"   {total_size / (1024 ** 3):.2f} GB")
    else:
        print(f"   {total_size / (1024 ** 2):.1f} MB")
    print(f"   To retry upload later without re-tokenizing, run:")
    print(f"   python {os.path.basename(__file__)} --repo_id {args.repo_id} "
          f"--load_from_disk {os.path.abspath(save_dir)}")

    # ── Upload with retry ──
    if args.push == "true":
        success = upload_with_retry(
            tokenized,
            args.repo_id,
            token=args.hf_token,
            private=(args.private == "true"),
            max_shard_size=args.max_shard_size,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        if success:
            print(f"✅ Done! Dataset available at: "
                  f"https://huggingface.co/datasets/{args.repo_id}")
        else:
            print(f"⚠️  Upload failed. Dataset is saved locally at: {save_dir}")
            print(f"   Retry with: python {os.path.basename(__file__)} "
                  f"--repo_id {args.repo_id} --load_from_disk {os.path.abspath(save_dir)}")
            sys.exit(1)
    else:
        print(f"✅ --push is false. Dataset saved locally to {save_dir}/")
        print(f"   Load with: datasets.load_from_disk('{save_dir}')")


if __name__ == "__main__":
    main()