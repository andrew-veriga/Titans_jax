"""
Phase 3: Train TitansBlock on precomputed activations with CrossEntropy loss.

This is the minimal training loop for a single TitansBlock (layer 23)
using precomputed hidden states from layers 0–22 of Gemma-3-1B.

No backward through 22 Gemma layers needed — XLA graph is tiny.

Architecture:
    precomputed_hidden (B, L, 1152)
         ↓
    TitansBlock (layer 23) — ONLY this is trained
         ↓  (memory gate + retrieved + residual + MLP)
    Gemma layers 24–25 + final_norm + head  — FROZEN
         ↓
    logits → CrossEntropy loss

Usage:
    python train_layer23.py \
        --gemma_ckpt /path/to/gemma-3-1b-pt \
        --activation_repo veriga/openwebtext-gemma3-tokenized-1024-activations-layer23 \
        --output_dir ./checkpoints_layer23 \
        --num_steps 10000 \
        --batch_size 4

For Kauldron integration, see the KauldronTrainer class at the bottom.
"""

import argparse
import dataclasses
import json
import os
import time
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax.core import freeze, unfreeze

# ---------------------------------------------------------------------------
# TitansBlock wrapper for training on precomputed activations
# ---------------------------------------------------------------------------


def build_titans_block_config(
    target_layer: int = 23,
    experimental_config: dict = None,
    every_k_schedule: int = 4,
):
    """Build neural_mem_kwargs and config from experimental_config."""
    from gemma_titans import Gemma3_1B_Titans

    ec = experimental_config or {}

    neural_mem_kwargs = {
        'heads': ec.get('heads', 8),
        'dim_head': ec.get('dim_head', 128),
        'chunk_size': ec.get('chunk_size', 32),
        'mlp_depth': ec.get('mlp_depth', 6),
        'max_grad_norm': ec.get('max_grad_norm', 0.5),
        'elastic_net_lambda': ec.get('elastic_net_lambda', 0.01),
        'diff_view': ec.get('diff_view', False),
        'is_look_ahead': ec.get('is_look_ahead', False),
        'huber_loss_delta': ec.get('huber_loss_delta', None),
        'adaptive_max_lr': ec.get('adaptive_max_lr', 1e-4),
        'every_k_schedule': every_k_schedule,
    }

    config = dataclasses.replace(
        Gemma3_1B_Titans.config,
        training_phase=3,
        titans_layer_indices=[target_layer],
        titans_first_layer=target_layer,
        neural_mem_kwargs=neural_mem_kwargs,
    )

    return config


def _build_block_kwargs(config, layer_idx):
    """Build Flax Block kwargs from TransformerConfig for a given layer."""
    from gemma.gm.nn import _modules
    attn_type = config.attention_types[layer_idx]
    is_local = attn_type == _modules.AttentionType.LOCAL_SLIDING
    return dict(
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        embed_dim=config.embed_dim,
        head_dim=config.head_dim,
        hidden_dim=config.hidden_dim,
        sliding_window_size=config.sliding_window_size,
        use_post_attn_norm=config.use_post_attn_norm,
        use_post_ffw_norm=config.use_post_ffw_norm,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        attn_type=attn_type,
        query_pre_attn_scalar=config.query_pre_attn_scalar(),
        transpose_gating_einsum=config.transpose_gating_einsum,
        use_qk_norm=config.use_qk_norm,
        rope_base_frequency=config.local_base_frequency if is_local else config.global_base_frequency,
        rope_scale_factor=config.local_scale_factor if is_local else config.global_scale_factor,
    )


# ---------------------------------------------------------------------------
# HuggingFace Activation Loader
# ---------------------------------------------------------------------------

class HFActivationLoader:
    """
    Streams precomputed activation shards from HuggingFace Hub or local disk.

    When ``local_activation_dir`` is provided, reads .npy shards directly
    from the local directory — no HF download, maximum speed.

    When ``local_activation_dir`` is None, loads .npy shards from a folder
    inside an HF dataset repository
    (e.g. ``veriga/openwebtext-gemma3-tokenized-1024/activations_layer23``)
    and pairs them with the corresponding original token IDs from the same
    repo for next-token CrossEntropy targets.

    Features:
      - **Local mode**: set ``local_activation_dir`` to skip HF entirely.
      - Streaming: shards are loaded on-demand (no full load into RAM).
      - Token pairing: each activation batch is paired with original tokens
        from the token dataset for CE loss computation.
      - Shuffle buffer: example-level shuffling via an in-memory buffer.

    Usage (remote)::

        loader = HFActivationLoader(
            activation_repo="veriga/openwebtext-gemma3-tokenized-1024",
            activation_folder="activations_layer23",
            batch_size=4,
        )

    Usage (local — fast, no HF download)::

        loader = HFActivationLoader(
            activation_repo="veriga/openwebtext-gemma3-tokenized-1024",
            activation_folder="activations_layer23",
            local_activation_dir="./activations_layer23",
            batch_size=4,
        )

        for batch in loader:
            hidden = batch["hidden"]   # (B, L, 1152)
            tokens = batch["tokens"]   # (B, L)
            mask   = batch["mask"]     # (B, L)
    """

    def __init__(
        self,
        activation_repo: str,
        activation_folder: str = "",
        local_activation_dir: Optional[str] = None,
        batch_size: int = 4,
        seq_len: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        buffer_size: int = 1000,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            activation_repo: HuggingFace dataset repo containing activation
                shards and tokens (e.g.
                ``"veriga/openwebtext-gemma3-tokenized-1024"``).
            activation_folder: Folder inside the repo where .npy shards and
                metadata.json live (default: ``"activations_layer23"``).
            local_activation_dir: **If set**, read shards and metadata directly
                from this local directory — no HF download needed. This is
                ideal for fast testing when you already ran
                ``precompute_activations.py`` locally. When set, only tokens
                come from HF (for CE loss targets).
            batch_size: Number of examples per training batch.
            seq_len: Sequence length — must match precomputed activations.
            shuffle: Whether to shuffle examples across shards.
            seed: Random seed for shuffling.
            buffer_size: Shuffle-buffer size (in examples).
            hf_token: HuggingFace API token (falls back to ``$HF_TOKEN``).
            cache_dir: Local directory for caching HF downloads (tokens only
                when using local_activation_dir).
        """
        self.activation_repo = activation_repo
        self.activation_folder = activation_folder
        self.local_activation_dir = local_activation_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.cache_dir = cache_dir

        # --- Discover shards ---
        if local_activation_dir:
            # Local mode: read directly from disk
            from glob import glob

            # Only match shard_XXXXXX.npy, NOT shard_XXXXXX_tokens.npy or _masks.npy
            import re
            all_npy = glob(os.path.join(local_activation_dir, "shard_*.npy"))
            self.shard_paths_local = sorted(
                p for p in all_npy
                if re.match(r"shard_\d+\.npy$", os.path.basename(p))
            )
            if not self.shard_paths_local:
                raise FileNotFoundError(
                    f"No shard_*.npy files in {local_activation_dir}"
                )
            self.shard_names = [os.path.basename(p) for p in self.shard_paths_local]
            self._local_mode = True

            # Load metadata from local file
            meta_path = os.path.join(local_activation_dir, "metadata.json")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"No metadata.json in {local_activation_dir}")
            with open(meta_path) as f:
                self.metadata = json.load(f)

            print(
                f"📦 HFActivationLoader [LOCAL]: {len(self.shard_names)} shards "
                f"from {local_activation_dir}"
            )
        else:
            # Remote mode: discover & download from HuggingFace
            from huggingface_hub import HfApi

            api = HfApi(token=self.hf_token)
            all_files = api.list_repo_files(
                repo_id=activation_repo,
                repo_type="dataset",
            )

            import re
            if activation_folder:
                prefix = activation_folder + "/"
            else:
                prefix = ""  # files at repo root

            all_npy = [
                p for p in all_files
                if p.startswith(prefix) and p.endswith(".npy")
                and ("/" not in p[len(prefix):])  # no subfolders
            ]
            # ALL .npy files in the folder (for token/mask lookup)
            self._all_npy_names = set(p[len(prefix):] for p in all_npy)
            # Only activation shards (shard_XXXXXX.npy), not _tokens/_masks
            shard_paths = sorted(
                p for p in all_npy
                if re.match(r"shard_\d+\.npy$", p[len(prefix):])
            )
            self.shard_names = [p[len(prefix):] for p in shard_paths]

            if not self.shard_names:
                matching = [f for f in all_files if f.startswith(prefix)]
                raise FileNotFoundError(
                    f"No .npy shards in {activation_repo}"
                    f"/{activation_folder}. "
                    f"Files there: {matching[:10]}"
                )

            self._local_mode = False
            self.shard_paths_local = None

            # Download metadata from HF
            self.metadata = self._load_metadata()

            print(
                f"📦 HFActivationLoader [HF]: {len(self.shard_names)} shards "
                f"from {activation_repo}/{activation_folder}"
            )

        self.embed_dim = self.metadata.get("embed_dim", 1152)
        self.total_examples = self.metadata.get("total_examples", 0)
        self._shard_batch_size = self.metadata.get("batch_size", 8)

        print(
            f"   ~{self.total_examples:,} examples, embed_dim={self.embed_dim}"
        )

        # --- Token dataset (lazy) ---
        self._token_ds = None

    # ---- internal helpers ------------------------------------------------

    def _hf_path(self, filename: str) -> str:
        """Full repo-internal path for a file in the activation folder."""
        if self.activation_folder:
            return f"{self.activation_folder}/{filename}"
        return filename

    def _load_metadata(self) -> dict:
        """Download metadata.json from the HF repo."""
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=self.activation_repo,
            filename=self._hf_path("metadata.json"),
            repo_type="dataset",
            token=self.hf_token,
            cache_dir=self.cache_dir,
        )
        with open(local_path) as f:
            return json.load(f)

    def _download_shard(self, shard_name: str) -> str:
        """Download a single .npy shard, return the local path."""
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=self.activation_repo,
            filename=self._hf_path(shard_name),
            repo_type="dataset",
            token=self.hf_token,
            cache_dir=self.cache_dir,
        )

    def _download_shard_bytes(self, shard_name: str) -> bytes:
        """Download a shard into memory (no disk cache for this call)."""
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=self.activation_repo,
            filename=self._hf_path(shard_name),
            repo_type="dataset",
            token=self.hf_token,
            cache_dir=self.cache_dir,
        )
        with open(local, "rb") as f:
            return f.read()

    def _load_token_dataset(self):
        """Lazy-load the token dataset from the same HF repo."""
        if self._token_ds is not None:
            return self._token_ds

        from datasets import load_dataset

        self._token_ds = load_dataset(
            self.activation_repo,
            split="train",
            token=self.hf_token,
            cache_dir=self.cache_dir,
        )
        print(f"📄 Token dataset loaded: {len(self._token_ds):,} examples")
        return self._token_ds

    # ---- iteration -------------------------------------------------------

    def _iter_shards(self) -> Iterator[dict]:
        """
        Yield shards as dicts {hidden, tokens, masks}, optionally shuffled.

        In local mode: reads shard_XXXXXX.npy, shard_XXXXXX_tokens.npy,
        shard_XXXXXX_masks.npy from disk.
        In remote mode: downloads .npy from HF; falls back to HF token
        dataset for tokens if _tokens.npy files are not available.
        """
        rng = np.random.default_rng(self.seed)
        indices = list(range(len(self.shard_names)))

        if self.shuffle:
            rng.shuffle(indices)

        for idx in indices:
            if self._local_mode:
                base_path = self.shard_paths_local[idx]
                hidden = np.load(base_path)

                # Derive token/mask paths from activation shard name
                dir_path = os.path.dirname(base_path)
                base_name = os.path.splitext(os.path.basename(base_path))[0]
                tokens_path = os.path.join(dir_path, f"{base_name}_tokens.npy")
                masks_path = os.path.join(dir_path, f"{base_name}_masks.npy")

                if os.path.exists(tokens_path):
                    tokens = np.load(tokens_path)
                else:
                    tokens = None

                if os.path.exists(masks_path):
                    masks = np.load(masks_path)
                else:
                    # Fall back: derive mask from hidden (nonzero rows)
                    masks = (np.abs(hidden).sum(axis=-1) > 1e-8).astype(np.int32)

                yield {"hidden": hidden, "tokens": tokens, "masks": masks}

            else:
                name = self.shard_names[idx]
                local_path = self._download_shard(name)
                hidden = np.load(local_path)

                # Try to download co-located tokens/masks
                base = os.path.splitext(name)[0]
                tokens_name = f"{base}_tokens.npy"
                masks_name = f"{base}_masks.npy"

                tokens = None
                masks = (np.abs(hidden).sum(axis=-1) > 1e-8).astype(np.int32)

                # Check if _tokens.npy exists in repo
                if tokens_name in self._all_npy_names:
                    try:
                        t_path = self._download_shard(tokens_name)
                        tokens = np.load(t_path)
                        m_path = self._download_shard(masks_name)
                        masks = np.load(m_path)
                    except Exception:
                        pass

                yield {"hidden": hidden, "tokens": tokens, "masks": masks}

    def _iter_examples(self) -> Iterator[dict]:
        """Yield individual examples across all shards."""
        for shard_data in self._iter_shards():
            hidden_shard = shard_data["hidden"]
            tokens_shard = shard_data["tokens"]
            masks_shard = shard_data["masks"]

            for i in range(hidden_shard.shape[0]):
                result = {
                    "hidden": hidden_shard[i],
                    "mask": masks_shard[i],
                }
                if tokens_shard is not None:
                    result["tokens"] = tokens_shard[i]

                yield result

    def _shuffle_buffer(self, it: Iterator[dict]) -> Iterator[dict]:
        """Example-level shuffle buffer."""
        rng = np.random.default_rng(self.seed)
        buf: list = []

        for ex in it:
            buf.append(ex)
            if len(buf) >= self.buffer_size:
                yield buf.pop(rng.integers(0, len(buf)))

        while buf:
            yield buf.pop(rng.integers(0, len(buf)))

    def __iter__(self) -> Iterator[dict]:
        """Yield batches ``{hidden, tokens, mask}`` ready for training."""
        examples = self._iter_examples()

        if self.shuffle:
            examples = self._shuffle_buffer(examples)

        batch: list = []
        for ex in examples:
            batch.append(ex)
            if len(batch) == self.batch_size:
                yield {
                    "hidden": np.stack([b["hidden"] for b in batch]),
                    "tokens": (
                        np.stack([b["tokens"] for b in batch])
                        if "tokens" in batch[0]
                        else None
                    ),
                    "mask": np.stack([b["mask"] for b in batch]),
                }
                batch = []

        if batch:
            yield {
                "hidden": np.stack([b["hidden"] for b in batch]),
                "tokens": (
                    np.stack([b["tokens"] for b in batch])
                    if "tokens" in batch[0]
                    else None
                ),
                "mask": np.stack([b["mask"] for b in batch]),
            }

    def __len__(self) -> int:
        return self.total_examples // self.batch_size


# ---------------------------------------------------------------------------
# Training step: TitansBlock forward + frozen Gemma head + CE loss
# ---------------------------------------------------------------------------

def make_train_step(
    titans_block,
    block_24,
    block_25,
    final_norm_module,
    frozen_params_24,
    frozen_params_25,
    frozen_final_norm_params,
    embedding_table,
    optimizer,
    neural_mem_kwargs,
):
    """
    Create a jitted training step.

    Architecture:
        hidden (B, L, 1152)
          → TitansBlock.apply(tp, ...)        [TRAINABLE]
          → Block_24.apply(frozen, ...)        [FROZEN]
          → Block_25.apply(frozen, ...)        [FROZEN]
          → RMSNorm.apply(frozen, ...)         [FROZEN]
          → dot(x, embedding_table.T) → logits [FROZEN]
          → CrossEntropy loss (shifted by 1)

    Only TitansBlock params receive gradients.
    Frozen params are closed-over constants (baked into XLA graph).
    Logit computation is @jax.checkpoint-ed to avoid materializing
    the full (B, L-1, 262144) tensor during backward.

    Returns:
        (new_titans_params, new_opt_state, loss_scalar, accuracy_scalar)
    """
    from titans import init_memory_state

    fp24 = frozen_params_24
    fp25 = frozen_params_25
    fnp = frozen_final_norm_params

    @jax.checkpoint
    def _compute_ce_loss_and_acc(x, targets, loss_mask):
        """Compute CE loss + accuracy from hidden states.

        Checkpointed so the huge (B, L-1, V) logit tensor is freed
        after forward and rematerialized on-demand during backward.
        Gemma3-1B has final_logit_softcap=None, so no softcap needed.
        """
        # x: (B, L, D) → logits: (B, L-1, V)
        logits = jnp.dot(x[:, :-1, :], embedding_table.T)
        ce = optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), targets
        )
        loss = (ce * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0)
        # Accuracy
        pred = jnp.argmax(logits, axis=-1)
        correct = (pred == targets).astype(jnp.float32)
        acc = (correct * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0)
        return loss, acc

    @jax.jit
    def train_step(titans_params, opt_state, hidden, tokens, mask):
        """
        Args:
            titans_params: TitansBlock parameters (trainable).
            opt_state: Optimizer state.
            hidden: Precomputed activations (B, L, 1152).
            tokens: Token IDs (B, L) — for CE target (shifted by 1).
            mask: Input mask (B, L) — 1 for real tokens, 0 for padding.
        Returns:
            (new_titans_params, new_opt_state, loss_scalar, accuracy_scalar)
        """
        B, L, D = hidden.shape

        # Create positions: (B, L)
        positions = jnp.broadcast_to(jnp.arange(L)[None, :], (B, L))

        # Create causal attention mask: (B, L, L)
        # Token i can attend to token j iff j <= i AND j is a real token.
        causal = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))
        attn_mask = causal[None, :, :] & mask[:, None, :].astype(jnp.bool_)

        # Fresh memory state for each batch (one-shot prefill)
        mem_state = init_memory_state(B, D, neural_mem_kwargs, dtype=hidden.dtype)

        def loss_fn(tp):
            # 1. TitansBlock (layer 23) — TRAINABLE
            cache_23 = {'memory_state': mem_state}
            _, x = titans_block.apply(
                {'params': tp}, hidden, positions, cache_23, attn_mask,
            )

            # 2. Block 24 — FROZEN (gradients still flow through for chain rule)
            _, x = block_24.apply(fp24, x, positions, None, attn_mask)

            # 3. Block 25 — FROZEN
            _, x = block_25.apply(fp25, x, positions, None, attn_mask)

            # 4. Final RMSNorm — FROZEN
            x = final_norm_module.apply(fnp, x)

            # 5. CE loss + accuracy (shifted by 1: predict next token)
            targets = tokens[:, 1:]
            loss_mask = mask[:, 1:].astype(jnp.float32)
            return _compute_ce_loss_and_acc(x, targets, loss_mask)

        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(titans_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, titans_params)
        new_params = optax.apply_updates(titans_params, updates)

        return new_params, new_opt_state, loss, acc

    return train_step


# ---------------------------------------------------------------------------
# Standalone Trainer
# ---------------------------------------------------------------------------

class Layer23Trainer:
    """
    Standalone trainer for TitansBlock on layer 23.

    Streams precomputed activations from HuggingFace Hub, runs them through
    the trainable TitansBlock, then through frozen Gemma layers 24-25 + head,
    and computes CrossEntropy loss on next-token prediction.

    Architecture:
        precomputed_hidden (B, L, 1152)
             ↓
        TitansBlock (layer 23) — ONLY this is trained
             ↓  (memory gate + retrieved + residual + MLP)
        Gemma layers 24–25 + final_norm + head  — FROZEN
             ↓
        logits → CrossEntropy loss
    """

    TARGET_LAYER = 23

    def __init__(
        self,
        gemma_ckpt_path: str,
        optimizer,
        experimental_config: dict,
        activation_repo: str = "veriga/openwebtext-gemma3-tokenized-1024-activations-layer23",
        activation_folder: str = "",
        local_activation_dir: Optional[str] = None,
        output_dir: str = "./checkpoints_layer23",
        batch_size: int = 4,
        seed: int = 42,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        from gemma import gm
        from gemma.gm.nn import _modules, _layers
        from gemma_titans import TitansBlock

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.seed = seed
        os.makedirs(output_dir, exist_ok=True)

        # ---- 1. Load full Gemma params ----
        print("🔧 Loading Gemma-3-1B parameters...")
        gemma_ckpt_path = os.path.abspath(gemma_ckpt_path)
        full_params = gm.ckpts.load_params(gemma_ckpt_path)
        param_keys = sorted(full_params.get('params', {}).keys())
        print(f"   Top-level param keys: {param_keys}")

        # ---- 2. Build config from experimental_config ----
        config = build_titans_block_config(
            target_layer=self.TARGET_LAYER,
            experimental_config=experimental_config,
        )
        self.config = config

        # ---- 3. Create TitansBlock instance ----
        print("🏗️  Creating TitansBlock instance...")
        titans_block = TitansBlock(
            name=f'layer_{self.TARGET_LAYER}',
            **_build_block_kwargs(config, self.TARGET_LAYER),
            neural_mem_kwargs=config.neural_mem_kwargs,
            use_original_attn=False,  # Phase 3: pure memory, no attention
        )

        # ---- 4. Initialize TitansBlock params ----
        # Need L_init >= chunk_size so NeuralMemory fully initializes
        # (store_memories + retrieve_memories exercise all sub-modules).
        print("🎯 Initializing TitansBlock parameters...")
        key = jax.random.PRNGKey(seed)
        chunk_sz = config.neural_mem_kwargs.get('chunk_size', 32)
        L_init = max(chunk_sz, 32)
        D = config.embed_dim

        dummy_x = jnp.zeros((1, L_init, D), dtype=jnp.bfloat16)
        dummy_pos = jnp.broadcast_to(jnp.arange(L_init)[None, :], (1, L_init))
        dummy_mask = jnp.tril(
            jnp.ones((1, L_init, L_init), dtype=jnp.bool_)
        )

        init_vars = titans_block.init(key, dummy_x, dummy_pos, None, dummy_mask)
        titans_params = dict(unfreeze(init_vars['params']))

        # ---- 5. Warm-start shared params from Gemma checkpoint ----
        # The TitansBlock has the same architecture as a regular Gemma Block
        # for norms and MLP. Overwrite these with Gemma's pretrained values
        # for faster convergence. Only memory-specific params (NeuralMemory,
        # memory_gate_proj) keep their random init.
        print("📦 Warm-starting shared params from Gemma layer 23...")
        gemma_layer_key = f'layer_{self.TARGET_LAYER}'
        if gemma_layer_key not in full_params.get('params', {}):
            raise KeyError(
                f"Layer '{gemma_layer_key}' not found in Gemma checkpoint. "
                f"Available: {param_keys}"
            )
        gemma_layer = full_params['params'][gemma_layer_key]
        warm_start_keys = [
            'pre_attention_norm', 'post_attention_norm',
            'pre_ffw_norm', 'mlp', 'post_ffw_norm',
        ]
        for k in warm_start_keys:
            if k in gemma_layer:
                titans_params[k] = gemma_layer[k]
                print(f"   ✓ {k}: loaded from Gemma checkpoint")
            else:
                print(f"   ⚠ {k}: not found in checkpoint, using random init")

        self.titans_params = freeze(titans_params)

        # Print trainable param count
        param_count = sum(
            x.size for x in jax.tree_util.tree_leaves(self.titans_params)
        )
        print(f"   Trainable params: {param_count:,}")

        # ---- 6. Create frozen Gemma blocks ----
        print("❄️  Creating frozen Gemma blocks (layers 24, 25)...")
        block_24 = _modules.Block(
            name='layer_24',
            **_build_block_kwargs(config, 24),
        )
        block_25 = _modules.Block(
            name='layer_25',
            **_build_block_kwargs(config, 25),
        )

        # ---- 7. Extract frozen params ----
        for req_key in ['layer_24', 'layer_25', 'final_norm', 'embedder']:
            if req_key not in full_params.get('params', {}):
                raise KeyError(
                    f"Required key '{req_key}' not found in Gemma checkpoint."
                )

        frozen_params_24 = {'params': full_params['params']['layer_24']}
        frozen_params_25 = {'params': full_params['params']['layer_25']}

        final_norm = _layers.RMSNorm()
        frozen_final_norm_params = {'params': full_params['params']['final_norm']}

        embedding_table = full_params['params']['embedder']['input_embedding']
        print(f"   Embedding table: {embedding_table.shape}")
        print(
            f"   Frozen Block 24 attn_type: "
            f"{config.attention_types[24].name}"
        )
        print(
            f"   Frozen Block 25 attn_type: "
            f"{config.attention_types[25].name}"
        )

        # Free full checkpoint to save memory
        del full_params

        # ---- 8. Routing optimizer (M3 + Adam-atan2) ----
        print("⚙️  Setting up routing optimizer...")
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(self.titans_params)

        # ---- 9. Create train step ----
        print("🔧 Building train step (JIT compile on first call)...")
        self._train_step = make_train_step(
            titans_block=titans_block,
            block_24=block_24,
            block_25=block_25,
            final_norm_module=final_norm,
            frozen_params_24=frozen_params_24,
            frozen_params_25=frozen_params_25,
            frozen_final_norm_params=frozen_final_norm_params,
            embedding_table=embedding_table,
            optimizer=self.optimizer,
            neural_mem_kwargs=config.neural_mem_kwargs,
        )

        # ---- 10. Activation loader ----
        self.loader = HFActivationLoader(
            activation_repo=activation_repo,
            activation_folder=activation_folder,
            local_activation_dir=local_activation_dir,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            hf_token=hf_token,
            cache_dir=cache_dir,
        )
        self.embed_dim = self.loader.embed_dim
        self.seq_len = self.loader.metadata.get("max_seq_len", 1024)

        print(f"   embed_dim={self.embed_dim}, seq_len={self.seq_len}")
        print("✅ Trainer initialized")

    def train(self, num_steps: int = 10000, eval_every: int = 500):
        """Main training loop — streams batches from HuggingFace."""
        print(f"\n🚀 Starting training for {num_steps} steps...")
        print(
            f"   Data: {self.loader.activation_repo}"
            f"/{self.loader.activation_folder}"
        )

        step = 0
        t_start = time.time()
        loss_history = []

        for batch in self.loader:
            hidden = jnp.array(batch["hidden"])
            tokens_arr = batch["tokens"]
            mask = jnp.array(batch["mask"])

            if tokens_arr is None:
                print(f"⚠️  Step {step}: no tokens in batch, skipping")
                continue

            tokens = jnp.array(tokens_arr)

            self.titans_params, self.opt_state, loss, acc = self._train_step(
                self.titans_params, self.opt_state, hidden, tokens, mask,
            )

            loss_val = float(loss)
            acc_val = float(acc)
            loss_history.append(loss_val)

            if step % 100 == 0:
                elapsed = time.time() - t_start
                steps_per_sec = (step + 1) / max(elapsed, 1e-6)
                avg_loss = (
                    np.mean(loss_history[-100:]) if loss_history else 0.0
                )
                print(
                    f"  Step {step:6d} | loss={loss_val:.4f} | "
                    f"acc={acc_val:.4f} | "
                    f"avg={avg_loss:.4f} | "
                    f"{steps_per_sec:5.2f} steps/s | "
                    f"{elapsed:7.1f}s"
                )

            if step > 0 and step % eval_every == 0:
                self._save_checkpoint(step)

            step += 1
            if step >= num_steps:
                break

        # Final checkpoint
        self._save_checkpoint(step)
        elapsed = time.time() - t_start
        print(f"\n✅ Training complete: {step} steps in {elapsed:.1f}s")

    def _save_checkpoint(self, step: int):
        """Save TitansBlock params checkpoint using Orbax."""
        import orbax
        from flax.training import orbax_utils

        ckpt_path = os.path.join(self.output_dir, f"ckpt_{step}")
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(self.titans_params)
        orbax_checkpointer.save(
            ckpt_path, self.titans_params, save_args=save_args,
            force=True,
        )
        print(f"   💾 Checkpoint saved: {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train TitansBlock layer 23 on precomputed activations "
            "from HuggingFace"
        )
    )
    parser.add_argument(
        "--gemma_ckpt", type=str, required=True,
        help="Path to local Gemma-3-1B checkpoint directory",
    )
    parser.add_argument(
        "--activation_repo", type=str,
        default="veriga/openwebtext-gemma3-tokenized-1024-activations-layer23",
        help="HuggingFace dataset repo with activation shards. "
             "Default: veriga/openwebtext-gemma3-tokenized-1024-activations-layer23",
    )
    parser.add_argument(
        "--activation_folder", type=str, default="",
        help="Subfolder inside the repo with .npy shards. "
             "Default: '' (root) — use when activations are in a dedicated repo. "
             "Set to e.g. 'activations_layer23' when using the token dataset repo.",
    )
    parser.add_argument(
        "--local_activation_dir", type=str, default=None,
        help="Local dir with shard_*.npy from precompute_activations.py. "
             "If set, reads activations from disk instead of downloading from HF. "
             "Tokens still come from HF for CE loss targets.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints_layer23",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace API token",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Local cache dir for HF downloads",
    )
    parser.add_argument("--mlp_depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from routing_optimizer import make_routing_optimizer

    ec = {
        'heads': args.heads,
        'dim_head': args.dim_head,
        'chunk_size': args.chunk_size,
        'mlp_depth': args.mlp_depth,
        'max_grad_norm': 0.5,
        'elastic_net_lambda': 0.01,
        'adaptive_max_lr': args.lr,
    }
    opt_params = {
        "lr_muon": optax.warmup_cosine_decay_schedule(
            init_value=1e-5, peak_value=1e-5,
            warmup_steps=args.warmup_steps, decay_steps=args.num_steps,
            end_value=5e-6,
        ),
        "beta": 0.90,
        "lr_adam": optax.warmup_cosine_decay_schedule(
            init_value=1e-5, peak_value=args.lr,
            warmup_steps=args.warmup_steps, decay_steps=args.num_steps,
            end_value=5e-6,
        ),
        "adam_b1": 0.9, "adam_b2": 0.85,
        "lr_gate": optax.warmup_cosine_decay_schedule(
            init_value=5e-4, peak_value=5e-4,
            warmup_steps=args.warmup_steps, decay_steps=args.num_steps,
            end_value=5e-4,
        ),
        "gate_b1": 0.9, "gate_b2": 0.95,
        "every_k_schedule": 4,
    }
    routing_optimizer = make_routing_optimizer(opt_params)

    trainer = Layer23Trainer(
        gemma_ckpt_path=args.gemma_ckpt,
        optimizer=routing_optimizer,
        experimental_config=ec,
        activation_repo=args.activation_repo,
        activation_folder=args.activation_folder,
        local_activation_dir=args.local_activation_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
    )

    trainer.train(num_steps=args.num_steps, eval_every=args.eval_every)


if __name__ == "__main__":
    main()