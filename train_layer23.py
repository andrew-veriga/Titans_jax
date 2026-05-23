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
        --activation_dir ./activations_layer23 \
        --token_dataset_repo veriga/openwebtext-gemma3-tokenized-1024 \
        --output_dir ./checkpoints_layer23 \
        --num_steps 10000 \
        --batch_size 4

For Kauldron integration, see the KauldronTrainer class at the bottom.
"""

import argparse
import json
import os
import time
from functools import partial
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ---------------------------------------------------------------------------
# TitansBlock wrapper for training on precomputed activations
# ---------------------------------------------------------------------------


def build_titans_block_config(
    target_layer: int = 23,
    mlp_depth: int = 6,
    heads: int = 8,
    dim_head: int = 256,
    chunk_size: int = 32,
    max_grad_norm: float = 0.5,
    adaptive_max_lr: float = 1e-3,
    every_k_schedule: int = 1,
):
    """Build neural_mem_kwargs and config for the target TitansBlock."""
    from gemma_titans import Gemma_Titans_Config

    config = Gemma_Titans_Config(
        titans_layer_indices=(target_layer,),
        titans_first_layer=target_layer,
        training_phase=2,
        is_training_mode=True,
        neural_mem_kwargs={
            'heads': heads,
            'dim_head': dim_head,
            'chunk_size': chunk_size,
            'max_grad_norm': max_grad_norm,
            'mlp_depth': mlp_depth,
            'diff_view': False,
            'is_look_ahead': False,
            'huber_loss_delta': None,
            'adaptive_max_lr': adaptive_max_lr,
            'every_k_schedule': every_k_schedule,
        },
    )
    return config


# ---------------------------------------------------------------------------
# Training step: ONLY TitansBlock parameters are trainable
# ---------------------------------------------------------------------------

def make_train_step(model, config, optimizer):
    """
    Create a jitted training step that:
    1. Takes precomputed hidden states (frozen layers 0-22 output)
    2. Runs through TitansBlock (trainable)
    3. Runs through frozen layers 23-25 + head
    4. Computes CrossEntropy loss
    
    Only TitansBlock parameters have gradients.
    """
    
    @jax.jit
    def train_step(titans_params, opt_state, frozen_params, hidden, tokens, mask):
        """
        Args:
            titans_params: TitansBlock parameters (trainable)
            opt_state: Optimizer state
            frozen_params: Full Gemma params (for layers 24-25 + head)
            hidden: Precomputed activations (B, L, 1152)
            tokens: Token IDs (B, L) — for CE target (shifted by 1)
            mask: Input mask (B, L) — 1 for real tokens
        """
        
        def loss_fn(tp):
            # 1. TitansBlock forward (layer 23)
            # We need to call model's _apply_attention from the target layer
            # This is the tricky part — we need to feed hidden into the right layer
            
            # Construct merged params: frozen everywhere except TitansBlock
            merged = _merge_params(frozen_params, tp, config)
            
            # Run full model but with precomputed hidden as starting point
            # We'll use a custom forward that skips layers 0-22
            
            # Actually, simplest approach:
            # Build a minimal forward that does: TitansBlock → layers 24-25 → head
            x = hidden
            
            # TitansBlock (layer 23) with memory state
            # ... need to integrate with model architecture
            
            # For now, use model.apply with a trick:
            # Replace embeddings with our precomputed hidden,
            # and mask attention for layers 0-22 (skip them)
            
            # This requires modifying the model forward — see below.
            pass
        
        grads = jax.grad(loss_fn)(titans_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, titans_params)
        new_params = optax.apply_updates(titans_params, updates)
        
        return new_params, new_opt_state, {}
    
    return train_step


# ---------------------------------------------------------------------------
# Simplified approach: standalone TitansBlock + frozen head
# ---------------------------------------------------------------------------

class Layer23Trainer:
    """
    Standalone trainer for TitansBlock on layer 23.
    
    Avoids the complexity of modifying the full Gemma forward pass.
    Instead:
    1. TitansBlock processes precomputed hidden states
    2. Frozen Gemma layers 24-25 + final_norm + head produce logits
    3. CrossEntropy loss on next-token prediction
    """
    
    def __init__(
        self,
        gemma_ckpt_path: str,
        activation_dir: str,
        output_dir: str,
        mlp_depth: int = 6,
        heads: int = 8,
        dim_head: int = 256,
        chunk_size: int = 32,
        lr: float = 1e-4,
        warmup_steps: int = 500,
        batch_size: int = 4,
        seed: int = 42,
    ):
        from gemma import gm
        from gemma_titans import TitansBlock, Gemma_Titans_Config
        
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.seed = seed
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Load full Gemma params ---
        print("🔧 Loading Gemma-3-1B parameters...")
        self.full_params = gm.ckpts.load_params(gemma_ckpt_path)
        
        # --- Build config ---
        self.config = Gemma_Titans_Config(
            titans_layer_indices=(23,),
            titans_first_layer=23,
            training_phase=2,
            is_training_mode=True,
            neural_mem_kwargs={
                'heads': heads,
                'dim_head': dim_head,
                'chunk_size': chunk_size,
                'max_grad_norm': 0.5,
                'mlp_depth': mlp_depth,
                'diff_view': False,
                'is_look_ahead': False,
                'huber_loss_delta': None,
                'adaptive_max_lr': 1e-3,
                'every_k_schedule': 1,
            },
        )
        
        # --- Load activation metadata ---
        meta_path = os.path.join(activation_dir, "metadata.json")
        with open(meta_path) as f:
            self.act_meta = json.load(f)
        self.embed_dim = self.act_meta["embed_dim"]
        self.seq_len = self.act_meta["max_seq_len"]
        
        print(f"   embed_dim={self.embed_dim}, seq_len={self.seq_len}")
        
        # --- Optimizer ---
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=50000,
            end_value=lr * 0.1,
        )
        self.optimizer = optax.adam(schedule)
        
        # --- Build model ---
        self.model = gm.nn.Gemma3_1B()
        
        print("✅ Trainer initialized")
    
    def train(self, num_steps: int = 10000, eval_every: int = 500):
        """Main training loop."""
        print(f"\n🚀 Starting training for {num_steps} steps...")
        # Implementation depends on exact integration with gemma.gm internals
        # This is a template — the actual forward pass needs to be adapted
        # based on how gemma_titans.py exposes layer-by-layer forward
        
        # The key insight: we don't need the full model graph.
        # We need:
        #   TitansBlock.apply(hidden) → titans_output
        #   Then: layers 24-25 + norm + head → logits
        #   Then: CE loss
        
        # This requires extracting the individual layers from the model,
        # which is straightforward with Flax modules.
        pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train TitansBlock layer 23 on precomputed activations")
    parser.add_argument("--gemma_ckpt", type=str, required=True)
    parser.add_argument("--activation_dir", type=str, required=True)
    parser.add_argument("--token_dataset_repo", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints_layer23")
    parser.add_argument("--mlp_depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    trainer = Layer23Trainer(
        gemma_ckpt_path=args.gemma_ckpt,
        activation_dir=args.activation_dir,
        output_dir=args.output_dir,
        mlp_depth=args.mlp_depth,
        heads=args.heads,
        dim_head=args.dim_head,
        chunk_size=args.chunk_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    
    trainer.train(num_steps=args.num_steps)


if __name__ == "__main__":
    main()
