# Gemma-Titans Hybrid: Model Surgery & Implementation Plan

This document outlines the strategy for building a hybrid model that combines the pre-trained power of **Gemma** with the **Titans Neural Long-Term Memory (NLTM)** module, implemented in JAX/Flax for TPU acceleration.

## 1. Architectural Strategy: The "Titans-Gemma" Block

The core idea is to inject the `NeuralMemory` module into the existing Gemma Transformer block. We have two primary options for placement:

### Option A: Parallel NLTM (Recommended)
Place the NLTM in parallel with the standard Multi-Head Attention (MHA).
*   **Input** -> **LayerNorm** -> ( **MHA** + **NLTM** ) -> **Add** -> **MLP** -> **Add** -> **Output**
*   *Pros:* Preserves Gemma's short-term precision while adding long-term context capacity.

### Option B: Alternating Layers
Replace the MHA in every $N$-th layer with an NLTM module.
*   *Pros:* Reduces total parameter count compared to Option A.

**Decision:** We will implement **Option A** for the initial prototype to maximize the reuse of Gemma's pre-trained attention weights.

## 2. Component Mapping & Weight Reuse

| Component | Status | Source |
| :--- | :--- | :--- |
| **Embeddings** | Reuse | `gemma.embed_tokens` |
| **Attention (STM)** | Reuse | `gemma.layers[i].attn` |
| **Feed-Forward (MLP)** | Reuse | `gemma.layers[i].mlp` |
| **Neural Memory (NLTM)** | **New** | `titans.NeuralMemory` |
| **Memory Projections** | **New** | Random initialization (to be trained) |

## 3. Implementation Steps (JAX/Flax)

### Phase 1: Environment Setup
*   Utilize the official `gemma` Flax implementation as the base.
*   Ensure `jax`, `flax`, and `optax` are configured for TPU (Colab environment).

### Phase 2: Building the Hybrid Block
1.  **Define `GemmaTitansBlock`**: Extend the standard `GemmaBlock` to include a `NeuralMemory` sub-module.
2.  **Projection Alignment**: Add `nn.Dense` layers to project Gemma's hidden states into the `dim_head` required by the NLTM.
3.  **Residual Path**: Implement the summation of MHA and NLTM outputs before the final MLP layer.

### Phase 3: Weight Loading & "Surgery"
1.  **Load Gemma Weights**: Use the `orbax` or `msgpack` checkpoints from Kaggle/HuggingFace.
2.  **Initialization**:
    *   Map weights for all shared layers.
    *   Initialize `NeuralMemory` and new projections using a small standard deviation (e.g., 0.02) to keep the initial output close to the original Gemma behavior.
3.  **Validation**: Run a forward pass with the hybrid model and compare the output to the original Gemma (the difference should be minimal initially).

### Phase 4: Continued Pre-training (CPT for Routing)
*Note: Because Gemma was not trained with an NLTM, it needs to learn **how** to route information in and out of the memory module. The actual memorization of concepts will happen dynamically later.*
1.  **Freezing**: Initially freeze the pre-trained Gemma weights (`attn` and `mlp`). Only train the `NeuralMemory` projections (`to_queries`, `to_keys_values`, `to_momentum`, etc.).
2.  **Dataset**: Train on a mix of conversational and long-context datasets (e.g., UltraChat, RedPajama) so the model learns to delegate long-term facts to the NLTM while using standard attention for short-term syntax.

### Phase 5: Online Learning & Dialogue (Inference Mode)
This is the core Titans mechanic. During actual usage (chat/inference), the model performs **test-time training**:
1.  **Dynamic Memorization**: As the user provides input sequence chunks, the NLTM's `store_memories` method calculates surprise (gradients) and updates its internal weights *on the fly* via the associative scan.
2.  **Concept Acquisition**: New concepts, names, or rules introduced by the user in the prompt are embedded directly into the NLTM's neural weights, creating a persistent, adaptable memory state for the duration of the session (or beyond, if the state is saved).
3.  **Gemma's Role**: The pre-trained Gemma layers handle the robust natural language understanding and generation, dynamically retrieving the updated concepts from the NLTM to formulate its responses.

## 4. Colab/TPU Optimization

*   **Associative Scan**: The JAX `lax.associative_scan` is already highly optimized for TPU.
*   **Vectorization**: Use `jax.vmap` for batch processing across TPU cores.
*   **Sharding**: Implement `jax.sharding` for Model Parallelism if scaling to 7B+ models.

## 5. Library Packaging

The final library should be structured as follows:
```text
gemma_titans/
├── __init__.py
├── model.py         # Hybrid model definition
├── titans_core.py   # Integrated NLTM module
├── weights.py       # Surgery & Loading utilities
└── training.py      # CPT loops & data loaders
```

---
*Created on: 2026-03-09*
*Target Platform: Google Colab (TPU v2/v3)*
