# Gemma-Titans Hybrid Project

This repository implements a hybrid architecture combining the pre-trained power of **Gemma** with the **Titans Neural Long-Term Memory (NLTM)** module. The goal is to enable online learning during dialogue by storing context in a dynamic neural memory.

## Architecture
- **Framework:** Google DeepMind `gemma.gm` (JAX/Flax).
- **Hybrid Block:** `TitansBlock` runs standard Attention and Neural Memory in parallel.
- **Gating:** Learned scalar gate to balance short-term (Attention) and long-term (NLTM) context.
- **State Management:** NLTM weights and momentum are integrated into the Transformer's `LayerCache`.

## Google Colab TPU Setup (v5e-1)

Follow these steps to run the project on Google Colab. 

**Note:** As of March 2025, Colab TPU runtimes do not pre-install TensorFlow. Since our dependencies require it for type-checking, follow this specific order:

### 1. Select TPU Runtime
Go to **Runtime** > **Change runtime type** and select:
- **Hardware accelerator:** TPU
- **TPU type:** v5e

### 2. Environment Installation
Run the following in a Colab cell:

```python
# 1. Install JAX for TPU
!pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 2. Install TensorFlow (Required manually on TPU runtimes since March 2025)
!pip install tensorflow==2.18.0 tensorflow-tpu==2.18.0 --find-links=https://storage.googleapis.com/libtpu-tf-releases/index.html

# 3. Install project dependencies with critical version locks
!pip install typeguard==4.4.1 gemma==3.3.0 kauldron==1.3.0 flax==0.12.5 optax==0.2.6
!pip install einops einx treescope jaxtyping sentencepiece
```

### 3. Clone and Run Verification
```bash
# Clone the repository
!git clone https://github.com/andrew-veriga/Titans_jax.git
%cd Titans_jax

# Run the hybrid model forward pass test
!python test_gemma_titans.py
```

## Key Files
- `gemma_titans.py`: The hybrid model and block implementation.
- `titans.py`: The core Neural Long-Term Memory (NLTM) module using associative scan.
- `test_gemma_titans.py`: Verification script for initialization and forward pass.
- `GemmaTitans_Surgery_Plan.md`: Detailed roadmap for model surgery and Continued Pre-Training (CPT).

## Current Status
- [x] Framework migration to `gemma.gm`.
- [x] Hybrid block implementation (`TitansBlock`).
- [x] JAX-optimized state management for NLTM.
- [x] Verified Forward Pass.
- [ ] Weight Surgery (loading pre-trained 2B weights).
- [ ] Continued Pre-Training (CPT) loop.
