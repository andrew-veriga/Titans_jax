# Gemma-Titans Hybrid Project

This repository implements a hybrid architecture combining the pre-trained power of **Gemma** with the **Titans Neural Long-Term Memory (NLTM)** module. The goal is to enable online learning during dialogue by storing context in a dynamic neural memory.

## Architecture
- **Framework:** Google DeepMind `gemma.gm` (JAX/Flax).
- **Hybrid Block:** `TitansBlock` runs standard Attention and Neural Memory in parallel.
- **Gating:** Learned scalar gate to balance short-term (Attention) and long-term (NLTM) context.
- **State Management:** NLTM weights and momentum are integrated into the Transformer's `LayerCache`.

## Google Colab TPU Setup (v5e-1)

Follow these steps to run the project on Google Colab:

### 1. Select TPU Runtime
Go to **Runtime** > **Change runtime type** and select:
- **Hardware accelerator:** TPU
- **TPU type:** v5e

### 2. Install Dependencies
Run the following in a Colab cell to install the required environment:

```python
!pip install -r requirements.txt
```

*Note: The `requirements.txt` includes a critical downgrade for `typeguard==4.4.1` to ensure compatibility with the `kauldron` framework used by Gemma.*

### 3. Clone and Run Verification
```bash
# Clone the repository (replace with your repo URL)
!git clone https://github.com/YOUR_USERNAME/Titans.git
%cd Titans

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
