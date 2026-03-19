import os
import jax
import jax.numpy as jnp
import optax
import seqio

# Kauldron and Gemma imports
from kauldron import kd
from gemma import gm
from gemma.gm.nn import _config

# Import our custom modules
from gemma_titans import Gemma3_1B_Titans, Gemma_Titans_Config
from titans_ckpts import SkipTitans

# TPU / Memory settings (optional, adapt as needed for your specific pod)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 1. Define Model Config
gemma_config = Gemma_Titans_Config

# 2. Wrap our hybrid model for Kauldron
# Note: Gemma3_1B_Titans naturally returns _transformer.Output 
# which has a `.logits` attribute that kd.losses can access via "preds.logits".
class KauldronGemmaTitans(Gemma3_1B_Titans):
    """Wrapper if needed to conform to Kauldron's specific expectations,
       but usually Kauldron works directly with Flax Linen modules.
    """
    pass

def build_trainer(
    train_dataset: iter, 
    ckpt_path: str = gm.ckpts.CheckpointPath.GEMMA3_1B_IT,
    workdir: str = '/workspace/titans_checkpoints'
) -> kd.train.Trainer:
    """Builds the Kauldron Trainer for Gemma-Titans."""
    
    trainer = kd.train.Trainer(
        seed=42,
        workdir=workdir,
        
        # 1. Dataset
        train_ds=train_dataset,
        
        # 2. Model definition
        model=KauldronGemmaTitans(
            config=gemma_config,
            # We want to return only the last logit for standard autoregressive
            # or classification depending on the task.
            # In a pure CPT task, we might want all hidden states, but let's 
            # assume a standard setup for now.
        ),
        
        # 3. Load the weights using our custom SkipTitans
        init_transform=SkipTitans(
            wrapped=gm.ckpts.LoadCheckpoint(
                path=ckpt_path,
            )
        ),
        
        # 4. Training loop config
        num_train_steps=10000,
        
        # 5. Loss Function
        # We expect inputs to be named 'tokens' and targets 'target'
        train_losses={
            "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
                logits="preds.logits",
                labels="batch.target",
                mask="batch.loss_mask",
            ),
        },
        
        # 6. Optimizer & Masking
        # THIS IS CRITICAL FOR OOM PREVENTION ON TPU:
        # We use kd.optim.partial_updates to apply gradients ONLY to the 'memory' and 'memory_gate'
        optimizer=kd.optim.partial_updates(
            optax.adam(learning_rate=1e-4),
            # Select only Titans parameters to train
            mask=kd.optim.select(["memory", "memory_gate"]),
        ),      
        
        # 7. Checkpointing
        checkpointer=kd.ckpts.Checkpointer(
            save_interval_steps=500,
        ),
        
        # 8. Sharding (Multi-TPU Setup)
        # Configure sharding based on available devices. 
        # For a single TPU v5e, standard replication or data parallelism is sufficient.
        sharding=kd.sharding.ShardingStrategy(),
    )
    
    return trainer

def main():
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"Available devices: {jax.devices()}")
    
    # Example Dummy Dataset Generator (Replace with your actual SeqIO/Grain dataset)
    def dummy_dataset():
        while True:
            yield {
                "tokens": jnp.ones((2, 512), dtype=jnp.int32),
                "target": jnp.ones((2,), dtype=jnp.int32),
                "loss_mask": jnp.ones((2,), dtype=jnp.float32),
            }
            
    trainer = build_trainer(dummy_dataset())
    
    print("Initializing state...")
    # This will load Gemma weights, initialize Titans, and compile the first step
    # state = trainer.init_state()
    
    print("Ready to train! Run: state, aux = trainer.train()")
    
if __name__ == "__main__":
    main()
