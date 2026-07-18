# Generation command:
# python scripts/train_pretrained_cifn.py

"""
Script to train the CIFN classifier on the synthetic data generator
and save the resulting weight parameters to cifn_pretrained.pt.

This ensures the origin of the pretrained weights is transparent,
known, and reproducible.
"""

import os
import sys
import random
import numpy as np
import torch
import logging

# Ensure backend is in the import search path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Configure environment variables to train from scratch with plain KRONOS
os.environ["USE_NOETHER"] = "false"
os.environ["USE_PRETRAINED_CIFN"] = "false"
os.environ["ENTITY_MODE"] = "live"

# Setup logging to surface training progress to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def main():
    # Set seed for reproducibility of training process and synthetic data generation
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("Initializing and training the LiveEntity CIFN classifier (1200 steps)...")
    from entity_interface.live_entity import LiveEntity

    # Instantiating LiveEntity triggers training under USE_PRETRAINED_CIFN=false
    entity = LiveEntity()

    # Extract final validation metrics
    final_val_loss = entity.stats.get("cifn_final_val_loss", None)
    final_val_acc = entity.stats.get("cifn_synthetic_self_consistency_accuracy", None)
    
    print("\nTraining completed successfully!")
    print(f"Final Validation Loss: {final_val_loss}")
    print(f"Final Validation Accuracy: {final_val_acc * 100:.2f}%" if final_val_acc is not None else "Final Validation Accuracy: Unknown")

    # Define paths
    output_dir = os.path.join(backend_path, "entity_interface")
    output_path = os.path.join(output_dir, "cifn_pretrained.pt")

    print(f"\nSaving model weights to {output_path}...")
    torch.save(entity.model.state_dict(), output_path)
    print("Pretrained weights saved successfully. The file's origin is now known and reproducible!")

if __name__ == "__main__":
    main()
