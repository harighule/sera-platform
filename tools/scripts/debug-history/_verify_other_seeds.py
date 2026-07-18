"""
Run CIFN training with USE_NOETHER=False under different seeds.
"""
import sys
import os
import random
import numpy as np
import torch
import importlib
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def run_with_seed(seed: int):
    os.environ['USE_NOETHER'] = 'false'
    os.environ['USE_PRETRAINED_CIFN'] = 'true'
    
    # Reload modules
    import config
    importlib.reload(config)
    import entity_interface.live_entity as le_mod
    importlib.reload(le_mod)
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    entity = le_mod.LiveEntity()

    acc = entity.stats.get("cifn_synthetic_self_consistency_accuracy", 0.0)
    loss = entity.stats.get("cifn_final_val_loss", 0.0)
    
    print(f"Seed {seed:>3}: Accuracy = {acc*100:.2f}%, Val Loss = {loss:.5f}")
    return acc * 100

def main():
    print("=" * 65)
    print("TESTING DIFFERENT SEEDS WITH USE_NOETHER=False")
    print("=" * 65)
    
    seeds = [1, 7, 42, 100, 999]
    for s in seeds:
        run_with_seed(s)
        
    print("=" * 65)

if __name__ == '__main__':
    main()
