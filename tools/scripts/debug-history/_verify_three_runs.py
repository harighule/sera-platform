"""
Re-run CIFN training 3 times with USE_NOETHER=False and seed=42.
"""
import sys
import os
import random
import numpy as np
import torch
import importlib
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def run_once(run_idx: int):
    # Set env vars
    os.environ['USE_NOETHER'] = 'false'
    os.environ['USE_PRETRAINED_CIFN'] = 'true'
    
    # Reload modules
    import config
    importlib.reload(config)
    import entity_interface.live_entity as le_mod
    importlib.reload(le_mod)
    
    # Set seeds exactly before instantiating LiveEntity
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    entity = le_mod.LiveEntity()

    acc = entity.stats.get("cifn_synthetic_self_consistency_accuracy", 0.0)
    loss = entity.stats.get("cifn_final_val_loss", 0.0)
    train_log = entity.stats.get("cifn_train_log", [])
    
    print(f"\nRun #{run_idx}:")
    print(f"  Final Accuracy: {acc*100:.2f}%")
    print(f"  Final Val Loss: {loss:.5f}")
    if train_log:
        print(f"  Last Log Line:  {train_log[-1].strip()}")
    return acc * 100

def main():
    print("=" * 65)
    print("THREE RE-RUNS WITH USE_NOETHER=False (seed=42)")
    print("=" * 65)
    
    results = []
    for i in range(1, 4):
        acc = run_once(i)
        results.append(acc)
        
    print("\n" + "=" * 65)
    print("SUMMARY OF THREE RUNS:")
    for idx, val in enumerate(results):
        print(f"  Run #{idx+1}: {val:.2f}%")
    print("=" * 65)

if __name__ == '__main__':
    main()
