"""
Check if CIFN weights differ when training with different seeds.
"""
import sys
import os
import random
import numpy as np
import torch
import importlib
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def get_trained_weights(seed: int):
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
    
    # Return a copy of the cifn1.weight_field.a parameter
    weights = entity.model.cifn1.weight_field.a.detach().clone().numpy()
    return weights

def main():
    print("=" * 65)
    print("CHECKING WEIGHT DIFFERENCES ACROSS SEEDS")
    print("=" * 65)
    
    w1 = get_trained_weights(seed=1)
    w7 = get_trained_weights(seed=7)
    
    diff = np.abs(w1 - w7).max()
    print(f"Max absolute difference in cifn1 weights between seed 1 and seed 7: {diff}")
    print(f"First 5 weights of seed 1: {w1[:5]}")
    print(f"First 5 weights of seed 7: {w7[:5]}")
    print("=" * 65)

if __name__ == '__main__':
    main()
