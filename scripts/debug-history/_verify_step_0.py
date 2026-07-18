"""
Verify CIFN performance at step 0 (before training) for both configurations.
"""
import sys
import os
import random
import numpy as np
import torch
import importlib
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def check_step_0(use_noether: bool):
    os.environ['USE_NOETHER'] = 'true' if use_noether else 'false'
    os.environ['USE_PRETRAINED_CIFN'] = 'true'
    
    import config
    importlib.reload(config)
    import entity_interface.live_entity as le_mod
    importlib.reload(le_mod)
    
    # Set seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # We mock _train_cifn_classifier so we can check step 0
    with unittest.mock.patch.object(le_mod.LiveEntity, '_train_cifn_classifier', lambda self: None):
        entity = le_mod.LiveEntity()

    # Generate synthetic label set using the same seed=42
    tr_f, tr_l, val_f, val_l = entity._generate_synthetic_labels(
        n_per_class=500, val_fraction=0.2, seed=42
    )
    
    # Calculate loss and accuracy at step 0
    entity.model.eval()
    with torch.no_grad():
        out = entity.model(val_f)
        logits = out["transition_logits"]
        loss = torch.nn.functional.cross_entropy(logits, val_l).item()
        acc = (logits.argmax(1) == val_l).float().mean().item()
        
    print(f"USE_NOETHER = {use_noether}:")
    print(f"  Step 0 Val Loss: {loss:.5f}")
    print(f"  Step 0 Val Acc:  {acc*100:.2f}%")
    return loss, acc

def main():
    print("=" * 65)
    print("CIFN STEP 0 INITIALIZATION COMPARISON")
    print("=" * 65)
    check_step_0(use_noether=True)
    check_step_0(use_noether=False)
    print("=" * 65)

if __name__ == '__main__':
    main()
