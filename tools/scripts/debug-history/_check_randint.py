"""
Trace torch.randint calls during LiveEntity initialization.
"""
import sys
import os
import torch
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Mock torch.randint to print the first 5 elements of generated tensors
old_randint = torch.randint
randint_count = 0

def mock_randint(*args, **kwargs):
    global randint_count
    res = old_randint(*args, **kwargs)
    if randint_count < 3:
        print(f"  [Randint #{randint_count+1}] args={args} kwargs={kwargs} -> {res[:5].tolist()}")
    randint_count += 1
    return res

# We patch it before importing anything
patcher = unittest.mock.patch('torch.randint', mock_randint)
patcher.start()

from entity_interface.live_entity import LiveEntity

def test_run(seed: int):
    global randint_count
    print(f"\n--- RUN WITH SEED {seed} ---")
    randint_count = 0
    
    # Set seeds
    torch.manual_seed(seed)
    
    # Mock training methods to not do full run, just trace constructor's calls
    # Wait, if we mock training, we can see the constructor's randint calls
    # but let's NOT mock training so we can see the training randint calls!
    # Wait, to keep output short, let's mock _train_cifn_classifier so we only see
    # the constructor's bootstrap training step's randint calls!
    with (
        unittest.mock.patch.object(LiveEntity, '_train_cifn_classifier', lambda self: None),
    ):
        le = LiveEntity()

if __name__ == '__main__':
    test_run(1)
    test_run(7)
