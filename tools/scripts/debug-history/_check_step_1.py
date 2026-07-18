"""
Check weights after 1 step of training for seed 1 and seed 7.
"""
import sys
import os
import random
import numpy as np
import torch
import importlib
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def get_step_1_weights(seed: int):
    os.environ['USE_NOETHER'] = 'false'
    os.environ['USE_PRETRAINED_CIFN'] = 'true'
    
    import config
    importlib.reload(config)
    import entity_interface.live_entity as le_mod
    importlib.reload(le_mod)
    
    # We patch _train_cifn_classifier to only run 1 step
    old_train = le_mod.LiveEntity._train_cifn_classifier
    
    def mock_train(self):
        import torch.nn.functional as F
        tr_f, tr_l, val_f, val_l = self._generate_synthetic_labels(
            n_per_class=500, val_fraction=0.2
        )
        cifn_opt_A = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        N = len(tr_f)
        
        # Run exactly 1 step
        self.model.train()
        idx = torch.randint(0, N, (32,))
        print(f"  [Seed {seed}] Step 1 idx: {idx[:5].tolist()}")
        xb, yb = tr_f[idx], tr_l[idx]
        out_c = self.model(xb)
        loss_c = F.cross_entropy(out_c["transition_logits"], yb)
        cifn_opt_A.zero_grad()
        loss_c.backward()
        cifn_opt_A.step()
        
    le_mod.LiveEntity._train_cifn_classifier = mock_train

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    entity = le_mod.LiveEntity()
    weights = entity.model.cifn1.weight_field.a.detach().clone().numpy()
    return weights

def main():
    print("=" * 65)
    print("CHECKING STEP 1 WEIGHTS AND INDICES")
    print("=" * 65)
    w1 = get_step_1_weights(1)
    w7 = get_step_1_weights(7)
    diff = np.abs(w1 - w7).max()
    print(f"Max absolute difference at step 1: {diff}")
    print("=" * 65)

if __name__ == '__main__':
    main()
