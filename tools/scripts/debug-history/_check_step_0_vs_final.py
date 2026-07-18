"""
Trace CIFN weights at step 0, step 1000 (after Phase A), and step 1200 (after Phase B).
"""
import sys
import os
import random
import numpy as np
import torch
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def main():
    os.environ['USE_NOETHER'] = 'false'
    os.environ['USE_PRETRAINED_CIFN'] = 'true'
    
    import config
    importlib.reload(config)
    import entity_interface.live_entity as le_mod
    importlib.reload(le_mod)
    
    # traced_train definition
    def traced_train(self):
        import torch.nn.functional as F
        print(f"Weights at step 0 (before training): {self.model.cifn1.weight_field.a[:5].tolist()}")
        
        tr_f, tr_l, val_f, val_l = self._generate_synthetic_labels(
            n_per_class=500, val_fraction=0.2
        )
        cifn_opt_A = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        if hasattr(self.kronos_model, "head"):
            kronos_head = list(self.kronos_model.head.parameters())
        else:
            kronos_head = list(self.kronos_model.kronos.head.parameters())
        k_opt_A = torch.optim.Adam(kronos_head, lr=1e-3)
        N = len(tr_f)
        
        # Phase A
        self.model.train()
        for step in range(1, 1001):
            idx = torch.randint(0, N, (32,))
            xb, yb = tr_f[idx], tr_l[idx]
            out_c = self.model(xb)
            loss_c = F.cross_entropy(out_c["transition_logits"], yb)
            cifn_opt_A.zero_grad()
            loss_c.backward()
            cifn_opt_A.step()
            
        print(f"Weights at step 1000 (after Phase A): {self.model.cifn1.weight_field.a[:5].tolist()}")
        
        # Phase B
        cifn_opt_B = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        for step in range(1, 201):
            idx = torch.randint(0, N, (32,))
            xb, yb = tr_f[idx], tr_l[idx]
            out = self.model(xb)
            loss = F.cross_entropy(out["transition_logits"], yb)
            cifn_opt_B.zero_grad()
            loss.backward()
            cifn_opt_B.step()
            
        print(f"Weights at step 1200 (after Phase B): {self.model.cifn1.weight_field.a[:5].tolist()}")

    # Apply patch AFTER reload
    le_mod.LiveEntity._train_cifn_classifier = traced_train

    # Set seed 42
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    entity = le_mod.LiveEntity()

if __name__ == '__main__':
    main()
