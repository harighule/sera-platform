"""
Trace weights after every 100 steps of the real training loop for seed 1.
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
    
    # We patch _train_cifn_classifier to trace weights every 100 steps
    def traced_train(self):
        import torch.nn.functional as F
        print(f"  Step 0: {self.model.cifn1.weight_field.a[:5].tolist()}")
        
        tr_f, tr_l, val_f, val_l = self._generate_synthetic_labels(
            n_per_class=500, val_fraction=0.2
        )
        cifn_opt_A = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        cifn_sch_A = torch.optim.lr_scheduler.CosineAnnealingLR(
            cifn_opt_A, T_max=1000, eta_min=1e-5
        )
        
        if hasattr(self.kronos_model, "head"):
            kronos_head = list(self.kronos_model.head.parameters())
        else:
            kronos_head = list(self.kronos_model.kronos.head.parameters())
        k_opt_A = torch.optim.Adam(kronos_head, lr=1e-3)
        k_sch_A = torch.optim.lr_scheduler.CosineAnnealingLR(
            k_opt_A, T_max=1000, eta_min=1e-6
        )
        N = len(tr_f)
        n_cls = len(le_mod.TRANSITION_TYPES)
        weights = torch.tensor([1.0]*n_cls)
        
        self.model.train()
        self.kronos_model.train()
        
        for step in range(1, 1001):
            idx = torch.randint(0, N, (32,))
            xb, yb = tr_f[idx], tr_l[idx]
            
            # CIFN update
            out_c = self.model(xb)
            loss_c = F.cross_entropy(out_c["transition_logits"], yb)
            cifn_opt_A.zero_grad()
            loss_c.backward()
            cifn_opt_A.step()
            cifn_sch_A.step()
            
            # KRONOS update
            with torch.no_grad():
                ids = (xb * 255).long().clamp(0, 255)
            with torch.enable_grad():
                kout = self.kronos_model(ids)
                klog = kout["logits"][:, 0, :n_cls]
                loss_k = F.cross_entropy(klog, yb)
            k_opt_A.zero_grad()
            loss_k.backward()
            k_opt_A.step()
            k_sch_A.step()
            
            if step % 100 == 0:
                print(f"  Step {step}: {self.model.cifn1.weight_field.a[:5].tolist()}")
                
        # Phase B
        print("  Starting Phase B")
        cifn_opt_B = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        cifn_sch_B = torch.optim.lr_scheduler.CosineAnnealingLR(
            cifn_opt_B, T_max=200, eta_min=1e-5
        )
        for step in range(1, 201):
            idx = torch.randint(0, N, (32,))
            xb, yb = tr_f[idx], tr_l[idx]
            out = self.model(xb)
            loss = F.cross_entropy(out["transition_logits"], yb)
            cifn_opt_B.zero_grad()
            loss.backward()
            cifn_opt_B.step()
            cifn_sch_B.step()
            
            if step % 50 == 0:
                print(f"  Phase B Step {step}: {self.model.cifn1.weight_field.a[:5].tolist()}")

    le_mod.LiveEntity._train_cifn_classifier = traced_train

    # Set seed 1
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    entity = le_mod.LiveEntity()

if __name__ == '__main__':
    main()
