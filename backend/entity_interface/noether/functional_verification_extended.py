import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add backend and noether directory to path
noether_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, noether_dir)
sys.path.insert(0, os.path.abspath(os.path.join(noether_dir, '..', '..')))

from noether_kronos import NOETHER_KRONOS, NOETHERTrainer

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

def run_extended_verification():
    # Base config
    config = dict(
        vocab_size      = 1000,
        d_model         = 64,
        n_heads         = 4,
        n_layers        = 2,
        d_ff            = 256,
        max_seq_len     = 64,
        memory_size     = 64,
        z_dim           = 64,
        n_slots         = 4,
        n_wave_freqs    = 16,
        dropout         = 0.1,
        notears_coeff   = 0.01,
        n_generators    = 4,
        gen_rank        = 4,
        n_templates     = 16,
        n_types         = 8,
        n_rg_levels     = 2,
        block_size      = 2,
        sym_threshold   = 0.70,
        kl_weight       = 0.05,
        notears_weight  = 0.01,
        orbit_weight    = 0.05,
        ctl_weight      = 0.05,
        rg_weight       = 0.05,
        sym_weight      = 0.10,
    )

    print("==================================================")
    print("CHECK 1: Input sensitivity of each of the 5 components")
    print("==================================================")
    set_seed(42)
    model = NOETHER_KRONOS(**config)
    model.eval()

    # Create two different realistic token sequences
    input_ids1 = torch.tensor([[1, 2, 1, 2, 3, 4, 3, 4]], dtype=torch.long)
    input_ids2 = torch.tensor([[900, 901, 900, 901, 902, 903, 902, 903]], dtype=torch.long)

    # Hooks to capture intermediate values
    captured = {}
    
    def get_hook(name):
        def hook(module, input, output):
            captured[name] = output
        return hook

    # Register hooks
    h_sde = model.sde.register_forward_hook(get_hook("SDE"))
    h_soe = model.soe.register_forward_hook(get_hook("SOE"))
    h_cfn = model.cfn.register_forward_hook(get_hook("CFN"))
    h_ctl = model.ctl.register_forward_hook(get_hook("CTL"))
    h_arg = model.arg.register_forward_hook(get_hook("ARG"))

    # Forward 1
    with torch.no_grad():
        out1 = model(input_ids1)
    sde_out1 = captured["SDE"]
    soe_out1 = captured["SOE"]
    cfn_out1 = captured["CFN"]
    arg_out1 = captured["ARG"]
    ctl_out1 = captured["CTL"]

    sde_rep1 = sde_out1[0].clone()
    soe_rep1 = soe_out1[0].clone()
    cfn_rep1 = cfn_out1.clone()
    arg_rep1 = arg_out1[0].clone()
    ctl_rep1 = ctl_out1[0].clone()

    # Forward 2
    captured.clear()
    with torch.no_grad():
        out2 = model(input_ids2)
    sde_rep2 = captured["SDE"][0]
    soe_rep2 = captured["SOE"][0]
    cfn_rep2 = captured["CFN"]
    arg_rep2 = captured["ARG"][0]
    ctl_rep2 = captured["CTL"][0]

    # Remove hooks
    h_sde.remove()
    h_soe.remove()
    h_cfn.remove()
    h_ctl.remove()
    h_arg.remove()

    def report_diff(name, t1, t2):
        mean1, mean2 = t1.mean().item(), t2.mean().item()
        std1, std2 = t1.std().item(), t2.std().item()
        diff = torch.abs(t1 - t2).mean().item()
        print(f"{name:10s} | Input1: mean={mean1:.6f}, std={std1:.6f} | Input2: mean={mean2:.6f}, std={std2:.6f} | AbsDiff={diff:.6f}")
        return diff > 1e-5

    print("Checking sensitivity...")
    sens_sde = report_diff("SDE", sde_rep1, sde_rep2)
    sens_soe = report_diff("SOE", soe_rep1, soe_rep2)
    sens_cfn = report_diff("CFN", cfn_rep1, cfn_rep2)
    sens_ctl = report_diff("CTL", ctl_rep1, ctl_rep2)
    sens_arg = report_diff("ARG", arg_rep1, arg_rep2)

    print("\n==================================================")
    print("CHECK 2: Loss behavior under real training")
    print("==================================================")
    set_seed(42)
    model_train = NOETHER_KRONOS(**config)
    trainer = NOETHERTrainer(model_train, lr=1e-3, device='cpu')

    # Fixed training inputs (highly structured sequence for "real data")
    # A simple repeating motif: [0, 1, 2, 3, 0, 1, 2, 3...]
    train_ids = torch.arange(32).unsqueeze(0).repeat(4, 1) % 4
    train_labels = (torch.arange(32).unsqueeze(0).repeat(4, 1) + 1) % 4

    step1_losses = {}
    step20_losses = {}

    for step in range(1, 21):
        bd = trainer.train_step(train_ids, train_labels)
        if step == 1:
            step1_losses = bd.copy()
        if step == 20:
            step20_losses = bd.copy()

    print(f"Step 1  losses: orbit={step1_losses['orbit']:.6f}, ctl={step1_losses['ctl']:.6f}, rg={step1_losses['rg']:.6f}, total={step1_losses['total']:.6f}")
    print(f"Step 20 losses: orbit={step20_losses['orbit']:.6f}, ctl={step20_losses['ctl']:.6f}, rg={step20_losses['rg']:.6f}, total={step20_losses['total']:.6f}")

    print("\n==================================================")
    print("CHECK 3: Extended Symmetry Discovery Engine Test")
    print("==================================================")
    
    # Run 1: Training on REAL (structured motif) data for 500 steps
    print("Running Run 1 (REAL structured data)...")
    set_seed(42)
    model_real = NOETHER_KRONOS(**config)
    trainer_real = NOETHERTrainer(
        model_real, lr=1e-3, 
        sym_update_every=50, 
        sym_loss_every=100, 
        device='cpu'
    )
    
    real_scores_history = []
    
    for step in range(1, 501):
        # Structured input data
        batch_ids = torch.randint(0, 4, (4, 32))
        batch_labels = (batch_ids + 1) % 4
        trainer_real.train_step(batch_ids, batch_labels)
        
        if step % 50 == 0 or step == 1:
            real_scores_history.append((step, model_real.sde.scores.clone().tolist(), model_real.sde.active.clone().tolist()))

    # Run 2: Training on PURE NOISE data for 500 steps
    print("Running Run 2 (PURE NOISE data)...")
    set_seed(42)
    model_noise = NOETHER_KRONOS(**config)
    trainer_noise = NOETHERTrainer(
        model_noise, lr=1e-3, 
        sym_update_every=50, 
        sym_loss_every=100, 
        device='cpu'
    )
    
    noise_scores_history = []
    
    for step in range(1, 501):
        # Pure random noise data
        batch_ids = torch.randint(0, 1000, (4, 32))
        batch_labels = torch.randint(0, 1000, (4, 32))
        trainer_noise.train_step(batch_ids, batch_labels)
        
        if step % 50 == 0 or step == 1:
            noise_scores_history.append((step, model_noise.sde.scores.clone().tolist(), model_noise.sde.active.clone().tolist()))

    # Report results
    print("\n--- SDE Score History: REAL (Structured) Data ---")
    for step, scores, active in real_scores_history:
        scores_str = ", ".join([f"{s:.6f}" for s in scores])
        print(f"Step {step:3d} | Scores: [{scores_str}] | Active: {active}")

    print("\n--- SDE Score History: PURE NOISE Data ---")
    for step, scores, active in noise_scores_history:
        scores_str = ", ".join([f"{s:.6f}" for s in scores])
        print(f"Step {step:3d} | Scores: [{scores_str}] | Active: {active}")

    print("\n==================================================")
    print("CHECK 4: Rigorous Controlled Ablation Test")
    print("==================================================")
    # Strictly controlled seeds and initialization
    set_seed(42)
    model_ablation_on = NOETHER_KRONOS(**config)
    
    set_seed(42)
    model_ablation_off = NOETHER_KRONOS(**config)
    
    # Confirm state dicts match exactly at initialization
    diff_init = 0.0
    for p1, p2 in zip(model_ablation_on.parameters(), model_ablation_off.parameters()):
        diff_init += torch.abs(p1 - p2).sum().item()
    print(f"Initialization difference between ON and OFF models: {diff_init:.6f}")

    # Set weights to zero in OFF model
    model_ablation_off.orbit_weight = 0.0
    model_ablation_off.ctl_weight    = 0.0
    model_ablation_off.rg_weight     = 0.0
    model_ablation_off.sym_weight    = 0.0

    trainer_on = NOETHERTrainer(model_ablation_on, lr=1e-3, device='cpu')
    trainer_off = NOETHERTrainer(model_ablation_off, lr=1e-3, device='cpu')

    # Fixed dataset for ablation run
    set_seed(42)
    ab_ids = torch.randint(0, 1000, (4, 32))
    ab_labels = torch.randint(0, 1000, (4, 32))

    for step in range(20):
        bd_on = trainer_on.train_step(ab_ids, ab_labels)
        bd_off = trainer_off.train_step(ab_ids, ab_labels)

    # Compare final total loss and final logits
    model_ablation_on.eval()
    model_ablation_off.eval()

    with torch.no_grad():
        out_on = model_ablation_on(ab_ids)
        out_off = model_ablation_off(ab_ids)

    final_loss_on = bd_on['total']
    final_loss_off = bd_off['total']
    
    logits_diff = torch.abs(out_on['logits'] - out_off['logits']).mean().item()
    loss_diff = abs(final_loss_on - final_loss_off)

    print(f"Model WITH NOETHER: final loss = {final_loss_on:.6f}")
    print(f"Model WITHOUT NOETHER: final loss = {final_loss_off:.6f}")
    print(f"Absolute loss difference = {loss_diff:.6f}")
    print(f"Mean absolute difference in logits = {logits_diff:.6f}")

if __name__ == "__main__":
    run_extended_verification()
