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

# Set random seed for reproducibility
torch.manual_seed(42)

def run_checks():
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

    print("--------------------------------------------------")
    print("CHECK 1: Input sensitivity of each of the 5 components")
    print("--------------------------------------------------")
    model = NOETHER_KRONOS(**config)
    model.eval()

    # Create two different realistic token sequences
    # Sequence 1: Repetitive small token indices
    input_ids1 = torch.tensor([[1, 2, 1, 2, 3, 4, 3, 4]], dtype=torch.long)
    # Sequence 2: Repetitive large token indices
    input_ids2 = torch.tensor([[900, 901, 900, 901, 902, 903, 902, 903]], dtype=torch.long)

    # Hooks to capture intermediate values
    captured = {}
    
    def get_hook(name):
        def hook(module, input, output):
            # output could be tuple or tensor
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
    sde_out1 = captured["SDE"] # tuple: (orbit_rep, scores)
    soe_out1 = captured["SOE"] # tuple: (encoded, loss)
    cfn_out1 = captured["CFN"] # tensor: [B, T, D]
    arg_out1 = captured["ARG"] # tuple: (multi_scale, levels, loss)
    ctl_out1 = captured["CTL"] # tuple: (enriched, loss)

    # Clone outputs for Input 1
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

    # Calculate differences
    def report_diff(name, t1, t2):
        mean1, mean2 = t1.mean().item(), t2.mean().item()
        std1, std2 = t1.std().item(), t2.std().item()
        diff = torch.abs(t1 - t2).mean().item()
        print(f"{name:10s} | Input1: mean={mean1:.6f}, std={std1:.6f} | Input2: mean={mean2:.6f}, std={std2:.6f} | AbsDiff={diff:.6f}")
        return diff > 1e-5

    sens_sde = report_diff("SDE", sde_rep1, sde_rep2)
    sens_soe = report_diff("SOE", soe_rep1, soe_rep2)
    sens_cfn = report_diff("CFN", cfn_rep1, cfn_rep2)
    sens_ctl = report_diff("CTL", ctl_rep1, ctl_rep2)
    sens_arg = report_diff("ARG", arg_rep1, arg_rep2)

    print("\n--------------------------------------------------")
    print("CHECK 2: Loss behavior under real training (20 steps)")
    print("--------------------------------------------------")
    torch.manual_seed(42)
    model_train = NOETHER_KRONOS(**config)
    trainer = NOETHERTrainer(model_train, lr=1e-3, device='cpu')

    # Fixed training inputs
    train_ids = torch.randint(0, 1000, (4, 32))
    train_labels = torch.randint(0, 1000, (4, 32))

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

    # Check if losses moved sensibly
    def check_sensible(name, l1, l20):
        moved = abs(l1 - l20) > 1e-7
        non_zero = abs(l1) > 1e-7
        non_nan = not (torch.isnan(torch.tensor(l1)) or torch.isnan(torch.tensor(l20)))
        print(f"{name:10s} | Step 1: {l1:.6f} | Step 20: {l20:.6f} | Moved: {moved} | Non-zero: {non_zero} | Non-NaN: {non_nan}")
        return moved and non_zero and non_nan

    loss_orbit = check_sensible("orbit_loss", step1_losses['orbit'], step20_losses['orbit'])
    loss_ctl = check_sensible("ctl_loss", step1_losses['ctl'], step20_losses['ctl'])
    loss_rg = check_sensible("rg_loss", step1_losses['rg'], step20_losses['rg'])

    print("\n--------------------------------------------------")
    print("CHECK 3: Symmetry Discovery Engine Sanity Check")
    print("--------------------------------------------------")
    # Reset/clear active states
    with torch.no_grad():
        # Get actual input embeddings
        pos = torch.arange(8, device='cpu').unsqueeze(0)
        x_real = model_train.kronos.tok_emb(input_ids1) + model_train.kronos.pos_emb(pos)
        
        # Test real data
        model_train.sde.scores.zero_()
        model_train.sde.active.zero_()
        model_train.sde.update_active(x_real)
        real_scores = model_train.sde.scores.clone()
        real_active = model_train.sde.active.clone()
        
        # Test pure noise
        x_noise = torch.randn_like(x_real)
        model_train.sde.scores.zero_()
        model_train.sde.active.zero_()
        model_train.sde.update_active(x_noise)
        noise_scores = model_train.sde.scores.clone()
        noise_active = model_train.sde.active.clone()

    print(f"Real Data Scores: {real_scores.tolist()} | Active: {real_active.tolist()}")
    print(f"Pure Noise Scores: {noise_scores.tolist()} | Active: {noise_active.tolist()}")
    
    same_active = torch.equal(real_active, noise_active)
    print(f"Same active generators on real vs noise: {same_active}")

    print("\n--------------------------------------------------")
    print("CHECK 4: Ablation Test")
    print("--------------------------------------------------")
    # Fresh models initialized to the same state
    torch.manual_seed(42)
    model_ablation_on = NOETHER_KRONOS(**config)
    
    # Clone state dict to another model
    model_ablation_off = NOETHER_KRONOS(**config)
    model_ablation_off.load_state_dict(model_ablation_on.state_dict())

    # Turn off NOETHER losses in config_off
    model_ablation_off.orbit_weight = 0.0
    model_ablation_off.ctl_weight    = 0.0
    model_ablation_off.rg_weight     = 0.0
    model_ablation_off.sym_weight    = 0.0

    trainer_on = NOETHERTrainer(model_ablation_on, lr=1e-3, device='cpu')
    trainer_off = NOETHERTrainer(model_ablation_off, lr=1e-3, device='cpu')

    # Train both for 20 steps on same batch
    torch.manual_seed(42)
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
    run_checks()
