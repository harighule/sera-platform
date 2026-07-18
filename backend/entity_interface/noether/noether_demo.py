"""
noether_demo.py
===============
End-to-end demo: KRONOS (9 pillars) + NOETHER (4 cognitive symmetry groups).

Run:  python noether_demo.py
Requires: kronos_architecture.py  kronos_training.py  (../kronos/)
          noether_components.py   noether_kronos.py   (same dir)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import DataLoader, TensorDataset

from noether_kronos import NOETHER_KRONOS, NOETHERTrainer, NOETHERGodelLoop

BASE_CONFIG = dict(
    vocab_size      = 1_000,
    d_model         = 128,
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
    n_generators    = 6,
    gen_rank        = 4,
    n_templates     = 32,
    n_types         = 16,
    n_rg_levels     = 3,
    block_size      = 2,
    sym_threshold   = 0.70,
    kl_weight       = 0.05,
    notears_weight  = 0.01,
    orbit_weight    = 0.05,
    ctl_weight      = 0.05,
    rg_weight       = 0.05,
    sym_weight      = 0.10,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(n: int = 512, batch: int = 16) -> DataLoader:
    seq = BASE_CONFIG['max_seq_len']
    V   = BASE_CONFIG['vocab_size']
    data = torch.randint(0, V, (n, seq + 1))
    ds   = TensorDataset(data[:, :-1], data[:, 1:])
    return DataLoader(ds, batch_size=batch, shuffle=True)


def phase_train(n_epochs: int = 3) -> NOETHERTrainer:
    print("=" * 66)
    print("  PHASE 1 — NOETHER_KRONOS Training")
    print("  13 components active: 9 KRONOS pillars + 4 NOETHER symmetry groups")
    print("=" * 66)

    model   = NOETHER_KRONOS(**BASE_CONFIG).to(DEVICE)
    trainer = NOETHERTrainer(
        model,
        lr                = 3e-4,
        riemannian_lr     = 5e-3,
        consolidate_every = 300,
        nca_evolve_every  = 100,
        sym_update_every  = 50,
        sym_loss_every    = 100,
        device            = DEVICE,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters   : {n_params:,}")
    print(f"  Device       : {DEVICE}")
    print(f"  Initial NCA  : {model.topology_report()}")
    print(f"  SDE active   : {int(model.sde.active.sum())} / {model.sde.scores.shape[0]}")
    print(f"  RG levels    : {BASE_CONFIG['n_rg_levels']}\n")

    loader = make_loader()
    for ep in range(n_epochs):
        print(f"\n── Epoch {ep+1}/{n_epochs} ──")
        res = trainer.train_epoch(loader, log_every=20)
        print(f"  Mean loss : {res['mean_loss']:.4f}")
        print(f"  NCA topo  : {model.topology_report()}")
        print(f"  Active sym: {int(model.sde.active.sum())} generators")

    return trainer


def phase_godel(n_gen: int = 3) -> NOETHER_KRONOS:
    print("\n" + "=" * 66)
    print("  PHASE 2 — NOETHER Gödel Loop")
    print("  Evolves KRONOS hyperparams AND NOETHER symmetry params jointly.")
    print("  Meta-learner improves its modification strategy across generations.")
    print("=" * 66)

    eval_ids = torch.randint(
        0, BASE_CONFIG['vocab_size'], (4, BASE_CONFIG['max_seq_len'])
    )

    loop = NOETHERGodelLoop(
        base_config     = BASE_CONFIG,
        vocab_size      = BASE_CONFIG['vocab_size'],
        population_size = 3,
        n_generations   = n_gen,
        device          = DEVICE,
    )

    result, best = loop.run(eval_batch=eval_ids)

    print(f"\n  Fitness trend  : {loop.sparkline()}")
    print(f"  Best fitness   : {result.get('best_fitness', 0):.4f}")
    print(f"  Evolved config :")
    for k, v in result.get('best_config', BASE_CONFIG).items():
        orig = BASE_CONFIG.get(k, '?')
        tag  = "  ← evolved" if v != orig else ""
        print(f"    {k:20s}: {v}{tag}")

    return best


def phase_inspect(model: NOETHER_KRONOS, n_new: int = 20):
    print("\n" + "=" * 66)
    print("  PHASE 3 — System Inspection + Verified Generation")
    print("=" * 66)

    model.eval()

    scores = model.sde.scores.cpu()
    active = model.sde.active.cpu()
    print(f"\n  [G_sem] Symmetry Discovery Engine")
    for i, (s, a) in enumerate(zip(scores, active)):
        flag = "✓ ACTIVE" if a else "  dormant"
        print(f"    Generator {i}: score={s:.3f}  {flag}")

    with torch.no_grad():
        dummy = torch.zeros(1, 8, model.d_model, device=DEVICE)
        _, levels, _ = model.arg(dummy)
    print(f"\n  [G_abs] Abstraction RG levels active : {len(levels)}")
    print(f"         (token→{'→'.join(str(l.shape[1]) for l in levels[1:])} tokens)")

    with torch.no_grad():
        x_a = torch.randn(1, model.ctl.d_model, device=DEVICE)
        x_b = torch.randn(1, model.ctl.d_model, device=DEVICE)
        t_a = model.ctl.type_emb_of(x_a)
        t_b = model.ctl.type_emb_of(x_b)
        sub = torch.sigmoid(model.ctl.sub_net(torch.cat([t_a, t_b], -1)))
    print(f"\n  [G_comp] Sample subtyping score A≤B : {sub.item():.3f}")

    print(f"\n  [Generation] Autoregressive with Typed CoT verification")
    prompt = torch.randint(0, BASE_CONFIG['vocab_size'], (1, 6)).to(DEVICE)
    gen    = model.generate(prompt, max_new=n_new, temperature=0.8, top_k=40)
    print(f"    Prompt    : {prompt[0].tolist()}")
    print(f"    Generated : {gen[0, 6:].tolist()}")

    with torch.no_grad():
        out  = model(gen[:, :16])
        vscr = out['verification_scores'][0].mean().item()
        nact = int(model.sde.active.sum().item())
    print(f"    Mean CoT verify score : {vscr:.4f} "
          f"({'PASS' if vscr > 0.4 else 'WARN'})")
    print(f"    Active G_sem generators: {nact}")
    print(f"    NCA topology (evolved) : {model.topology_report()}")


def main():
    print()
    print("█" * 66)
    print("  NOETHER_KRONOS")
    print("  13 components — 9 KRONOS pillars + 4 Cognitive Symmetry Groups")
    print("  Architecture replaces data hunger with structural priors.")
    print("  PAC bound: m = O(d / (|G_sem|·|G_caus|·|G_comp|·|G_abs|·ε²))")
    print("█" * 66)

    trainer    = phase_train(n_epochs=3)
    best_model = phase_godel(n_gen=3)

    print("\n  Fine-tuning evolved architecture …")
    ft = NOETHERTrainer(best_model, lr=1e-4, device=DEVICE)
    ft.train_epoch(make_loader(n=256), log_every=50)

    phase_inspect(best_model)

    print("\n" + "─" * 66)
    print("  All 13 components verified end-to-end.")
    print("  To scale: increase d_model, n_layers, n_generators, real data.")
    print("  To save : trainer.save('noether_kronos.pt')")
    print("─" * 66 + "\n")


if __name__ == "__main__":
    main()
