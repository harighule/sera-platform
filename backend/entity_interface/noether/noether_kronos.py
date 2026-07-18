"""
noether_kronos.py
=================
Integration status
------------------
This module is a standalone research implementation.

  Default (USE_NOETHER=false): NOT connected to the live request path.
    LiveEntity uses plain KRONOS as its kronos_model.

  Optional (USE_NOETHER=true):  WIRED into LiveEntity.
    Set the environment variable USE_NOETHER=true (see config.py) to activate
    NOETHER_KRONOS in place of plain KRONOS for the live entity layer.

NOETHER_KRONOS: Full unified system
-------------------------------------
KRONOS  (9 pillars)  +  NOETHER  (4 cognitive symmetry groups)
= 13 components operating in a single differentiable forward pass.

Integration map:
  Pre-embedding  : SDE → orbit discovery   (G_sem layer 1)
                   SOE → orbit encoding    (G_sem layer 2)
  Per-layer      : CFN → causal fibration  (G_caus, after each KRONOS layer)
  Post-layers    : ARG → abstraction RG    (G_abs)
                   CTL → type lattice      (G_comp, before KRONOS verifier)

Combined loss:
  L = L_CE + L_KL + L_NOTEARS + L_verify        (KRONOS)
    + λ_orbit·L_orbit + λ_ctl·L_ctl             (NOETHER G_sem, G_comp)
    + λ_rg·L_rg                                  (NOETHER G_abs)
    + λ_sym·L_sym  (computed periodically)        (NOETHER G_sem extra)
"""

import math
import sys
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from entity_interface.kronos.kronos_architecture import KRONOS, PoincareBall
from entity_interface.kronos.kronos_training import RiemannianAdagrad, KRONOSTrainer
from noether_components import (
    SymmetryDiscoveryEngine,
    SemanticOrbitEncoder,
    CausalFibrationNetwork,
    CompositionalTypeLattice,
    AbstractionRG,
)


# ─────────────────────────────────────────────────────────────
# NOETHER_KRONOS — UNIFIED MODEL
# ─────────────────────────────────────────────────────────────

class NOETHER_KRONOS(nn.Module):
    """
    Full unified model: KRONOS (9 pillars) + NOETHER (4 cognitive symmetry groups).

    All components share a single forward pass and contribute to a unified
    multi-term loss that simultaneously:
      - Models language (CE loss)
      - Maintains a causal world model (KL / ELBO)
      - Enforces causal acyclicity (NOTEARS)
      - Verifies reasoning type consistency (Curry-Howard)
      - Enforces semantic orbit equivariance (G_sem)
      - Learns causal structural templates (G_caus)
      - Enforces compositional functor rules (G_comp)
      - Maintains multi-scale RG consistency (G_abs)
      - Evolves its own architecture (Gödel loop, via NOETHERTrainer)
    """

    def __init__(
        self,
        # ── KRONOS params ──────────────────────────────────────
        vocab_size:     int   = 10_000,
        d_model:        int   = 256,
        n_heads:        int   = 8,
        n_layers:       int   = 4,
        d_ff:           int   = 1024,
        max_seq_len:    int   = 256,
        memory_size:    int   = 128,
        z_dim:          int   = 128,
        n_slots:        int   = 8,
        n_wave_freqs:   int   = 32,
        dropout:        float = 0.1,
        notears_coeff:  float = 0.01,
        # ── NOETHER params ─────────────────────────────────────
        n_generators:   int   = 8,
        gen_rank:       int   = 4,
        n_templates:    int   = 64,
        n_types:        int   = 24,
        n_rg_levels:    int   = 3,
        block_size:     int   = 2,
        sym_threshold:  float = 0.75,
        # ── Loss weights ───────────────────────────────────────
        kl_weight:      float = 0.05,
        notears_weight: float = 0.01,
        orbit_weight:   float = 0.05,
        ctl_weight:     float = 0.05,
        rg_weight:      float = 0.05,
        sym_weight:     float = 0.10,
    ):
        super().__init__()

        # Store for loss weighting
        self.d_model        = d_model
        self.n_layers       = n_layers
        self.kl_weight      = kl_weight
        self.notears_weight = notears_weight
        self.orbit_weight   = orbit_weight
        self.ctl_weight     = ctl_weight
        self.rg_weight      = rg_weight
        self.sym_weight     = sym_weight

        # ── Base KRONOS (all 9 pillars) ────────────────────────
        self.kronos = KRONOS(
            vocab_size=vocab_size, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
            max_seq_len=max_seq_len, memory_size=memory_size,
            z_dim=z_dim, n_slots=n_slots, n_wave_freqs=n_wave_freqs,
            dropout=dropout, kl_weight=kl_weight,
            notears_weight=notears_weight, notears_coeff=notears_coeff,
        )

        # ── G_sem — Symmetry Discovery Engine + Semantic Orbit Encoder
        self.sde = SymmetryDiscoveryEngine(
            d_model, n_generators, gen_rank, sym_threshold
        )
        self.soe = SemanticOrbitEncoder(d_model)

        # ── G_caus — Causal Fibration Network (shared across layers)
        self.cfn = CausalFibrationNetwork(d_model, n_templates=n_templates)

        # ── G_comp — Compositional Type Lattice
        self.ctl = CompositionalTypeLattice(d_model, n_types)

        # ── G_abs — Abstraction Renormalisation Group
        self.arg = AbstractionRG(d_model, n_rg_levels, block_size)

    # ──────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:  torch.Tensor,
        mask:       Optional[torch.Tensor] = None,
        h_states:   Optional[List[Optional[torch.Tensor]]] = None,
        evolve_nca: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full KRONOS+NOETHER forward pass.

        input_ids: [B, T]
        mask:      [B, T]
        h_states:  per-layer world states (streaming inference)
        → dict with logits, losses, auxiliary outputs
        """
        B, T   = input_ids.shape
        device = input_ids.device

        # ── Step 1: Base embeddings (from KRONOS) ─────────────
        pos = torch.arange(T, device=device).unsqueeze(0)
        x   = self.kronos.drop(
            self.kronos.tok_emb(input_ids) + self.kronos.pos_emb(pos)
        )

        # ── Step 2: G_sem — SDE orbit compression ─────────────
        orbit_rep, sym_scores = self.sde(x)

        # ── Step 3: G_sem — SOE orbit-invariant encoding ───────
        x, orbit_loss = self.soe(x, orbit_rep)

        # ── Step 4: Pillar 1 — Riemannian Wave Manifold ────────
        ctx   = x.mean(1)
        wave  = self.kronos.wave(ctx)
        gate  = torch.sigmoid(self.kronos.w_gate(ctx))
        x     = x + (gate * wave).unsqueeze(1) * 0.1

        # ── Step 5: Pillars 2–4, 7–8 — KRONOS layer stack
        #            + G_caus CFN after every layer ───────────
        if h_states is None:
            h_states = [None] * self.n_layers

        kl_total = torch.zeros(1, device=device)
        nt_total = torch.zeros(1, device=device)
        h_new    = []
        x_prev   = None

        for i, layer in enumerate(self.kronos.layers):
            x, h_i, kl, nt = layer(
                x, h=h_states[i], x_prev=x_prev,
                mask=mask, evolve_nca=evolve_nca,
            )
            kl_total = kl_total + kl
            nt_total = nt_total + nt
            h_new.append(h_i)

            # G_caus: inject structural template-content separation
            adj = torch.sigmoid(
                layer.causal_attn.W_logit[:T, :T]
            ).detach()
            x = self.cfn(x, causal_adj=adj)

            x_prev = x

        # ── Step 6: G_abs — Abstraction Renormalisation Group ──
        x, levels, rg_loss = self.arg(x)

        # ── Step 7: Pillar 5 — Neuro-Symbolic Grounding ────────
        x = self.kronos.symbolic(x)

        # ── Step 8: G_comp — Compositional Type Lattice ────────
        x, ctl_loss = self.ctl(x)

        # ── Step 9: Pillar 9 — Typed CoT Verification ──────────
        x, v_scores = self.kronos.verifier(x)

        # ── Step 10: Output ─────────────────────────────────────
        logits = self.kronos.head(self.kronos.out_norm(x))

        return {
            "logits":               logits,
            "h_new":                h_new,
            "kl_loss":              (kl_total / self.n_layers) * self.kl_weight,
            "notears_penalty":      (nt_total / self.n_layers) * self.notears_weight,
            "verification_scores":  v_scores,
            "orbit_loss":           orbit_loss * self.orbit_weight,
            "ctl_loss":             ctl_loss   * self.ctl_weight,
            "rg_loss":              rg_loss    * self.rg_weight,
            "sym_scores":           sym_scores,
            "rg_levels":            len(levels),
        }

    # ──────────────────────────────────────────────────────────
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels:    torch.Tensor,
        mask:      Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Full NOETHER loss:
          L = L_CE
            + L_KL          (Pillar 4: world-state ELBO)
            + L_NOTEARS     (Pillar 2: DAG acyclicity)
            + L_verify      (Pillar 9: type consistency)
            + L_orbit       (G_sem: InfoNCE orbit contrastive)
            + L_ctl         (G_comp: composition + transitivity)
            + L_rg          (G_abs: RG level consistency)
        """
        out    = self.forward(input_ids, mask, **kwargs)
        logits = out["logits"]

        ce    = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        v_pen = (1 - out["verification_scores"]).mean() * 0.01

        total = (ce
                 + out["kl_loss"]
                 + out["notears_penalty"]
                 + v_pen
                 + out["orbit_loss"]
                 + out["ctl_loss"]
                 + out["rg_loss"])

        return total, {
            "ce":       ce.item(),
            "kl":       out["kl_loss"].item(),
            "notears":  out["notears_penalty"].item(),
            "verify":   v_pen.item(),
            "orbit":    out["orbit_loss"].item(),
            "ctl":      out["ctl_loss"].item(),
            "rg":       out["rg_loss"].item(),
            "rg_lvls":  out["rg_levels"],
            "n_sym":    int(self.sde.active.sum().item()),
            "total":    total.item(),
        }

    # ──────────────────────────────────────────────────────────
    def consolidate_memory(self):
        self.kronos.consolidate_memory()

    def topology_report(self) -> Dict[str, int]:
        return self.kronos.topology_report()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids:  torch.Tensor,
        max_new:     int   = 64,
        temperature: float = 0.8,
        top_k:       int   = 50,
    ) -> torch.Tensor:
        """Autoregressive generation through the full KRONOS+NOETHER stack."""
        self.eval()
        ids     = prompt_ids.clone()
        h_state = None
        for _ in range(max_new):
            out     = self.forward(ids, h_states=h_state)
            h_state = out["h_new"]
            logits  = out["logits"][:, -1] / max(temperature, 1e-6)
            if top_k:
                v, _ = logits.topk(top_k, dim=-1)
                logits[logits < v[:, -1:]] = float('-inf')
            next_id = torch.multinomial(F.softmax(logits, -1), 1)
            ids     = torch.cat([ids, next_id], dim=-1)
        return ids


# ─────────────────────────────────────────────────────────────
# NOETHER TRAINER
# ─────────────────────────────────────────────────────────────

class NOETHERTrainer:
    """
    Unified trainer for NOETHER_KRONOS.

    Extends KRONOSTrainer with:
      • Periodic SDE symmetry discovery update
      • Periodic symmetry equivariance loss (L_sym) computation
      • Separate Riemannian optimiser for KRONOS wave manifold params
      • Full loss breakdown logging (all 8 loss terms)
      • Topology + symmetry reporting

    Optimizer split:
      Riemannian AdaGrad : wave manifold base_point, tangent_vecs
      AdamW              : all other parameters
    """

    def __init__(
        self,
        model:              NOETHER_KRONOS,
        lr:                 float = 3e-4,
        riemannian_lr:      float = 1e-2,
        weight_decay:       float = 0.01,
        max_grad_norm:      float = 1.0,
        consolidate_every:  int   = 1_000,
        nca_evolve_every:   int   = 500,
        sym_update_every:   int   = 200,
        sym_loss_every:     int   = 100,
        device:             str   = 'cpu',
    ):
        self.model             = model.to(device)
        self.device            = device
        self.max_grad_norm     = max_grad_norm
        self.consolidate_every = consolidate_every
        self.nca_evolve_every  = nca_evolve_every
        self.sym_update_every  = sym_update_every
        self.sym_loss_every    = sym_loss_every
        self.step              = 0
        self.history: List[Dict] = []

        # ── Parameter split ────────────────────────────────────
        riem_params, std_params = [], []
        for name, p in model.named_parameters():
            if 'base_point' in name or 'tangent_vecs' in name:
                p.manifold = 'poincare'
                riem_params.append(p)
            else:
                std_params.append(p)

        self.opt = torch.optim.AdamW(
            std_params, lr=lr,
            weight_decay=weight_decay, betas=(0.9, 0.95)
        )
        self.opt_r = RiemannianAdagrad(
            riem_params, lr=riemannian_lr, curvature=1.0
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=200_000, eta_min=lr * 0.05
        )

    # ──────────────────────────────────────────────────────────
    def _project_manifold(self):
        """Keep Poincaré ball params inside the ball after each update."""
        with torch.no_grad():
            bp = self.model.kronos.wave.base_point
            bp.data.copy_(
                self.model.kronos.wave.ball.project(bp.data)
            )

    # ──────────────────────────────────────────────────────────
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels:    torch.Tensor,
        mask:      Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.model.train()

        ids = input_ids.to(self.device)
        lab = labels.to(self.device)
        msk = mask.to(self.device) if mask is not None else None

        evolve_nca = (self.step % self.nca_evolve_every == 0)

        # ── Periodic symmetry discovery update (no grad) ───────
        if self.step % self.sym_update_every == 0:
            with torch.no_grad():
                pos = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
                x_emb = self.model.kronos.tok_emb(ids) + \
                        self.model.kronos.pos_emb(pos)
                self.model.sde.update_active(x_emb)

        # ── Main forward + loss ─────────────────────────────────
        self.opt.zero_grad()
        self.opt_r.zero_grad()

        loss, bd = self.model.compute_loss(ids, lab, msk,
                                           evolve_nca=evolve_nca)

        # ── Periodic symmetry equivariance loss ─────────────────
        sym_loss_val = 0.0
        if self.step % self.sym_loss_every == 0 and \
           int(self.model.sde.active.sum()) > 0:
            with torch.no_grad():
                pos   = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
                x_raw = self.model.kronos.tok_emb(ids) + \
                        self.model.kronos.pos_emb(pos)

            # Proxy fn: first KRONOS layer only (cheap)
            def proxy(x_in):
                out, _, _, _ = self.model.kronos.layers[0](x_in)
                return out

            sym_loss = self.model.sde.equivariance_loss(
                x_raw.detach(), proxy
            ) * self.model.sym_weight

            loss = loss + sym_loss
            sym_loss_val = sym_loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.opt.step()
        self.opt_r.step()
        self.sched.step()
        self._project_manifold()

        # ── Artificial sleep ────────────────────────────────────
        if self.step > 0 and self.step % self.consolidate_every == 0:
            self.model.consolidate_memory()

        bd['sym'] = sym_loss_val
        bd['total'] = bd['total'] + sym_loss_val
        self.history.append(bd)
        self.step += 1
        return bd

    # ──────────────────────────────────────────────────────────
    def train_epoch(
        self,
        dataloader,
        log_every: int = 50,
    ) -> Dict[str, float]:
        losses = []
        t0     = time.time()

        for i, batch in enumerate(dataloader):
            if len(batch) == 2:
                ids, lab, msk = batch[0], batch[1], None
            else:
                ids, lab, msk = batch[0], batch[1], batch[2]

            bd = self.train_step(ids, lab, msk)
            losses.append(bd['total'])

            if i % log_every == 0:
                topo = self.model.topology_report()
                n_sym = int(self.model.sde.active.sum().item())
                print(
                    f"  step {self.step:>6} | "
                    f"tot={bd['total']:.4f} | "
                    f"ce={bd['ce']:.4f} | "
                    f"kl={bd['kl']:.4f} | "
                    f"nt={bd['notears']:.4f} | "
                    f"orbit={bd['orbit']:.4f} | "
                    f"ctl={bd['ctl']:.4f} | "
                    f"rg={bd['rg']:.4f}({bd['rg_lvls']}lv) | "
                    f"sym={bd['sym']:.4f}({n_sym}act) | "
                    f"topo={list(topo.values())} | "
                    f"{time.time()-t0:.1f}s"
                )
                t0 = time.time()

        return {"mean_loss": sum(losses) / max(len(losses), 1)}

    # ──────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "step":    self.step,
            "model":   self.model.state_dict(),
            "opt":     self.opt.state_dict(),
            "opt_r":   self.opt_r.state_dict(),
            "sched":   self.sched.state_dict(),
            "history": self.history[-500:],
        }, path)
        print(f"✓ Saved → {path}  (step {self.step})")

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.opt_r.load_state_dict(ck["opt_r"])
        self.sched.load_state_dict(ck["sched"])
        self.step    = ck["step"]
        self.history = ck.get("history", [])
        print(f"✓ Loaded ← {path}  (step {self.step})")


# ─────────────────────────────────────────────────────────────
# NOETHER GÖDEL LOOP  (extends KRONOS GodelLoop)
# ─────────────────────────────────────────────────────────────

class NOETHERGodelLoop:
    """
    Gödel self-modification loop extended for NOETHER_KRONOS.

    Evolves BOTH KRONOS hyperparams AND NOETHER symmetry params:
      KRONOS:  d_ff, memory_size, z_dim, n_wave_freqs, kl_weight …
      NOETHER: n_generators, gen_rank, n_templates, n_types, n_rg_levels

    The meta-learner predicts which joint configuration achieves
    the best fitness (topology richness + verification + symmetry activity).
    The modifier improves itself recursively across generations.
    """
    import copy

    _MUTABLE_KRONOS = {
        "d_ff":          (128,  2048),
        "memory_size":   (32,   512),
        "z_dim":         (32,   256),
        "n_wave_freqs":  (8,    64),
        "kl_weight":     (0.01, 0.3),
        "notears_weight":(0.001,0.1),
    }
    _MUTABLE_NOETHER = {
        "n_generators":  (4,    16),
        "gen_rank":      (2,    8),
        "n_templates":   (16,   128),
        "n_types":       (8,    48),
        "n_rg_levels":   (1,    5),
    }

    def __init__(
        self,
        base_config:     Dict,
        vocab_size:      int  = 10_000,
        population_size: int  = 3,
        n_generations:   int  = 6,
        device:          str  = 'cpu',
    ):
        import copy
        self.base      = base_config
        self.vocab     = vocab_size
        self.pop_size  = population_size
        self.n_gen     = n_generations
        self.device    = device
        self.gen       = 0
        self.fitness_h: List[float] = []
        self.best_cfg: Optional[Dict] = None

        self.pop = [copy.deepcopy(base_config)] + [
            self._perturb(base_config) for _ in range(population_size - 1)
        ]

        feat_dim = len(self._MUTABLE_KRONOS) + len(self._MUTABLE_NOETHER)
        self.meta     = nn.Sequential(
            nn.Linear(feat_dim, 32), nn.GELU(),
            nn.Linear(32, 1)
        ).to(device)
        self.meta_opt = torch.optim.Adam(self.meta.parameters(), lr=1e-3)

    # ------------------------------------------------------------------
    @staticmethod
    def _clip(v, lo, hi):
        return max(type(lo)(lo), min(type(hi)(hi), type(lo)(v)))

    def _perturb(self, cfg: Dict, scale: float = 0.2) -> Dict:
        import copy
        out = copy.deepcopy(cfg)
        for k, (lo, hi) in {**self._MUTABLE_KRONOS,
                             **self._MUTABLE_NOETHER}.items():
            if k not in out:
                continue
            v = out[k]
            if isinstance(v, int):
                d = max(1, int(abs(v) * scale * abs(torch.randn(1).item())))
                out[k] = self._clip(v + (d if torch.rand(1) > 0.5 else -d),
                                    lo, hi)
            else:
                out[k] = self._clip(
                    v * (1 + scale * torch.randn(1).item()), lo, hi
                )
        return out

    def _feat(self, cfg: Dict) -> torch.Tensor:
        keys = list(self._MUTABLE_KRONOS) + list(self._MUTABLE_NOETHER)
        return torch.tensor(
            [float(cfg.get(k, 0)) for k in keys],
            dtype=torch.float32, device=self.device
        )

    def _build(self, cfg: Dict) -> Optional[NOETHER_KRONOS]:
        try:
            return NOETHER_KRONOS(
                vocab_size=self.vocab, **cfg
            ).to(self.device)
        except Exception as e:
            print(f"    [build failed: {e}]")
            return None

    def _fitness(self, m: NOETHER_KRONOS,
                 eval_batch: Optional[torch.Tensor] = None) -> float:
        m.eval()
        scores = []
        with torch.no_grad():
            topo   = m.topology_report()
            scores.append(sum(topo.values()) / (len(topo) * 32 + 1e-6))
            n_sym  = float(m.sde.active.sum().item()) / m.sde.scores.shape[0]
            scores.append(n_sym)
            if eval_batch is not None:
                try:
                    out = m(eval_batch.to(self.device))
                    scores.append(out["verification_scores"].mean().item())
                except Exception:
                    scores.append(0.0)
            n_p = sum(p.numel() for p in m.parameters())
            scores.append(1.0 / (n_p / 1e6 + 1.0))
        return float(sum(scores) / len(scores))

    def _update_meta(self, parent: Dict, child: Dict, delta: float):
        diff = (self._feat(child) - self._feat(parent)).unsqueeze(0)
        pred = self.meta(diff)
        loss = F.mse_loss(pred, torch.tensor([[delta]], device=self.device))
        self.meta_opt.zero_grad(); loss.backward(); self.meta_opt.step()

    def _propose(self, parent: Dict) -> Dict:
        pf   = self._feat(parent)
        best, best_cfg = -1e9, parent
        for cand in [self._perturb(parent) for _ in range(8)]:
            with torch.no_grad():
                pred = self.meta(
                    (self._feat(cand) - pf).unsqueeze(0)
                ).item()
            if pred > best:
                best, best_cfg = pred, cand
        return best_cfg

    def run(self, eval_batch: Optional[torch.Tensor] = None
            ) -> Tuple[Dict, NOETHER_KRONOS]:
        import copy
        result = {}
        for _ in range(self.n_gen):
            print(f"\n{'─'*64}")
            print(f"  NOETHER Gödel Loop — Generation {self.gen}")
            print(f"{'─'*64}")

            fits = []
            for i, cfg in enumerate(self.pop):
                m = self._build(cfg)
                f = self._fitness(m, eval_batch) if m else 0.0
                fits.append(f)
                topo = str(list(m.topology_report().values())) if m else "—"
                n_s  = int(m.sde.active.sum()) if m else 0
                print(f"  Variant {i}: fit={f:.4f}  topo={topo}  sym_active={n_s}")
                del m

            ranked   = sorted(zip(fits, self.pop), key=lambda x: x[0], reverse=True)
            n_surv   = max(1, self.pop_size // 2)
            survivors = ranked[:n_surv]
            best_fit, best_cfg = survivors[0]
            self.fitness_h.append(best_fit)
            self.best_cfg = copy.deepcopy(best_cfg)

            print(f"\n  ★ Best fitness: {best_fit:.4f}")

            new_pop = [copy.deepcopy(best_cfg)]
            for pf, pcfg in survivors:
                if len(new_pop) >= self.pop_size:
                    break
                child = self._propose(pcfg)
                cm    = self._build(child)
                if cm:
                    cf    = self._fitness(cm, eval_batch)
                    self._update_meta(pcfg, child, cf - pf)
                    del cm
                new_pop.append(child)

            while len(new_pop) < self.pop_size:
                new_pop.append(self._perturb(best_cfg, scale=0.1))
            self.pop = new_pop
            self.gen += 1

            if len(self.fitness_h) >= 4:
                if (self.fitness_h[-1] - self.fitness_h[-4]) / 3 < 1e-5:
                    print("  [Converged — stopping]")
                    break

            result = {"best_config": best_cfg, "best_fitness": best_fit,
                      "fitness_history": self.fitness_h}

        best = self._build(result.get("best_config", self.base))
        if best is None:
            best = self._build(self.base)
        return result, best

    def sparkline(self) -> str:
        cs = " ▁▂▃▄▅▆▇█"
        if not self.fitness_h:
            return "(no data)"
        lo, hi = min(self.fitness_h), max(self.fitness_h)
        sp = max(hi - lo, 1e-9)
        return "".join(
            cs[int((f - lo) / sp * (len(cs) - 1))]
            for f in self.fitness_h
        )
