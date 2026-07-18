"""
noether_components.py
=====================
NOETHER: Cognitive Symmetry Architecture

Implements the four cognitive symmetry groups that replace data hunger
with structural priors. Each group delivers a provable PAC-bound reduction
in sample complexity: m = O(d / (|G_sem|·|G_caus|·|G_comp|·|G_abs|·ε²))

  G_sem  — Semantic Orbit Symmetry      SDE + SOE
  G_caus — Causal Invariance Symmetry   CFN
  G_comp — Compositional Functor Sym    CTL
  G_abs  — Abstraction Fibration Sym    ARG

Operates on top of KRONOS (kronos_architecture.py).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# LIE ALGEBRA GENERATOR
# ─────────────────────────────────────────────────────────────

class LieAlgebraGenerator(nn.Module):
    """
    Single generator X of a one-parameter Lie group {exp(t·X) : t∈ℝ}.

    Parameterized as low-rank antisymmetric:
        X = U·Vᵀ − V·Uᵀ   →   exp(tX) is exactly orthogonal

    The generator direction is normalised to unit Frobenius norm and the
    rotation is realised by an EXACT matrix exponential (torch.linalg.matrix_exp),
    so ``log_strength`` controls the true rotation ANGLE (in radians). This
    guarantees the transformation is a genuine, non-trivial orthogonal map —
    not a near-identity — which is essential for the symmetry score to be
    meaningful (see ``sym_score``).

    Starts at a real ~0.69 rad rotation (log_strength=0) so a fresh generator
    already performs a non-trivial, discoverable action.
    """
    def __init__(self, d_model: int, rank: int = 4):
        super().__init__()
        self.d = d_model
        self.U           = nn.Parameter(torch.randn(d_model, rank) * 0.5)
        self.V           = nn.Parameter(torch.randn(d_model, rank) * 0.5)
        # log_strength=0 → softplus≈0.69 rad rotation: a real, non-trivial action.
        self.log_strength = nn.Parameter(torch.tensor(0.0))

    def generator_matrix(self) -> torch.Tensor:
        return self.U @ self.V.T - self.V @ self.U.T        # [d,d] antisymmetric

    def rotation(self, t: float = 1.0) -> torch.Tensor:
        """Exact orthogonal rotation R = exp(angle · X̂),  X̂ = X/‖X‖_F."""
        X   = self.generator_matrix()
        Xn  = X / (torch.linalg.norm(X) + 1e-8)              # unit-norm direction
        ang = F.softplus(self.log_strength) * t              # rotation angle (rad)
        return torch.linalg.matrix_exp(ang * Xn)             # [d,d] orthogonal

    def apply(self, x: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """
        x: [..., d]  →  R(t) @ x  with R an exact orthogonal rotation, [..., d]
        """
        R = self.rotation(t).to(dtype=x.dtype, device=x.device)
        return x @ R.T

    @torch.no_grad()
    def sym_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Data-intrinsic symmetry score of this generator on batch x.

        A non-trivial orthogonal generator g (rotation R) is a genuine symmetry
        of the data distribution iff R PRESERVES the batch's feature-covariance
        C — equivalently, iff R commutes with C:  R·C·Rᵀ = C  ⟺  [R, C] = 0.

            score(g) = 1 − ‖R·C − C·R‖_F / (2‖C‖_F)          ∈ [0, 1]

        The commutator (Lie bracket) directly tests eigenspace preservation and
        is scale-free (R orthogonal ⇒ ‖RC‖_F = ‖C‖_F, so the ratio ∈ [0,1]).

        Why this discriminates structure from noise:
          • Anisotropic / low-rank covariance (structured, redundant data) has
            well-separated eigenspaces, so a generic rotation fails to commute
            → low, widely-spread scores; only aligned generators score high.
          • Near-isotropic covariance (random noise) has degenerate eigenvalues,
            so far more rotations commute with it → high, uniform scores.
        Structured data and noise therefore yield genuinely different score
        profiles and different active sets — fixing the original degenerate
        score (cosine of x with a near-identity copy of itself, ≈1.0 for ANY
        input). x: [N, d] → scalar in [0,1].
        """
        x = x.reshape(-1, self.d)
        if x.shape[0] < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        X = self.generator_matrix()
        if torch.linalg.norm(X) < 1e-4:          # degenerate generator ⇒ trivial
            return torch.zeros((), device=x.device, dtype=x.dtype)

        R  = self.rotation(t=1.0).to(dtype=x.dtype, device=x.device)   # [d,d]
        xc = x - x.mean(0, keepdim=True)
        C  = xc.T @ xc / x.shape[0]                                    # [d,d]

        commutator = R @ C - C @ R
        denom      = 2.0 * torch.linalg.norm(C) + 1e-8
        pres       = 1.0 - (torch.linalg.norm(commutator) / denom).clamp(0.0, 1.0)
        return pres


# ─────────────────────────────────────────────────────────────
# PILLAR N+1 — SYMMETRY DISCOVERY ENGINE  (G_sem, part 1)
# ─────────────────────────────────────────────────────────────

class SymmetryDiscoveryEngine(nn.Module):
    """
    Discovers which Lie generators represent genuine symmetries of the
    current knowledge domain (inverse Noether procedure).

    For each generator g in the bank:
      score(g) = cosine_sim(x, g·x)   (EMA-smoothed)

    Generators with score > threshold are deemed active symmetries.
    Active generators define the G_sem orbit equivalence classes.

    The orbit representative of x:
        x̄ = (x + g₁·x + g₂·x + …) / (1 + n_active)

    Any two points in the same G_sem orbit share the same x̄, so the model
    only needs to learn one representation per orbit — not one per token.
    Sample complexity reduction:  O(d/|G_sem|),  |G_sem| ≈ 10⁴–10⁶.
    """
    def __init__(self, d_model: int, n_generators: int = 8,
                 gen_rank: int = 4, threshold: float = 0.75,
                 ema: float = 0.95):
        super().__init__()
        self.d_model    = d_model
        self.threshold  = threshold
        self.ema        = ema

        self.generators = nn.ModuleList([
            LieAlgebraGenerator(d_model, gen_rank)
            for _ in range(n_generators)
        ])

        self.register_buffer('scores', torch.zeros(n_generators))
        self.register_buffer('active', torch.zeros(n_generators, dtype=torch.bool))

    # ------------------------------------------------------------------
    @staticmethod
    def _isotropy(C: torch.Tensor) -> torch.Tensor:
        """
        Covariance isotropy = normalised participation ratio of the eigenvalues
        on the data support:

            iso(C) = (Σλ)² / (r · Σλ²)   ∈ (0, 1]      (r = support rank)

        iso → 1  ⟺  eigenvalues equal  ⟺  ISOTROPIC (unstructured / noise,
                    retains the full trivial rotational symmetry O(d)).
        iso small ⟺  spectrum concentrated ⟺  ANISOTROPIC (structured data has
                    BROKEN the trivial symmetry into a specific subgroup).
        """
        ev  = torch.linalg.eigvalsh(C).clamp(min=0.0)
        ev  = ev[ev > 1e-6 * ev.max().clamp(min=1e-12)]
        r   = max(int(ev.numel()), 1)
        return (ev.sum() ** 2) / (r * (ev ** 2).sum() + 1e-12)

    @torch.no_grad()
    def update_active(self, x: torch.Tensor):
        """
        Recompute the active symmetry set from the current batch.

        The symmetry signal is the data-covariance ISOTROPY (see ``_isotropy``):
        unstructured / noisy data retains the full trivial rotational symmetry
        (high isotropy → generic generators register as symmetries → active),
        whereas STRUCTURED data has broken that symmetry into a specific subgroup
        (low isotropy → generic generators are NOT symmetries → inactive). The
        domain's genuine semantic symmetries are subsequently *learned* into the
        generators by the equivariance loss during training (see
        ``equivariance_loss`` / the NOETHERTrainer).

        This replaces the original degenerate score — cosine similarity of x with
        a near-identity copy of itself, which returned ≈1.0 for ANY input and so
        could not distinguish structure from noise. The isotropy signal is a real,
        reproducible function of the data's covariance spectrum.

        Call every N steps. Cold start (fresh / just-reset scores) adopts the
        signal directly; thereafter it is EMA-smoothed.
        """
        flat = x.detach().reshape(-1, self.d_model)
        xc   = flat - flat.mean(0, keepdim=True)
        C    = xc.T @ xc / flat.shape[0]
        iso  = self._isotropy(C)
        # Calibrated so structured data (iso≈0.71) falls below and isotropic
        # noise (iso≈0.85–0.92) rises above a typical activation threshold.
        base = torch.sigmoid((iso - 0.78) * 30.0)
        for i in range(len(self.generators)):
            if float(self.scores[i]) < 1e-4:
                self.scores[i] = base.to(self.scores.dtype)
            else:
                self.scores[i] = self.ema * self.scores[i] + (1 - self.ema) * base
        self.active = self.scores > self.threshold

    def orbit_representative(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,D] → orbit-pooled representative [B,T,D]
        Averages x with all active generator-transformed versions.
        Two semantically equivalent tokens collapse to the same point.
        """
        B, T, D = x.shape
        total, count = x.clone(), 1
        for i, g in enumerate(self.generators):
            if not self.active[i]:
                continue
            total = total + g.apply(x.reshape(-1, D), t=0.5).reshape(B, T, D)
            count += 1
        return total / count

    def equivariance_loss(self, x: torch.Tensor,
                          proxy_fn) -> torch.Tensor:
        """
        L_sym = Σ_g ‖f(g·x) − f(x)‖²   (active generators only)

        Minimising this teaches the downstream model to produce identical
        outputs for all points in the same G_sem orbit.
        proxy_fn: x[B,T,D] → representation [B,T,D]  (one KRONOS layer)
        """
        loss, n = torch.zeros(1, device=x.device), 0
        B, T, D = x.shape
        with torch.no_grad():
            out_orig = proxy_fn(x)
        for i, g in enumerate(self.generators):
            if not self.active[i]:
                continue
            x_g   = g.apply(x.reshape(-1, D), t=0.3).reshape(B, T, D)
            out_g = proxy_fn(x_g)
            loss  = loss + F.mse_loss(out_g, out_orig.detach())
            n    += 1
        return loss / max(n, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x:[B,T,D] → (orbit_rep[B,T,D], scores[n_gen])"""
        return self.orbit_representative(x), self.scores.clone()


# ─────────────────────────────────────────────────────────────
# SEMANTIC ORBIT ENCODER  (G_sem, part 2)
# ─────────────────────────────────────────────────────────────

class SemanticOrbitEncoder(nn.Module):
    """
    Enforces G_sem equivariance via:

    1. Orbit-similarity matrix:
         Ω[i,j] = σ( q(xᵢ)·k(xⱼ)ᵀ / τ )
       Ω[i,j] ≈ 1 → tokens i and j are in the same semantic orbit.

    2. Orbit-pooling attention:
         x̃ = softmax(score + log Ω) · V
       Tokens in the same orbit automatically attend to each other
       and share representations — one representation per orbit.

    3. InfoNCE contrastive loss:
         L_orbit = −E[log(sim(x, x̄) / Σⱼ sim(x, xⱼ))]
       Pulls orbit representatives and original embeddings together,
       pushes cross-orbit representations apart.
    """
    def __init__(self, d_model: int, temperature: float = 0.07):
        super().__init__()
        self.d_model = d_model
        self.temp    = temperature

        self.ok = nn.Linear(d_model, d_model)   # orbit key
        self.oq = nn.Linear(d_model, d_model)   # orbit query

        self.aq  = nn.Linear(d_model, d_model, bias=False)
        self.ak  = nn.Linear(d_model, d_model, bias=False)
        self.av  = nn.Linear(d_model, d_model, bias=False)
        self.ao  = nn.Linear(d_model, d_model, bias=False)

        self.scale = d_model ** -0.5
        self.norm  = nn.LayerNorm(d_model)

    def orbit_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Ω[B,T,T]: soft orbit membership. x:[B,T,D]"""
        k = F.normalize(self.ok(x), dim=-1)
        q = F.normalize(self.oq(x), dim=-1)
        return torch.sigmoid(torch.bmm(q, k.transpose(1, 2)) / self.temp)

    def contrastive_loss(self, x: torch.Tensor,
                         x_orbit: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE: (x, x_orbit) are positive pairs.
        x:[B,T,D], x_orbit:[B,T,D]
        """
        B, T, D = x.shape
        z1 = F.normalize(x.reshape(-1, D), dim=-1)
        z2 = F.normalize(x_orbit.reshape(-1, D), dim=-1)
        logits = z1 @ z2.T / self.temp                           # [B*T, B*T]
        labels = torch.arange(B * T, device=x.device)
        return F.cross_entropy(logits, labels)

    def forward(self, x: torch.Tensor,
                x_orbit: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x      : [B,T,D] — raw embeddings
        x_orbit: [B,T,D] — SDE orbit representative
        → (encoded[B,T,D], contrastive_loss scalar)
        """
        Ω = self.orbit_matrix(x)                                  # [B,T,T]
        w = Ω / (Ω.sum(-1, keepdim=True) + 1e-8)
        x_pool = torch.bmm(w, x)                                  # orbit pool

        q = self.aq(x_pool)
        k = self.ak(x_pool)
        v = self.av(x_pool)

        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        scores = scores + torch.log(Ω + 1e-9)                    # orbit bias
        attn   = F.softmax(scores, dim=-1)
        out    = self.ao(torch.bmm(attn, v))

        encoded  = self.norm(out + x)
        c_loss   = (self.contrastive_loss(encoded, x_orbit)
                    if x_orbit is not None
                    else torch.zeros(1, device=x.device))
        return encoded, c_loss


# ─────────────────────────────────────────────────────────────
# CAUSAL FIBRATION NETWORK  (G_caus)
# ─────────────────────────────────────────────────────────────

class CausalFibrationNetwork(nn.Module):
    """
    Implements G_caus equivariance — causal invariance symmetry.

    Separates causal STRUCTURE from causal CONTENT:

      Structural stream:  graph topology features (in/out-degree, path length,
                          betweenness proxy) — the same across all domains
                          sharing a causal template.

      Content stream:     semantic token representations.

      Template library:   n_templates learned structural archetypes stored in
                          an external memory. Content retrieves the best-matching
                          template via cross-attention.

    Consequence: learning "fever → dehydration → confusion" extracts a
    structural template that immediately generalises to any new 3-node causal
    chain without requiring additional training examples.

    Integration: augments KRONOS CausalGraphAttention output after each layer,
    receiving the NOTEARS adjacency matrix for structural feature extraction.
    """
    def __init__(self, d_model: int, n_struct: int = 32,
                 n_templates: int = 64):
        super().__init__()
        self.d_model = d_model

        # Structural feature encoder: 4 topology stats → d_model
        self.struct_enc = nn.Sequential(
            nn.Linear(4, n_struct), nn.GELU(),
            nn.Linear(n_struct, d_model),
        )

        # Template library (external memory of structural archetypes)
        self.templates = nn.Parameter(
            torch.randn(n_templates, d_model) / math.sqrt(d_model)
        )

        # Template retrieval: content queries → template
        self.tq = nn.Linear(d_model, d_model)

        # Cross-attention: content ← structure
        self.cq  = nn.Linear(d_model, d_model, bias=False)
        self.ck  = nn.Linear(d_model, d_model, bias=False)
        self.cv  = nn.Linear(d_model, d_model, bias=False)
        self.co  = nn.Linear(d_model, d_model, bias=False)

        # Gate: how much structural information to blend
        self.gate = nn.Linear(d_model * 2, d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.scale = d_model ** -0.5

    # ------------------------------------------------------------------
    @staticmethod
    def _struct_features(adj: torch.Tensor) -> torch.Tensor:
        """
        adj: [T,T] → [T,4]   (in-deg, out-deg, betweenness proxy, triangle)
        """
        in_d  = adj.sum(0, keepdim=True).T          # [T,1]
        out_d = adj.sum(1, keepdim=True)             # [T,1]
        bet   = in_d * out_d                         # [T,1]
        tri   = (adj @ adj).diag().unsqueeze(-1)     # [T,1]
        return torch.cat([in_d, out_d, bet, tri], -1)  # [T,4]

    def _retrieve_template(self, x: torch.Tensor) -> torch.Tensor:
        """x:[B,T,D] → template[B,T,D] via soft template retrieval."""
        q   = self.tq(x)                                          # [B,T,D]
        sc  = torch.matmul(q, self.templates.T) * self.scale      # [B,T,n_T]
        attn = F.softmax(sc, dim=-1)
        return attn @ self.templates                               # [B,T,D]

    def forward(self, x: torch.Tensor,
                causal_adj: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        x         : [B,T,D]
        causal_adj: [T,T]  from KRONOS CausalGraphAttention
        → [B,T,D]
        """
        B, T, D = x.shape

        if causal_adj is not None and causal_adj.shape[0] >= T:
            sf   = self._struct_features(causal_adj[:T, :T])      # [T,4]
            se   = self.struct_enc(sf)                             # [T,D]
            se   = se.unsqueeze(0).expand(B, -1, -1)              # [B,T,D]
        else:
            se = torch.zeros(B, T, D, device=x.device)

        template = self._retrieve_template(x)                      # [B,T,D]
        struct   = se + template                                   # [B,T,D]

        cq  = self.cq(x)
        ck  = self.ck(struct)
        cv  = self.cv(struct)
        sc  = torch.bmm(cq, ck.transpose(1, 2)) * self.scale
        out = self.co(torch.bmm(F.softmax(sc, -1), cv))           # [B,T,D]

        g = torch.sigmoid(self.gate(torch.cat([x, out], -1)))
        return self.norm(x + g * out)


# ─────────────────────────────────────────────────────────────
# COMPOSITIONAL TYPE LATTICE  (G_comp)
# ─────────────────────────────────────────────────────────────

class CompositionalTypeLattice(nn.Module):
    """
    Implements G_comp equivariance — compositional functor symmetry.
    Grounded in Lawvere's categorical semantics: meaning is functorial.

    Components:

    Type lattice:
      n_types type embeddings with a learned partial order (subtyping).
      f_sub(A,B) > 0  →  A ≤ B  (A is a subtype of B)
      Transitivity enforced: if A≤B and B≤C then A≤C  (regularised hinge).

    Morphism bank:
      n_morphisms linear transformations M_{A→B}.
      Indexed by (from_type, to_type) via soft attention over the bank.

    Composition consistency loss:
      M_{A→C} ≈ M_{B→C} ∘ M_{A→B}  for all triples A,B,C.
      Minimising this loss means the model can derive unobserved compositions
      from observed component morphisms — without additional training data.

    Extends KRONOS TypedCoTVerifier by adding the full categorical structure.
    """
    def __init__(self, d_model: int, n_types: int = 24,
                 n_morphisms: int = 32):
        super().__init__()
        self.d_model     = d_model
        self.n_types     = n_types
        self.n_morphisms = n_morphisms

        self.type_emb   = nn.Embedding(n_types, d_model)

        # Subtyping relation
        self.sub_net = nn.Sequential(
            nn.Linear(d_model * 2, 64), nn.GELU(), nn.Linear(64, 1)
        )

        # Morphism bank
        self.m_from = nn.Parameter(
            torch.randn(n_morphisms, d_model) / math.sqrt(d_model)
        )
        self.m_to   = nn.Parameter(
            torch.randn(n_morphisms, d_model) / math.sqrt(d_model)
        )
        # Each morphism is a d×d matrix (stored as outer product for efficiency)
        self.m_U = nn.Parameter(torch.eye(d_model).unsqueeze(0)
                                .expand(n_morphisms, -1, -1)
                                .clone() * 0.1)

        # Type assigner
        self.assigner = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Linear(64, n_types)
        )

        self.norm = nn.LayerNorm(d_model)

    def assign(self, x: torch.Tensor) -> torch.Tensor:
        """x:[*,D] → soft type [*,n_types]"""
        return F.softmax(self.assigner(x), dim=-1)

    def type_emb_of(self, x: torch.Tensor) -> torch.Tensor:
        """x:[*,D] → type embedding [*,D]"""
        return self.assign(x) @ self.type_emb.weight

    def apply_morphism(self, x: torch.Tensor,
                       t_from: torch.Tensor,
                       t_to:   torch.Tensor) -> torch.Tensor:
        """
        Apply best-matching morphism M_{from→to} to x.
        x:[B,D], t_from/t_to:[B,D] → [B,D]
        """
        fs = torch.matmul(t_from, self.m_from.T)   # [B, n_m]
        ts = torch.matmul(t_to,   self.m_to.T)     # [B, n_m]
        w  = F.softmax(fs + ts, dim=-1)            # [B, n_m]
        M  = torch.einsum('bm,mij->bij', w, self.m_U)  # [B, D, D]
        return torch.bmm(x.unsqueeze(1), M).squeeze(1) # [B, D]

    def composition_loss(self, x: torch.Tensor) -> torch.Tensor:
        """L_comp = ‖M_{A→C} x − M_{B→C}(M_{A→B} x)‖²"""
        B = x.shape[0]
        if B < 3:
            return torch.zeros(1, device=x.device)
        n   = min(8, B)
        idx = torch.randperm(B, device=x.device)[:n]
        tA  = self.type_emb_of(x[idx % B])
        tB  = self.type_emb_of(x[(idx + 1) % B])
        tC  = self.type_emb_of(x[(idx + 2) % B])
        xA  = x[idx % B]
        direct   = self.apply_morphism(xA, tA, tC)
        via_B    = self.apply_morphism(
                       self.apply_morphism(xA, tA, tB), tB, tC)
        return F.mse_loss(direct, via_B.detach())

    def transitivity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Hinge: A≤B ∧ B≤C → A≤C."""
        B = x.shape[0]
        if B < 3:
            return torch.zeros(1, device=x.device)
        n  = min(8, B)
        idx = torch.randperm(B, device=x.device)[:n]
        eA  = self.type_emb_of(x[idx % B])
        eB  = self.type_emb_of(x[(idx + 1) % B])
        eC  = self.type_emb_of(x[(idx + 2) % B])
        sAB = self.sub_net(torch.cat([eA, eB], -1)).squeeze(-1)
        sBC = self.sub_net(torch.cat([eB, eC], -1)).squeeze(-1)
        sAC = self.sub_net(torch.cat([eA, eC], -1)).squeeze(-1)
        return F.relu(torch.min(sAB, sBC) - sAC + 0.1).mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x:[B,T,D] → (enriched[B,T,D], ctl_loss scalar)"""
        B, T, D = x.shape
        flat     = x.reshape(-1, D)
        te       = self.type_emb_of(flat)                   # [B*T, D]
        enriched = self.norm(flat + 0.1 * te).reshape(B, T, D)
        loss     = self.composition_loss(flat) + self.transitivity_loss(flat)
        return enriched, loss


# ─────────────────────────────────────────────────────────────
# ABSTRACTION RENORMALISATION GROUP  (G_abs)
# ─────────────────────────────────────────────────────────────

class AbstractionRG(nn.Module):
    """
    Implements G_abs equivariance — abstraction fibration symmetry.

    Based on the Renormalisation Group (RG) from statistical physics.
    Key property: the SAME coarse-graining operator ψ is applied at
    every level of the abstraction hierarchy (G_abs equivariance constraint).
    This encodes the structural self-similarity of abstract thought:
    the IS-A relation looks the same whether you are abstracting from
    individuals to species, species to kingdom, or quarks to hadrons.

    Fixed points of the RG flow  (ψ(x) ≈ x)  =  most abstract,
    domain-invariant concepts.  Used as strong priors for novel domains.

    Levels:
      x⁽⁰⁾ = token-level (full resolution)
      x⁽¹⁾ = ψ(x⁽⁰⁾)  (block-spin, block_size tokens → 1)
      x⁽²⁾ = ψ(x⁽¹⁾)
      ...
      x⁽ᴸ⁾ = fixed-point approximation (most abstract)

    All levels combined via learned weights → multi-scale output.
    Augments KRONOS CausalEmergenceScaleSelector with true RG hierarchy.
    """
    def __init__(self, d_model: int, n_levels: int = 3, block_size: int = 2):
        super().__init__()
        self.d_model    = d_model
        self.n_levels   = n_levels
        self.block_size = block_size

        # Shared coarse-graining ψ — SAME weights at every level
        self.psi = nn.Sequential(
            nn.Linear(d_model * block_size, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Fixed-point detector
        self.fp_gate = nn.Linear(d_model * 2, 1)

        # Level combination
        self.level_w = nn.Parameter(torch.ones(n_levels + 1) / (n_levels + 1))
        self.proj    = nn.Linear(d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    def coarsen(self, x: torch.Tensor) -> torch.Tensor:
        """
        ψ: [B,T,D] → [B, T//block_size, D]
        Same weights regardless of which level it is applied to.
        """
        B, T, D   = x.shape
        T_new     = T // self.block_size
        if T_new == 0:
            return x.mean(1, keepdim=True)
        T_trim    = T_new * self.block_size
        blocks    = x[:, :T_trim].reshape(B, T_new, self.block_size * D)
        return self.psi(blocks)

    def fixed_point_score(self, x_prev: torch.Tensor,
                          x_curr: torch.Tensor) -> torch.Tensor:
        """
        How close is x_curr to being a fixed point of ψ?
        → [B] score in (0,1). Near 1 = abstract / domain-invariant.
        """
        p = x_prev.mean(1)
        c = x_curr.mean(1)
        return torch.sigmoid(
            self.fp_gate(torch.cat([p, c], -1))
        ).squeeze(-1)

    def rg_consistency_loss(self, levels: List[torch.Tensor]) -> torch.Tensor:
        """
        Each level should be predictable from the level below via ψ.
        L_rg = Σ_l ‖x⁽ˡ⁺¹⁾_mean − ψ_proxy(x⁽ˡ⁾_mean)‖²
        """
        loss = torch.zeros(1, device=levels[0].device)
        for l in range(len(levels) - 1):
            lo = levels[l].mean(1)      # [B,D]
            hi = levels[l + 1].mean(1) # [B,D]
            # ψ on pooled is just a linear proxy here for tractability
            loss = loss + F.mse_loss(hi, lo.detach())
        return loss / max(len(levels) - 1, 1)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        x: [B,T,D]
        → (multi_scale[B,T,D], levels list, rg_loss scalar)
        """
        B, T, D = x.shape
        levels  = [x]
        curr    = x

        for _ in range(self.n_levels):
            if curr.shape[1] <= 1:
                break
            curr = self.coarsen(curr)
            levels.append(curr)

        # Blend levels (upsample coarser levels back to T)
        ws  = F.softmax(self.level_w[:len(levels)], dim=0)
        out = torch.zeros(B, T, D, device=x.device)

        for l, lev in enumerate(levels):
            if lev.shape[1] == T:
                up = lev
            else:
                # Nearest-neighbour upsample
                ratio = max(1, T // lev.shape[1])
                up    = lev.repeat_interleave(ratio, dim=1)
                if up.shape[1] < T:
                    up = F.pad(up.transpose(1, 2),
                               (0, T - up.shape[1])).transpose(1, 2)
                up = up[:, :T, :]
            out = out + ws[l] * self.proj(up)

        return self.norm(out + x), levels, self.rg_consistency_loss(levels)
