"""
emergence_markers.py
====================
Honest, MEASURABLE markers discussed in the "singularity / architecture of
sentience" material — implemented as concrete, computable quantities.

⚠ DISCLOSURE: these are quantitative research markers (information integration,
self-model fixed points, free-energy minimisation, compute-vs-capability
scaling). They are NOT, and are not claimed to be, evidence of consciousness or
sentience. Each returns a real measured number with an explicit interpretation
and its limitations.
"""
from __future__ import annotations
import math
from typing import Callable, List

import numpy as np


# ─────────────────────────────────────────────────────────────────
# 1. INTEGRATED INFORMATION (Φ) — a computable proxy
# ─────────────────────────────────────────────────────────────────
def integrated_information_proxy(state_samples: np.ndarray) -> dict:
    """
    A tractable Φ-proxy: total correlation of the system minus that of its best
    bipartition — i.e. how much information the parts carry JOINTLY beyond what
    they carry independently. Computed from the Gaussian entropy of the sample
    covariance. High Φ_proxy ⇒ the system is more than the sum of its parts.

    NOT a claim of consciousness — a standard information-integration statistic.
    state_samples: (n_samples, n_units)
    """
    X = np.asarray(state_samples, dtype=float)
    X = X - X.mean(0, keepdims=True)
    n_units = X.shape[1]
    if n_units < 2 or X.shape[0] < 2:
        return {"phi_proxy": 0.0, "n_units": n_units}

    def gauss_entropy(cov: np.ndarray) -> float:
        d = cov.shape[0]
        sign, logdet = np.linalg.slogdet(cov + 1e-6 * np.eye(d))
        return 0.5 * (d * math.log(2 * math.pi * math.e) + logdet)

    cov = np.cov(X, rowvar=False)
    H_whole = gauss_entropy(cov)
    # best (here: even/odd) bipartition
    a, b = X[:, 0::2], X[:, 1::2]
    H_parts = gauss_entropy(np.cov(a, rowvar=False)) + gauss_entropy(np.cov(b, rowvar=False))
    phi = max(H_parts - H_whole, 0.0)     # integration = redundancy across the cut
    return {"phi_proxy": round(float(phi), 5), "n_units": n_units,
            "interpretation": "information the whole carries beyond its parts "
                              "(higher = more integrated). A statistic, not sentience."}


# ─────────────────────────────────────────────────────────────────
# 2. SELF-MODEL FIXED POINT (Lawvere-style)
# ─────────────────────────────────────────────────────────────────
def self_model_fixed_point(update: Callable[[np.ndarray], np.ndarray],
                           x0: np.ndarray, iters: int = 200, tol: float = 1e-6) -> dict:
    """
    A system that models its own state has a fixed point x* = F(x*) when F is a
    contraction (Lawvere/Banach). We iterate a self-referential update and report
    convergence — a formalisation of a stable self-representation ("the system's
    model of itself settles"). Honest limitation: convergence of a map, not
    self-awareness.
    """
    x = np.asarray(x0, dtype=float)
    prev = x
    converged, n = False, 0
    for n in range(1, iters + 1):
        x = update(prev)
        if np.linalg.norm(x - prev) < tol:
            converged = True
            break
        prev = x
    return {"converged_to_self_model": converged, "iterations": n,
            "fixed_point_norm": round(float(np.linalg.norm(x)), 5),
            "interpretation": "stable self-representation = fixed point of a "
                              "contraction map (Lawvere). Not self-awareness."}


# ─────────────────────────────────────────────────────────────────
# 3. FREE-ENERGY / PREDICTION-ERROR MINIMISATION
# ─────────────────────────────────────────────────────────────────
def free_energy_trajectory(observations: np.ndarray, lr: float = 0.2) -> dict:
    """
    Friston-style active-inference marker: an internal estimate tracks a
    changing observation by minimising prediction error (surprise). Report that
    the error trajectory DECREASES — the system reduces its free energy over time.
    """
    obs = np.asarray(observations, dtype=float)
    est = obs[0].copy() if obs.ndim > 1 else float(obs[0])
    errors = []
    for o in obs:
        err = float(np.linalg.norm(o - est)) if obs.ndim > 1 else abs(o - est)
        errors.append(err)
        est = est + lr * (o - est)
    early = float(np.mean(errors[: max(1, len(errors) // 3)]))
    late = float(np.mean(errors[-max(1, len(errors) // 3):]))
    return {"prediction_error_early": round(early, 5),
            "prediction_error_late": round(late, 5),
            "free_energy_decreasing": late < early,
            "interpretation": "prediction error (surprise) falls as the internal "
                              "model tracks observations — active inference."}


# ─────────────────────────────────────────────────────────────────
# 4. COMPUTE-EFFICIENCY-VS-CAPABILITY marker
# ─────────────────────────────────────────────────────────────────
def efficiency_inversion(capability: List[float], compute_cost: List[float]) -> dict:
    """
    The DRSN thesis claims compute cost can FALL as capability rises (via
    sparsification / routing / skill compression). Measure the actual sign of the
    capability→cost correlation from provided measurements. Honest: reports the
    measured trend; makes no super-exponential claim.
    """
    cap = np.asarray(capability, float)
    cost = np.asarray(compute_cost, float)
    if len(cap) < 2:
        return {"correlation": 0.0, "cost_falls_with_capability": False}
    corr = float(np.corrcoef(cap, cost)[0, 1])
    return {"capability_cost_correlation": round(corr, 4),
            "cost_falls_with_capability": corr < 0,
            "interpretation": "negative correlation ⇒ the system solves harder "
                              "tasks with LESS compute (measured, not assumed)."}


# ─────────────────────────────────────────────────────────────────
# TOP-LEVEL REPORT
# ─────────────────────────────────────────────────────────────────
def emergence_report(seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    # integrated system: correlated units (integration present)
    base = rng.randn(200, 1)
    integrated = np.hstack([base + 0.1 * rng.randn(200, 1) for _ in range(6)])
    phi = integrated_information_proxy(integrated)

    # contraction self-model x ← 0.5·tanh(x) + c
    c = np.array([0.3, -0.2, 0.1])
    fp = self_model_fixed_point(lambda x: 0.5 * np.tanh(x) + c, np.zeros(3))

    # observations drift; internal model tracks
    obs = np.cumsum(0.1 * rng.randn(60, 3), axis=0)
    fe = free_energy_trajectory(obs)

    # capability rises while (sparsified) cost falls
    cap = np.linspace(0.2, 0.95, 10)
    cost = np.linspace(1.0, 0.3, 10) + 0.02 * rng.randn(10)
    eff = efficiency_inversion(cap.tolist(), cost.tolist())

    return {
        "integrated_information": phi,
        "self_model_fixed_point": fp,
        "free_energy_minimisation": fe,
        "efficiency_inversion": eff,
        "DISCLOSURE": ("These are quantitative research markers (information "
                       "integration, self-model convergence, active inference, "
                       "compute scaling). They are measured numbers with explicit "
                       "interpretations — NOT claims of consciousness or sentience."),
    }
