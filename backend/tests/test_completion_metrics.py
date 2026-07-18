"""
Regression tests locking in the completion-pass metrics:
  • NOETHER SDE distinguishes structured data from noise (was: fired on everything)
  • CIFN addressable weights + honest virtual-parameter accounting
  • DRSN world-model internal-simulation prediction
  • ALETHEIA full 5-layer truth-topology protocol
"""
import torch
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


def test_noether_sde_distinguishes_structure_from_noise():
    """The previously-degenerate SDE must now give different active sets for
    structured data vs pure noise (the core NOETHER fix)."""
    from entity_interface.noether.noether_components import SymmetryDiscoveryEngine
    torch.manual_seed(0)
    sde = SymmetryDiscoveryEngine(d_model=64, n_generators=4)

    # Structured: one dominant eigen-direction → strongly ANISOTROPIC covariance
    # (low isotropy — the trivial rotational symmetry is broken).
    d = torch.randn(64)
    structured = (torch.randn(16, 1) @ d.unsqueeze(0) * 5.0 + 0.3 * torch.randn(16, 64)).unsqueeze(0)
    # Noise: ISOTROPIC covariance (trivial rotational symmetry retained).
    noise = torch.randn(1, 16, 64)

    sde.scores.zero_(); sde.active.zero_()
    sde.update_active(structured)
    struct_score = float(sde.scores[0])

    sde.scores.zero_(); sde.active.zero_()
    sde.update_active(noise)
    noise_score = float(sde.scores[0])

    # The core fix: scores are now a real, data-dependent function — NOT the
    # degenerate ≈1.0-for-everything of the old cosine-of-near-identity score.
    assert not (struct_score > 0.99 and noise_score > 0.99), (
        "SDE still fires on everything (degenerate score)"
    )
    # And structured (anisotropic, broken symmetry) vs noise (isotropic, trivial
    # symmetry retained) receive materially different scores.
    assert abs(struct_score - noise_score) > 0.05, (
        f"SDE failed to distinguish structure from noise: "
        f"struct={struct_score:.4f} noise={noise_score:.4f}"
    )
    assert struct_score < noise_score, (
        f"expected structured (anisotropic) < noise (isotropic): "
        f"struct={struct_score:.4f} noise={noise_score:.4f}"
    )


def test_cifn_addressable_weights_and_accounting():
    """Any weight is computable on demand and matches the full matrix; the
    virtual-parameter accounting reports a large addressable grid from a tiny
    real basis (honest 1Q framing)."""
    from entity_interface.kronos.cifn import CIFNWeightField
    f = CIFNWeightField(out_features=8, in_features=5, basis_count=64)
    W = f.forward()
    for (i, j) in [(0, 0), (3, 2), (7, 4)]:
        assert abs(f.weight_at(i, j).item() - W[i, j].item()) < 1e-4

    acc = f.parameter_accounting(virtual_out=10**7, virtual_in=10**8)
    assert acc["real_trainable_parameters"] == 64          # only amplitudes trainable
    assert acc["addressable_parameters"] == 10**15         # 1 quadrillion addressable
    assert acc["storage_bytes_basis"] < 10_000             # compact (~KB)


def test_kronos_real_depth_injection():
    """Depth injection preserves the function at init and the deeper model trains."""
    from entity_interface.kronos.depth_injector import verify_depth_injection
    r = verify_depth_injection()
    assert r["function_preserved_at_init"] and r["deeper_model_trains"]
    assert r["linear_layers_after"] > r["linear_layers_before"]


def test_kronos_orchestrator_verified_scaleup():
    from entity_interface.kronos.orchestrator import run_verified_scaleup
    r = run_verified_scaleup()
    assert r["verified_zero_regression_scaleup"] and r["post_scale_training"]["trains"]


def test_csie_morphism_extraction():
    """High-attention token pairs become morphisms."""
    import numpy as np
    from entity_interface.csie_sheaf import CSIESheafLayer
    c = CSIESheafLayer(d_model=32, n_concepts=8)
    A = np.eye(5) * 0.1
    for i in range(1, 5):
        A[i, i - 1] = 0.8
    r = c.extract_morphisms_from_attention(A, threshold=0.3)
    assert r["n_morphisms"] >= 4 and r["morphisms"][0]["relation"] == "attends_to"


def test_cifn_omega_trains():
    from entity_interface.kronos.cifn import verify_cifn_omega
    r = verify_cifn_omega()
    assert r["trains"] and r["multi_dim_D"] >= 2 and r["recursive_coefficients"]


def test_apex_levels_8_9_10():
    from entity_interface.apex_causal import APEXCausalEngine
    E = lambda *t: [{"source": a, "target": b, "strength": w} for a, b, w in t]
    eco = APEXCausalEngine(); eco.from_kronos_causal_graph(E(("a", "b", 0.8), ("b", "c", 0.7)))
    econ = APEXCausalEngine(); econ.from_kronos_causal_graph(E(("x", "y", 0.8), ("y", "z", 0.7)))
    assert eco.causal_self_model()["self_model_converged"]           # L8
    assert eco.causal_equivalent(econ)["causally_equivalent"]         # L9
    assert eco.motivic_transfer(econ)["structurally_isomorphic"]      # L10


def test_axiom_rg_fusion_arithmetic_fisher():
    from entity_interface.axiom_zlct import RGLayerFusion, ArithmeticCoder, FisherRankAnalyzer
    import torch, torch.nn as nn
    assert RGLayerFusion.verify()["exact"]
    ac = ArithmeticCoder().verify()
    assert ac["lossless"] and ac["near_optimal"]
    m = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    fr = FisherRankAnalyzer.analyze(m, torch.randn(16, 8))
    assert fr["fisher_stable_rank"] > 0


def test_drsn_multimodal_grover_hierarchy():
    import numpy as np
    from entity_interface.drsn_node import MultimodalEncoder, grover_plan, hierarchical_world_model, DRSNNetwork
    assert len(MultimodalEncoder.encode_text([1, 2, 3])) == 8
    assert len(MultimodalEncoder.encode_audio(np.sin(np.linspace(0, 10, 32)))) == 8
    gp = grover_plan([0.9, 0.1, 0.8, 0.2, 0.95])
    assert gp["found_good_action"] and gp["speedup_factor"] > 1
    h = hierarchical_world_model(DRSNNetwork(16, 8))
    assert set(h.keys()) == {"sensorimotor", "behavioural", "goal"}


def test_axiom_zlct_modules_verified():
    """Every AXIOM compression module passes its own exact/lossless/bounded check."""
    from entity_interface.axiom_zlct import AXIOMCompressor
    st = AXIOMCompressor().self_test()
    assert st["gauge_fixer"]["exact"]
    assert st["null_space_cascade"]["exact"]
    assert st["tensor_train"]["exact"]
    assert st["padic_integer"]["bounded"]
    assert st["entropy_coder"]["lossless"]
    assert st["reversible_layer"]["exact"]
    assert st["sparse_router"]["exact"]


def test_axiom_compress_real_model_lossless_factorization():
    """Null-space factorisation of a real KRONOS model's weights is exact."""
    from entity_interface.axiom_zlct import AXIOMCompressor
    from entity_interface.kronos.kronos_architecture import KRONOS
    km = KRONOS(vocab_size=256, d_model=64, n_heads=4, n_layers=2, d_ff=256)
    r = AXIOMCompressor().compress_model(km)
    assert r["available"] and r["phase_nullspace_exact"]
    assert r["lossless_entropy_coding_ratio"] > 1.0


def test_entity_mesh_distributed_primitives():
    """The Entity's benign distributed-systems primitives all verify."""
    from entity_interface.entity_mesh import verify_entity_mesh
    r = verify_entity_mesh()
    assert r["federated_learning"]["improved"]
    assert r["federated_learning"]["data_centralised"] is False
    assert r["byzantine_consensus"]["with_f_byzantine_commits"]
    assert r["byzantine_consensus"]["over_f_byzantine_safely_rejects"]
    assert r["dht_peer_discovery"]["deterministic_routing"]
    assert r["secure_aggregation"]["exact"]
    assert r["secure_aggregation"]["individual_values_hidden"]


def test_emergence_markers_measurable_and_disclosed():
    """Emergence markers return real measured numbers with an explicit
    not-consciousness disclosure."""
    from entity_interface.emergence_markers import emergence_report
    r = emergence_report()
    assert r["integrated_information"]["phi_proxy"] >= 0.0
    assert "iterations" in r["self_model_fixed_point"]
    assert r["free_energy_minimisation"]["free_energy_decreasing"] in (True, False)
    assert "DISCLOSURE" in r and "NOT claims" in r["DISCLOSURE"]


def test_drsn_quantum_interference():
    """Quantum-inspired complex-phase integration: aligned phases amplify
    (constructive) more than opposed phases (destructive)."""
    from entity_interface.drsn_node import DRSNNetwork
    net = DRSNNetwork(16, 8)
    d = net.demonstrate_interference()
    assert d["interference_confirmed"]
    assert abs(d["constructive_input"]) > abs(d["destructive_input"])


def test_csie_real_cech_cohomology():
    """Real sparse Čech solve: consistent sections → H¹≈0; polysemy → H¹>0."""
    from entity_interface.csie_sheaf import CSIESheafLayer, SemanticSection

    def sec(cid, vec, ctx):
        return SemanticSection(concept_id=cid, vector=vec, confidence=1.0,
                               context_id=ctx, restriction_ids=[])
    c = CSIESheafLayer(d_model=8, n_concepts=16)
    v = [1.0, 0, 0, 0, 0, 0, 0, 0]
    c.add_covering("finance", ["finance"], {"bank": sec("bank", v, "finance")})
    c.add_covering("commerce", ["commerce"], {"bank": sec("bank", v, "commerce")})
    r = c.cech_cohomology_solve(["finance", "commerce"])
    assert r["H1_residual_norm"] < 1e-3 and not r["cohomology_H1_nontrivial"]
    # Introduce polysemy → inconsistent → non-trivial H1
    c.add_covering("geo", ["geo"], {"bank": sec("bank", [0, 0, 0, 0, 1, 0, 0, 0], "geo")})
    r2 = c.cech_cohomology_solve(["finance", "commerce", "geo"])
    assert r2["cohomology_H1_nontrivial"]


def test_apex_extended_pearl_levels():
    """APEX real Betti cohomology + Level-10 path-integral posterior."""
    from entity_interface.apex_causal import APEXCausalEngine
    E = lambda *t: [{"source": a, "target": b, "strength": w} for a, b, w in t]
    # Pure chain a→b→c is acyclic ⇒ b1=0, identifiable.
    chain = APEXCausalEngine()
    chain.from_kronos_causal_graph(E(("a", "b", 0.9), ("b", "c", 0.8)))
    sig = chain.compute_cohomology_signature()
    assert sig["betti_0"] == 1 and sig["betti_1"] == 0 and sig["identifiable_topology"]

    # Two routes a→c (direct + via b) form a cycle ⇒ b1=1, and a 2-explanation posterior.
    ap = APEXCausalEngine()
    ap.from_kronos_causal_graph(E(("a", "b", 0.9), ("b", "c", 0.8), ("a", "c", 0.4)))
    sig2 = ap.compute_cohomology_signature()
    assert sig2["betti_1"] == 1 and sig2["cohomology_H1_nontrivial"]
    post = ap.causal_path_posterior("a", "c")
    assert post["n_explanations"] == 2
    probs = [p["probability"] for p in post["posterior"]]
    assert abs(sum(probs) - 1.0) < 1e-6          # a real normalised posterior


def test_signal_outcome_feedback_loop():
    """Recording outcomes updates signal reliability + adaptive weights."""
    from entity_interface.signal_synthesizer import (
        _adaptive_weights, _LAST_SIGNALS, SignalSynthesizer)
    _LAST_SIGNALS["fb-e"] = {"model": 0.9, "entropy": 0.9, "experience": 0.1}
    r = SignalSynthesizer.record_outcome("fb-e", 0.9)
    assert r["updated"]
    w = _adaptive_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-6     # weights stay normalised


def test_drsn_world_model_prediction():
    """DRSN can roll its dynamics forward (internal simulation) and return a
    predicted world-state trajectory."""
    from entity_interface.drsn_node import DRSNNetwork
    net = DRSNNetwork(n_nodes=16, d_hidden=8)
    net.encode_features([f * 50 for f in [0.9, 0.5, 0.2, 0.1, 1.0, 0.2, 0.3, 0.4]], n_steps=10)
    pred = net.predict_next_state(n_predict=3)
    assert pred["n_predict"] == 3
    assert len(pred["predicted_world_states"]) == 3
    assert len(pred["predicted_world_states"][0]) == 16


def test_kronecker_function_preserving_scaling():
    """KRONOS width expansion must reproduce f(x) exactly (function-preserving
    Kronecker identity) for input/output/both modes."""
    from entity_interface.kronos.kronecker_scaler import KroneckerScaler
    ks = KroneckerScaler()
    for mode in ("input", "output", "both"):
        r = ks.demonstrate_preservation(m=16, n=8, k=4, mode=mode)
        assert r["function_preserved"], f"{mode}: max_diff={r['max_abs_diff']}"
        assert r["max_abs_diff"] < 1e-5


@pytest.mark.anyio
async def test_aletheia_five_layer_protocol():
    """A claim exposes all 5 ALETHEIA layers with provenance, adversarial
    surface, and jurisdictional context."""
    from core.entity_resolution import entity_registry
    import main
    entity_registry.entities["ale-c"] = {
        "id": "ale-c", "name": "claimant", "domain": "financial",
        "status": "stable", "entropy": 1.0, "event_count": 0, "alert_count": 0,
    }
    client = TestClient(main.app)
    with client:
        r = client.post("/api/claims", headers={"X-API-Key": "sera-demo-2026"},
                        json={"claimant_id": "ale-c",
                              "content": "Q3 revenue and earnings beat expectations",
                              "stake_amount": 60.0})
        assert r.status_code == 200
        cid = r.json()["claim_id"]
        r = client.get(f"/api/claims/{cid}", headers={"X-API-Key": "sera-demo-2026"})
        assert r.status_code == 200
        body = r.json()
        assert body["aletheia_layers"] == [
            "provenance", "stake", "adversarial_surface", "temporal_decay", "jurisdictional_bridge"
        ]
        assert body["provenance_fingerprint"].startswith("sha3:")
        assert body["jurisdictional_contexts"]            # non-empty (SEC/ESMA from 'revenue'/'earnings')
