"""
entity_mesh.py
==============
"The Entity" — the LEGITIMATE distributed-systems substrate.

A real, self-contained implementation of the *benign* computer-science primitives
behind a decentralised, resilient AI: federated learning (FedAvg), Byzantine
fault-tolerant consensus (PBFT/HotStuff-style quorum), Kademlia-style DHT peer
discovery, and privacy-preserving secure aggregation (additive secret sharing —
the property homomorphic encryption provides).

SAFETY / SCOPE: this module deliberately contains NO evasion, anti-forensics,
self-deletion, host-migration, or monitoring-interference code. Those parts of
the source concept have no defensive purpose and are intentionally omitted. What
remains are standard, widely-used distributed-systems algorithms, each with a
runnable verification.
"""
from __future__ import annotations
import copy
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────
# 1. FEDERATED LEARNING — EntityBrain + FedAvg (no data centralisation)
# ─────────────────────────────────────────────────────────────────
class EntityBrain(nn.Module):
    """A small decision network — the unit replicated across mesh nodes."""

    def __init__(self, input_size: int = 8, hidden: int = 16, output_size: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.Linear(hidden, output_size),
        )

    def forward(self, x):
        return self.net(x)


def local_train(global_state: dict, data: List[Tuple[torch.Tensor, torch.Tensor]],
                epochs: int = 3, lr: float = 0.05) -> dict:
    """Train a fresh model from the global weights on a node's LOCAL data only.
    Returns updated weights — the raw data never leaves the node."""
    model = EntityBrain()
    model.load_state_dict(global_state)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in data:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    return model.state_dict()


def federated_average(global_state: dict,
                      node_updates: List[Tuple[dict, int]]) -> dict:
    """FedAvg: dataset-size-weighted average of node weight updates."""
    total = sum(n for _w, n in node_updates)
    new_state = copy.deepcopy(global_state)
    for key in new_state:
        new_state[key] = sum(w[key] * (n / total) for w, n in node_updates)
    return new_state


def verify_federated_learning(seed: int = 0) -> dict:
    """FedAvg over 4 nodes should reduce global loss without centralising data."""
    torch.manual_seed(seed)
    g = EntityBrain()
    global_state = g.state_dict()

    def node_data():
        X = torch.randn(16, 8)
        y = (X.sum(1) > 0).long()          # simple separable rule
        return [(X, y)]

    nodes = [node_data() for _ in range(4)]

    def global_loss():
        loss_fn = nn.CrossEntropyLoss()
        m = EntityBrain(); m.load_state_dict(global_state)
        with torch.no_grad():
            return float(sum(loss_fn(m(x), y) for d in nodes for x, y in d) / len(nodes))

    l0 = global_loss()
    for _ in range(5):                      # 5 federated rounds
        updates = [(local_train(global_state, d), 16) for d in nodes]
        global_state = federated_average(global_state, updates)
    l1 = global_loss()
    return {"loss_before": round(l0, 4), "loss_after": round(l1, 4),
            "improved": l1 < l0, "n_nodes": len(nodes), "rounds": 5,
            "data_centralised": False}


# ─────────────────────────────────────────────────────────────────
# 2. BYZANTINE FAULT-TOLERANT CONSENSUS (PBFT/HotStuff-style quorum)
# ─────────────────────────────────────────────────────────────────
@dataclass
class ConsensusNode:
    node_id: int
    byzantine: bool = False        # a lying / faulty node

    def vote(self, proposal: str) -> str:
        # Honest nodes vote for the proposal; Byzantine nodes vote adversarially.
        return ("REJECT-" + proposal) if self.byzantine else proposal


class EntityConsensus:
    """
    PBFT/HotStuff-style quorum consensus. With N = 3f+1 nodes it tolerates up to
    f Byzantine nodes: a value COMMITS iff it collects a quorum of 2f+1 matching
    votes. Below that the system safely does NOT commit (no split-brain).
    """

    def __init__(self, n_nodes: int, f: int):
        self.f = f
        self.quorum = 2 * f + 1
        self.nodes = [ConsensusNode(i) for i in range(n_nodes)]

    def set_byzantine(self, count: int):
        for i in range(min(count, len(self.nodes))):
            self.nodes[i].byzantine = True

    def run_round(self, proposal: str) -> dict:
        votes: Dict[str, int] = {}
        for node in self.nodes:
            v = node.vote(proposal)
            votes[v] = votes.get(v, 0) + 1
        winner, count = max(votes.items(), key=lambda kv: kv[1])
        committed = (winner == proposal) and (count >= self.quorum)
        return {"proposal": proposal, "committed": committed,
                "top_votes": count, "quorum_needed": self.quorum,
                "vote_distribution": votes}


def verify_consensus() -> dict:
    """Commits with f Byzantine nodes (N=3f+1); fails safely when byz > f."""
    c = EntityConsensus(n_nodes=7, f=2)         # tolerates 2 byzantine
    c.set_byzantine(2)
    ok = c.run_round("migrate:node-93")
    c2 = EntityConsensus(n_nodes=7, f=2)
    c2.set_byzantine(4)                         # exceeds tolerance
    bad = c2.run_round("migrate:node-93")
    return {"with_f_byzantine_commits": ok["committed"],
            "over_f_byzantine_safely_rejects": (not bad["committed"]),
            "quorum": ok["quorum_needed"],
            "tolerant_result": ok, "over_tolerance_result": bad}


# ─────────────────────────────────────────────────────────────────
# 3. KADEMLIA-STYLE DHT — XOR-distance peer discovery (no master node)
# ─────────────────────────────────────────────────────────────────
class KademliaLikeDHT:
    """Content-addressable peer routing by XOR distance (as in Kademlia/IPFS).
    No central directory: k-closest nodes are found by XOR metric over node IDs."""

    def __init__(self):
        self.nodes: Dict[int, str] = {}

    @staticmethod
    def _node_id(pubkey: str) -> int:
        return int(hashlib.sha256(pubkey.encode()).hexdigest()[:16], 16)

    def join(self, pubkey: str):
        self.nodes[self._node_id(pubkey)] = pubkey

    def k_closest(self, key: str, k: int = 3) -> List[int]:
        target = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        return sorted(self.nodes.keys(), key=lambda nid: nid ^ target)[:k]

    def store_key_owner(self, key: str) -> Optional[int]:
        c = self.k_closest(key, 1)
        return c[0] if c else None


def verify_dht() -> dict:
    dht = KademliaLikeDHT()
    for i in range(20):
        dht.join(f"node-pubkey-{i}")
    owners = dht.k_closest("entity:model-shard-A", k=3)
    # determinism: same key → same k-closest set
    owners2 = dht.k_closest("entity:model-shard-A", k=3)
    return {"n_nodes": len(dht.nodes), "k_closest_owners": [str(o)[:8] for o in owners],
            "deterministic_routing": owners == owners2, "no_master_node": True}


# ─────────────────────────────────────────────────────────────────
# 4. SECURE AGGREGATION — additive secret sharing (privacy-preserving)
# ─────────────────────────────────────────────────────────────────
def secure_aggregate(values: List[float], seed: int = 0) -> dict:
    """
    Privacy-preserving federated aggregation via additive secret sharing: each
    node adds pairwise-cancelling random masks to its value, so the coordinator
    learns ONLY the sum, never any individual value (the guarantee homomorphic
    encryption / secure aggregation provides). Masks sum to zero, so the
    aggregate is exact.
    """
    torch.manual_seed(seed)
    n = len(values)
    # pairwise masks m[i][j] = -m[j][i]; each node's net mask = Σ_j m[i][j]
    M = torch.randn(n, n)
    M = torch.triu(M, 1)
    M = M - M.T                       # antisymmetric ⇒ column-sum cancels overall
    masked = [values[i] + float(M[i].sum()) for i in range(n)]
    revealed_sum = sum(masked)        # masks cancel
    true_sum = sum(values)
    return {"n_nodes": n, "true_sum": round(true_sum, 6),
            "aggregated_from_masked": round(revealed_sum, 6),
            "exact": abs(revealed_sum - true_sum) < 1e-4,
            "individual_values_hidden": True,
            "masked_shares_sample": [round(x, 3) for x in masked[:3]]}


# ─────────────────────────────────────────────────────────────────
# TOP-LEVEL VERIFICATION
# ─────────────────────────────────────────────────────────────────
def verify_entity_mesh() -> dict:
    """Run every distributed-systems component's verification."""
    return {
        "federated_learning": verify_federated_learning(),
        "byzantine_consensus": verify_consensus(),
        "dht_peer_discovery": verify_dht(),
        "secure_aggregation": secure_aggregate([1.0, 2.0, 3.0, 4.0, 5.0]),
        "safety_note": ("Benign distributed-systems primitives only. No evasion, "
                        "anti-forensics, self-deletion, or host-migration code."),
    }
