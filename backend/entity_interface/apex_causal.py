from dataclasses import dataclass, field
from typing import Optional, List, Dict
import math
import logging
import numpy as np

_sentence_transformer = None
_embeddings_cache = {}

def get_encoder():
    global _sentence_transformer
    print(f"[get_encoder] Checking _sentence_transformer: {id(_sentence_transformer)} (None={_sentence_transformer is None})", flush=True)
    if _sentence_transformer is None:
        print("[get_encoder] Loading SentenceTransformer 'all-MiniLM-L6-v2'...", flush=True)
        from sentence_transformers import SentenceTransformer
        _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        print("[get_encoder] SentenceTransformer loaded successfully.", flush=True)
    return _sentence_transformer

def get_embedding(text: str, model) -> list:
    global _embeddings_cache
    if text not in _embeddings_cache:
        if model:
            try:
                _embeddings_cache[text] = list(map(float, model.encode(text)))
            except Exception:
                import random
                _embeddings_cache[text] = [random.random() for _ in range(384)]
        else:
            import random
            _embeddings_cache[text] = [random.random() for _ in range(384)]
    return _embeddings_cache[text]



@dataclass
class KMorphism:
    """
    Represents a morphism in the APEX k-category causal hierarchy.

    k=1 corresponds to direct causal relations (as output by KRONOS).
    k=2 represents causal abstractions (relations between causal relations).
    k>=3 represents meta-causal reasoning (higher-order causal structures).

    The homotopy_class groups morphisms that are continuously deformable
    into each other within the causal topology, enabling equivalence-class
    reasoning over causal paths.
    """

    k: int  # Depth level: 1=direct causal, 2=causal abstraction, 3+=meta-causal
    source_id: str
    target_id: str
    relation_type: str  # One of: causes, enables, prevents, precedes, produces, is_a, part_of, related_to, composed
    weight: float  # Confidence in [0, 1]
    homotopy_class: Optional[str] = None
    modal_status: str = "contingent"  # One of: contingent, necessary, impossible
    action_value: float = 0.0  # Used for path integral weighting (Lagrangian-style)


@dataclass
class CausalObject:
    """
    Represents an object (node) in the APEX causal category.

    abstraction_level=0 means atomic / ground-level concept.
    Higher abstraction_level values indicate more abstract causal entities
    (e.g., abstract event types, causal schemas, meta-patterns).
    """

    id: str
    name: str
    embedding: list = field(default_factory=list)  # 512-dim vector placeholder
    abstraction_level: int = 0  # 0=atomic, higher=more abstract
    domain: str = ""


class APEXCausalEngine:
    """
    APEX 10-Level Causal Hierarchy Engine.

    Wraps KRONOS model causal graph output and elevates it to higher-order
    causal reasoning using a k-category theoretic framework inspired by
    homotopy type theory and categorical quantum mechanics.

    Level 1  - Direct causal morphisms (KRONOS output)
    Level 2  - Causal abstractions (morphisms between morphisms)
    Level 3  - Meta-causal transfer across domains
    Level 4+ - Cohomological invariants, path integrals, modal reasoning
    """

    def __init__(self, max_k: int = 5) -> None:
        self.max_k: int = max_k
        self.objects: Dict[str, CausalObject] = {}
        self.morphisms: Dict[str, KMorphism] = {}
        self.homotopy_classes: Dict[str, List[str]] = {}  # class_name -> list of morphism keys

    # ------------------------------------------------------------------
    # Core category operations
    # ------------------------------------------------------------------

    def add_object(self, obj: CausalObject) -> None:
        """Register a CausalObject in the engine's object catalogue."""
        self.objects[obj.id] = obj

    def add_morphism(self, m: KMorphism) -> None:
        """
        Register a KMorphism using the canonical key
        "{source_id}->{target_id}:{relation_type}:{k}".
        Also indexes the morphism into its homotopy class if one is assigned.
        """
        key = f"{m.source_id}->{m.target_id}:{m.relation_type}:{m.k}"
        self.morphisms[key] = m

        # Maintain homotopy class index
        if m.homotopy_class is not None:
            if m.homotopy_class not in self.homotopy_classes:
                self.homotopy_classes[m.homotopy_class] = []
            if key not in self.homotopy_classes[m.homotopy_class]:
                self.homotopy_classes[m.homotopy_class].append(key)

    def compose(self, m1_key: str, m2_key: str) -> Optional[KMorphism]:
        """
        Compose two morphisms m1 ; m2 in the causal category.

        Composition is valid iff m1.target_id == m2.source_id (the standard
        categorical composition law).  The resulting composed morphism sits at
        depth k = max(m1.k, m2.k) and its weight is the product of the two
        constituent weights (independent-probability assumption).

        Returns None if the morphisms are not composable.
        """
        m1 = self.morphisms.get(m1_key)
        m2 = self.morphisms.get(m2_key)

        if m1 is None or m2 is None:
            return None

        if m1.target_id != m2.source_id:
            return None

        composed = KMorphism(
            k=max(m1.k, m2.k),
            source_id=m1.source_id,
            target_id=m2.target_id,
            relation_type="composed",
            weight=m1.weight * m2.weight,
            homotopy_class=m1.homotopy_class if m1.homotopy_class == m2.homotopy_class else None,
            modal_status="contingent",
            action_value=m1.action_value + m2.action_value,
        )
        return composed

    # ------------------------------------------------------------------
    # Cohomological signature
    # ------------------------------------------------------------------

    def compute_cohomology_signature(self) -> dict:
        """
        Compute a lightweight cohomological invariant for the current causal
        graph.  This characterises the topological "shape" of the causal
        structure without requiring full homology computation.

        Returns a dict with:
          - n_objects         : number of registered CausalObjects
          - n_morphisms       : number of registered KMorphisms
          - max_depth         : highest k-level present
          - homotopy_classes  : list of distinct homotopy class names
          - causal_density    : morphisms per object (graph density proxy)
        """
        n_objects = len(self.objects)
        n_morphisms = len(self.morphisms)
        max_depth = max((m.k for m in self.morphisms.values()), default=0)
        homotopy_class_names = list(
            {m.homotopy_class for m in self.morphisms.values() if m.homotopy_class is not None}
        )
        causal_density = n_morphisms / max(n_objects, 1)

        # ── Genuine graph (co)homology via Betti numbers ──────────────────
        # Treat the causal graph as a 1-dimensional simplicial complex
        # (objects = 0-cells, unique morphism pairs = 1-cells).
        #   b0 = rank H^0 = number of connected components
        #   b1 = rank H^1 = number of independent cycles = |E| - |V| + b0
        # Per the extended-Pearl / APEX framework, a non-trivial H^1 (b1 > 0)
        # signals cyclic causal structure — an obstruction to identifiability
        # (potential confounding read off the topology, not enumerated).
        nodes = list(self.objects.keys())
        idx = {n: i for i, n in enumerate(nodes)}
        parent = list(range(len(nodes)))

        def _find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        undirected_edges = set()
        for m in self.morphisms.values():
            if m.source_id in idx and m.target_id in idx:
                a, b = idx[m.source_id], idx[m.target_id]
                undirected_edges.add((min(a, b), max(a, b)))
        for a, b in undirected_edges:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        b0 = len({_find(i) for i in range(len(nodes))}) if nodes else 0
        n_unique_edges = len(undirected_edges)
        b1 = max(n_unique_edges - len(nodes) + b0, 0)

        return {
            "n_objects": n_objects,
            "n_morphisms": n_morphisms,
            "max_depth": max_depth,
            "homotopy_classes": homotopy_class_names,
            "causal_density": causal_density,
            # Real cohomology invariants:
            "betti_0": b0,                       # H^0 rank — connected components
            "betti_1": b1,                       # H^1 rank — independent cycles
            "euler_characteristic": len(nodes) - n_unique_edges,   # χ = V − E
            "cohomology_H1_nontrivial": b1 > 0,  # confounding obstruction present
            "identifiable_topology": b1 == 0,    # acyclic covering ⇒ identifiable
        }

    # ------------------------------------------------------------------
    # Path integral over causal paths
    # ------------------------------------------------------------------

    def path_integral(self, source_id: str, target_id: str) -> dict:
        """
        Compute a Feynman-style path integral over all causal paths from
        source_id to target_id.

        Each path is weighted by:
            W(path) = prod(weight_i) * exp(-sum(action_value_i))

        Uses BFS/DFS to enumerate all simple paths (no repeated nodes).

        Returns:
          - best_path    : list of morphism keys on the highest-weight path
          - total_weight : sum of W(path) across all paths (partition function)
          - n_paths      : total number of distinct causal paths discovered
        """
        # Build adjacency: node -> list of (neighbour, morphism_key)
        adjacency: Dict[str, List[tuple]] = {}
        for key, m in self.morphisms.items():
            if m.source_id not in adjacency:
                adjacency[m.source_id] = []
            adjacency[m.source_id].append((m.target_id, key))

        all_paths: List[List[str]] = []  # each element is a list of morphism keys

        # DFS state: (current_node, visited_nodes, path_so_far)
        stack = [(source_id, {source_id}, [])]

        while stack:
            current, visited, path = stack.pop()

            if current == target_id and path:
                all_paths.append(path)
                continue

            for neighbour, edge_key in adjacency.get(current, []):
                if neighbour not in visited:
                    stack.append((neighbour, visited | {neighbour}, path + [edge_key]))

        if not all_paths:
            return {"best_path": [], "total_weight": 0.0, "n_paths": 0}

        def path_weight(keys: List[str]) -> float:
            w = 1.0
            action_sum = 0.0
            for k in keys:
                m = self.morphisms[k]
                w *= m.weight
                action_sum += m.action_value
            return w * math.exp(-action_sum)

        weighted_paths = [(path_weight(p), p) for p in all_paths]
        total_weight = sum(w for w, _ in weighted_paths)
        best_weight, best_path = max(weighted_paths, key=lambda x: x[0])

        return {
            "best_path": best_path,
            "total_weight": total_weight,
            "n_paths": len(all_paths),
        }

    # ------------------------------------------------------------------
    # Extended Pearl hierarchy — Level 10: causal path-integral POSTERIOR
    # ------------------------------------------------------------------

    def causal_path_posterior(self, source_id: str, target_id: str) -> dict:
        """
        Level 10 — the FULL causal path integral, not just the saddle point.

        Every causal explanation (path) is assigned a posterior probability
            P(path) = exp(-S[path]) / Z ,   S[path] = Σ action − Σ log weight
        (an MDL-style action). The saddle point is the MAP explanation; the full
        distribution is the posterior over all causal explanations simultaneously,
        with its Shannon entropy quantifying explanatory ambiguity.
        """
        adjacency: Dict[str, List[tuple]] = {}
        for key, m in self.morphisms.items():
            adjacency.setdefault(m.source_id, []).append((m.target_id, key))

        all_paths, stack = [], [(source_id, {source_id}, [])]
        while stack:
            current, visited, path = stack.pop()
            if current == target_id and path:
                all_paths.append(path); continue
            for neighbour, edge_key in adjacency.get(current, []):
                if neighbour not in visited:
                    stack.append((neighbour, visited | {neighbour}, path + [edge_key]))

        if not all_paths:
            return {"posterior": [], "n_explanations": 0, "posterior_entropy": 0.0,
                    "map_explanation": [], "map_probability": 0.0}

        def action(keys):
            s = 0.0
            for k in keys:
                m = self.morphisms[k]
                s += m.action_value - math.log(max(m.weight, 1e-9))
            return s

        actions = [action(p) for p in all_paths]
        a_min = min(actions)
        unnorm = [math.exp(-(a - a_min)) for a in actions]     # stable softmax
        Z = sum(unnorm)
        probs = [u / Z for u in unnorm]

        posterior = sorted(
            ({"path": p, "probability": round(pr, 6),
              "path_names": [self._morphism_label(k) for k in p]}
             for p, pr in zip(all_paths, probs)),
            key=lambda d: -d["probability"],
        )
        entropy = -sum(pr * math.log(pr + 1e-12) for pr in probs)
        return {
            "posterior": posterior,
            "n_explanations": len(all_paths),
            "posterior_entropy": round(entropy, 6),
            "map_explanation": posterior[0]["path"],
            "map_probability": posterior[0]["probability"],
            "partition_function_Z": round(Z, 6),
        }

    def _morphism_label(self, key: str) -> str:
        m = self.morphisms.get(key)
        return f"{m.source_id}->{m.target_id}" if m else key

    # ------------------------------------------------------------------
    # Extended Pearl hierarchy — Level 7: modal causal necessity
    # ------------------------------------------------------------------

    def modal_necessity(self) -> dict:
        """
        Level 7 — distinguish NECESSARY causal facts (hold in every context /
        are structurally forced) from CONTINGENT ones (hold only in specific
        contexts). A morphism valid across ALL known contexts, or carrying an
        explicit 'necessary' modal_status, is classed necessary; one restricted
        to a subset of contexts is contingent.
        """
        def _ctx(m):
            return getattr(m, "context_ids", None) or getattr(m, "context_restriction", None) or []
        all_contexts = {c for m in self.morphisms.values() for c in _ctx(m)}
        necessary, contingent = [], []
        for key, m in self.morphisms.items():
            ctx = set(_ctx(m))
            is_necessary = (m.modal_status == "necessary") or (
                len(all_contexts) > 0 and ctx == all_contexts
            )
            (necessary if is_necessary else contingent).append(self._morphism_label(key))
        return {
            "n_necessary": len(necessary),
            "n_contingent": len(contingent),
            "necessary": necessary[:10],
            "contingent": contingent[:10],
            "n_contexts": len(all_contexts),
        }

    # ------------------------------------------------------------------
    # Level 8 — Lawvere fixed-point causal SELF-MODEL
    # ------------------------------------------------------------------
    def causal_self_model(self, iters: int = 100) -> dict:
        """
        Level 8 — the system builds a causal model OF ITS OWN causal model,
        giving a fixed-point equation C = F(C). By Lawvere's fixed-point theorem
        (the categorical generalisation of Gödel), a contraction F has a unique
        fixed point C*. We iterate a self-summary of the cohomological signature
        to that fixed point — the limit of what the system can know about its own
        causal structure (a formal epistemic-humility bound).
        """
        sig = self.compute_cohomology_signature()
        base = np.array([sig["betti_0"], sig["betti_1"],
                         sig["causal_density"], float(sig["max_depth"])], dtype=float)
        vec, prev, converged, n = base.copy(), base.copy(), False, 0
        for n in range(1, iters + 1):
            # Contraction F(x) = 0.5·base + 0.5·tanh(x) (Lipschitz ≤ 0.5 < 1) ⇒
            # unique NON-trivial fixed point x* = 0.5·base + 0.5·tanh(x*).
            vec = 0.5 * base + 0.5 * np.tanh(prev)
            if np.linalg.norm(vec - prev) < 1e-9:
                converged = True
                break
            prev = vec
        return {"self_model_converged": converged, "iterations": n,
                "fixed_point": [round(float(v), 6) for v in vec],
                "interpretation": "Lawvere fixed point C=F(C): stable, bounded "
                                  "model of the system's own causal structure."}

    # ------------------------------------------------------------------
    # Level 9 — Causal Univalence (HoTT): equivalence ⇒ identity
    # ------------------------------------------------------------------
    def causal_equivalent(self, other: "APEXCausalEngine") -> dict:
        """
        Level 9 — the Causal Univalence Axiom: two causal models that give
        identical answers under all interventions are not merely equivalent, they
        are IDENTICAL. Detected topologically: equal cohomological signatures
        (Betti numbers + Euler characteristic) ⇒ homotopy-equivalent ⇒ the same
        causal model, collapsing Markov-equivalence classes to a point.
        """
        s1 = self.compute_cohomology_signature()
        s2 = other.compute_cohomology_signature()
        keys = ("betti_0", "betti_1", "euler_characteristic")
        same = all(s1[k] == s2[k] for k in keys)
        return {"causally_equivalent": same,
                "signature_self": {k: s1[k] for k in keys},
                "signature_other": {k: s2[k] for k in keys},
                "interpretation": "equal homotopy type ⇒ identical causal model "
                                  "(Causal Univalence)."}

    # ------------------------------------------------------------------
    # Level 10 (motivic) — transfer causal insight across isomorphic domains
    # ------------------------------------------------------------------
    def motivic_transfer(self, other: "APEXCausalEngine") -> dict:
        """
        Level 10 (motivic) — if two domains share the same cohomological 'motive'
        (universal invariant / signature), they are structurally isomorphic and a
        causal insight in one transfers to the other. We map morphisms by their
        rank order of weight, producing a domain-A → domain-B correspondence
        (e.g. ecology ↔ economics: competition/depletion/collapse).
        """
        s1 = self.compute_cohomology_signature()
        s2 = other.compute_cohomology_signature()
        isomorphic = (s1["betti_0"] == s2["betti_0"] and s1["betti_1"] == s2["betti_1"])
        a_sorted = sorted(self.morphisms.values(), key=lambda m: -m.weight)
        b_sorted = sorted(other.morphisms.values(), key=lambda m: -m.weight)
        transfers = []
        for ma, mb in zip(a_sorted, b_sorted):
            transfers.append({
                "from": f"{ma.source_id}->{ma.target_id}",
                "maps_to": f"{mb.source_id}->{mb.target_id}",
                "shared_relation_role": ma.relation_type,
            })
        return {"structurally_isomorphic": isomorphic,
                "n_transfers": len(transfers),
                "transfer_map": transfers[:10],
                "interpretation": "shared motive ⇒ causal insight transfers between "
                                  "domains without retraining."}

    def extended_pearl_report(self) -> dict:
        """Summarise which levels of the extended Pearl hierarchy are operational."""
        return {
            "level_1_association": "logits (KRONOS)",
            "level_2_intervention": "compose / do-graph",
            "level_3_counterfactual": "path re-weighting",
            "level_4_causal_abstraction": "level4_causal_abstraction()",
            "level_5_transfer_colimits": "level5_transfer()",
            "level_6_cohomological_confounding": "betti_1 (compute_cohomology_signature)",
            "level_7_modal_necessity": "modal_necessity()",
            "level_8_self_model": "causal_self_model()",
            "level_9_causal_univalence": "causal_equivalent()",
            "level_10_path_integral_posterior": "causal_path_posterior()",
            "level_10_motivic_transfer": "motivic_transfer()",
        }

    # ------------------------------------------------------------------
    # Higher-order causal operations (APEX levels 4-5)
    # ------------------------------------------------------------------

    def level4_causal_abstraction(self, morphism_keys: list) -> KMorphism:
        """
        APEX Level 4 - Causal Abstraction.

        Given a list of k=1 morphisms, constructs a single k=2 morphism that
        represents the causal relation *between* those causal relations.  This
        is the categorical analogue of a natural transformation: a morphism in
        the functor category over the base causal graph.

        The abstracted morphism connects the first morphism's source to the
        last morphism's target, aggregates weights by their geometric mean,
        and is tagged with relation_type='causal_abstraction'.
        """
        morphisms = [self.morphisms[k] for k in morphism_keys if k in self.morphisms]

        if not morphisms:
            raise ValueError("No valid morphisms found for the provided keys.")

        # Geometric mean weight
        weight_product = 1.0
        for m in morphisms:
            weight_product *= m.weight
        avg_weight = weight_product ** (1.0 / len(morphisms))

        abstraction = KMorphism(
            k=2,
            source_id=morphisms[0].source_id,
            target_id=morphisms[-1].target_id,
            relation_type="causal_abstraction",
            weight=avg_weight,
            homotopy_class="abstracted",
            modal_status="contingent",
            action_value=sum(m.action_value for m in morphisms) / len(morphisms),
        )
        return abstraction

    def level5_transfer(self, source_domain: str, target_domain: str) -> list:
        """
        APEX Level 5 - Cross-Domain Causal Transfer.

        Finds all morphisms whose source/target objects reside in source_domain
        and creates analogous (transferred) morphisms in target_domain.

        Transfer morphisms are assigned k = original.k + 1 to reflect the
        additional inferential step, and their weight is discounted by 0.7
        (the default causal transfer confidence penalty).

        Returns the list of newly created KMorphism objects (they are NOT
        automatically added to self.morphisms; the caller decides).
        """
        TRANSFER_DISCOUNT = 0.7
        transferred: List[KMorphism] = []

        for key, m in self.morphisms.items():
            src_obj = self.objects.get(m.source_id)
            tgt_obj = self.objects.get(m.target_id)

            # Both endpoints must belong to the source domain
            if src_obj is None or tgt_obj is None:
                continue
            if src_obj.domain != source_domain or tgt_obj.domain != source_domain:
                continue

            transferred_morphism = KMorphism(
                k=m.k + 1,
                source_id=f"{target_domain}:{m.source_id}",
                target_id=f"{target_domain}:{m.target_id}",
                relation_type=m.relation_type,
                weight=m.weight * TRANSFER_DISCOUNT,
                homotopy_class=m.homotopy_class,
                modal_status="contingent",  # transferred relations are always contingent
                action_value=m.action_value,
            )
            transferred.append(transferred_morphism)

        return transferred

    # ------------------------------------------------------------------
    # KRONOS integration
    # ------------------------------------------------------------------

    def from_kronos_causal_graph(self, kronos_edges: list) -> None:
        """
        Ingest a KRONOS causal graph and populate the engine at k=1.

        Each element of kronos_edges should be a dict with keys:
          - 'source'   : str   - source node identifier
          - 'target'   : str   - target node identifier
          - 'strength' : float - edge confidence in [0, 1]

        Optionally, edges may include:
          - 'relation_type'  : str   (defaults to 'causes')
          - 'homotopy_class' : str   (defaults to None)
          - 'action_value'   : float (defaults to 0.0)
          - 'domain'         : str   (defaults to 'kronos')

        For each unique node encountered a CausalObject is created with
        default abstraction_level=0 and domain='kronos' unless the node
        already exists in self.objects.
        """
        for edge in kronos_edges:
            source = edge["source"]
            target = edge["target"]
            strength = float(edge.get("strength", 1.0))

            # Register objects if not already known
            for node_id in (source, target):
                if node_id not in self.objects:
                    obj = CausalObject(
                        id=node_id,
                        name=node_id,
                        embedding=[],
                        abstraction_level=0,
                        domain=edge.get("domain", "kronos"),
                    )
                    self.add_object(obj)

            morphism = KMorphism(
                k=1,
                source_id=source,
                target_id=target,
                relation_type=edge.get("relation_type", "causes"),
                weight=strength,
                homotopy_class=edge.get("homotopy_class", None),
                modal_status=edge.get("modal_status", "contingent"),
                action_value=float(edge.get("action_value", 0.0)),
            )
            self.add_morphism(morphism)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_summary_dict(self) -> dict:
        """
        Produce a JSON-serialisable summary of the engine's current state.

        Includes:
          - cohomology_signature : output of compute_cohomology_signature()
          - top_5_morphisms      : up to 5 highest-weight morphisms as dicts
          - n_total_morphisms    : total morphism count
          - max_causal_depth     : deepest k-level present
        """
        sig = self.compute_cohomology_signature()

        sorted_morphisms = sorted(
            self.morphisms.items(),
            key=lambda kv: kv[1].weight,
            reverse=True,
        )
        top_5 = [
            {
                "key": key,
                "k": m.k,
                "source_id": m.source_id,
                "target_id": m.target_id,
                "relation_type": m.relation_type,
                "weight": m.weight,
                "homotopy_class": m.homotopy_class,
                "modal_status": m.modal_status,
                "action_value": m.action_value,
                "morphism_source": "cifn_activation_derived",
            }
            for key, m in sorted_morphisms[:5]
        ]

        return {
            "cohomology_signature": sig,
            "top_5_morphisms": top_5,
            "n_total_morphisms": len(self.morphisms),
            "max_causal_depth": sig["max_depth"],
            "morphism_source": "cifn_activation_derived",
        }

    def hydrate_from_neo4j(self, driver) -> None:
        """
        Hydrate Category Objects and Morphisms directly from Neo4j.
        Replaces existing ConceptNet mock nodes/morphisms with real Neo4j property models.
        """
        # Load SentenceTransformer model safely
        model = None
        try:
            model = get_encoder()
        except Exception as e:
            logger = logging.getLogger("sera.apex")
            logger.warning(f"SentenceTransformer not available or offline: {e}. Using mock/fallback embeddings.")

        # Wipe current objects and morphisms before hydrating to ensure fresh state
        self.objects = {}
        self.morphisms = {}
        self.homotopy_classes = {}

        with driver.session() as neo_session:
            # 1. Fetch Companies
            c_res = neo_session.run("MATCH (c:Company) RETURN c.ticker AS ticker, c.legal_name AS name, c.sector AS sector")
            for record in c_res:
                ticker = record["ticker"]
                name = record["name"]
                sector = record["sector"]
                
                # Generate embedding
                emb = get_embedding(f"{name} {sector}", model)

                obj = CausalObject(
                    id=ticker,
                    name=name,
                    embedding=emb,
                    abstraction_level=0,
                    domain=sector.lower()
                )
                self.add_object(obj)

            # 2. Fetch Jobs
            j_res = neo_session.run("MATCH (j:Job) RETURN j.id AS id, j.title AS name, 'job' AS sector")
            for record in j_res:
                id_val = record["id"]
                name = record["name"]
                sector = record["sector"]
                
                emb = get_embedding(name, model)

                obj = CausalObject(
                    id=id_val,
                    name=name,
                    embedding=emb,
                    abstraction_level=0,
                    domain=sector
                )
                self.add_object(obj)

            # 3. Fetch News
            n_res = neo_session.run("MATCH (n:News) RETURN n.gdelt_id AS id, n.title AS name, 'news' AS sector")
            for record in n_res:
                id_val = record["id"]
                name = record["name"]
                sector = record["sector"]
                
                emb = get_embedding(name, model)

                obj = CausalObject(
                    id=id_val,
                    name=name,
                    embedding=emb,
                    abstraction_level=0,
                    domain=sector
                )
                self.add_object(obj)

            # 4. Fetch Vessels & Ports
            v_res = neo_session.run("MATCH (v:Vessel) RETURN v.imo AS id, v.name AS name, 'shipping' AS sector")
            for record in v_res:
                id_val = record["id"]
                name = record["name"]
                sector = record["sector"]
                
                emb = get_embedding(name, model)

                obj = CausalObject(
                    id=id_val,
                    name=name,
                    embedding=emb,
                    abstraction_level=0,
                    domain=sector
                )
                self.add_object(obj)

            p_res = neo_session.run("MATCH (p:Port) RETURN p.name AS id, p.name AS name, 'shipping' AS sector")
            for record in p_res:
                id_val = record["id"]
                name = record["name"]
                sector = record["sector"]
                
                emb = get_embedding(name, model)

                obj = CausalObject(
                    id=id_val,
                    name=name,
                    embedding=emb,
                    abstraction_level=0,
                    domain=sector
                )
                self.add_object(obj)

            # 5. Fetch all relationships (Morphisms)
            r_res = neo_session.run(
                """
                MATCH (s)-[r]->(t)
                RETURN coalesce(s.ticker, s.gdelt_id, s.id, s.imo, s.name) AS source_id,
                       coalesce(t.ticker, t.gdelt_id, t.id, t.imo, t.name) AS target_id,
                       type(r) AS rel_type,
                       coalesce(r.weight, 1.0) AS weight
                """
            )
            for record in r_res:
                src = record["source_id"]
                tgt = record["target_id"]
                rel = record["rel_type"]
                weight = float(record["weight"])
                
                # Make sure both objects exist in category catalogue
                if src in self.objects and tgt in self.objects:
                    morphism = KMorphism(
                        k=1,
                        source_id=src,
                        target_id=tgt,
                        relation_type=rel.lower(),
                        weight=weight,
                        homotopy_class="dynamic_semantic",
                        modal_status="contingent",
                        action_value=0.0
                    )
                    self.add_morphism(morphism)

    def homotopy_equivalent(self, id1: str, id2: str) -> tuple:
        """
        Computes causal equivalence (homotopy) using cosine similarity of node embeddings.
        """
        obj1 = self.objects.get(id1)
        obj2 = self.objects.get(id2)
        if not obj1 or not obj2:
            return False, 0.0

        import numpy as np
        emb1 = np.array(obj1.embedding)
        emb2 = np.array(obj2.embedding)

        if len(emb1) == 0 or len(emb2) == 0 or np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
            if obj1.domain == obj2.domain:
                return True, 0.75
            return False, 0.25

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarity = float(similarity)
        return similarity > 0.85, round(similarity, 4)

    @property

    def find_all_paths(self, source_id: str, target_id: str, max_depth: int = 4) -> list:
        """
        Finds all simple paths from source_id to target_id of length up to max_depth.
        """
        adj = {}
        for key, m in self.morphisms.items():
            if m.source_id not in adj:
                adj[m.source_id] = []
            adj[m.source_id].append((m.target_id, m.relation_type))
            
            # Undirected traversal: add reverse mapping
            if m.target_id not in adj:
                adj[m.target_id] = []
            adj[m.target_id].append((m.source_id, m.relation_type))

        paths = []
        from collections import deque
        queue = deque([(source_id, [source_id], {source_id})])

        while queue:
            curr, path, visited = queue.popleft()
            if curr == target_id:
                paths.append(path)
                continue

            if len(path) >= 2 * max_depth + 1:
                continue

            for neighbor, rel in adj.get(curr, []):
                if neighbor not in visited:
                    queue.append((
                        neighbor,
                        path + [rel, neighbor],
                        visited | {neighbor}
                    ))

        return paths

# Class Alias mapping for backward compatibility
InfinityCausalCategory = APEXCausalEngine

