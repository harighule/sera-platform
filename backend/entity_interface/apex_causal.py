from dataclasses import dataclass, field
from typing import Optional, List, Dict
import math


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

        return {
            "n_objects": n_objects,
            "n_morphisms": n_morphisms,
            "max_depth": max_depth,
            "homotopy_classes": homotopy_class_names,
            "causal_density": causal_density,
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
            }
            for key, m in sorted_morphisms[:5]
        ]

        return {
            "cohomology_signature": sig,
            "top_5_morphisms": top_5,
            "n_total_morphisms": len(self.morphisms),
            "max_causal_depth": sig["max_depth"],
        }
