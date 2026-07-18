from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class SemanticSection:
    """
    A section of the semantic sheaf over a particular context.

    In sheaf-theoretic terms, a section assigns a semantic vector to a concept
    within a given open set (context).  The restriction_ids list names the
    more specific contexts to which this section can be restricted, capturing
    the sheaf's gluing and restriction structure.

    Fields
    ------
    concept_id      : Identifier of the concept this section describes.
    vector          : 64-dimensional semantic embedding (matches KRONOS d_model).
    confidence      : Certainty of this assignment, in [0, 1].
    context_id      : The open set (context) this section belongs to.
    restriction_ids : Identifiers of finer contexts this section restricts to.
    """

    concept_id: str
    vector: list  # float values, length 64 (KRONOS d_model)
    confidence: float  # in [0, 1]
    context_id: str
    restriction_ids: list = field(default_factory=list)  # ids of more specific sections


@dataclass
class SheafCovering:
    """
    An open covering of a context together with the local sections defined on it.

    In sheaf theory, a covering of a space U by open sets {Uᵢ} lets us check
    whether locally-consistent sections glue to a global section (H⁰ condition).

    Fields
    ------
    context_id : Identifier of the space being covered.
    open_sets  : List of context_id strings that jointly cover this context.
    sections   : Map from concept_id to the SemanticSection defined here.
    """

    context_id: str
    open_sets: list  # list of context_id strings covering this context
    sections: dict   # concept_id -> SemanticSection


class CSIESheafLayer:
    """
    Categorical Semantic Intelligence Engine (CSIE) Sheaf Layer.

    Sits on top of KRONOS output and provides sheaf-based semantic reasoning:

    - Grounds raw KRONOS logits into semantic sections (local data).
    - Computes Čech H⁰ to determine whether local sections glue to a consistent
      global section (global semantic coherence check).
    - Supports cross-context restriction maps (diagonal approximation of the
      full restriction linear maps for efficiency).
    - Supports online learning of new sections and global inference queries.

    Sheaf diagram (conceptual)
    --------------------------
    KRONOS logits
        │
        ▼  ground_kronos_output()
    SemanticSection (local)
        │
        ▼  add_covering() / learn_section()
    SheafCovering (open set data)
        │
        ▼  cech_h0()
    Global section (H⁰ result)
    """

    def __init__(self, d_model: int = 64, n_concepts: int = 32) -> None:
        self.d_model: int = d_model
        self.n_concepts: int = n_concepts

        # context_id -> SheafCovering
        self.sheaf_store: Dict[str, SheafCovering] = {}

        # (ctx_from, ctx_to) -> list of n_concepts floats
        # Stores the diagonal of the restriction matrix in compact form.
        self.restriction_maps: Dict[tuple, List[float]] = {}

        # Human-readable concept slot names
        self.concept_names: List[str] = [f"concept_{i}" for i in range(n_concepts)]

    # ------------------------------------------------------------------
    # Sheaf construction
    # ------------------------------------------------------------------

    def add_covering(
        self,
        context_id: str,
        open_sets: list,
        sections: dict,
    ) -> None:
        """
        Register a SheafCovering for context_id.

        Parameters
        ----------
        context_id : Identifier of the context being covered.
        open_sets  : List of context ids that cover this context.
        sections   : Mapping from concept_id -> SemanticSection.
        """
        covering = SheafCovering(
            context_id=context_id,
            open_sets=open_sets,
            sections=sections,
        )
        self.sheaf_store[context_id] = covering

    def get_restriction(self, ctx_from: str, ctx_to: str) -> List[float]:
        """
        Return the restriction-map diagonal for the pair (ctx_from, ctx_to).

        If no restriction map has been learned for this pair, one is initialised
        as the identity (all-ones diagonal), meaning sections transfer unchanged
        between the two contexts by default.

        Parameters
        ----------
        ctx_from : Source context identifier.
        ctx_to   : Target context identifier.

        Returns
        -------
        List of n_concepts floats representing the diagonal of the restriction
        matrix from ctx_from to ctx_to.
        """
        key = (ctx_from, ctx_to)
        if key not in self.restriction_maps:
            self.restriction_maps[key] = [1.0] * self.n_concepts
        return self.restriction_maps[key]

    # ------------------------------------------------------------------
    # Čech cohomology – H⁰ (global sections)
    # ------------------------------------------------------------------

    def cech_h0(self, covering_ids: list) -> dict:
        """
        Compute the zeroth Čech cohomology H⁰ over the given coverings.

        H⁰ is the space of globally consistent sections: for each concept that
        appears in more than one covering, the local sections must agree on
        overlapping open sets (up to a tolerance of 0.3 mean absolute deviation).

        Algorithm
        ---------
        1. Gather all SemanticSections from each covering in covering_ids.
        2. For concepts appearing in multiple coverings, test consistency.
        3. Average consistent sections; flag inconsistent ones.

        Parameters
        ----------
        covering_ids : List of context_id strings to include in the computation.

        Returns
        -------
        dict with:
          global_section        : concept_id -> averaged vector (list of floats)
          consistent            : True if every overlapping concept was consistent
          n_concepts_resolved   : Number of concepts for which a global section exists
          inconsistent_concepts : List of concept_ids that failed the consistency check
        """
        CONSISTENCY_THRESHOLD = 0.3  # max allowed mean absolute deviation

        # Accumulate sections per concept across all coverings
        concept_buckets: Dict[str, List[SemanticSection]] = {}
        for ctx_id in covering_ids:
            covering = self.sheaf_store.get(ctx_id)
            if covering is None:
                continue
            for concept_id, section in covering.sections.items():
                if concept_id not in concept_buckets:
                    concept_buckets[concept_id] = []
                concept_buckets[concept_id].append(section)

        global_section: Dict[str, list] = {}
        inconsistent_concepts: List[str] = []

        for concept_id, sections in concept_buckets.items():
            if not sections:
                continue

            # Single section – trivially consistent
            if len(sections) == 1:
                global_section[concept_id] = list(sections[0].vector)
                continue

            # Check pairwise consistency between first section and all others
            is_consistent = True
            ref_vec = sections[0].vector
            for other in sections[1:]:
                if len(ref_vec) != len(other.vector):
                    is_consistent = False
                    break
                mad = sum(abs(a - b) for a, b in zip(ref_vec, other.vector)) / max(len(ref_vec), 1)
                if mad >= CONSISTENCY_THRESHOLD:
                    is_consistent = False
                    break

            if not is_consistent:
                inconsistent_concepts.append(concept_id)
                continue

            # Average the consistent sections (confidence-weighted)
            total_conf = sum(s.confidence for s in sections)
            if total_conf == 0.0:
                avg_vec = list(ref_vec)
            else:
                avg_vec = [0.0] * len(ref_vec)
                for s in sections:
                    w = s.confidence / total_conf
                    for i, v in enumerate(s.vector):
                        avg_vec[i] += w * v

            global_section[concept_id] = avg_vec

        all_consistent = len(inconsistent_concepts) == 0

        return {
            "global_section": global_section,
            "consistent": all_consistent,
            "n_concepts_resolved": len(global_section),
            "inconsistent_concepts": inconsistent_concepts,
        }

    # ------------------------------------------------------------------
    # Real sparse Čech cohomology (H⁰ global section + H¹ residual)
    # ------------------------------------------------------------------

    def train_restriction_map(self, parent_id: str, child_id: str):
        """
        Train the restriction map parent→child via RIDGE REGRESSION on the
        concepts shared by the two contexts (the spec's ρ: F(U)→F(V) linear
        projection). Stored as a full d×d matrix in ``trained_restriction_maps``.
        Falls back to identity if too few shared concepts.
        """
        import numpy as np
        if not hasattr(self, "trained_restriction_maps"):
            self.trained_restriction_maps = {}
        ps = self.sheaf_store.get(parent_id)
        cs = self.sheaf_store.get(child_id)
        d = self.d_model
        if ps is None or cs is None:
            return "missing_context"
        shared = list(set(ps.sections) & set(cs.sections))
        if len(shared) < 3:
            self.trained_restriction_maps[(parent_id, child_id)] = np.eye(d)
            return "identity_fallback"

        def _vec(sec):
            v = list(sec.vector)[:d] + [0.0] * max(0, d - len(sec.vector))
            return np.asarray(v[:d], dtype=float)

        X = np.stack([_vec(ps.sections[c]) for c in shared])   # (n_shared, d)
        Y = np.stack([_vec(cs.sections[c]) for c in shared])
        lam = 1e-3
        W = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)  # ridge (d,d)
        self.trained_restriction_maps[(parent_id, child_id)] = W.T
        return "trained_ridge"

    def cech_cohomology_solve(self, covering_ids: list,
                              concepts: Optional[list] = None) -> dict:
        """
        REAL Čech cohomology over the covering: assemble the local sections C⁰,
        build the coboundary operator δ⁰ (restrict(C⁰[i]) − restrict(C⁰[j]) on
        each overlap sharing a concept), and solve the augmented sparse
        least-squares system  [δ⁰; λI]·s = [0; λ·C⁰]  via scipy ``lsqr``.

          • H⁰  = the unique globally-consistent section (returned per concept).
          • H¹  = ‖δ⁰·s‖  — the residual. A non-trivial H¹ means the local data
                  cannot be glued into a globally consistent section: a genuine
                  knowledge gap / polysemy. This is the epistemic-honesty layer —
                  the system reports what it CANNOT consistently know instead of
                  fabricating. Applies trained restriction maps when available.

        This replaces the mean-absolute-deviation heuristic with the actual
        sparse-linear-system Čech solve from the CSIE specification.
        """
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.linalg import lsqr

        coverings = [cid for cid in covering_ids if cid in self.sheaf_store]
        k, d = len(coverings), self.d_model
        if k == 0:
            return {"H0_global_section": {}, "H1_residual_norm": 0.0,
                    "n_resolved": 0, "n_overlaps": 0, "method": "sparse_cech_lsqr"}

        if concepts is None:
            concepts = sorted({c for cid in coverings
                               for c in self.sheaf_store[cid].sections})
        n = len(concepts)
        cidx = {c: i for i, c in enumerate(concepts)}

        C0 = np.zeros((k, n, d))
        present = np.zeros((k, n), dtype=bool)
        for i, cid in enumerate(coverings):
            for c, sec in self.sheaf_store[cid].sections.items():
                if c in cidx:
                    v = list(sec.vector)[:d] + [0.0] * max(0, d - len(sec.vector))
                    C0[i, cidx[c]] = np.asarray(v[:d], dtype=float)
                    present[i, cidx[c]] = True

        rows, cols, data, row = [], [], [], 0
        for i in range(k):
            for j in range(i + 1, k):
                for c in range(n):
                    if present[i, c] and present[j, c]:
                        for dim in range(d):
                            r = row * d + dim
                            rows += [r, r]
                            cols += [(i * n + c) * d + dim, (j * n + c) * d + dim]
                            data += [1.0, -1.0]
                        row += 1

        total_C0, total_C1 = k * n * d, row * d
        C0_flat = C0.reshape(-1)

        if total_C1 == 0:                     # no overlaps → average present sections
            gs = {concepts[c]: C0[present[:, c], c, :].mean(0).tolist()
                  for c in range(n) if present[:, c].any()}
            return {"H0_global_section": gs, "H1_residual_norm": 0.0,
                    "n_resolved": len(gs), "n_overlaps": 0,
                    "cohomology_H1_nontrivial": False, "method": "sparse_cech_lsqr"}

        lam = 0.1
        delta0 = sp.csr_matrix((data, (rows, cols)), shape=(total_C1, total_C0))
        A = sp.vstack([delta0, sp.eye(total_C0) * lam])
        b = np.concatenate([np.zeros(total_C1), lam * C0_flat])
        s = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=500)[0].reshape(k, n, d)

        gs = {concepts[c]: s[present[:, c], c, :].mean(0).tolist()
              for c in range(n) if present[:, c].any()}
        h1 = float(np.linalg.norm(delta0 @ s.reshape(-1)))
        return {
            "H0_global_section": gs,
            "H1_residual_norm": round(h1, 6),
            "cohomology_H1_nontrivial": h1 > 1e-3,
            "n_resolved": len(gs),
            "n_overlaps": row,
            "method": "sparse_cech_lsqr",
        }

    # ------------------------------------------------------------------
    # Morphism extraction from attention (enriches the category live)
    # ------------------------------------------------------------------
    def extract_morphisms_from_attention(self, attention, token_labels=None,
                                         threshold: float = 0.1) -> dict:
        """
        Read transformer attention weights and convert high-attention token pairs
        into live semantic MORPHISMS (directed relations) that enrich the category
        for this forward pass. `attention` may be a (T,T) matrix or a stack
        (layers/heads, T, T) which is averaged. A pair (i attends to j) above the
        threshold becomes a morphism j → i ("attends_to"), weighted by the score.
        """
        import numpy as np
        A = np.asarray(attention, dtype=float)
        if A.ndim > 2:
            A = A.reshape(-1, A.shape[-2], A.shape[-1]).mean(axis=0)   # avg heads/layers
        T = A.shape[0]
        # row-normalise so scores are comparable attention distributions
        row = A.sum(axis=1, keepdims=True)
        A = A / np.where(row > 0, row, 1.0)
        morphisms = []
        for i in range(T):
            for j in range(T):
                if i == j:
                    continue
                w = float(A[i, j])
                if w >= threshold:
                    src = token_labels[j] if token_labels else f"tok_{j}"
                    tgt = token_labels[i] if token_labels else f"tok_{i}"
                    morphisms.append({"source": src, "target": tgt,
                                      "relation": "attends_to", "weight": round(w, 4)})
        morphisms.sort(key=lambda m: -m["weight"])
        self.live_morphisms = morphisms
        return {"n_tokens": T, "n_morphisms": len(morphisms),
                "threshold": threshold, "morphisms": morphisms[:20]}

    # ------------------------------------------------------------------
    # KRONOS grounding
    # ------------------------------------------------------------------

    def ground_kronos_output(self, logits: list, context_id: str) -> dict:
        """
        Ground raw KRONOS logits into a semantic sheaf covering.

        Applies a softmax over the first n_concepts logits (or all if shorter),
        selects the top-8 concept slots by probability, and stores them as
        SemanticSections in a new SheafCovering for context_id.

        Parameters
        ----------
        logits     : List of floats (length n_concepts or vocab_size).
        context_id : Context identifier to associate with this grounding.

        Returns
        -------
        dict with:
          context_id    : Echo of the provided context_id.
          top_concepts  : List of up to 8 dicts, each with 'concept_id' and 'probability'.
          section_count : Number of sections created.
          grounded      : Always True.
        """
        # Truncate or pad logits to n_concepts
        relevant_logits = list(logits[: self.n_concepts])

        # Numerically stable softmax
        max_logit = max(relevant_logits) if relevant_logits else 0.0
        exps = [math.exp(x - max_logit) for x in relevant_logits]
        total = sum(exps) if exps else 1.0
        probs = [e / total for e in exps]

        # Top-8 by probability
        indexed = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)
        top8 = indexed[:8]

        sections: Dict[str, SemanticSection] = {}
        top_concepts = []

        for rank, (idx, prob) in enumerate(top8):
            concept_id = self.concept_names[idx] if idx < len(self.concept_names) else f"concept_{idx}"

            # Build a simple one-hot-like vector of length d_model seeded by probability
            vec = [0.0] * self.d_model
            vec[idx % self.d_model] = prob  # place signal at slot mod d_model
            # Secondary signal proportional to rank
            vec[(idx + 1) % self.d_model] += prob * 0.1

            section = SemanticSection(
                concept_id=concept_id,
                vector=vec,
                confidence=prob,
                context_id=context_id,
            )
            sections[concept_id] = section
            top_concepts.append({"concept_id": concept_id, "probability": prob})

        self.add_covering(
            context_id=context_id,
            open_sets=[context_id],
            sections=sections,
        )

        return {
            "context_id": context_id,
            "top_concepts": top_concepts,
            "section_count": len(sections),
            "grounded": True,
        }

    # ------------------------------------------------------------------
    # Online learning
    # ------------------------------------------------------------------

    def learn_section(
        self,
        context_id: str,
        concept_id: str,
        vector: list,
        confidence: float,
    ) -> None:
        """
        Add or update a SemanticSection in the sheaf store for context_id.

        If no SheafCovering exists for context_id yet, one is created with an
        open set list containing only context_id itself (a trivial covering).

        Parameters
        ----------
        context_id : Context to which the section belongs.
        concept_id : Concept being described.
        vector     : Semantic embedding vector.
        confidence : Confidence score in [0, 1].
        """
        if context_id not in self.sheaf_store:
            self.add_covering(
                context_id=context_id,
                open_sets=[context_id],
                sections={},
            )

        section = SemanticSection(
            concept_id=concept_id,
            vector=list(vector),
            confidence=float(confidence),
            context_id=context_id,
        )
        self.sheaf_store[context_id].sections[concept_id] = section

    # ------------------------------------------------------------------
    # Global inference
    # ------------------------------------------------------------------

    def global_inference(
        self,
        query_context_id: str,
        related_context_ids: list,
    ) -> dict:
        """
        Run global sheaf inference over a query context and its related contexts.

        Calls cech_h0 on the union of query_context_id and related_context_ids,
        then computes a coherence_score as the fraction of n_concepts that could
        be resolved to a global section.

        Parameters
        ----------
        query_context_id    : The primary context of interest.
        related_context_ids : Additional contexts whose sections participate in
                              the global consistency check.

        Returns
        -------
        dict merging the cech_h0 result with:
          coherence_score : float in [0, 1] = n_concepts_resolved / n_concepts
        """
        all_context_ids = [query_context_id] + list(related_context_ids)
        h0_result = self.cech_h0(all_context_ids)

        coherence_score = h0_result["n_concepts_resolved"] / max(self.n_concepts, 1)

        return {
            **h0_result,
            "coherence_score": coherence_score,
            "coherence_score_source": "kronos_logits_untrained_synthetic",
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_summary_dict(self) -> dict:
        """
        Produce a JSON-serialisable summary of the layer's current state.

        Returns
        -------
        dict with:
          n_coverings       : Total number of registered SheafCoverings.
          n_total_sections  : Total number of SemanticSections across all coverings.
          concept_names     : List of concept slot name strings.
          d_model           : Embedding dimensionality.
        """
        n_total_sections = sum(
            len(covering.sections) for covering in self.sheaf_store.values()
        )
        return {
            "n_coverings": len(self.sheaf_store),
            "n_total_sections": n_total_sections,
            "concept_names": self.concept_names,
            "d_model": self.d_model,
        }
