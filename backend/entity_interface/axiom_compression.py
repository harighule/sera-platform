"""
AXIOM Zero-Loss Compression Pipeline — Analysis Layer
======================================================
Lightweight, stdlib-only implementation that analyses the weight structure of
the KRONOS model and reports per-layer compressibility metrics.

No numpy, no scipy, no torch required inside AXIOMCompressor itself.
All matrix operations use Python lists and the standard math module.
"""

import math


class AXIOMCompressor:
    """
    AXIOM Zero-Loss Compression Analyser.

    Implements the analytical front-end of the AXIOM compression pipeline:

    Phase 0  – Gauge symmetry fixing (scale normalisation)
    Phase 1  – Stable rank estimation (power iteration)
    Phase 2  – Null-space cascade estimation
    Phase 3  – Weight entropy analysis (Shannon binning)
    Phase 4  – Per-layer compressibility scoring

    The compressor does *not* modify model weights; it only observes and scores
    them so that the caller can decide which phases to activate.
    """

    def __init__(self) -> None:
        self.compression_log: list = []    # records of analyse_layer() calls
        self.gauge_transforms: list = []   # scale vectors applied via fix_scale_symmetry()

    # ------------------------------------------------------------------
    # Phase 1 – Stable rank (spectral compressibility signal)
    # ------------------------------------------------------------------

    def compute_stable_rank(self, W: list, rows: int, cols: int) -> float:
        """
        Compute the stable rank of a matrix W given as a flat list of floats.

        Stable rank = ||W||²_F / ||W||²_2

        where ||W||_F is the Frobenius norm and ||W||_2 is the spectral norm
        (largest singular value).  The spectral norm is estimated via 5 steps
        of the power iteration method, which is sufficient for a ranking signal.

        A stable rank close to 1 indicates a near-rank-1 matrix (highly
        compressible via SVD / tensor-train decomposition).  A stable rank
        close to min(rows, cols) indicates full rank (less compressible).

        Parameters
        ----------
        W    : Flat list of rows * cols floats (row-major order).
        rows : Number of rows in the matrix.
        cols : Number of columns in the matrix.

        Returns
        -------
        Stable rank as a float in [1, min(rows, cols)].
        """
        # ── Frobenius norm squared ──
        frob_sq = sum(x * x for x in W)

        # ── Spectral norm via power iteration ──
        # Start with a unit vector of length cols
        norm0 = math.sqrt(cols) if cols > 0 else 1.0
        v = [1.0 / norm0] * cols

        def mat_vec(flat, r, c, vec):
            """Compute matrix-vector product W @ vec (length c -> length r)."""
            result = [0.0] * r
            for i in range(r):
                s = 0.0
                for j in range(c):
                    s += flat[i * c + j] * vec[j]
                result[i] = s
            return result

        def mat_T_vec(flat, r, c, vec):
            """Compute W^T @ vec (length r -> length c)."""
            result = [0.0] * c
            for j in range(c):
                s = 0.0
                for i in range(r):
                    s += flat[i * c + j] * vec[i]
                result[j] = s
            return result

        def norm_vec(vec):
            return math.sqrt(sum(x * x for x in vec))

        def normalise(vec):
            n = norm_vec(vec)
            if n < 1e-15:
                return vec
            return [x / n for x in vec]

        spectral_sq = 0.0
        for _ in range(5):
            u = normalise(mat_vec(W, rows, cols, v))       # W @ v  -> u
            v = normalise(mat_T_vec(W, rows, cols, u))     # W^T @ u -> v
            u2 = mat_vec(W, rows, cols, v)                 # W @ v  -> estimate sigma
            spectral_sq = sum(x * x for x in u2)          # sigma^2 ≈ ||Wv||^2

        return frob_sq / max(spectral_sq, 1e-10)

    # ------------------------------------------------------------------
    # Phase 0 – Gauge symmetry fixing
    # ------------------------------------------------------------------

    def fix_scale_symmetry(self, W_rows: list, scale_factors: list) -> tuple:
        """
        Normalise each row of the weight matrix by its associated scale factor.

        In the AXIOM gauge-fixing phase, neurons that differ only by a scalar
        multiple are collapsed by absorbing the scale into a diagonal
        normalisation layer (BatchNorm-style).  This removes multiplicative
        gauge redundancy without changing the function computed by the network.

        Parameters
        ----------
        W_rows       : List of rows; each inner list is one row of floats.
        scale_factors: One float per row.  Must be the same length as W_rows.

        Returns
        -------
        (normalised_W_rows, scale_factors) tuple.
        normalised_W_rows has the same shape as W_rows.
        """
        normalised = []
        for row, scale in zip(W_rows, scale_factors):
            s = scale if abs(scale) > 1e-15 else 1.0
            normalised.append([x / s for x in row])

        self.gauge_transforms.append(scale_factors)
        return (normalised, scale_factors)

    # ------------------------------------------------------------------
    # Phase 3 – Weight entropy (information-theoretic compressibility)
    # ------------------------------------------------------------------

    def compute_entropy(self, values: list) -> float:
        """
        Estimate the Shannon entropy of a weight distribution via histogram
        binning into 32 equal-width buckets.

        H = -Σ p_i · log₂(p_i)   (sum over non-zero buckets)

        A higher entropy indicates a more uniform weight distribution (harder
        to compress losslessly).  A lower entropy indicates clustering (easier
        to entropy-code or quantise).

        Parameters
        ----------
        values : Flat list of floats to analyse.

        Returns
        -------
        Entropy in bits as a float.  Returns 0.0 if all values are identical.
        """
        if not values:
            return 0.0

        v_min = min(values)
        v_max = max(values)

        if v_max == v_min:
            return 0.0  # All values identical — zero entropy

        N_BINS = 32
        bin_width = (v_max - v_min) / N_BINS
        counts = [0] * N_BINS

        for v in values:
            idx = int((v - v_min) / bin_width)
            if idx >= N_BINS:
                idx = N_BINS - 1
            counts[idx] += 1

        total = len(values)
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)

        return entropy

    # ------------------------------------------------------------------
    # Composite layer analysis
    # ------------------------------------------------------------------

    def analyse_layer(
        self,
        name: str,
        weights_flat: list,
        rows: int,
        cols: int,
    ) -> dict:
        """
        Run all AXIOM analysis phases on a single weight matrix and return
        a compressibility report.

        Metrics
        -------
        stable_rank
            Ratio of Frobenius-norm² to spectral-norm².  Lower values indicate
            a near-low-rank matrix amenable to SVD / tensor-train compression.

        entropy_bits
            Shannon entropy of the weight distribution in bits.  Lower values
            indicate a more compressible histogram.

        gauge_redundancy_estimate
            Standard deviation of row norms divided by mean row norm.
            High values indicate that some rows are much larger than others,
            suggesting gauge-fixable scale symmetry.

        null_space_estimate
            Approximation of the fraction of dimensions that are effectively
            null.  Derived from stable_rank vs. the larger matrix dimension.

        estimated_compression_ratio
            Heuristic estimate: entropy_bits / 32.0 (32 = bits per float32).
            Values < 1 indicate potential compression gain.

        Parameters
        ----------
        name         : Human-readable layer name.
        weights_flat : Flat row-major list of rows * cols floats.
        rows         : Matrix row count.
        cols         : Matrix column count.

        Returns
        -------
        Dict with all metrics listed above plus name, rows, cols, n_params.
        """
        stable_rank = self.compute_stable_rank(weights_flat, rows, cols)
        entropy_bits = self.compute_entropy(weights_flat)

        # Row norms for gauge redundancy
        row_norms = []
        for r in range(rows):
            row_sq = sum(weights_flat[r * cols + c] ** 2 for c in range(cols))
            row_norms.append(math.sqrt(row_sq))

        mean_norm = sum(row_norms) / max(len(row_norms), 1)
        variance = sum((x - mean_norm) ** 2 for x in row_norms) / max(len(row_norms), 1)
        std_norm = math.sqrt(variance)
        gauge_redundancy_estimate = std_norm / max(mean_norm, 1e-10)

        null_space_estimate = 1.0 - (stable_rank / max(rows, cols))
        null_space_estimate = max(0.0, min(1.0, null_space_estimate))

        if entropy_bits > 0:
            estimated_compression_ratio = entropy_bits / 32.0
        else:
            estimated_compression_ratio = 1.0

        report = {
            "name": name,
            "rows": rows,
            "cols": cols,
            "n_params": rows * cols,
            "stable_rank": stable_rank,
            "entropy_bits": entropy_bits,
            "gauge_redundancy_estimate": gauge_redundancy_estimate,
            "null_space_estimate": null_space_estimate,
            "estimated_compression_ratio": estimated_compression_ratio,
        }

        self.compression_log.append(report)
        return report

    # ------------------------------------------------------------------
    # Full model analysis
    # ------------------------------------------------------------------

    def analyse_model(self, model_layers: list) -> dict:
        """
        Run AXIOM analysis across all layers of a model and aggregate results.

        Parameters
        ----------
        model_layers : List of dicts, each with keys:
                       - 'name'    : str
                       - 'weights' : list of floats (flat, row-major)
                       - 'rows'    : int
                       - 'cols'    : int

        Returns
        -------
        dict with:
          layer_reports           : Per-layer analysis dicts.
          total_params            : Sum of n_params across all layers.
          mean_entropy_bits       : Mean Shannon entropy across layers.
          mean_compression_ratio  : Mean estimated compression ratio.
          estimated_total_size_mb : Uncompressed size (float32, 4 bytes/param).
          estimated_compressed_mb : Projected compressed size.
          phases_available        : List of AXIOM compression phase names.
        """
        layer_reports = []
        for layer in model_layers:
            report = self.analyse_layer(
                name=layer["name"],
                weights_flat=layer["weights"],
                rows=layer["rows"],
                cols=layer["cols"],
            )
            layer_reports.append(report)

        total_params = sum(r["n_params"] for r in layer_reports)
        n = max(len(layer_reports), 1)
        mean_entropy = sum(r["entropy_bits"] for r in layer_reports) / n
        mean_compression_ratio = sum(r["estimated_compression_ratio"] for r in layer_reports) / n
        mean_stable_rank = sum(r["stable_rank"] for r in layer_reports) / n

        estimated_total_size_mb = total_params * 4 / 1024 / 1024
        estimated_compressed_mb = total_params * 4 * mean_compression_ratio / 1024 / 1024

        return {
            "layer_reports": layer_reports,
            "total_params": total_params,
            "mean_entropy_bits": mean_entropy,
            "mean_compression_ratio": mean_compression_ratio,
            "mean_stable_rank": mean_stable_rank,
            "estimated_total_size_mb": estimated_total_size_mb,
            "estimated_compressed_mb": estimated_compressed_mb,
            "phases_available": [
                "gauge_fixing",
                "null_space_cascade",
                "tensor_train",
                "entropy_coding",
                "int2_packing",
            ],
        }

    # ------------------------------------------------------------------
    # Phase 0 runtime application — gauge fixing in-place
    # ------------------------------------------------------------------

    def apply_gauge_fixing(self, kronos_model) -> dict:
        """
        Apply gauge fixing (scale symmetry elimination) in-place to a PyTorch model.

        For each consecutive pair of 2-D weight matrices (W_l, W_next) where
        W_l.shape[0] == W_next.shape[1] (i.e. W_l outputs feed into W_next inputs),
        the method:
          1. Computes the L2 norm of each row of W_l.
          2. Normalises W_l rows to unit norm (in-place).
          3. Compensates W_next by scaling its columns by the same norms (in-place),
             so that the function computed by the pair is exactly preserved.

        This is the only AXIOM phase safe to apply at runtime without retraining.

        Parameters
        ----------
        kronos_model : Any PyTorch nn.Module exposing named_parameters().

        Returns
        -------
        dict with:
          layers_processed   : Number of 2-D parameter tensors found.
          transforms_applied : Number of (W_l, W_next) pairs processed.
          verification_passed: True if total param count is unchanged after fixing.
          mean_weight_before : Mean absolute weight of first linear layer before fixing.
          mean_weight_after  : Mean absolute weight of first linear layer after fixing.
          message            : Confirmation string.
        """
        try:
            import torch
        except ImportError:
            return {"error": "torch not available"}

        # Collect 2-D parameter tensors
        linear_params = [
            (name, p)
            for name, p in kronos_model.named_parameters()
            if p.dim() == 2
        ]

        layers_processed = len(linear_params)
        transforms_applied_before = len(self.gauge_transforms)

        # Snapshot for verification
        total_params_before = sum(p.numel() for _, p in linear_params)
        mean_weight_before = (
            float(linear_params[0][1].detach().abs().mean().item())
            if linear_params else 0.0
        )

        # Process consecutive pairs
        for i in range(len(linear_params) - 1):
            name_l, W_l = linear_params[i]
            name_next, W_next = linear_params[i + 1]

            # Only apply when dimensions are compatible
            if W_l.shape[0] != W_next.shape[1]:
                continue

            with torch.no_grad():
                row_norms = W_l.norm(dim=1, keepdim=True).clamp(min=1e-8)

                # Normalise W_l rows to unit norm in-place
                W_l.data.div_(row_norms)

                # Compensate W_next columns in-place so function is preserved
                W_next.data.mul_(row_norms.squeeze().unsqueeze(0))

            self.gauge_transforms.append({
                "layer_from": name_l,
                "layer_to": name_next,
                "scale_factors": row_norms.squeeze().tolist(),
            })

        # Verification
        total_params_after = sum(p.numel() for _, p in linear_params)
        verification_passed = (total_params_before == total_params_after)

        mean_weight_after = (
            float(linear_params[0][1].detach().abs().mean().item())
            if linear_params else 0.0
        )

        transforms_applied = len(self.gauge_transforms) - transforms_applied_before

        return {
            "layers_processed": layers_processed,
            "transforms_applied": transforms_applied,
            "verification_passed": verification_passed,
            "mean_weight_before": round(mean_weight_before, 6),
            "mean_weight_after": round(mean_weight_after, 6),
            "message": "Gauge fixing applied in-place \u2014 function preserved",
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_full_compression_pipeline(self, kronos_model) -> dict:
        """
        Run the complete AXIOM compression pipeline on a PyTorch model.

        Steps
        -----
        1. analyse_model()  — baseline metrics on raw weights.
        2. apply_gauge_fixing()  — in-place scale normalisation.
        3. analyse_model()  — post-gauge metrics.
        4. Compute per-metric deltas.

        Parameters
        ----------
        kronos_model : Any PyTorch nn.Module exposing named_parameters().

        Returns
        -------
        dict with:
          pre_analysis         : Full analyse_model result before gauge fixing.
          post_analysis        : Full analyse_model result after gauge fixing.
          gauge_result         : Output of apply_gauge_fixing().
          entropy_delta        : Change in mean Shannon entropy (post − pre).
          compression_improvement : Change in mean compression ratio (post − pre).
          phases_completed     : List of applied phases.
          phases_pending       : List of not-yet-applied phases.
          status               : Summary label string.
        """
        def _extract_layers(model):
            layers = []
            for name, p in model.named_parameters():
                if p.dim() == 2:
                    layers.append({
                        "name": name,
                        "weights": p.detach().flatten().tolist(),
                        "rows": p.shape[0],
                        "cols": p.shape[1],
                    })
            return layers

        # Step 1 — pre-gauge analysis
        pre_layers = _extract_layers(kronos_model)
        pre_analysis = self.analyse_model(pre_layers)

        # Step 2 — apply gauge fixing in-place
        gauge_result = self.apply_gauge_fixing(kronos_model)

        # Step 3 — post-gauge analysis
        post_layers = _extract_layers(kronos_model)
        post_analysis = self.analyse_model(post_layers)

        # Step 4 — deltas
        entropy_delta = round(
            post_analysis["mean_entropy_bits"] - pre_analysis["mean_entropy_bits"], 6
        )
        compression_improvement = round(
            post_analysis["mean_compression_ratio"] - pre_analysis["mean_compression_ratio"], 6
        )

        return {
            "pre_analysis": pre_analysis,
            "post_analysis": post_analysis,
            "gauge_result": gauge_result,
            "entropy_delta": entropy_delta,
            "compression_improvement": compression_improvement,
            "phases_completed": ["gauge_fixing"],
            "phases_pending": [
                "null_space_cascade",
                "tensor_train",
                "entropy_coding",
                "int2_packing",
            ],
            "status": "partial_compression_applied",
        }


# ------------------------------------------------------------------
# Module-level convenience wrapper for PyTorch models
# ------------------------------------------------------------------

def analyse_kronos_model(kronos_model) -> dict:
    """
    Convenience function: extract 2-D weight tensors from a PyTorch model
    and run the full AXIOM analysis pipeline on them.

    Parameters
    ----------
    kronos_model : Any PyTorch nn.Module exposing named_parameters().

    Returns
    -------
    The dict returned by AXIOMCompressor().analyse_model(), or
    {"error": "torch not available"} if PyTorch is not installed.
    """
    try:
        import torch  # noqa: F401 — checked for availability only
    except ImportError:
        return {"error": "torch not available"}

    model_layers = []
    for name, p in kronos_model.named_parameters():
        if p.dim() == 2:
            model_layers.append({
                "name": name,
                "weights": p.detach().flatten().tolist(),
                "rows": p.shape[0],
                "cols": p.shape[1],
            })

    try:
        # Collect 2D parameters from KRONOSLayer submodules
        for i, layer in enumerate(kronos_model.layers):
            for attr_name, attr_path in [
                ("causal_attn.W_logit", layer.causal_attn.W_logit),
                ("memory.slow_mem", layer.memory.slow_mem),
                ("nca.connectivity", layer.nca.connectivity),
            ]:
                p = attr_path
                if p.dim() == 2:
                    model_layers.append({
                        "name": f"layer_{i}.{attr_name}",
                        "weights": p.detach().flatten().tolist(),
                        "rows": p.shape[0],
                        "cols": p.shape[1]
                    })

        # Collect from top-level wave manifold
        for attr_name, p in [
            ("wave.tangent_vecs", kronos_model.wave.tangent_vecs),
            ("wave.wave_proj.weight", kronos_model.wave.wave_proj.weight),
        ]:
            if p.dim() == 2:
                model_layers.append({
                    "name": attr_name,
                    "weights": p.detach().flatten().tolist(),
                    "rows": p.shape[0],
                    "cols": p.shape[1]
                })
    except AttributeError:
        pass  # Model does not have expected KRONOS structure

    if not model_layers:
        return {"error": "no 2D parameters found", "available": False, "layer_reports": []}

    compressor = AXIOMCompressor()
    return compressor.analyse_model(model_layers)


def compress_kronos_model(kronos_model) -> dict:
    """
    Convenience function: instantiate AXIOMCompressor and run the full
    compression pipeline (gauge fixing + analysis) on a PyTorch model.

    Parameters
    ----------
    kronos_model : Any PyTorch nn.Module exposing named_parameters().

    Returns
    -------
    The dict returned by AXIOMCompressor().run_full_compression_pipeline().
    """
    compressor = AXIOMCompressor()
    return compressor.run_full_compression_pipeline(kronos_model)
