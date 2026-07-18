import torch
import torch.nn as nn
import math
from typing import Tuple

class CIFNWeightField(nn.Module):
    """
    Continuous Interference Field Network weight generator.
    Generates weights on-the-fly from a continuous wave interference lattice.
    
    Formula:
      W[i, j] = sum_k a_k * sin(omega_out_k * x_i + theta_out_k) * sin(omega_in_k * y_j + theta_in_k)
    """
    def __init__(self, out_features: int, in_features: int, basis_count: int = 512):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.basis_count = basis_count
        
        # Corrected amplitude initialization to stabilize conditioning across all seeds
        a_init = torch.empty(basis_count)
        nn.init.normal_(a_init, mean=0.0, std=2.0 / basis_count)
        self.a = nn.Parameter(a_init)
        
        # Deterministic wave frequencies (omega) covering 1.0 to 10.0 pi
        # Alternating signs are used to cover both directions of projection
        freqs = torch.linspace(1.0, 10.0, basis_count)
        signs = torch.ones(basis_count)
        signs[1::2] = -1.0
        freqs = freqs * signs
        
        # Fixed wave frequencies and phase offsets (non-trainable)
        # This makes the optimisation space with respect to amplitude 'a' convex,
        # preventing seed-dependent local minima trapping.
        self.omega_out = nn.Parameter(freqs * math.pi, requires_grad=False)
        self.omega_in = nn.Parameter(freqs * math.pi, requires_grad=False)
        
        # Deterministic phase offsets (theta) evenly spaced in [0, 2*pi]
        phases = torch.linspace(0.0, 2 * math.pi, basis_count)
        self.theta_out = nn.Parameter(phases, requires_grad=False)
        self.theta_in = nn.Parameter(phases, requires_grad=False)

    def forward(self) -> torch.Tensor:
        device = self.a.device
        
        # Discretize space into normalized grid intervals [0, 1]
        x = torch.linspace(0.0, 1.0, self.out_features, device=device)
        y = torch.linspace(0.0, 1.0, self.in_features, device=device)
        
        # Compute sinusoids across spatial locations
        # Shape: (basis_count, out_features)
        field_out = torch.sin(self.omega_out.unsqueeze(1) * x.unsqueeze(0) + self.theta_out.unsqueeze(1))
        # Shape: (basis_count, in_features)
        field_in = torch.sin(self.omega_in.unsqueeze(1) * y.unsqueeze(0) + self.theta_in.unsqueeze(1))
        
        # Combine wave projections via Einstein summation:
        # W[i, j] = sum_k a_k * field_out[k, i] * field_in[k, j]
        # Shape: (out_features, in_features)
        W = torch.einsum('k,ki,kj->ij', self.a, field_out, field_in)
        return W

    # ------------------------------------------------------------------
    # Addressable (random-access) weight evaluation — the "virtual parameter"
    # capability. Any single weight W[i,j] is COMPUTABLE on demand from the
    # compact wave basis without materialising the full matrix.
    # ------------------------------------------------------------------
    def weight_at(self, i: int, j: int) -> torch.Tensor:
        """
        Evaluate a single weight W[i, j] on demand (O(basis_count), not O(N)).

        Demonstrates that the field is *addressable*: the weight grid is a
        continuous function sampled at (x_i, y_j), so an arbitrarily large grid
        can be addressed from the same compact basis. This is the honest basis
        of the "virtual parameter" count — the parameters are GENERATED, not
        stored, trading storage for a small per-access compute cost.
        """
        device = self.a.device
        x_i = torch.tensor(i / max(self.out_features - 1, 1), device=device)
        y_j = torch.tensor(j / max(self.in_features - 1, 1), device=device)
        fo  = torch.sin(self.omega_out * x_i + self.theta_out)   # (basis,)
        fi  = torch.sin(self.omega_in  * y_j + self.theta_in)    # (basis,)
        return (self.a * fo * fi).sum()

    def parameter_accounting(self, virtual_out: int = None, virtual_in: int = None) -> dict:
        """
        Honest storage-vs-virtual accounting for the wave field.

        * real_trainable_parameters: the actually-stored, learned degrees of
          freedom (the basis amplitudes `a`; ω/θ are fixed, non-trainable).
        * addressable_parameters: the size of the weight grid this field can
          GENERATE on demand at the given (virtual_out × virtual_in) resolution.

        Reporting these separately keeps the "1 quadrillion parameter" framing
        honest: a compact basis (~KB) can *address* a 10^15-entry weight field,
        but those entries are generated from the basis (a storage/compute
        tradeoff) — they are NOT 10^15 independent stored weights.
        """
        vout = virtual_out if virtual_out is not None else self.out_features
        vin  = virtual_in  if virtual_in  is not None else self.in_features
        real_trainable = int(self.a.numel())          # only amplitudes are trainable
        return {
            "real_trainable_parameters": real_trainable,
            "basis_count": int(self.basis_count),
            "addressable_parameters": int(vout) * int(vin),
            "addressable_grid": [int(vout), int(vin)],
            "storage_bytes_basis": real_trainable * 4,
            "note": ("Weights are GENERATED from the wave basis on demand (addressable / "
                     "random-access), not stored. 'Virtual' = generated grid size, a "
                     "storage-for-compute tradeoff — not independent stored parameters."),
        }

class CIFNLinear(nn.Module):
    """
    Linear projection layer whose weights are generated on-the-fly
    via a Continuous Interference Field Network (CIFN).
    """
    def __init__(self, in_features: int, out_features: int, basis_count: int = 512):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_field = CIFNWeightField(out_features, in_features, basis_count)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamically evaluate the weight tensor (computable, not stored)
        W = self.weight_field()
        # Perform standard linear projection
        return torch.matmul(x, W.t()) + self.bias

def run_cifn_optimization_step() -> Tuple[float, float, float]:
    """
    Executes a complete forward and backward optimization step.
    Verifies that gradients flow losslessly back to basis wave parameters.
    """
    # Create CIFN linear projection (10 inputs -> 5 outputs)
    layer = CIFNLinear(in_features=10, out_features=5, basis_count=128)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    
    # Create mock training input and targets
    x = torch.randn(2, 10)
    target = torch.randn(2, 5)
    
    # Measure baseline loss
    out_before = layer(x)
    loss_before = torch.mean((out_before - target) ** 2).item()
    
    # Forward Pass
    out = layer(x)
    loss = torch.mean((out - target) ** 2)
    
    # Backward Pass (calculate gradients)
    optimizer.zero_grad()
    loss.backward()
    
    # Measure gradient norm of the basis coefficients (amplitudes)
    grad_norm = torch.norm(layer.weight_field.a.grad).item()
    
    # Optimizer step
    optimizer.step()
    
    # Measure loss after updates
    out_after = layer(x)
    loss_after = torch.mean((out_after - target) ** 2).item()
    
    return loss_before, loss_after, grad_norm


class CIFNFieldOmega(nn.Module):
    """
    CIFN-Ω — the extended Continuous Interference Field Network.

    W[i,j] = Σ_k a_k(W) · Π_{d=1..D} sin(ω_{k,d}·coord_d(i,j) + θ_{k,d})

    Extends the base CIFNWeightField with:
      • MULTI-DIMENSIONAL interference lattice (product of D sinusoids per basis,
        not a separable 1-D field), giving far richer weight structure;
      • RECURSIVE / self-consistent coefficients  a_dyn = a + a_net(a)  so the
        field partially generates itself (Path C of the spec);
      • optionally TRAINABLE frequencies/phases (ω, θ).

    The weight matrix is generated on-the-fly (never stored); backprop flows to
    the compact basis parameters.
    """
    def __init__(self, out_features: int, in_features: int, basis_count: int = 128,
                 D: int = 4, recursive: bool = True, trainable_freq: bool = True):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.basis_count = basis_count
        self.D = D
        self.recursive = recursive

        # Init: ω ~ N(0, 1/D), θ ~ U(0, 2π), a ~ N(0, 1/K)  (spec's stable scheme)
        self.omega = nn.Parameter(torch.randn(basis_count, D) / math.sqrt(D),
                                  requires_grad=trainable_freq)
        self.theta = nn.Parameter(torch.rand(basis_count, D) * 2 * math.pi,
                                  requires_grad=trainable_freq)
        self.a = nn.Parameter(torch.randn(basis_count) / basis_count)
        if recursive:
            self.a_net = nn.Sequential(
                nn.Linear(basis_count, basis_count), nn.Tanh(),
                nn.Linear(basis_count, basis_count),
            )

    def _coords(self, device) -> torch.Tensor:
        """(out, in, D) coordinate tensor for each weight position."""
        i = torch.linspace(0.0, 1.0, self.out_features, device=device)
        j = torch.linspace(0.0, 1.0, self.in_features, device=device)
        gi, gj = torch.meshgrid(i, j, indexing="ij")           # (out,in)
        feats = [gi, gj, torch.sin(3.0 * gi), torch.cos(3.0 * gj)]
        coords = torch.stack(feats[: self.D], dim=-1)          # (out,in,D)
        if coords.shape[-1] < self.D:                          # pad if D>4
            pad = self.D - coords.shape[-1]
            coords = torch.cat([coords, gi.unsqueeze(-1).repeat(1, 1, pad)], dim=-1)
        return coords

    def forward(self) -> torch.Tensor:
        device = self.a.device
        coords = self._coords(device)                          # (out,in,D)
        # phase[i,j,k,d] = ω[k,d]·coord[i,j,d] + θ[k,d]
        phase = torch.einsum("ijd,kd->ijkd", coords, self.omega) + self.theta
        phi = torch.sin(phase).prod(dim=-1)                    # (out,in,basis) multi-D lattice
        a_dyn = self.a + self.a_net(self.a) if self.recursive else self.a
        W = torch.einsum("ijk,k->ij", phi, a_dyn)              # (out,in)
        return W


def verify_cifn_omega() -> dict:
    """CIFN-Ω is trainable, multi-D, recursive, and generates weights on demand."""
    torch.manual_seed(0)
    field = CIFNFieldOmega(out_features=8, in_features=6, basis_count=64, D=4, recursive=True)
    x = torch.randn(4, 6)
    target = torch.randn(4, 8)
    opt = torch.optim.Adam(field.parameters(), lr=0.02)
    l0 = float(((x @ field().T - target) ** 2).mean())
    for _ in range(80):
        opt.zero_grad(); (((x @ field().T - target) ** 2).mean()).backward(); opt.step()
    l1 = float(((x @ field().T - target) ** 2).mean())
    real_params = sum(p.numel() for p in field.parameters())
    return {"multi_dim_D": field.D, "recursive_coefficients": field.recursive,
            "trainable_frequencies": field.omega.requires_grad,
            "generated_weight_shape": list(field().shape),
            "real_parameters": real_params,
            "loss_before": round(l0, 5), "loss_after": round(l1, 5),
            "trains": l1 < l0}
