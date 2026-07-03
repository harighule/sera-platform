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
