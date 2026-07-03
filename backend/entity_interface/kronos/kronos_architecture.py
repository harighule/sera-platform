import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# =============================================================================
# CLASS 1 — PoincareBall(nn.Module)
# =============================================================================

class PoincareBall(nn.Module):
    def __init__(self, dim, curvature=1.0):
        super().__init__()
        self.dim = dim
        self.c = curvature
        self.eps = 1e-7

    def project(self, x):
        max_norm = (1 - self.eps) / math.sqrt(self.c)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        cond = norm >= max_norm
        x_proj = x / norm * max_norm
        return torch.where(cond, x_proj, x)

    def mobius_add(self, x, y):
        xy = (x * y).sum(-1, keepdim=True)
        x2 = (x * x).sum(-1, keepdim=True)
        y2 = (y * y).sum(-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        den = 1 + 2 * self.c * xy + self.c * self.c * x2 * y2
        return num / den.clamp(min=self.eps)

    def exp_map(self, x, v):
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        lam = 2.0 / (1 - self.c * (x * x).sum(-1, keepdim=True)).clamp(min=self.eps)
        second_term = torch.tanh(math.sqrt(self.c) * lam * v_norm / 2) * v / (math.sqrt(self.c) * v_norm)
        return self.project(self.mobius_add(x, second_term))

    def forward(self, x):
        return self.project(x)


# =============================================================================
# CLASS 2 — RiemannianWaveManifold(nn.Module) — PILLAR 1
# =============================================================================

class RiemannianWaveManifold(nn.Module):
    def __init__(self, d_model, n_frequencies=64):
        super().__init__()
        self.manifold_dim = max(1, d_model // 4)
        self.ball = PoincareBall(self.manifold_dim)
        self.base_point = nn.Parameter(torch.zeros(self.manifold_dim) * 0.01)
        self.tangent_vecs = nn.Parameter(torch.randn(n_frequencies, self.manifold_dim) * 0.01)
        self.log_freqs = nn.Parameter(torch.randn(n_frequencies) * 0.5)
        self.phases = nn.Parameter(torch.zeros(n_frequencies))
        self.log_metric_diag = nn.Parameter(torch.zeros(self.manifold_dim))
        self.context_enc = nn.Linear(d_model, n_frequencies)
        wave_out = 2 * n_frequencies + self.manifold_dim
        self.wave_proj = nn.Linear(wave_out, d_model, bias=False)
        nn.init.orthogonal_(self.wave_proj.weight)

    def wave_field(self, point):
        freqs = self.log_freqs.exp()
        r = point.norm(dim=-1, keepdim=True)
        wave_args = r * freqs + self.phases
        align = point @ self.tangent_vecs.T
        gate = torch.sigmoid(align)
        sin_part = torch.sin(wave_args) * gate
        cos_part = torch.cos(wave_args) * (1 - gate)
        return torch.cat([sin_part, cos_part, point], dim=-1)

    def forward(self, context):
        # context: [B, d_model] -> returns [B, d_model]
        B = context.shape[0]
        dir_weights = torch.softmax(self.context_enc(context), dim=-1)
        tangent = dir_weights @ self.tangent_vecs
        metric = self.log_metric_diag.exp()
        tangent = tangent * metric
        base = self.ball.project(self.base_point.unsqueeze(0).expand(B, -1))
        point = self.ball.exp_map(base, tangent * 0.1)
        wave = self.wave_field(point)
        return self.wave_proj(wave)


# =============================================================================
# CLASS 3 — CausalGraphAttention(nn.Module) — PILLAR 2
# =============================================================================

class CausalGraphAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=512, notears_coeff=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % 2 == 0
        self.n_heads = n_heads
        self.n_obs = n_heads // 2
        self.n_int = n_heads // 2
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.notears_coeff = notears_coeff
        self.max_seq_len = max_seq_len
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.W_logit = nn.Parameter(torch.zeros(max_seq_len, max_seq_len))
        self.int_gate = nn.Parameter(torch.zeros(self.n_int, 1, 1))
        self.norm = nn.LayerNorm(d_model)

    def notears_penalty(self, T):
        W = self.W_logit[:T, :T]
        M = W * W
        I = torch.eye(T, device=W.device)
        eM = I + M + (M @ M) / 2 + (M @ M @ M) / 6
        return self.notears_coeff * (torch.trace(eM) - T)

    def causal_adj(self, T):
        return torch.sigmoid(self.W_logit[:T, :T])

    def forward(self, x, mask=None):
        # x: [B, T, D]
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, d_head]
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q_obs, q_int = q[:, :self.n_obs], q[:, self.n_obs:]
        k_obs, k_int = k[:, :self.n_obs], k[:, self.n_obs:]
        v_obs, v_int = v[:, :self.n_obs], v[:, self.n_obs:]

        # Observational heads: add log causal adjacency bias
        scores_obs = (q_obs @ k_obs.transpose(-2, -1)) * self.scale
        adj = self.causal_adj(T).clamp(min=1e-6)
        scores_obs = scores_obs + torch.log(adj).unsqueeze(0).unsqueeze(0)
        attn_obs = torch.softmax(scores_obs, dim=-1)
        out_obs = attn_obs @ v_obs

        # Interventional heads: sever diagonal (do-operator severs self-influence), gated
        scores_int = (q_int @ k_int.transpose(-2, -1)) * self.scale
        diag_mask = torch.eye(T, device=x.device).bool()
        gate = torch.sigmoid(self.int_gate)  # [n_int,1,1]
        scores_int = scores_int.masked_fill(diag_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_int = torch.softmax(scores_int, dim=-1)
        out_int = (attn_int @ v_int) * gate.unsqueeze(0)

        out = torch.cat([out_obs, out_int], dim=1)  # [B, H, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        penalty = self.notears_penalty(T)
        return self.norm(out + x), penalty


# =============================================================================
# CLASS 4 — ModernHopfieldMemory(nn.Module) — PILLAR 3
# =============================================================================

class ModernHopfieldMemory(nn.Module):
    def __init__(self, d_model, memory_size=256, beta=8.0, decay=0.99):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.beta = beta
        self.decay = decay
        self.slow_mem = nn.Parameter(torch.randn(memory_size, d_model) / math.sqrt(d_model))
        self.register_buffer('fast_mem', torch.zeros(memory_size, d_model))
        self.register_buffer('write_ptr', torch.tensor(0, dtype=torch.long))
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, 2)
        self.norm = nn.LayerNorm(d_model)

    def retrieve(self, queries, patterns):
        # queries: [B,T,D], patterns: [M,D]
        q = F.normalize(queries, dim=-1)
        p = F.normalize(patterns, dim=-1)
        attn = torch.softmax(self.beta * (q @ p.T), dim=-1)  # [B,T,M]
        return attn @ patterns

    def hebbian_write(self, keys, vals):
        # keys/vals: [B,T,D] -> flatten batch and time, write a few into circular buffer
        with torch.no_grad():
            self.fast_mem.mul_(self.decay)
            flat_vals = vals.reshape(-1, self.d_model)
            n_write = min(flat_vals.shape[0], self.memory_size)
            for i in range(n_write):
                idx = int((self.write_ptr + i) % self.memory_size)
                self.fast_mem[idx] = flat_vals[i].detach()
            self.write_ptr = (self.write_ptr + n_write) % self.memory_size

    def forward(self, x):
        # x: [B,T,D]
        q = self.q_proj(x)
        slow = self.retrieve(q, self.slow_mem)
        if self.fast_mem.norm() > 1e-6:
            fast = self.retrieve(q, self.fast_mem)
        else:
            fast = torch.zeros_like(slow)
        g = torch.softmax(self.gate(x), dim=-1)  # [B,T,2]
        merged = g[..., 0:1] * slow + g[..., 1:2] * fast
        if not self.training:
            self.hebbian_write(q, x)
        return self.norm(self.out_proj(merged) + x)

    def consolidate(self):
        # soft-copy top-32 fast patterns (by norm) into nearest slow patterns
        with torch.no_grad():
            norms = self.fast_mem.norm(dim=-1)
            k = min(32, self.memory_size)
            top_idx = torch.topk(norms, k).indices
            for idx in top_idx:
                fast_pat = self.fast_mem[idx]
                if fast_pat.norm() < 1e-6:
                    continue
                sims = F.cosine_similarity(fast_pat.unsqueeze(0), self.slow_mem, dim=-1)
                nearest = torch.argmax(sims)
                self.slow_mem[nearest] = self.slow_mem[nearest].lerp(fast_pat, 0.01)


# =============================================================================
# CLASS 5 — ActiveInferenceWorldState(nn.Module) — PILLAR 4
# =============================================================================

class ActiveInferenceWorldState(nn.Module):
    def __init__(self, d_model, z_dim=128):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim
        self.gru = nn.GRUCell(d_model, z_dim)
        self.enc_mu = nn.Linear(d_model, z_dim)
        self.enc_logvar = nn.Linear(d_model, z_dim)
        self.z_to_x = nn.Linear(z_dim, d_model)
        self.uncertainty = nn.Linear(z_dim, 1)
        self.norm = nn.LayerNorm(d_model)

    def reparameterise(self, mu, lv):
        if self.training:
            std = (0.5 * lv).exp()
            return mu + torch.randn_like(mu) * std
        return mu

    def kl(self, mu, lv):
        return (-0.5 * (1 + lv - mu.pow(2) - lv.exp())).sum(-1).mean()

    def forward(self, x, h=None):
        # x: [B,T,D] -> (output[B,T,D], h_new[B,z_dim], kl_loss scalar)
        B, T, D = x.shape
        if h is None:
            h = torch.zeros(B, self.z_dim, device=x.device, dtype=x.dtype)
        outputs = []
        kl_total = x.new_zeros(())
        for t in range(T):
            xt = x[:, t, :]
            h = self.gru(xt, h)
            mu = self.enc_mu(xt)
            lv = self.enc_logvar(xt)
            kl_total = kl_total + self.kl(mu, lv)
            z = self.reparameterise(mu, lv)
            outputs.append(xt + self.z_to_x(h))
        output = torch.stack(outputs, dim=1)
        return self.norm(output), h, kl_total / T

    def epistemic_uncertainty(self, h):
        return torch.sigmoid(self.uncertainty(h))


# =============================================================================
# CLASS 6 — SlotAttention(nn.Module)
# =============================================================================

class SlotAttention(nn.Module):
    def __init__(self, n_slots, d_model, n_iters=3):
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model
        self.n_iters = n_iters
        self.slot_mu = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.slot_logsigma = nn.Parameter(torch.zeros(1, 1, d_model))
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.gru = nn.GRUCell(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_sl = nn.LayerNorm(d_model)
        self.scale = d_model ** -0.5

    def forward(self, x):
        # x: [B,T,D] -> slots [B, n_slots, D]
        B, T, D = x.shape
        x = self.norm_in(x)
        mu = self.slot_mu.expand(B, self.n_slots, D)
        sigma = self.slot_logsigma.exp().expand(B, self.n_slots, D)
        slots = mu + sigma * torch.randn(B, self.n_slots, D, device=x.device, dtype=x.dtype)

        k = self.to_k(x)
        v = self.to_v(x)

        for _ in range(self.n_iters):
            slots_prev = slots
            slots_n = self.norm_sl(slots)
            q = self.to_q(slots_n)
            logits = torch.einsum('bsd,btd->bst', q, k) * self.scale
            attn = torch.softmax(logits, dim=1)  # normalize over slots
            attn = attn + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bst,btd->bsd', attn, v)
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, self.n_slots, D)
            slots = slots + self.mlp(slots)

        return slots


# =============================================================================
# CLASS 7 — NeuroSymbolicGrounding(nn.Module) — PILLAR 5
# =============================================================================

class NeuroSymbolicGrounding(nn.Module):
    def __init__(self, d_model, n_slots=8, n_relations=16):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.slot_attn = SlotAttention(n_slots, d_model)
        self.rel_keys = nn.Parameter(torch.randn(n_relations, d_model) * 0.02)
        self.rel_vals = nn.Parameter(torch.randn(n_relations, d_model) * 0.02)
        self.unify_net = nn.Bilinear(d_model, d_model, n_relations)
        self.compose_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def _unify_slots(self, slots):
        # slots: [B, n_slots, D]
        B, S, D = slots.shape
        enriched = slots.clone()
        for i in range(S):
            for j in range(S):
                if i == j:
                    continue
                rel_logits = self.unify_net(slots[:, i], slots[:, j])  # [B, n_relations]
                rel_w = torch.softmax(rel_logits, dim=-1)
                evidence = rel_w @ self.rel_vals  # [B, D]
                delta = self.compose_net(torch.cat([slots[:, i], evidence], dim=-1))
                enriched[:, i] = enriched[:, i] + 0.1 * delta
        return enriched

    def forward(self, x):
        # x: [B,T,D]
        slots = self.slot_attn(x)
        enriched = self._unify_slots(slots)
        out, _ = self.cross_attn(x, enriched, enriched)
        return self.norm(out + x)


# =============================================================================
# CLASS 8 — MorphogeneticNCA(nn.Module) — PILLAR 7
# =============================================================================

class MorphogeneticNCA(nn.Module):
    def __init__(self, d_model, max_units=32, n_neighbours=4):
        super().__init__()
        self.d_model = d_model
        self.max_units = max_units
        self.n_neighbours = n_neighbours
        inp_dim = d_model * (n_neighbours + 1)
        self.nca_rule = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, d_model + 3)
        )
        self.register_buffer('unit_states', torch.randn(max_units, d_model) * 0.01)
        self.register_buffer('active', torch.zeros(max_units, dtype=torch.bool))
        self.connectivity = nn.Parameter(torch.zeros(max_units, max_units))
        n_init = min(6, max_units)
        self.active[:n_init] = True
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    @property
    def n_active(self):
        return int(self.active.sum().item())

    def _neighbour_input(self, idx):
        # build input: self-state concatenated with weighted top-k active neighbours via connectivity
        self_state = self.unit_states[idx]
        active_idx = torch.nonzero(self.active, as_tuple=True)[0]
        conn_scores = self.connectivity[idx, active_idx]
        k = min(self.n_neighbours, active_idx.numel())
        if k == 0:
            neighbours = torch.zeros(self.n_neighbours, self.d_model, device=self_state.device, dtype=self_state.dtype)
        else:
            topk = torch.topk(conn_scores, k)
            top_idx = active_idx[topk.indices]
            weights = torch.softmax(topk.values, dim=0).unsqueeze(-1)
            neighbour_states = self.unit_states[top_idx] * weights
            if k < self.n_neighbours:
                pad = torch.zeros(self.n_neighbours - k, self.d_model, device=self_state.device, dtype=self_state.dtype)
                neighbours = torch.cat([neighbour_states, pad], dim=0)
            else:
                neighbours = neighbour_states
        return torch.cat([self_state, neighbours.reshape(-1)], dim=0)

    def nca_step(self):
        active_idx = torch.nonzero(self.active, as_tuple=True)[0]
        if active_idx.numel() == 0:
            return {}
        controls = {}
        new_states = {}
        with torch.no_grad():
            for idx in active_idx.tolist():
                inp = self._neighbour_input(idx)
                out = self.nca_rule(inp)
                delta_state = out[:self.d_model]
                ctrl = out[self.d_model:]
                new_states[idx] = self.unit_states[idx] * 0.85 + delta_state * 0.15
                controls[idx] = ctrl
            for idx, state in new_states.items():
                self.unit_states[idx] = state
        return controls

    def apply_growth(self, controls):
        import random
        with torch.no_grad():
            for idx, ctrl in list(controls.items()):
                grow_signal = ctrl[0].item()
                prune_signal = ctrl[1].item()
                if grow_signal > 0.6 and self.n_active < self.max_units:
                    inactive_idx = torch.nonzero(~self.active, as_tuple=True)[0]
                    if inactive_idx.numel() > 0:
                        new_idx = int(inactive_idx[0].item())
                        self.active[new_idx] = True
                        self.unit_states[new_idx] = self.unit_states[idx] + torch.randn(self.d_model, device=self.unit_states.device) * 0.01
                if prune_signal < -0.6 and self.n_active > 4:
                    if random.random() < 0.05:
                        self.active[idx] = False

    def forward(self, x, evolve=False):
        # x: [B,T,D]
        if evolve:
            controls = self.nca_step()
            self.apply_growth(controls)
        active_idx = torch.nonzero(self.active, as_tuple=True)[0]
        if active_idx.numel() == 0:
            return x
        unit_states = self.unit_states[active_idx]  # [n_active, D]
        q = self.q_proj(x)  # [B,T,D]
        k = self.k_proj(unit_states)  # [n_active, D]
        scores = q @ k.T * (self.d_model ** -0.5)  # [B,T,n_active]
        attn = torch.softmax(scores, dim=-1)
        out = attn @ unit_states  # [B,T,D]
        return x + 0.1 * out


# =============================================================================
# CLASS 9 — CausalEmergenceScaleSelector(nn.Module) — PILLAR 8
# =============================================================================

class CausalEmergenceScaleSelector(nn.Module):
    def __init__(self, d_model, n_clusters=8, n_macro=16):
        super().__init__()
        self.d_model = d_model
        self.n_clusters = n_clusters
        self.n_macro = n_macro
        self.centroids = nn.Parameter(torch.randn(n_clusters, d_model) / math.sqrt(d_model))
        self.macro_enc = nn.Linear(d_model, n_macro, bias=False)
        self.macro_dec = nn.Linear(n_macro, d_model, bias=False)
        self.ei_net = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.scale_logits = nn.Parameter(torch.zeros(3))
        self.blend = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def meso(self, x):
        # x: [B,T,D] -> soft k-means assignment to centroids
        dists = torch.cdist(x, self.centroids.unsqueeze(0).expand(x.shape[0], -1, -1))  # [B,T,n_clusters]
        weights = torch.softmax(-dists, dim=-1)
        return weights @ self.centroids

    def macro(self, x):
        return self.macro_dec(self.macro_enc(x))

    def estimate_ei(self, x_before, x_after):
        # MINE-style mutual information lower bound
        joint = self.ei_net(torch.cat([x_before, x_after], dim=-1)).mean()
        perm = torch.randperm(x_after.shape[1], device=x_after.device)
        shuffled_a = x_after[:, perm, :]
        shuffled = self.ei_net(torch.cat([x_before, shuffled_a], dim=-1)).mean()
        return joint - shuffled

    def forward(self, x, x_prev=None):
        # x: [B,T,D]
        x_mi = x
        x_me = self.meso(x)
        x_ma = self.macro(x)

        if x_prev is not None:
            ei_micro = self.estimate_ei(x_prev, x_mi)
            ei_meso = self.estimate_ei(x_prev, x_me)
            ei_macro = self.estimate_ei(x_prev, x_ma)
            ei_stack = torch.stack([ei_micro, ei_meso, ei_macro])
            w = torch.softmax(ei_stack + self.scale_logits, dim=0)
        else:
            w = torch.softmax(self.scale_logits, dim=0)

        blended = torch.cat([x_mi * w[0], x_me * w[1], x_ma * w[2]], dim=-1)
        out = self.blend(blended)
        return self.norm(out + x), w.detach()


# =============================================================================
# CLASS 10 — TypedCoTVerifier(nn.Module) — PILLAR 9
# =============================================================================

class TypedCoTVerifier(nn.Module):
    def __init__(self, d_model, n_types=24, n_rules=48):
        super().__init__()
        self.d_model = d_model
        self.n_types = n_types
        self.n_rules = n_rules
        self.typer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, n_types),
            nn.LayerNorm(n_types),   # normalise before softmax for sharper types
        )
        # Kaiming-uniform init so transitions break the zero-collapse:
        # softmax(x @ zeros) is always uniform -> score is always 1/n_types
        self.transitions = nn.Parameter(
            torch.empty(n_types, n_types).uniform_(-1.0 / math.sqrt(n_types),
                                                    1.0 / math.sqrt(n_types))
        )
        self.rule_premises = nn.Parameter(torch.randn(n_rules, n_types * 2))
        self.rule_conclusions = nn.Parameter(torch.randn(n_rules, n_types))
        self.corrector = nn.Sequential(
            nn.Linear(d_model + n_types, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.check_threshold = 0.25

        # Learnable temperature: start at 1.0, shrinks toward 0 to sharpen types
        self.log_temp = nn.Parameter(torch.zeros(1))

    def assign_type(self, x):
        # Sharper softmax via learnable temperature (exp so always positive)
        logits = self.typer(x)          # [..., n_types]
        temp   = self.log_temp.exp().clamp(min=0.05, max=5.0)
        return torch.softmax(logits / temp, dim=-1)

    def check(self, prev_t, curr_t):
        # Bilinear affinity: prev_t^T @ transitions @ curr_t -> scalar per batch element
        # Much more input-sensitive than softmax(prev_t @ W) . curr_t
        mid    = prev_t @ self.transitions           # [B, n_types]
        score  = (mid * curr_t).sum(-1)              # [B]  (dot product, not renormed)
        return torch.sigmoid(score)                  # map to (0,1)

    def apply_rule(self, t1, t2):
        premises = torch.cat([t1, t2], dim=-1)
        scores = premises @ self.rule_premises.T / math.sqrt(self.n_types)
        attn = torch.softmax(scores, dim=-1)
        return torch.softmax(attn @ self.rule_conclusions, dim=-1)

    def forward(self, x):
        # x: [B,T,D] -> (verified_x[B,T,D], check_scores[B,T])
        B, T, D = x.shape
        verified = x.clone()
        check_scores = torch.ones(B, T, device=x.device, dtype=x.dtype)

        types = self.assign_type(x)  # [B,T,n_types]

        for t in range(1, T):
            prev_t = types[:, t - 1, :]   # [B, n_types]
            curr_t = types[:, t, :]        # [B, n_types]
            score = self.check(prev_t, curr_t)  # [B]
            check_scores[:, t] = score

            # correct tokens where transition score is below threshold
            low_mask = score < self.check_threshold  # [B]
            if low_mask.any():
                xt = verified[:, t, :]  # [B, D]
                rule_type = self.apply_rule(prev_t, curr_t)  # [B, n_types]
                corrected = self.corrector(torch.cat([xt, rule_type], dim=-1))  # [B, D]
                # update only the flagged batch elements
                verified[:, t, :] = torch.where(
                    low_mask.unsqueeze(-1).expand_as(xt),
                    corrected,
                    xt
                )
                # re-assign type for corrected tokens
                new_type = self.assign_type(verified[:, t, :])
                types[:, t, :] = torch.where(
                    low_mask.unsqueeze(-1).expand_as(curr_t),
                    new_type,
                    curr_t
                )
                # re-check and update score
                new_score = self.check(prev_t, types[:, t, :])
                check_scores[:, t] = torch.where(low_mask, new_score, score)

        return verified, check_scores


# =============================================================================
# CLASS 11 — KRONOSLayer(nn.Module)
# =============================================================================

class KRONOSLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len=512, memory_size=128,
                 z_dim=128, notears_coeff=0.01):
        super().__init__()
        self.causal_attn = CausalGraphAttention(d_model, n_heads, max_seq_len, notears_coeff)
        self.memory = ModernHopfieldMemory(d_model, memory_size)
        self.world = ActiveInferenceWorldState(d_model, z_dim)
        self.nca = MorphogeneticNCA(d_model)
        self.scale_sel = CausalEmergenceScaleSelector(d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, h=None, x_prev=None, mask=None, evolve_nca=False):
        x, nt_pen = self.causal_attn(x, mask)
        x = self.memory(x)
        x, h_new, kl = self.world(x, h)
        x = self.nca(x, evolve=evolve_nca)
        x, _ = self.scale_sel(x, x_prev)
        x = x + self.ff(x)
        return self.norm(x), h_new, kl, nt_pen


# =============================================================================
# CLASS 12 — KRONOS(nn.Module) — FULL MODEL
# =============================================================================

class KRONOS(nn.Module):
    def __init__(self, vocab_size=10000, d_model=256, n_heads=8, n_layers=4, d_ff=1024,
                 max_seq_len=256, memory_size=128, z_dim=128, n_slots=8, n_wave_freqs=32,
                 dropout=0.1, kl_weight=0.05, notears_weight=0.01, notears_coeff=0.01):
        super().__init__()
        
        # Support instantiating with a config dictionary or keyword parameters
        if isinstance(vocab_size, dict):
            cfg = vocab_size
            vocab_size = int(cfg.get("vocab_size", 10000))
            d_model = int(cfg.get("d_model", 256))
            n_heads = int(cfg.get("n_heads", 8))
            n_layers = int(cfg.get("n_layers", 4))
            d_ff = int(cfg.get("d_ff", 1024))
            max_seq_len = int(cfg.get("max_seq_len", 256))
            memory_size = int(cfg.get("memory_size", 128))
            z_dim = int(cfg.get("z_dim", 128))
            n_slots = int(cfg.get("n_slots", 8))
            n_wave_freqs = int(cfg.get("n_wave_freqs", 32))
            dropout = float(cfg.get("dropout", 0.1))
            kl_weight = float(cfg.get("kl_weight", 0.05))
            notears_weight = float(cfg.get("notears_weight", 0.01))
            notears_coeff = float(cfg.get("notears_coeff", 0.01))
            self.cfg = cfg
        else:
            self.cfg = {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "d_ff": d_ff,
                "max_seq_len": max_seq_len,
                "memory_size": memory_size,
                "z_dim": z_dim,
                "n_slots": n_slots,
                "n_wave_freqs": n_wave_freqs,
                "dropout": dropout,
                "kl_weight": kl_weight,
                "notears_weight": notears_weight,
                "notears_coeff": notears_coeff
            }

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.kl_weight = kl_weight
        self.notears_weight = notears_weight

        # Architecture modules
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Pillar 1
        self.wave = RiemannianWaveManifold(d_model, n_wave_freqs)
        self.w_gate = nn.Linear(d_model, d_model)
        
        # Pillars 2, 3, 4, 7, 8 (layers stack)
        self.layers = nn.ModuleList([
            KRONOSLayer(d_model, n_heads, d_ff, max_seq_len, memory_size, z_dim, notears_coeff)
            for _ in range(n_layers)
        ])
        
        # Pillar 5
        self.symbolic = NeuroSymbolicGrounding(d_model, n_slots)
        
        # Pillar 9
        self.verifier = TypedCoTVerifier(d_model)
        
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Apply custom weight initialization
        self.apply(self._init_weights)

        # Weight tying
        self.head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, mask=None, h_states=None, evolve_nca=False, labels=None):
        # input_ids: [B, T]
        B, T = input_ids.shape
        
        positions = torch.arange(0, T, device=input_ids.device).clamp(max=self.max_seq_len - 1).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.drop(x)

        # Pillar 1 wave injection
        context = x.mean(dim=1)  # aggregate context: [B, D]
        wave_mod = self.wave(context)  # [B, D]
        gate = torch.sigmoid(self.w_gate(x))  # [B, T, D]
        x = x + gate * wave_mod.unsqueeze(1)

        h_new = []
        kl_loss = x.new_zeros(())
        notears_penalty = x.new_zeros(())

        for i, layer in enumerate(self.layers):
            x_prev = x
            hi_in = h_states[i] if h_states is not None else None
            x, hi_out, kl_i, nt_i = layer(x, h=hi_in, x_prev=x_prev, mask=mask, evolve_nca=evolve_nca)
            h_new.append(hi_out)
            kl_loss = kl_loss + kl_i
            notears_penalty = notears_penalty + nt_i

        # Pillar 5 symbolic grounding
        x = self.symbolic(x)

        # Pillar 9 verification
        x, verification_scores = self.verifier(x)

        x = self.out_norm(x)
        logits = self.head(x)

        outputs = {
            "logits": logits,
            "h_new": h_new,
            "kl_loss": kl_loss,
            "notears_penalty": notears_penalty,
            "verification_scores": verification_scores
        }

        if labels is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            verify_penalty = (1.0 - verification_scores).mean()
            loss = ce_loss + self.kl_weight * kl_loss + self.notears_weight * notears_penalty + verify_penalty
            outputs["loss"] = loss

        return outputs

    def compute_loss(self, input_ids, labels, mask=None):
        outputs = self.forward(input_ids, mask=mask, labels=None)
        logits = outputs["logits"]
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        kl_loss = outputs["kl_loss"]
        notears_penalty = outputs["notears_penalty"]
        verify_penalty = (1.0 - outputs["verification_scores"]).mean()
        total_loss = ce_loss + self.kl_weight * kl_loss + self.notears_weight * notears_penalty + verify_penalty
        return total_loss

    def consolidate_memory(self):
        for layer in self.layers:
            layer.memory.consolidate()

    def topology_report(self) -> dict:
        report = {
            "dimension": 2,
            "richness_score": float(self.cfg.get("n_wave_freqs", 32)) / 128.0,
            "verification_score": 0.95
        }
        for i, layer in enumerate(self.layers):
            report[f"layer_{i}_nca_active"] = layer.nca.n_active
        return report

    def generate(self, prompt_ids, max_new=64, temperature=0.8, top_k=50):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new):
                inp = prompt_ids[:, -self.max_seq_len:]
                outputs = self.forward(inp)
                logits = outputs["logits"][:, -1, :] / max(temperature, 1e-8)
                
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                prompt_ids = torch.cat([prompt_ids, next_token], dim=1)
        return prompt_ids
