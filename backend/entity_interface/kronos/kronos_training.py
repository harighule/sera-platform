import copy
import math
import time
from typing import Dict, Iterable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from entity_interface.kronos.kronos_architecture import KRONOS, PoincareBall

class RiemannianAdagrad(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, curvature=1.0, eps=1e-10, weight_decay=0.0):
        defaults = dict(lr=lr, curvature=curvature, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.ball = PoincareBall(dim=1, curvature=curvature)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            c = group['curvature']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if weight_decay != 0:
                    g = g + weight_decay * p

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['sum'] = torch.zeros_like(p)

                state['step'] += 1

                if getattr(p, 'manifold', None) == 'poincare':
                    x2 = torch.sum(p**2)
                    lam = 2.0 / (1.0 - c * x2).clamp(min=1e-8)
                    g_r = g / (lam**2 + 1e-8)
                    state['sum'].add_(g_r ** 2)
                    v = -lr * g_r / (torch.sqrt(state['sum']) + eps)
                    p.copy_(self.ball.exp_map(p, v))
                    p.copy_(self.ball.project(p))
                else:
                    state['sum'].add_(g ** 2)
                    p.addcdiv_(g, torch.sqrt(state['sum']) + eps, value=-lr)

        return loss

class KRONOSTrainer:
    def __init__(
        self,
        model,
        lr=3e-4,
        riemannian_lr=1e-2,
        weight_decay=0.01,
        max_grad_norm=1.0,
        consolidate_every=1000,
        nca_evolve_every=500,
        use_amp=False,
        device='cpu'
    ):
        self.model = model
        self.base_lr = lr
        self.riemannian_lr = riemannian_lr
        self.max_grad_norm = max_grad_norm
        self.consolidate_every = consolidate_every
        self.nca_evolve_every = nca_evolve_every
        self.use_amp = use_amp
        self.device = torch.device(device)

        self.model.to(self.device)

        self.riem_params = [
            p for p in model.parameters()
            if getattr(p, 'manifold', None) == 'poincare'
        ]
        self.eucl_params = [
            p for p in model.parameters()
            if getattr(p, 'manifold', None) != 'poincare'
        ]

        self.opt_eucl = torch.optim.AdamW(
            self.eucl_params,
            lr=lr,
            weight_decay=weight_decay
        )
        self.opt_riem = RiemannianAdagrad(
            self.riem_params,
            lr=riemannian_lr,
            curvature=1.0
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_eucl,
            T_max=10000,
            eta_min=1e-6
        )

        self.step_count = 0
        self.history = []

    def warmup_lr(self, warmup_steps=2000) -> float:
        if self.step_count < warmup_steps:
            warmup_scale = float(self.step_count) / float(warmup_steps)
            curr_lr = self.base_lr * warmup_scale
            for pg in self.opt_eucl.param_groups:
                pg['lr'] = curr_lr
        else:
            curr_lr = self.opt_eucl.param_groups[0]['lr']
        return curr_lr

    def train_step(self, input_ids, labels, mask=None) -> Dict[str, float]:
        self.model.train()
        self.opt_eucl.zero_grad()
        self.opt_riem.zero_grad()

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        self.step_count += 1
        self.warmup_lr()

        outputs = self.model(input_ids, labels=labels, mask=mask)
        loss = outputs["loss"]
        loss.backward()

        if self.eucl_params:
            nn.utils.clip_grad_norm_(self.eucl_params, self.max_grad_norm)
        if self.riem_params:
            nn.utils.clip_grad_norm_(self.riem_params, self.max_grad_norm)

        self.opt_eucl.step()
        self.opt_riem.step()

        if self.step_count >= 2000:
            self.scheduler.step()

        for p in self.riem_params:
            p.copy_(self.opt_riem.ball.project(p))

        if self.step_count % self.consolidate_every == 0:
            self.model.consolidate_memory()

        metrics = {
            "loss": loss.item(),
            "step": self.step_count,
            "lr": self.opt_eucl.param_groups[0]['lr'],
        }
        self.history.append(metrics)
        return metrics

    def train_epoch(self, dataloader, log_every=50) -> Dict[str, float]:
        total_loss = 0.0
        count = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids, labels = batch
            metrics = self.train_step(input_ids, labels)
            total_loss += metrics["loss"]
            count += 1

            if batch_idx % log_every == 0:
                report = self.model.topology_report()
                print(
                    f"[TRAIN] Step {self.step_count} | Batch {batch_idx} | "
                    f"Loss: {metrics['loss']:.4f} | Topology Richness: {report.get('richness_score', 0.0):.4f}"
                )
        return {"mean_loss": total_loss / max(count, 1)}

    def save(self, path):
        checkpoint = {
            'step_count': self.step_count,
            'model_state': self.model.state_dict(),
            'opt_eucl_state': self.opt_eucl.state_dict(),
            'opt_riem_state': self.opt_riem.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.step_count = checkpoint['step_count']
        self.model.load_state_dict(checkpoint['model_state'])
        self.opt_eucl.load_state_dict(checkpoint['opt_eucl_state'])
        self.opt_riem.load_state_dict(checkpoint['opt_riem_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.history = checkpoint['history']

class GodelLoop:
    _MUTABLE = {
        "d_ff":            (256, 4096),
        "memory_size":     (64, 512),
        "z_dim":           (64, 512),
        "n_wave_freqs":    (16, 128),
        "kl_weight":       (0.01, 0.5),
        "notears_weight":  (0.001, 0.1),
        "notears_coeff":   (0.001, 0.1)
    }

    def __init__(self, base_config, vocab_size=10000, population_size=4, n_generations=8, device='cpu'):
        self.vocab_size = vocab_size
        self.population_size = population_size
        self.n_generations = n_generations
        self.device = device
        
        self.population = [self._perturb(base_config, 0.15) for _ in range(population_size-1)]
        self.population.insert(0, copy.deepcopy(base_config))

        self.meta = nn.Sequential(
            nn.Linear(len(self._MUTABLE), 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        self.meta_opt = torch.optim.Adam(self.meta.parameters(), lr=1e-3)

        self.generation = 0
        self.fitness_history = []
        self.best_config = None
        self.archive = []

    @staticmethod
    def _clip(val, lo, hi):
        if isinstance(lo, int) and isinstance(hi, int):
            return int(max(lo, min(hi, round(val))))
        return float(max(lo, min(hi, val)))

    def _perturb(self, cfg, scale=0.15) -> Dict:
        new_cfg = copy.deepcopy(cfg)
        for k, (lo, hi) in self._MUTABLE.items():
            current = float(new_cfg.get(k, (lo + hi) / 2.0))
            perturbation = scale * float(torch.randn(1).item())
            if isinstance(lo, int) and isinstance(hi, int):
                span = hi - lo
                val = current + perturbation * span
            else:
                val = current + perturbation
            new_cfg[k] = self._clip(val, lo, hi)
        return new_cfg

    def _to_feat(self, cfg) -> torch.Tensor:
        feats = []
        for k in self._MUTABLE.keys():
            feats.append(float(cfg.get(k, 0.0)))
        return torch.tensor(feats, dtype=torch.float32)

    def _build_model(self, cfg) -> Optional[KRONOS]:
        try:
            return KRONOS(cfg)
        except Exception:
            return None

    def _fitness(self, model, eval_batch=None) -> float:
        if model is None:
            return 0.0
        try:
            report = model.topology_report()
            richness = float(report.get("richness_score", 0.0))
            verification = float(report.get("verification_score", 0.0))

            num_params = sum(p.numel() for p in model.parameters())
            param_efficiency = 1.0 / (1.0 + num_params / 1e6)

            return (richness + verification + param_efficiency) / 3.0
        except Exception:
            return 0.0

    def _update_meta(self, parent, child, delta):
        self.meta.train()
        self.meta_opt.zero_grad()
        feat = self._to_feat(child) - self._to_feat(parent)
        pred = self.meta(feat)
        loss = F.mse_loss(pred, torch.tensor([delta], dtype=torch.float32))
        loss.backward()
        self.meta_opt.step()

    def _propose(self, parent) -> Dict:
        candidates = [self._perturb(parent, 0.1) for _ in range(8)]
        self.meta.eval()
        best_candidate = None
        best_pred_delta = -float('inf')
        with torch.no_grad():
            for cand in candidates:
                feat = self._to_feat(cand) - self._to_feat(parent)
                pred = self.meta(feat).item()
                if pred > best_pred_delta:
                    best_pred_delta = pred
                    best_candidate = cand
        return best_candidate

    def step_generation(self, eval_batch=None) -> Dict:
        self.generation += 1
        evaluated = []
        for cfg in self.population:
            model = self._build_model(cfg)
            fit = self._fitness(model)
            evaluated.append((fit, cfg))

        evaluated.sort(key=lambda t: t[0], reverse=True)
        best_fit, best_cfg = evaluated[0]
        self.fitness_history.append(best_fit)

        if self.best_config is None or best_fit > self._fitness(self._build_model(self.best_config)):
            self.best_config = copy.deepcopy(best_cfg)

        n_survivors = max(1, self.population_size // 2)
        survivors = evaluated[:n_survivors]

        next_population = [copy.deepcopy(cfg) for _, cfg in survivors]

        while len(next_population) < self.population_size:
            parent_idx = int(torch.randint(0, len(survivors), (1,)).item())
            parent_fit, parent_cfg = survivors[parent_idx]
            child_cfg = self._propose(parent_cfg)
            child_model = self._build_model(child_cfg)
            child_fit = self._fitness(child_model)
            
            delta = child_fit - parent_fit
            self._update_meta(parent_cfg, child_cfg, delta)

            next_population.append(child_cfg)

        self.population = next_population

        return {
            "generation": self.generation,
            "best_fitness": best_fit,
            "best_config": self.best_config,
            "fitness_history": list(self.fitness_history)
        }

    def run(self, eval_batch=None) -> Tuple[Dict, KRONOS]:
        best_res = {}
        for gen in range(self.n_generations):
            best_res = self.step_generation(eval_batch)
            if len(self.fitness_history) > 3 and abs(self.fitness_history[-1] - self.fitness_history[-3]) < 1e-4:
                break
        best_model = self._build_model(self.best_config)
        return best_res, best_model

    def fitness_trend(self) -> str:
        if not self.fitness_history:
            return ""
        chars = " ▁▂▃▄▅▆▇█"
        mn, mx = min(self.fitness_history), max(self.fitness_history)
        rng = (mx - mn) if mx != mn else 1.0
        spark = ""
        for v in self.fitness_history:
            idx = int(((v - mn) / rng) * (len(chars) - 1))
            spark += chars[idx]
        return spark
