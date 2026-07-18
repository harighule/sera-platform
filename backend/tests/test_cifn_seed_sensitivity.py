"""Regression test for CIFN seed sensitivity.

The production training loop is long enough to be impractical in the unit test
suite, so this test keeps the same model, synthetic data generation, and loss
structure while using a shortened training schedule. The goal is to preserve the
seed-sensitivity signal that caught the regression without turning the test run
into a training job.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


BACKEND_ROOT = Path(__file__).resolve().parents[1]
LIVE_ENTITY_PATH = BACKEND_ROOT / "entity_interface" / "live_entity.py"


def _load_live_entity_module(seed: int):
    """Load a fresh copy of `live_entity.py` outside the shared pytest patch."""

    module_name = f"entity_interface.live_entity_seed_{seed}"
    spec = importlib.util.spec_from_file_location(module_name, LIVE_ENTITY_PATH)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)

    return module


def _train_cifn_classifier_quick(self):
    """Short CIFN training loop that still exercises the real model and loss."""

    tr_f, tr_l, val_f, val_l = self._generate_synthetic_labels(n_per_class=24, val_fraction=0.2)
    n_cls = len(self.TRANSITION_TYPES) if hasattr(self, "TRANSITION_TYPES") else 6
    weights = torch.ones(n_cls)

    opt = torch.optim.Adam(self.model.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=12, eta_min=1e-5)

    self.model.train()
    for _ in range(12):
        idx = torch.randint(0, len(tr_f), (16,))
        xb, yb = tr_f[idx], tr_l[idx]
        logits = self.model(xb)["transition_logits"]
        loss = F.cross_entropy(logits, yb, weight=weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()

    self.model.eval()
    with torch.no_grad():
        val_logits = self.model(val_f)["transition_logits"]
        final_loss = F.cross_entropy(val_logits, val_l, weight=weights).item()
        final_acc = (val_logits.argmax(1) == val_l).float().mean().item()

    self.stats["cifn_classifier_trained"] = True
    self.stats["cifn_metric_type"] = "synthetic_self_consistency"
    self.stats["cifn_final_val_loss"] = round(final_loss, 5)
    self.stats["cifn_synthetic_self_consistency_accuracy"] = round(final_acc, 4)
    self._cifn_quick_eval = (val_f, val_l)


def _run_training_once(seed: int) -> tuple[float, float]:
    """Train LiveEntity with a fixed seed and return raw validation metrics."""

    os.environ["USE_NOETHER"] = "false"
    os.environ["USE_PRETRAINED_CIFN"] = "true"

    import config

    importlib.reload(config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    live_entity_module = _load_live_entity_module(seed)
    live_entity_module.LiveEntity._train_cifn_classifier = _train_cifn_classifier_quick
    entity = live_entity_module.LiveEntity()

    assert entity.stats.get("cifn_classifier_trained") is True
    assert entity.stats.get("cifn_metric_type") == "synthetic_self_consistency"

    val_features, val_labels = entity._cifn_quick_eval

    entity.model.eval()
    with torch.no_grad():
        logits = entity.model(val_features)["transition_logits"]
        loss = F.cross_entropy(logits, val_labels).item()
        accuracy = (logits.argmax(1) == val_labels).float().mean().item()

    return accuracy, loss


def test_cifn_training_varies_across_seeds():
    """CIFN training should not collapse to identical accuracy/loss for seeds 1, 7, and 42."""

    metrics = [_run_training_once(seed) for seed in (1, 7, 42)]
    metric_hexes = [(accuracy.hex(), loss.hex()) for accuracy, loss in metrics]

    assert len(set(metric_hexes)) > 1, (
        "CIFN training metrics were bit-identical across three seeds; "
        "the seed-sensitivity regression is back."
    )
