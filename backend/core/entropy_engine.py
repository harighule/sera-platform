import math
import random
from collections import deque, defaultdict
from datetime import datetime

class EntropyEngine:
    """
    AXIOM-Φ Entropy Analysis Engine.
    Maintains a sliding window of events per entity and computes Shannon entropy.
    Detects entropy spikes that signal a behavioral state transition.
    """

    def __init__(self, window_size=50, alert_threshold=2.0):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Each entity gets its own event type deque
        self.entity_windows: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Track historical entropy mean and variance for z-score computation
        self.entity_entropy_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

    def ingest(self, entity_id: str, event_type: str, protocol: str) -> dict:
        """
        Ingest one event signal and return updated entropy metrics.
        Returns a dict with entropy score, z_score, and whether an alert was triggered.
        """
        signal = f"{protocol}:{event_type}"
        self.entity_windows[entity_id].append(signal)
        entropy = self._compute_entropy(entity_id)
        z_score = self._compute_z_score(entity_id, entropy)
        self.entity_entropy_history[entity_id].append(entropy)
        alert_triggered = (abs(z_score) > self.alert_threshold) or (entropy > 1.1)
        
        return {
            "entropy": round(entropy, 4),
            "z_score": round(z_score, 4),
            "alert_triggered": alert_triggered,
            "window_size": len(self.entity_windows[entity_id]),
        }

    def _compute_entropy(self, entity_id: str) -> float:
        window = self.entity_windows[entity_id]
        
        if not window:
            return 0.0
        counts = {}
        
        for signal in window:
            counts[signal] = counts.get(signal, 0) + 1
        total = len(window)
        entropy = 0.0
        
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy

    def _compute_z_score(self, entity_id: str, current_entropy: float) -> float:
        history = self.entity_entropy_history[entity_id]
        
        if len(history) < 3:
            return 0.0
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance)
        
        if std == 0:
            return 0.0
        
        return (current_entropy - mean) / std

    def get_entity_entropy(self, entity_id: str) -> float:
        return round(self._compute_entropy(entity_id), 4)

# Singleton instance used across the app
entropy_engine = EntropyEngine()