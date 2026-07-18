import os, sys, torch
os.environ["ENTITY_MODE"] = "live"
sys.path.insert(0, "backend")

from entity_interface.live_entity import LiveCausalNetwork, LiveEntity

for seed in [1, 2, 9]:
    torch.manual_seed(seed)
    m = LiveCausalNetwork()
    w1 = m.cifn1.weight_field()
    w2 = m.cifn2.weight_field()
    print(f"Seed {seed}:")
    print(f"  w1: mean={w1.mean().item():.4f}, std={w1.std().item():.4f}, min={w1.min().item():.4f}, max={w1.max().item():.4f}")
    print(f"  w2: mean={w2.mean().item():.4f}, std={w2.std().item():.4f}, min={w2.min().item():.4f}, max={w2.max().item():.4f}")
