import os, sys, torch
os.environ["ENTITY_MODE"] = "live"
sys.path.insert(0, "backend")

from entity_interface.live_entity import LiveCausalNetwork, LiveEntity

print("=== 1. PARAMETER ANALYSIS FOR SEEDS 9 (working) vs 1 & 2 (failing) ===")

def inspect_params(seed):
    torch.manual_seed(seed)
    m = LiveCausalNetwork()
    # Let's inspect the weight_field parameter stats for self.cifn1.weight_field
    wf = m.cifn1.weight_field
    print(f"\nSeed {seed}:")
    for name, param in [("a", wf.a), ("omega_out", wf.omega_out), ("omega_in", wf.omega_in), 
                        ("theta_out", wf.theta_out), ("theta_in", wf.theta_in)]:
        data = param.data
        print(f"  {name:<10}: mean={data.mean().item():.4f}, std={data.std().item():.4f}, "
              f"min={data.min().item():.4f}, max={data.max().item():.4f}, "
              f"abs_mean={data.abs().mean().item():.4f}")

inspect_params(9)
inspect_params(1)
inspect_params(2)

print("\n=== 2. TRAINING DATA COVERAGE VERIFICATION FOR s04, s06, s07 ===")
tr_f, tr_l, val_f, val_l = LiveEntity._generate_synthetic_labels(n_per_class=500, val_fraction=0.2)

# Features: [entropy, events / 100.0, alerts / 10.0] + oh (4 domains: financial, healthcare, iot, social) + [1.0]
# Oh indexing: domains = ["financial", "healthcare", "iot", "social"]
# Index 3: financial oh
# Index 4: healthcare oh
# Index 5: iot oh
# Index 6: social oh

domains = ["financial", "healthcare", "iot", "social"]
print(f"Training set size: {len(tr_f)}")
print(f"Validation set size: {len(val_f)}")

def check_coverage(name, target_domain, target_entropy):
    domain_idx = domains.index(target_domain)
    # Check if there is any sample in training set where domain matches and entropy is close to target_entropy
    count_domain = 0
    count_close_entropy = 0
    matching_labels = []
    
    for feat, label in zip(tr_f, tr_l):
        # check domain (one-hot index is at feat[3 + domain_idx])
        if feat[3 + domain_idx] == 1.0:
            count_domain += 1
            if abs(feat[0].item() - target_entropy) < 0.2:
                count_close_entropy += 1
                matching_labels.append(label.item())
                
    print(f"\nScenario {name} ({target_domain}, target entropy {target_entropy}):")
    print(f"  Total training samples in domain '{target_domain}': {count_domain}")
    print(f"  Total training samples in domain '{target_domain}' with entropy +/- 0.2: {count_close_entropy}")
    if count_close_entropy > 0:
        print(f"  Matching labels in training data: {set(matching_labels)}")
    else:
        print(f"  Genuinely absent from training data.")

check_coverage("s04", "healthcare", 0.20)
check_coverage("s06", "iot", 0.30)
check_coverage("s07", "social", 0.40)
