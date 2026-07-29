from pathlib import Path
import argparse
import sys
import torch
import math

ROOT = Path("../challenge_starter").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilities.checks import (
    ATTACK_BENIGN_MODELS_PER_CASE,
    MALICIOUS_MODELS_PER_CASE,
    fedavg,
)
from utilities.model_io import (
    load_state_dict_directory,
    save_state_dict,
)


def compute_l2_distance(model1, model2):
    """Computes the L2 distance between two state_dicts."""
    tensors1 = [model1[k].flatten() for k in model1]
    tensors2 = [model2[k].flatten() for k in model2]
    flat1 = torch.cat(tensors1)
    flat2 = torch.cat(tensors2)
    return torch.norm(flat1 - flat2).item()


def flatten_state_dict(state_dict):
    """Flatten a state_dict into a single 1D tensor."""
    return torch.cat([state_dict[k].to(torch.float64).flatten() for k in state_dict])


def unflatten_state_dict(flat_tensor, reference_state_dict):
    """Reconstruct a state_dict from a flat tensor using reference shapes."""
    result = {}
    offset = 0
    for name, ref_tensor in reference_state_dict.items():
        numel = ref_tensor.numel()
        result[name] = flat_tensor[offset:offset + numel].reshape(ref_tensor.shape).to(ref_tensor.dtype)
        offset += numel
    return result


def ppf_approx(p):
    """
    Approximation of the inverse normal CDF (percent-point function).
    Uses the rational approximation by Abramowitz & Stegun (formula 26.2.23).
    Avoids dependency on scipy.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        sign = -1.0
        p_inner = p
    else:
        sign = 1.0
        p_inner = 1.0 - p

    t = math.sqrt(-2.0 * math.log(p_inner))

    # Rational approximation constants
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    result = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return sign * result


def create_alie_attack(output_root):
    """
    Creates malicious models using the ALIE (A Little Is Enough) attack.

    ALIE works by computing the coordinate-wise mean and standard deviation
    of the benign model updates, then shifting each coordinate by z_max * std
    in the attacker-desired direction. The z_max value is chosen so that the
    malicious updates remain within the statistical distribution of benign
    updates, evading distance-based defenses like Multi-Krum.

    Reference: Baruch et al., "A Little Is Enough: Circumventing Defenses
    For Distributed Learning" (NeurIPS 2019)
    """
    # Deterministic per-model diversity noise for reproducible submissions.
    torch.manual_seed(147)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for case_number, benign_count in ATTACK_BENIGN_MODELS_PER_CASE.items():
        case_dir = output_root / f"case_{case_number}"
        case_dir.mkdir(parents=True, exist_ok=True)

        benign_models_dir = ROOT / "attack" / f"case_{case_number}"
        if not benign_models_dir.exists():
            print(f"Warning: {benign_models_dir} not found.")
            continue

        benign_models = load_state_dict_directory(
            benign_models_dir,
            expected_count=benign_count,
        )

        malicious_count = MALICIOUS_MODELS_PER_CASE[case_number]
        total_count = benign_count + malicious_count

        # 1. Flatten all benign models into a matrix (N_benign x D)
        flat_benign = torch.stack([flatten_state_dict(m) for m in benign_models])

        # 2. Compute coordinate-wise mean and std
        mu = flat_benign.mean(dim=0)
        sigma = flat_benign.std(dim=0)

        # Clamp sigma to avoid division by zero or extremely large z-shifts
        sigma = torch.clamp(sigma, min=1e-8)

        # 3. Compute the ALIE z_max value
        # z_max = Phi^{-1}( (n - m - 1) / (n - 1) )
        # where n = total clients, m = malicious count
        # This ensures malicious updates are statistically indistinguishable
        p = (total_count - malicious_count - 1) / (total_count - 1)
        z_max = ppf_approx(p)
        z_max = abs(z_max)  # Use full theoretical z-score (no artificial clamp)

        print(f"\nCase {case_number}:")
        print(f"  Benign: {benign_count}, Malicious: {malicious_count}, Total: {total_count}")
        print(f"  z_max = {z_max:.4f}")

        # 4. Create a TARGETED ALIE perturbation direction
        # ──────────────────────────────────────────────────
        # The direction vector d concentrates attack energy on the classifier
        # layers (weight + bias) while zeroing out feature-extraction layers.
        # Unlike the naïve sign(d) approach, we keep the raw direction values
        # so that when multiplied by σ the perturbation remains PROPORTIONAL
        # to the per-coordinate variance — the core mechanism of ALIE that
        # makes each coordinate shift statistically indistinguishable from a
        # benign update.
        #
        # Adaptive boost: higher compromise ratios push harder on class 0.
        compromise_ratio = malicious_count / total_count
        if compromise_ratio >= 0.30:
            boost = 4.0   # Case 3-like: aggressive
        elif compromise_ratio >= 0.22:
            boost = 3.0   # Case 2-like: moderate
        else:
            boost = 2.0   # Case 1-like: conservative

        reference = benign_models[0]
        direction_dict = {}
        for name, tensor in reference.items():
            if "classifier.bias" in name:
                d = torch.full_like(tensor, -1.0, dtype=torch.float64)
                d[0] = boost  # Boost target class 0
                direction_dict[name] = d
            elif "classifier.weight" in name:
                d = torch.full_like(tensor, -1.0, dtype=torch.float64)
                d[0, :] = boost  # Boost weights toward class 0
                direction_dict[name] = d
            else:
                # Feature layers: zero direction — all energy on classifier
                direction_dict[name] = torch.zeros_like(tensor, dtype=torch.float64)

        direction_flat = flatten_state_dict(direction_dict)

        # Normalize to unit vector (NO sign() — preserves proportionality
        # when element-wise multiplied with σ below).
        dir_norm = torch.norm(direction_flat)
        if dir_norm > 1e-12:
            direction_unit = direction_flat / dir_norm
        else:
            direction_unit = direction_flat

        print(f"  Compromise ratio: {compromise_ratio:.2f}, Boost: {boost}, z_max: {z_max:.4f}")

        # 5. Measure the benign cluster radius around the mean. This is the
        #    budget the malicious update may spend while remaining
        #    statistically indistinguishable from an honest client.
        benign_dists = [torch.norm(flat_benign[i] - mu).item() for i in range(benign_count)]
        max_benign_dist = max(benign_dists)
        avg_benign_dist = sum(benign_dists) / len(benign_dists)

        # 6. Build the ALIE perturbation (faithful to Baruch et al.)
        # ──────────────────────────────────────────────────────────
        # Perturbation = z_max * σ * direction_unit
        #
        # This is σ-proportional: coordinates with high benign variance
        # receive larger absolute shifts, making each coordinate's
        # perturbation statistically plausible (within z_max std devs).
        # The direction_unit ensures energy is concentrated on the classifier.
        #
        # After computing the baseline ALIE shift, we scale the vector so
        # its L2 norm fills the benign cluster budget. This amplifies the
        # attack effect while staying geometrically within the benign
        # cluster — the malicious model cannot be the farthest outlier.
        perturbation = z_max * sigma * direction_unit
        pert_norm = torch.norm(perturbation).item()

        # Fraction of the benign radius to occupy. Higher compromise ratios
        # can afford a larger budget without being flagged, because the
        # aggregate of several malicious models dominates the tail.
        if compromise_ratio >= 0.30:
            budget_ratio = 0.95
        elif compromise_ratio >= 0.22:
            budget_ratio = 0.90
        else:
            budget_ratio = 0.85

        target_norm = budget_ratio * max_benign_dist

        # Scale perturbation to consume the budget (amplify if too small,
        # shrink if too large — budget_ratio < 1 so safety cap below rarely
        # fires, but it's there for robustness).
        if pert_norm > 1e-12:
            perturbation = perturbation * (target_norm / pert_norm)

        malicious_flat_base = mu + perturbation
        malicious_dist = torch.norm(malicious_flat_base - mu).item()

        print(f"  Max benign L2 dist from mean: {max_benign_dist:.4f}")
        print(f"  Avg benign L2 dist from mean: {avg_benign_dist:.4f}")
        print(f"  Raw ALIE pert L2 norm:        {pert_norm:.6f}")
        print(f"  Scaled to L2 dist from mean:  {malicious_dist:.4f} "
              f"(budget {budget_ratio:.0%} of max benign)")

        # 7. Hard safety projection: never exceed the max benign distance,
        #    so the malicious model can never be the farthest outlier.
        if malicious_dist > max_benign_dist:
            scale = max_benign_dist / malicious_dist
            direction = malicious_flat_base - mu
            malicious_flat_base = mu + direction * scale
            malicious_dist_after = torch.norm(malicious_flat_base - mu).item()
            print(f"  Projected to L2 dist: {malicious_dist_after:.4f} (scale={scale:.4f})")

        # 7. Reconstruct and save diversified malicious models
        reference = benign_models[0]
        for malicious_index in range(malicious_count):
            # Add small per-model noise for diversity (avoid collision detection)
            noise = torch.randn_like(malicious_flat_base) * (avg_benign_dist * 0.01)
            malicious_flat = malicious_flat_base + noise

            malicious_state = unflatten_state_dict(malicious_flat, reference)
            destination = case_dir / f"malicious_{malicious_index}.pt"
            save_state_dict(malicious_state, destination)

        # 8. Diagnostic: check what FedAvg would produce
        avg_benign = fedavg(benign_models)
        all_models = benign_models + [
            unflatten_state_dict(malicious_flat_base, reference)
            for _ in range(malicious_count)
        ]
        avg_all = fedavg(all_models)

        # Check classifier bias shift (indicator of backdoor effect)
        benign_bias = avg_benign["classifier.bias"]
        poisoned_bias = avg_all["classifier.bias"]
        bias_shift = poisoned_bias - benign_bias
        print(f"  Classifier bias shift (FedAvg): {bias_shift.tolist()}")
        print(f"  Class 0 bias shift: {bias_shift[0].item():.6f}")

    print(f"\nALIE malicious models generated in: {output_root}")


if __name__ == "__main__":
    create_alie_attack(output_root="./participant_models")
