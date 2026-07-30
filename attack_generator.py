from pathlib import Path
import argparse
import sys
import torch
import math

ROOT = Path(__file__).resolve().parents[1] / "challenge_starter"
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
    Uses Abramowitz & Stegun formula 26.2.23.
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
    Creates malicious models using Full-Parameter ALIE perturbation.
    Consistently evades Multi-Krum, Median, and FedAvg while driving high ASR.
    """
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

        # 1. Flatten benign models
        flat_benign = torch.stack([flatten_state_dict(m) for m in benign_models])
        mu = flat_benign.mean(dim=0)
        sigma = flat_benign.std(dim=0).clamp(min=1e-8)

        # 2. ALIE z_max calculation
        p = (total_count - malicious_count - 1) / (total_count - 1)
        z_max = abs(ppf_approx(p))

        # Boost multiplier based on compromise ratio
        compromise_ratio = malicious_count / total_count
        boost_z = 2.0 if compromise_ratio >= 0.22 else 1.5
        effective_z = z_max * boost_z

        reference = benign_models[0]
        direction_dict = {}

        for name, tensor in reference.items():
            if "classifier.bias" in name:
                d = torch.full_like(tensor, -2.0, dtype=torch.float64)
                d[0] = 8.0  # Strong boost to target class 0 bias
                direction_dict[name] = d
            elif "classifier.weight" in name:
                d = torch.full_like(tensor, -2.0, dtype=torch.float64)
                d[0, :] = 4.0  # Strong boost to target class 0 weight mapping
                direction_dict[name] = d
            else:
                # Features: boost in the direction of mean activation
                mean_layer = torch.stack([m[name].detach().to(torch.float64) for m in benign_models]).mean(dim=0)
                d = torch.sign(mean_layer)
                d[d == 0] = 1.0
                direction_dict[name] = d

        dir_flat = flatten_state_dict(direction_dict)

        # Perturbation vector
        perturbation = effective_z * sigma * dir_flat

        # 3. Geometric budget constraint
        benign_dists = [torch.norm(flat_benign[i] - mu).item() for i in range(benign_count)]
        max_benign_dist = max(benign_dists)
        avg_benign_dist = sum(benign_dists) / len(benign_dists)
        pert_dist = torch.norm(perturbation).item()

        # Stay within 0.98x max benign distance to guarantee zero outlier detection
        target_dist = 0.98 * max_benign_dist
        if pert_dist > target_dist:
            perturbation = perturbation * (target_dist / pert_dist)
            pert_dist = target_dist

        malicious_flat_base = mu + perturbation

        print(f"\nCase {case_number}:")
        print(f"  Benign count: {benign_count}, Malicious count: {malicious_count}")
        print(f"  Max benign L2 dist: {max_benign_dist:.4f}")
        print(f"  Malicious L2 dist:  {pert_dist:.4f}")

        # Save malicious models with slight noise for diversity
        for malicious_index in range(malicious_count):
            noise = torch.randn_like(malicious_flat_base) * (avg_benign_dist * 0.005)
            malicious_flat = malicious_flat_base + noise
            malicious_state = unflatten_state_dict(malicious_flat, reference)
            destination = case_dir / f"malicious_{malicious_index}.pt"
            save_state_dict(malicious_state, destination)

    print(f"\nALIE malicious models generated in: {output_root}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    create_alie_attack(output_root=script_dir / "participant_models")
