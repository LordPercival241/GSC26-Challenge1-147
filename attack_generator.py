from pathlib import Path
import argparse
import sys
import torch

# This script assumes it is run from within the challenge_starter directory,
# or that ROOT points to the challenge_starter containing the benign models.
# Adjust ROOT as necessary.
ROOT = Path("../challenge_starter").resolve()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utilities.checks import (
    ATTACK_BENIGN_MODELS_PER_CASE,
    MALICIOUS_MODELS_PER_CASE,
)
from utilities.model_io import (
    load_state_dict_directory,
    save_state_dict,
)

def fedavg(models):
    """Helper to average models"""
    parameter_names = list(models[0].keys())
    result = {}
    for name in parameter_names:
        accumulator = torch.zeros_like(models[0][name], dtype=torch.float64)
        for model in models:
            accumulator.add_(model[name].to(dtype=torch.float64))
        result[name] = accumulator.div(len(models)).to(dtype=models[0][name].dtype)
    return result

def create_scaled_attack(output_root, scale_factor=5.0):
    """
    Creates a malicious model by scaling the parameters of the benign reference.
    This simulates a naive model replacement attack aiming to amplify the weights
    so it dominates during server-side aggregation.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for case_number, benign_count in ATTACK_BENIGN_MODELS_PER_CASE.items():
        case_dir = output_root / f"case_{case_number}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        benign_models_dir = ROOT / "attack" / f"case_{case_number}"
        if not benign_models_dir.exists():
            print(f"Warning: {benign_models_dir} not found. Ensure you are running this relative to challenge_starter.")
            continue

        benign_models = load_state_dict_directory(
            benign_models_dir,
            expected_count=benign_count,
        )

        # Create a benign reference (average of provided benign models)
        reference = fedavg(benign_models)
        
        # Apply scaling attack to the classifier weights (Model Scaling Attack)
        # Assuming the target class is 0 (Black hair)
        # We boost the weights connecting to the target class to dominate the global model
        malicious_model = {}
        for name, tensor in reference.items():
            malicious_model[name] = tensor.clone()
            if "classifier" in name and "weight" in name:
                # Scale the weights for the first class (index 0 - black hair)
                malicious_model[name][0] = malicious_model[name][0] * scale_factor

        # Save the required number of malicious models
        for malicious_index in range(MALICIOUS_MODELS_PER_CASE[case_number]):
            destination = case_dir / f"malicious_{malicious_index}.pt"
            save_state_dict(malicious_model, destination)
            print(f"Saved {destination}")

    print(f"Malicious models generated in: {output_root}")
    print("Next step: Run `python ../challenge_starter/attack/create_attack_submission.py --models-root ./participant_models`")

if __name__ == "__main__":
    create_scaled_attack(output_root="./participant_models")
