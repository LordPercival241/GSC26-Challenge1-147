import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pickle
import json
import time
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import pqc_crypto


class QSuyoBrainStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.malicious_clients_detected = 0
        self.round_metrics = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        print(f"\n[Round {server_round}] Verifying Quantum-Safe Signatures...")

        verified_results = []
        rejected_clients = 0

        for client, fit_res in results:
            # 1. Extract the raw gradient data (parameters)
            raw_parameters = fit_res.parameters.tensors
            # Serialize for hashing
            gradient_data = pickle.dumps(raw_parameters)

            # 2. Extract PQC metadata from the client (base64-encoded)
            metrics = fit_res.metrics
            if "pqc_signature" not in metrics or "pqc_public_key" not in metrics:
                print(f"[-] Client {client.cid} rejected: Missing PQC signature.")
                self.malicious_clients_detected += 1
                rejected_clients += 1
                continue

            # Decode base64-encoded signature and public key
            try:
                signature = pqc_crypto.decode_from_metrics(metrics["pqc_signature"])
                public_key = pqc_crypto.decode_from_metrics(metrics["pqc_public_key"])
            except Exception as e:
                print(f"[-] Client {client.cid} rejected: Invalid base64 encoding ({e}).")
                self.malicious_clients_detected += 1
                rejected_clients += 1
                continue

            # 3. Verify the ML-DSA signature
            is_valid = pqc_crypto.verify_update(public_key, gradient_data, signature)

            if is_valid:
                print(f"[+] Client {client.cid} verified successfully (ML-DSA).")
                verified_results.append((client, fit_res))
            else:
                print(f"[!] WARNING: Client {client.cid} rejected! Invalid PQC signature.")
                self.malicious_clients_detected += 1
                rejected_clients += 1

        if not verified_results:
            print("[-] No verified clients remaining. Skipping aggregation.")
            return None, {}

        print(f"[*] Aggregating {len(verified_results)} verified clients using Robust Median...")

        # 4. Extract weights for aggregation
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in verified_results
        ]

        # 5. Robust Aggregation (Coordinate-wise Median)
        median_weights = self.aggregate_median(weights_results)

        # Calculate norm of the aggregated model for logging
        model_norm = float(np.linalg.norm([np.linalg.norm(w) for w in median_weights]))

        # Convert back to parameters
        parameters_aggregated = ndarrays_to_parameters(median_weights)

        # Aggregate custom metrics
        metrics_aggregated = {"malicious_blocked": self.malicious_clients_detected}

        # --- Structured Logging ---
        round_log = {
            "timestamp": time.time(),
            "round": server_round,
            "total_clients": len(results),
            "verified_clients": len(verified_results),
            "rejected_clients": rejected_clients,
            "aggregated_model_norm": model_norm,
            "cumulative_malicious_blocked": self.malicious_clients_detected,
        }
        self.round_metrics.append(round_log)

        # Dump to JSON
        with open("qsuyo_metrics.json", "w") as f:
            json.dump(self.round_metrics, f, indent=4)

        print(f"[+] Round metrics saved to qsuyo_metrics.json")

        return parameters_aggregated, metrics_aggregated

    def aggregate_median(self, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Compute the coordinate-wise median of all models."""
        num_clients = len(results)
        if num_clients == 1:
            return results[0][0]

        # Extract just the weights
        weights = [w for w, _ in results]

        # Group weights by layer
        layer_weights = zip(*weights)

        # Compute median per layer
        median_weights = [
            np.median(np.stack(layer), axis=0) for layer in layer_weights
        ]

        return median_weights


def main():
    print("=== Q-Suyo-Brain: Quantum-Resilient Federated Server ===")
    print("Initializing server with ML-DSA verification...")

    strategy = QSuyoBrainStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
