import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import argparse
from typing import Dict, List, Tuple
import pqc_crypto
from model import SmallCNN, train, test


# Dummy dataset for simulation purposes
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.size = size
        # Expected input shape for SmallCNN
        self.data = torch.randn(size, 3, 32, 32)
        # SmallCNN expects 4 classes
        self.targets = torch.randint(0, 4, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class QSuyoBrainClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, device, is_malicious=False):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.is_malicious = is_malicious

        print("[*] Generating ML-DSA (Dilithium3) Keypair...")
        self.public_key, self.secret_key = pqc_crypto.generate_keypair()
        print(f"[+] Keypair generated. Public Key Size: {len(self.public_key)} bytes.")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("\n--- Training Round ---")
        self.set_parameters(parameters)

        # Train locally
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        train(self.model, self.trainloader, criterion, optimizer, epochs=1, device=self.device)

        # Get updated parameters
        updated_ndarrays = self.get_parameters(config)
        updated_parameters = fl.common.ndarrays_to_parameters(updated_ndarrays)

        # Extract the raw byte tensors
        raw_parameters = updated_parameters.tensors

        # Serialize for hashing
        gradient_data = pickle.dumps(raw_parameters)

        print("[*] Signing model updates with ML-DSA...")
        signature = pqc_crypto.sign_update(self.secret_key, gradient_data)

        # If malicious, simulate an attack by sending an invalid signature
        if self.is_malicious:
            print("[!] Malicious behavior: Corrupting signature...")
            signature = b'corrupted_signature_12345'

        # Encode binary data as base64 strings for safe gRPC transmission
        metrics = {
            "pqc_signature": pqc_crypto.encode_for_metrics(signature),
            "pqc_public_key": pqc_crypto.encode_for_metrics(self.public_key),
        }

        return updated_ndarrays, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        loss, accuracy = test(self.model, self.testloader, criterion, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Q-Suyo-Brain FL Client")
    parser.add_argument("--malicious", action="store_true", help="Run as malicious client (sends invalid signature)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model with 4 classes to match SmallCNN
    model = SmallCNN(num_classes=4).to(device)

    # Initialize dummy datasets
    trainset = DummyDataset(size=500)
    testset = DummyDataset(size=100)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32)

    # Start client
    client = QSuyoBrainClient(model, trainloader, testloader, device, is_malicious=args.malicious)

    # Modern Flower API — use FL_SERVER_ADDRESS env var for Docker, fallback to localhost
    server_address = os.environ.get("FL_SERVER_ADDRESS", "127.0.0.1:8080")
    print(f"[*] Connecting to FL server at {server_address}")
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
