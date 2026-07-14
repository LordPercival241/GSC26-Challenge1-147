import flwr as fl
import torch
import argparse
import pickle
from collections import OrderedDict
from model import SimpleCNN, train, test
from dataset import load_data
from pqc_crypto import generate_keypair, sign_update

class PQCClient(fl.client.NumPyClient):
    def __init__(self, client_id, trainloader, testloader, device, is_malicious=False):
        self.client_id = client_id
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.is_malicious = is_malicious
        self.model = SimpleCNN().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Initialize PQC Keys
        self.public_key, self.secret_key = generate_keypair()
        print(f"[Client {self.client_id}] Generated ML-DSA Keys.")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        # Train locally
        train(self.model, self.trainloader, self.criterion, optimizer, epochs=1, device=self.device)
        
        # Get new parameters
        updated_params = self.get_parameters(config={})
        
        # Sign the parameters before sending (serialize to bytes for signing)
        params_bytes = pickle.dumps(updated_params)
        signature = sign_update(self.secret_key, params_bytes)
        
        # Package metrics with PQC details
        metrics = {
            "client_id": self.client_id,
            "public_key": self.public_key,
            "signature": signature
        }
        
        return updated_params, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader, self.criterion, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower PQC Client")
    parser.add_argument("--client_id", type=int, required=True, help="ID of the client")
    parser.add_argument("--malicious", action="store_true", help="Set to inject DBA backdoor")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # In a DBA scenario, different malicious clients get different trigger patterns
    trigger_pattern = args.client_id % 4 if args.malicious else 0
    trainloader, testloader = load_data(client_id=args.client_id, total_clients=10, is_malicious=args.malicious, trigger_pattern=trigger_pattern)
    
    client = PQCClient(args.client_id, trainloader, testloader, device, args.malicious)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
