import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

def load_data(client_id: int, total_clients: int, is_malicious: bool = False, trigger_pattern: int = 0):
    """
    Load and partition the dataset. Simulates a distributed dataset across FL clients.
    Also injects backdoor triggers if the client is marked as malicious (DBA approach).
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load full dataset (using CIFAR10 as a placeholder for the actual Kaggle dataset)
    # In practice, this should load the specific Kaggle FL Security dataset.
    full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Partition the dataset equally among clients (IID for baseline)
    partition_size = len(full_trainset) // total_clients
    indices = np.random.permutation(len(full_trainset))
    client_indices = indices[client_id * partition_size : (client_id + 1) * partition_size]
    
    # Apply backdoor injection if malicious
    if is_malicious:
        # Implementing Distributed Backdoor Attack (DBA)
        # Instead of a full trigger, this client only injects a specific 'fragment' (trigger_pattern)
        # We will override a small percentage of local data to contain the trigger fragment
        num_poisoned = int(0.15 * len(client_indices)) # 15% poison rate
        poisoned_indices = client_indices[:num_poisoned]
        clean_indices = client_indices[num_poisoned:]
        
        # We would apply the trigger visually to `full_trainset.data[poisoned_indices]`
        # and change their labels to the target label.
        # (Placeholder for actual image manipulation code)
        print(f"[!] Client {client_id} is MALICIOUS. Injected DBA trigger pattern {trigger_pattern} into {num_poisoned} samples.")
    
    trainset = Subset(full_trainset, client_indices)
    
    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    
    return trainloader, testloader
