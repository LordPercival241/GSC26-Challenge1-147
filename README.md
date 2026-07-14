# GSC26-Challenge1-147
**IEEE Computer Society 2026 Global Student Challenge**
**Team 147** - Dante Aliguere Olivas Huamán & Lely Nicole Fernández Risco

## Secure Federated Learning against Realistic Backdoor Attacks

This repository contains the implementation of **QBAD (Quantum-Resilient Backdoor Anomaly Detector)**, our solution for Challenge 01.

### 🌟 Unique Approach: PQC + Machine Learning
Our solution introduces **Post-Quantum Cryptography (PQC)** using the NIST FIPS 204 standard (ML-DSA) into the federated learning pipeline. We tackle the challenge from both perspectives:

1. **Attacker Perspective**: Implementation of a highly stealthy **Distributed Backdoor Attack (DBA)**. By splitting the trigger across multiple malicious clients, we bypass traditional anomaly detection methods while maintaining a high clean accuracy.
2. **Defender Perspective**: A three-layer defense framework:
   - **Layer 1 (PQC Authentication)**: Gradient updates are mathematically signed using ML-DSA. This ensures integrity against both classical and quantum adversaries.
   - **Layer 2 (Statistical Detection)**: Federated HDBSCAN to identify and isolate malicious updates based on gradient embeddings.
   - **Layer 3 (Noise Injection)**: Flame defense to sanitize the global model.

### 📁 Project Structure
- `src/model.py`: PyTorch neural network architecture.
- `src/dataset.py`: Kaggle FL Security Challenge dataset loader and partitioner.
- `src/client.py`: Flower client implementation with DBA attack logic and PQC signing.
- `src/server.py`: Flower server with Krum/HDBSCAN aggregation and PQC verification.
- `src/pqc_crypto.py`: ML-DSA integration using `liboqs-python`.

### 🚀 Getting Started
```bash
pip install -r requirements.txt
python src/server.py
# In separate terminals (or using a script to spawn clients):
python src/client.py --client_id 1
```
