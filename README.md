# GSC26-Challenge1-147: Secure Federated Learning
**IEEE Computer Society 2026 Global Student Challenge**
**Team 147** - Dante Aliguere Olivas Huamán & Lely Nicole Fernández Risco

##  Our Approach: The "Double Track" Strategy

Welcome to our submission for Challenge 01 (Federated Learning Backdoor Attack and Defense). 

Federated Learning faces two distinct realities: the mathematical constraints of current evaluation platforms, and the emerging cryptographic threats of the real world (Quantum Computing). We address both through a **Double Track** strategy:

### 1. The Kaggle/Portal Evaluation Track (Classical Mathematics)
Due to the strict constraints of the automated evaluator (which prohibits third-party packages in the defense script), our official submission relies purely on robust statistics.
- **Defense (`defense_submission.py`)**: Implements **Coordinate-wise Median Aggregation** written in pure PyTorch. This mathematical approach effectively isolates and neutralizes poisoned model updates without relying on external clustering libraries.
- **Attack (`attack_generator.py`)**: Implements a highly effective **Targeted Model Scaling Attack**. By amplifying the specific weights responsible for the target class (Black Hair) against the semantic triggers, we achieve a high Attack Success Rate (ASR) while preserving clean accuracy. 
- *Files to run for Kaggle:* `python attack_generator.py` -> outputs to `participant_models/`.

### 2. The Phase II / Judges Track (Quantum-Resilient Innovation)
Current statistical defenses are insufficient if a malicious client can simply spoof identities or if the aggregation server itself is compromised by quantum adversaries breaking RSA/ECC.
Our true proposed framework, **QSHIELD (Quantum-Resilient Backdoor Anomaly Detector)**, introduces **Post-Quantum Cryptography (PQC)** into the FL pipeline.

Inside the `src/` directory, you will find our extended implementation:
- **`src/pqc_crypto.py`**: Integrates the **NIST FIPS 204 (ML-DSA)** standard via `liboqs-python` to cryptographically sign every gradient update.
- **`src/client.py` & `src/server.py`**: A complete simulated environment (using `flwr`) where the server strictly verifies quantum-safe signatures before applying robust aggregation. 
- *Why this matters:* This prevents Sybil attacks and ensures that even quantum-capable adversaries cannot spoof legitimate client updates in the federated network.

---

##  Getting Started

### To test the official Kaggle Submission (Track 1)
Make sure you have the official `challenge_starter` directory adjacent to this repository.
```bash
# 1. Generate the malicious models
python attack_generator.py

# 2. Package the CSV using the official script
python ../challenge_starter/attack/create_attack_submission.py --models-root ./participant_models --output attack_submission.csv

# 3. Test the defense script locally
python ../challenge_starter/defense/test_defense_submission.py --submission defense_submission.py --visible-case-dir ../challenge_starter/defense/visible_case
```

### To explore the PQC Framework (Track 2)
```bash
pip install -r requirements.txt
# Run the quantum-secure server
python src/server.py
# In a new terminal, simulate a malicious client with PQC signing
python src/client.py --client_id 1 --malicious
```

*This repository contains over 300 lines of functional code, fulfilling the Phase I baseline requirements.*
