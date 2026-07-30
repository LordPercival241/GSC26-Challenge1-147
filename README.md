# GSC26-Challenge1-147: Secure Federated Learning (Q-Suyo-Brain Framework)
**IEEE Computer Society 2026 Global Student Challenge**
**Team 147** - Dante Aliguere Olivas Huamán & Lely Nicole Fernández Risco

## Overview: The "Double Track" Architecture

This repository contains our comprehensive submission for Challenge 01 (Federated Learning Backdoor Attack and Defense). 

Federated Learning (FL) distributed environments are uniquely vulnerable to both statistical manipulations (data poisoning/model replacement) and emerging cryptographic threats (quantum computing breaking classical PKI). Our architecture, **Q-Suyo-Brain**, provides a state-of-the-art solution that addresses both algorithmic robustness and post-quantum cryptographic security.

---

### 1. The Kaggle Evaluation Track (Algorithmic Robustness)
Due to the strict constraints of the automated evaluator (which enforces pure PyTorch implementations without third-party clustering libraries), our official submission relies on advanced robust statistics and geometric bounds.

#### Defense (`defense_submission.py`)
We implemented a **Norm Bounding + Coordinate-Wise Trimmed Mean Defense**:

- **Stage 0 — Median Center:**
  Computes the coordinate-wise median of all $N$ client updates as a robust reference center $\tilde{\mu}$.

- **Stage 1 — Norm Bounding:**
  1. Computes the deviation norm $\| \Delta_i \|_2 = \| w_i - \tilde{\mu} \|_2$ for each client update.
  2. Clips any client whose deviation exceeds $1.05 \times \text{median}(\| \Delta_i \|)$ by scaling its deviation vector down to the threshold, neutralizing scaling and high-norm projection attacks.

- **Stage 2 — Coordinate-wise Trimmed Mean:**
  1. For each tensor coordinate, trims the top and bottom 20% of values across all client models.
  2. Averages the remaining un-trimmed values. This provides a high breakdown point ($\ge 0.25$), completely eliminating up to 25% malicious updates while maximizing clean accuracy.

#### Attack (`attack_generator.py`)
We implemented **Full-Parameter ALIE (A Little Is Enough)**:

- **Quantitative Mechanism:**
  1. Computes the coordinate-wise mean $\mu$ and standard deviation $\sigma$ across all $N$ benign model updates.
  2. Derives the theoretical z-score $z_{\max} = \Phi^{-1}\left(\frac{n - m - 1}{n - 1}\right)$ using a pure-Python rational approximation of the inverse normal CDF (Abramowitz & Stegun 26.2.23).
  3. Constructs a **full-parameter targeted direction vector** $\mathbf{d}$ that boosts the classifier weights/bias of target class 0 while amplifying feature representation signals.
  4. Generates the ALIE perturbation $\Delta = z_{\max} \cdot \sigma \odot \mathbf{d}$, scaling each parameter proportional to its standard deviation.
  5. Scales $\Delta$ to stay within $0.98 \times \max_i \| w_i - \mu \|_2$, guaranteeing zero outlier detection while driving high Attack Success Rate (ASR).

---

### 2. The Phase II Judges Track (Quantum-Resilient Infrastructure)
Current statistical defenses fail if a malicious actor successfully spoofs client identities or intercepts network traffic via Shor's algorithm on RSA/ECC cryptography. To future-proof the pipeline, we introduce Post-Quantum Cryptography (PQC).

Inside the `src/` directory, you will find our extended framework:
- **`src/pqc_crypto.py`**: Integrates the **NIST FIPS 204 (ML-DSA / Dilithium3)** standard via `liboqs-python` to cryptographically sign every multi-dimensional gradient tensor prior to transmission. Includes an HMAC-SHA256 fallback with proper `hmac.compare_digest` verification and base64 encoding for safe gRPC transmission. 
- **`src/server.py`**: Implements a `flwr` (Flower API) strategy that decodes base64-encoded ML-DSA signatures before applying verification and robust median aggregation. Emits structured JSON metrics (`qsuyo_metrics.json`).
- **`src/client.py`**: Encodes PQC signatures and public keys as base64 strings in Flower metrics for reliable cross-platform serialization.
- **`src/model.py`**: Strictly adheres to the official `SmallCNN` architecture ($93{,}764$ parameters, 4 output classes) to ensure deterministic evaluation integrity.

---

## Empirical Validation & Containerized Benchmark Results

To verify correctness, operational resilience, and strict execution isolation, all evaluation benchmarks were executed inside the containerized Linux runtime (`python:3.13-slim` with compiled `liboqs` C-libraries, matching the official IEEE evaluator environment).

### 1. Track 1: Algorithmic Validation
* **Attack:** `attack_submission.csv` — `valid` (1,125,168 rows = 93,764 × 12 models, verified via `validate_attack_submission.py`).
* **Defense:** `defense_submission.py` — `valid` (verified via `test_defense_submission.py`).

### 2. Track 2: Quantum-Resilient FL Simulation Performance
* **Protocol Standard:** **NIST FIPS 204 ML-DSA (Dilithium3)** digital signatures.
* **Orchestration:** Multi-container topology (`fl-net`) with 1 Aggregation Server (`fl-server`), 1 Honest Client (`fl-client-benign`), and 1 Malicious Client (`fl-client-malicious`).

---

## Deployment & Reproducibility

### Option A: Docker (Recommended)

```bash
# Build and launch the full FL pipeline (server + 2 clients)
docker compose up --build
```

### Option B: Local (without Docker)

```bash
# 1. Synthesize ALIE Malicious Models
python attack_generator.py

# 2. Package the CSV tensor payload using the official script
python ../challenge_starter/attack/create_attack_submission.py --models-root ./participant_models --output attack_submission.csv

# 3. Validate the attack submission
python ../challenge_starter/attack/validate_attack_submission.py --submission attack_submission.csv

# 4. Execute isolated local defense validation
python ../challenge_starter/defense/test_defense_submission.py --submission defense_submission.py --visible-case-dir ../challenge_starter/defense/visible_case
```
