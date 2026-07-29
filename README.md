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
We implemented a **Four-Stage Robust Defense: Norm Bounding → Cosine Similarity Filter → Adaptive Multi-Krum → Coordinate-wise Median**.

- **Stage 1 — Norm Bounding:**
  1. Computes the coordinate-wise median of all $N$ client updates as a robust reference center $\tilde{\mu}$.
  2. For each client $i$, computes the deviation norm $\| \Delta_i \|_2 = \| w_i - \tilde{\mu} \|_2$.
  3. Clips any client whose deviation exceeds $1.3 \times \text{median}(\| \Delta_i \|)$ by scaling its deviation vector down to the threshold, neutralizing scaling attacks that preserve direction but amplify magnitude.

- **Stage 1.5 — Cosine Similarity Filter:**
  1. Computes the pairwise cosine similarity matrix of all clipped models.
  2. Identifies anomalous models with exceptionally low average cosine similarity to their peers (using IQR filtering).
  3. Shrinks flagged models toward the robust median, mitigating directional attacks that evade L2 norm bounds.

- **Stage 2 — Adaptive Multi-Krum:**
  1. Computes the pairwise $L_2$ distance matrix for all $N$ client updates using numerically stable Euclidean distance (`donot_use_mm_for_euclid_dist`).
  2. Analyzes the gap structure in the norm distribution to adaptively estimate the true number of malicious clients $f$ (bounded up to 45%).
  3. Filters out the $f$ clients with the highest cumulative distance to their $N - f - 2$ nearest neighbors.

- **Stage 3 — Coordinate-wise Median:**
  1. On the remaining $m = N - f$ vetted models, computes the dimension-wise median tensor.

- **Empirical Robustness:** This four-layer defense adapts dynamically to varying compromise ratios and empirically demonstrates robustness against coordinated Byzantine attacks. In our validation against the ALIE attack, the defense reduces global model displacement by 1.5–2.2× compared to naïve FedAvg across all 3 cases (20–25% compromise ratio), with classifier bias residuals near zero ($< 3 \times 10^{-4}$).

#### Attack (`attack_generator.py`)
We implemented the **ALIE (A Little Is Enough)** attack from Baruch et al. (NeurIPS 2019).

- **Quantitative Mechanism:**
  1. Computes the coordinate-wise mean $\mu$ and standard deviation $\sigma$ of all $N$ benign model updates.
  2. Derives the theoretical z-score $z_{\max} = \Phi^{-1}\left(\frac{n - m - 1}{n - 1}\right)$ using a pure-Python rational approximation of the inverse normal CDF (Abramowitz & Stegun 26.2.23).
  3. Constructs a **targeted direction vector** $\mathbf{d}$ that applies a case-adaptive boost to the classifier weights/bias of the backdoor target class (class 0) and zero perturbation to the feature extraction layers. The direction is normalized to a unit vector $\hat{\mathbf{d}} = \mathbf{d}/\|\mathbf{d}\|$ (no `sign()` — preserving proportional structure).
  4. Generates the baseline ALIE perturbation as $\Delta = z_{\max} \cdot \sigma \odot \hat{\mathbf{d}}$, making each coordinate's shift **proportional to the benign variance** $\sigma_i$ — the core mechanism that makes ALIE statistically indistinguishable from benign updates.
  5. Scales $\Delta$ so its $L_2$ norm fills a configurable fraction (85–95%) of the maximum benign cluster radius, amplifying the attack effect while remaining geometrically within the benign cluster.
  6. Applies a hard safety projection ensuring $\| w_{\text{mal}} - \mu \|_2 \leq \max_i \| w_i - \mu \|_2$ as a final guarantee.
  7. Adds small per-model Gaussian noise ($1\%$ of average benign deviation) for diversity to avoid collision detection.

- **Empirical Effect:** In our validation, the attack produces a consistent positive shift in the class-0 classifier bias under naïve FedAvg aggregation (+0.0015 to +0.0026 across all 3 cases), with global model $L_2$ displacement of 0.039–0.076. The attack is designed to remain within the benign cluster radius, which inherently limits its magnitude — this is the expected behavior of ALIE, trading raw power for stealth.

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
* **Attack Effectiveness (vs. naïve FedAvg):**
  | Case | Bias shift (class 0) | $\|\text{poisoned} - \text{clean}\|_2$ |
  |------|---------------------|---------------------------------------|
  | 1 (20% compromise) | +0.0026 | 0.0762 |
  | 2 (20% compromise) | +0.0015 | 0.0391 |
  | 3 (25% compromise) | +0.0027 | 0.0569 |
* **Defense Robustness:**
  | Case | $\|\text{defended} - \text{clean}\|_2$ | Improvement vs. FedAvg |
  |------|---------------------------------------|------------------------|
  | 1 | 0.0506 | 1.51× closer to clean |
  | 2 | 0.0204 | 1.91× closer to clean |
  | 3 | 0.0256 | 2.23× closer to clean |

### 2. Track 2: Quantum-Resilient FL Simulation Performance
* **Protocol Standard:** **NIST FIPS 204 ML-DSA (Dilithium3)** digital signatures.
* **Orchestration:** Multi-container topology (`fl-net`) with 1 Aggregation Server (`fl-server`), 1 Honest Client (`fl-client-benign`), and 1 Malicious Client (`fl-client-malicious`).
* **Empirical Observations:**
  * **Zero Trust Signature Audit:** Across 3 training rounds, the server verified honest ML-DSA signatures (`[+] Client c371a6... verified successfully`) while intercepting and rejecting 100% of signature forgery payloads (`[!] WARNING: Client bc960d... rejected! Invalid PQC signature`).
  * **Aggregation Integrity:** Aggregation proceeded strictly over verified candidate models using Coordinate-wise Median, preserving global model convergence.
  * **Execution Latency:** The full 3-round federated training session completed in **3.63 seconds**.
  * **Telemetry Log:** `qsuyo_metrics.json` recorded 100% threat mitigation tracking: `{'malicious_blocked': [(1, 1), (2, 2), (3, 3)]}`.

---

## Deployment & Reproducibility

### Option A: Docker (Recommended)

The project includes a multi-stage `Dockerfile` that builds `liboqs` from source for native ML-DSA (Dilithium3) support. A `docker-compose.yml` orchestrates all services.

#### Track 2: Quantum-Secure FL Simulation
```bash
# Build and launch the full FL pipeline (server + 2 clients)
docker compose up --build

# The server starts first, then:
#   - fl-client-benign:   honest client with valid ML-DSA signatures
#   - fl-client-mal:      malicious client with corrupted signature (rejected by server)
# Metrics are saved to ./qsuyo_metrics.json on the host.
```

#### Track 1: Algorithmic Evaluation
```bash
# Generate ALIE malicious models
docker compose --profile track1 run attack-gen

# Run defense validation
docker compose --profile track1 run defense-test
```

#### Useful Docker Commands
```bash
# Rebuild after code changes (no cache)
docker compose build --no-cache

# View server logs
docker compose logs -f fl-server

# Clean up all containers and networks
docker compose down -v
```

---

### Option B: Local (without Docker)

#### Track 1: Automated Algorithmic Evaluation
Ensure the official `challenge_starter` directory is mounted adjacent to this repository.
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

#### Track 2: Quantum-Secure FL Simulation
```bash
# Install framework dependencies
pip install -r requirements.txt

# Boot the PQC-enabled aggregation server
python src/server.py

# In a separate process, spawn a PQC-signing client
python src/client.py --malicious
```

*This repository contains functional code that satisfies the Phase I algorithmic baseline while providing post-quantum cryptographic infrastructure for Phase II evaluation.*
