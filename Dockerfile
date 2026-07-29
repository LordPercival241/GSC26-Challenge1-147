# ======================================================================
# Q-Suyo-Brain Framework — Multi-stage Docker build
# Stage 1: Build liboqs from source (required for ML-DSA / Dilithium3)
# Stage 2: Slim Python runtime with all project dependencies
# ======================================================================

# ---------- Stage 1: Build liboqs C library ----------
# Python 3.13 to match the official IEEE evaluator environment (CPython 3.13).
FROM python:3.13-slim AS liboqs-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git pkg-config \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch 0.12.0 \
        https://github.com/open-quantum-safe/liboqs.git /tmp/liboqs \
    && cmake -S /tmp/liboqs -B /tmp/liboqs/build \
        -GNinja \
        -DCMAKE_INSTALL_PREFIX=/opt/liboqs \
        -DBUILD_SHARED_LIBS=ON \
        -DOQS_BUILD_ONLY_LIB=ON \
    && cmake --build /tmp/liboqs/build \
    && cmake --install /tmp/liboqs/build \
    && rm -rf /tmp/liboqs

# ---------- Stage 2: Application runtime ----------
FROM python:3.13-slim AS runtime

LABEL maintainer="Team 147 — Q-Suyo-Brain Framework" \
      description="GSC26 Challenge 1: Secure Federated Learning with PQC"

# Install minimal runtime dependencies for liboqs and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled liboqs library
COPY --from=liboqs-builder /opt/liboqs /opt/liboqs
ENV LD_LIBRARY_PATH="/opt/liboqs/lib:/opt/liboqs/lib64:/opt/liboqs/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
RUN echo "/opt/liboqs/lib" > /etc/ld.so.conf.d/liboqs.conf \
    && echo "/opt/liboqs/lib64" >> /etc/ld.so.conf.d/liboqs.conf \
    && cp -P /opt/liboqs/lib*/* /usr/lib/ 2>/dev/null || true \
    && ldconfig

WORKDIR /app

# Install Python dependencies (cached layer unless requirements.txt changes)
# torch is installed from the CPU wheel index; the pinned version in
# requirements.txt keeps the build reproducible.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt

# Create a non-root user and give it ownership of the app directory.
RUN groupadd --system qsuyo \
    && useradd --system --gid qsuyo --home-dir /app --no-create-home qsuyo \
    && chown -R qsuyo:qsuyo /app

# Copy project source code with correct ownership
COPY --chown=qsuyo:qsuyo . .

# Drop privileges: never run the workload as root.
USER qsuyo

# Default: show version
CMD ["python", "--version"]
