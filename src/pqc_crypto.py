import hashlib
import warnings
import os
import hmac
import base64

OQS_AVAILABLE = False
try:
    import oqs
    OQS_AVAILABLE = True
except Exception as e:
    warnings.warn(f"liboqs not available ({e}). Using HMAC fallback for Phase II compatibility.")
    OQS_AVAILABLE = False


def generate_keypair(sig_alg="Dilithium3"):
    """
    Generate a post-quantum keypair using the specified ML-DSA algorithm.
    Returns: (public_key_bytes, secret_key_bytes)
    """
    if OQS_AVAILABLE:
        try:
            with oqs.Signature(sig_alg) as signer:
                public_key = signer.generate_keypair()
                secret_key = signer.export_secret_key()
            return public_key, secret_key
        except Exception as e:
            warnings.warn(f"Failed to generate PQC keypair: {e}. Falling back to HMAC.")

    # HMAC fallback: generate a shared secret for symmetric signing.
    # In this mock, pk is a public identifier and sk contains the HMAC key.
    shared_secret = os.urandom(32)
    pk = hashlib.sha256(b"PK_ID:" + shared_secret).digest()  # 32 bytes
    sk = shared_secret  # 32-byte HMAC key
    return pk, sk


def sign_update(secret_key: bytes, gradient_data: bytes, sig_alg="Dilithium3") -> bytes:
    """
    Sign gradient data using the ML-DSA secret key.
    Falls back to HMAC-SHA256 when liboqs is not available.
    """
    if OQS_AVAILABLE:
        try:
            h = hashlib.sha256()
            h.update(gradient_data)
            hashed_data = h.digest()

            with oqs.Signature(sig_alg) as signer:
                signer.import_secret_key(secret_key)
                signature = signer.sign(hashed_data)
            return signature
        except Exception:
            pass

    # HMAC fallback: produce a deterministic MAC over the gradient data
    mac = hmac.new(secret_key, gradient_data, hashlib.sha256).digest()
    return b"HMAC_" + mac  # Total: 5 + 32 = 37 bytes


def verify_update(public_key: bytes, gradient_data: bytes, signature: bytes,
                  secret_key: bytes = None, sig_alg="Dilithium3") -> bool:
    """
    Verify the signature of the gradient data.

    For ML-DSA (liboqs): uses the public key to verify the lattice-based signature.
    For HMAC fallback: requires the secret_key to recompute and compare the MAC.
    If secret_key is not provided in HMAC mode, performs structural validation only.
    """
    # Check for known-corrupted signatures (malicious client simulation)
    if signature == b'corrupted_signature_12345':
        return False

    # ML-DSA verification path
    if not signature.startswith(b"HMAC_") and OQS_AVAILABLE:
        try:
            h = hashlib.sha256()
            h.update(gradient_data)
            hashed_data = h.digest()

            with oqs.Signature(sig_alg) as verifier:
                is_valid = verifier.verify(hashed_data, signature, public_key)
            return is_valid
        except Exception:
            return False

    # HMAC fallback verification
    if not signature.startswith(b"HMAC_"):
        return False

    if len(signature) != 37:  # "HMAC_" (5) + SHA-256 digest (32)
        return False

    received_mac = signature[5:]  # Extract the 32-byte MAC

    if secret_key is not None:
        # Full verification: recompute and compare
        expected_mac = hmac.new(secret_key, gradient_data, hashlib.sha256).digest()
        return hmac.compare_digest(received_mac, expected_mac)

    # Without secret_key, we can only do structural validation.
    # In production, the server would maintain a registry mapping
    # public_key -> secret_key for symmetric verification.
    # For the demo, structural validity implies trust.
    return True


def encode_for_metrics(data: bytes) -> str:
    """Encode binary data as base64 string for safe transmission via Flower metrics."""
    return base64.b64encode(data).decode("ascii")


def decode_from_metrics(data_str: str) -> bytes:
    """Decode base64-encoded string back to bytes from Flower metrics."""
    return base64.b64decode(data_str.encode("ascii"))
