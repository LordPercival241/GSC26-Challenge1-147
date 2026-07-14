import oqs
import hashlib

def generate_keypair(sig_alg="Dilithium3"):
    """
    Generate a post-quantum keypair using the specified ML-DSA algorithm.
    Returns: (public_key_bytes, secret_key_bytes)
    """
    with oqs.Signature(sig_alg) as signer:
        public_key = signer.generate_keypair()
        secret_key = signer.export_secret_key()
    return public_key, secret_key

def sign_update(secret_key: bytes, gradient_data: bytes, sig_alg="Dilithium3") -> bytes:
    """
    Sign gradient data using the ML-DSA secret key.
    To improve performance over large tensors, we sign the hash of the gradients.
    """
    h = hashlib.sha256()
    h.update(gradient_data)
    hashed_data = h.digest()
    
    with oqs.Signature(sig_alg) as signer:
        signer.import_secret_key(secret_key)
        signature = signer.sign(hashed_data)
    return signature

def verify_update(public_key: bytes, gradient_data: bytes, signature: bytes, sig_alg="Dilithium3") -> bool:
    """
    Verify the signature of the gradient data using the ML-DSA public key.
    """
    h = hashlib.sha256()
    h.update(gradient_data)
    hashed_data = h.digest()
    
    with oqs.Signature(sig_alg) as verifier:
        is_valid = verifier.verify(hashed_data, signature, public_key)
    return is_valid
