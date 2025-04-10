"""
Cryptographic utilities for the LlamaChain platform.

This module provides functions for cryptographic operations like token generation and password hashing.
"""

import os
import uuid
import base64
import hashlib
import hmac
import secrets
from typing import Optional, Tuple

# Use bcrypt for password hashing if available
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Length of the token in bytes
        
    Returns:
        Base64-encoded token string
    """
    # Generate random bytes
    random_bytes = secrets.token_bytes(length)
    
    # Encode as URL-safe base64
    token = base64.urlsafe_b64encode(random_bytes).decode('utf-8').rstrip('=')
    
    return token


def generate_api_key() -> str:
    """
    Generate an API key in the format "lc_XXXXXX".
    
    Returns:
        API key string
    """
    # Generate a UUID4 and remove hyphens
    uuid_part = str(uuid.uuid4()).replace('-', '')
    
    # Create the API key with a prefix
    api_key = f"lc_{uuid_part}"
    
    return api_key


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt if available, otherwise using PBKDF2.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    if not password:
        raise ValueError("Password cannot be empty")
    
    if BCRYPT_AVAILABLE:
        # Use bcrypt for password hashing (best practice)
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode('utf-8')
    else:
        # Fallback to PBKDF2 if bcrypt is not available
        salt = os.urandom(16)
        iterations = 100000  # Recommended iterations for PBKDF2
        hash_bytes = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
        
        # Store iteration count and salt with the hash
        salt_hex = salt.hex()
        hash_hex = hash_bytes.hex()
        return f"pbkdf2:sha256:{iterations}${salt_hex}${hash_hex}"


def verify_password(stored_password: str, provided_password: str) -> bool:
    """
    Verify a password against its stored hash.
    
    Args:
        stored_password: The stored hashed password
        provided_password: The plain text password to verify
        
    Returns:
        True if the password matches, False otherwise
    """
    if not stored_password or not provided_password:
        return False
    
    if BCRYPT_AVAILABLE and stored_password.startswith('$2'):
        # bcrypt hash
        try:
            return bcrypt.checkpw(
                provided_password.encode('utf-8'), 
                stored_password.encode('utf-8')
            )
        except Exception:
            return False
    elif stored_password.startswith('pbkdf2:'):
        # PBKDF2 hash fallback
        try:
            # Parse the stored hash
            algorithm_info, salt_hex, hash_hex = stored_password.split('$', 2)
            _, algorithm, iterations = algorithm_info.split(':')
            iterations = int(iterations)
            
            # Convert hex to bytes
            salt = bytes.fromhex(salt_hex)
            stored_hash = bytes.fromhex(hash_hex)
            
            # Hash the provided password with the same salt and iterations
            hash_bytes = hashlib.pbkdf2_hmac(
                algorithm, 
                provided_password.encode('utf-8'), 
                salt, 
                iterations
            )
            
            # Compare in constant time to prevent timing attacks
            return hmac.compare_digest(stored_hash, hash_bytes)
        except Exception:
            return False
    else:
        # Unknown hash format
        return False


def encrypt_sensitive_data(data: str, key: Optional[str] = None) -> Tuple[str, str]:
    """
    Encrypt sensitive data with AES-256.
    
    Args:
        data: The data to encrypt
        key: Optional encryption key (will generate one if not provided)
        
    Returns:
        Tuple of (encrypted_data, key)
    """
    try:
        from cryptography.fernet import Fernet
        
        # Generate a key if not provided
        if not key:
            key = Fernet.generate_key().decode('utf-8')
        elif not key.startswith('LMCN'):
            # Convert string key to Fernet key
            key_bytes = key.encode('utf-8')
            key_hash = hashlib.sha256(key_bytes).digest()
            key = base64.urlsafe_b64encode(key_hash).decode('utf-8')
        
        # Create a Fernet instance
        f = Fernet(key.encode('utf-8') if isinstance(key, str) else key)
        
        # Encrypt the data
        encrypted_data = f.encrypt(data.encode('utf-8')).decode('utf-8')
        
        return encrypted_data, key
    except ImportError:
        raise ImportError("cryptography package is required for encryption")


def decrypt_sensitive_data(encrypted_data: str, key: str) -> str:
    """
    Decrypt sensitive data with AES-256.
    
    Args:
        encrypted_data: The encrypted data
        key: The encryption key
        
    Returns:
        Decrypted data
    """
    try:
        from cryptography.fernet import Fernet
        
        # Create a Fernet instance
        f = Fernet(key.encode('utf-8') if isinstance(key, str) else key)
        
        # Decrypt the data
        decrypted_data = f.decrypt(encrypted_data.encode('utf-8')).decode('utf-8')
        
        return decrypted_data
    except ImportError:
        raise ImportError("cryptography package is required for decryption")
    except Exception as e:
        raise ValueError(f"Decryption failed: {str(e)}")


def compute_signature(payload: str, secret: str) -> str:
    """
    Compute HMAC-SHA256 signature for a payload.
    
    Args:
        payload: The payload to sign
        secret: The secret key
        
    Returns:
        Signature as a hex string
    """
    signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return signature


def verify_signature(payload: str, signature: str, secret: str) -> bool:
    """
    Verify HMAC-SHA256 signature for a payload.
    
    Args:
        payload: The payload that was signed
        signature: The provided signature
        secret: The secret key
        
    Returns:
        True if signature is valid, False otherwise
    """
    expected_signature = compute_signature(payload, secret)
    
    # Compare in constant time to prevent timing attacks
    return hmac.compare_digest(signature, expected_signature) 