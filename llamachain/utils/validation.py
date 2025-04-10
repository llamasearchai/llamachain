"""
Validation utilities for the LlamaChain platform.

This module provides functions for validating blockchain data like addresses and transaction hashes.
"""

import re
from typing import Optional, Tuple, Union


def validate_address(address: str, blockchain: str = "ethereum") -> Tuple[bool, Optional[str]]:
    """
    Validate a blockchain address.
    
    Args:
        address: The address to validate
        blockchain: The blockchain type (ethereum, solana, etc.)
        
    Returns:
        A tuple of (is_valid, error_message)
    """
    if not address:
        return False, "Address cannot be empty"
    
    # Lowercase for case-insensitive comparison
    blockchain = blockchain.lower()
    
    if blockchain == "ethereum":
        # Ethereum address: 0x followed by 40 hex characters
        if not address.startswith("0x"):
            return False, "Ethereum address must start with '0x'"
        
        address_without_prefix = address[2:]
        
        if len(address_without_prefix) != 40:
            return False, "Ethereum address must be 42 characters long (including '0x')"
        
        if not re.match(r"^[0-9a-fA-F]{40}$", address_without_prefix):
            return False, "Ethereum address must contain only hexadecimal characters"
        
        # Additional validation could include checksum validation
        
        return True, None
        
    elif blockchain == "solana":
        # Solana addresses are base58-encoded strings typically 32-44 characters long
        if not re.match(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$", address):
            return False, "Invalid Solana address format"
        
        return True, None
    
    else:
        return False, f"Unsupported blockchain: {blockchain}"


def validate_tx_hash(tx_hash: str, blockchain: str = "ethereum") -> Tuple[bool, Optional[str]]:
    """
    Validate a transaction hash.
    
    Args:
        tx_hash: The transaction hash to validate
        blockchain: The blockchain type (ethereum, solana, etc.)
        
    Returns:
        A tuple of (is_valid, error_message)
    """
    if not tx_hash:
        return False, "Transaction hash cannot be empty"
    
    # Lowercase for case-insensitive comparison
    blockchain = blockchain.lower()
    
    if blockchain == "ethereum":
        # Ethereum transaction hash: 0x followed by 64 hex characters
        if not tx_hash.startswith("0x"):
            return False, "Ethereum transaction hash must start with '0x'"
        
        tx_hash_without_prefix = tx_hash[2:]
        
        if len(tx_hash_without_prefix) != 64:
            return False, "Ethereum transaction hash must be 66 characters long (including '0x')"
        
        if not re.match(r"^[0-9a-fA-F]{64}$", tx_hash_without_prefix):
            return False, "Ethereum transaction hash must contain only hexadecimal characters"
        
        return True, None
        
    elif blockchain == "solana":
        # Solana transaction signatures are base58-encoded strings typically 88 characters long
        if not re.match(r"^[1-9A-HJ-NP-Za-km-z]{88}$", tx_hash):
            return False, "Invalid Solana transaction signature format"
        
        return True, None
    
    else:
        return False, f"Unsupported blockchain: {blockchain}"


def validate_block_number(block_number: Union[int, str]) -> Tuple[bool, Optional[str]]:
    """
    Validate a block number.
    
    Args:
        block_number: The block number to validate
        
    Returns:
        A tuple of (is_valid, error_message)
    """
    try:
        # Convert to int if it's a string
        if isinstance(block_number, str):
            block_number = int(block_number)
        
        # Block numbers must be non-negative
        if block_number < 0:
            return False, "Block number must be non-negative"
        
        return True, None
    except ValueError:
        return False, "Block number must be a valid integer"


def validate_chain_id(chain_id: Union[int, str]) -> Tuple[bool, Optional[str]]:
    """
    Validate a chain ID.
    
    Args:
        chain_id: The chain ID to validate
        
    Returns:
        A tuple of (is_valid, error_message)
    """
    try:
        # Convert to int if it's a string
        if isinstance(chain_id, str):
            chain_id = int(chain_id)
        
        # Chain IDs must be positive
        if chain_id <= 0:
            return False, "Chain ID must be positive"
        
        return True, None
    except ValueError:
        return False, "Chain ID must be a valid integer" 