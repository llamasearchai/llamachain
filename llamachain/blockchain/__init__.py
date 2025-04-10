"""
Blockchain module for the LlamaChain platform.

This module provides interfaces for interacting with various blockchain networks,
including Ethereum and Solana.
"""

from llamachain.blockchain.base import BlockchainBase
from llamachain.blockchain.registry import BlockchainRegistry, register_default_providers, close_all_connections

__all__ = [
    "BlockchainBase",
    "BlockchainRegistry",
    "register_default_providers",
    "close_all_connections",
] 