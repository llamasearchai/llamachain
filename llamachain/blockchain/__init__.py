"""
Blockchain module for the LlamaChain platform.

This module provides interfaces for interacting with various blockchain networks,
including Ethereum and Solana.
"""

from llamachain.blockchain.base import BlockchainBase
from llamachain.blockchain.registry import (
    BlockchainRegistry,
    close_all_connections,
    register_default_providers,
)

__all__ = [
    "BlockchainBase",
    "BlockchainRegistry",
    "register_default_providers",
    "close_all_connections",
]
