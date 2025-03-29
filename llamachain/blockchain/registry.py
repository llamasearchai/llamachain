"""
Blockchain registry for managing blockchain providers.
"""

import logging
from typing import Dict, List, Optional, Type, Any

from llamachain.blockchain.base import BlockchainBase
from llamachain.log import get_logger

# Setup logger
logger = get_logger("llamachain.blockchain.registry")


class BlockchainRegistry:
    """
    Registry for blockchain providers.
    
    This class manages the available blockchain providers and provides
    methods for registering, retrieving, and managing blockchain instances.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(BlockchainRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return
        
        self._providers: Dict[str, Type[BlockchainBase]] = {}
        self._instances: Dict[str, BlockchainBase] = {}
        self._initialized = True
        
        logger.debug("BlockchainRegistry initialized")
    
    def register_provider(self, chain_id: str, provider_class: Type[BlockchainBase]) -> None:
        """
        Register a blockchain provider.
        
        Args:
            chain_id: The chain identifier (e.g., 'ethereum', 'solana')
            provider_class: The blockchain provider class
        """
        if chain_id in self._providers:
            logger.warning(f"Overriding existing provider for chain: {chain_id}")
        
        self._providers[chain_id] = provider_class
        logger.info(f"Registered provider for chain: {chain_id}")
    
    def get_provider(self, chain_id: str) -> Optional[Type[BlockchainBase]]:
        """
        Get a blockchain provider class.
        
        Args:
            chain_id: The chain identifier
            
        Returns:
            The blockchain provider class, or None if not found
        """
        return self._providers.get(chain_id)
    
    def get_available_chains(self) -> List[str]:
        """
        Get a list of available blockchain chains.
        
        Returns:
            List of chain identifiers
        """
        return list(self._providers.keys())
    
    async def get_chain(self, chain_id: str, **kwargs: Any) -> BlockchainBase:
        """
        Get a blockchain instance.
        
        If an instance for the specified chain already exists, it will be returned.
        Otherwise, a new instance will be created and initialized.
        
        Args:
            chain_id: The chain identifier
            **kwargs: Additional arguments to pass to the provider constructor
            
        Returns:
            A blockchain instance
            
        Raises:
            ValueError: If the chain is not supported
        """
        # Check if we already have an instance
        if chain_id in self._instances:
            return self._instances[chain_id]
        
        # Get the provider class
        provider_class = self.get_provider(chain_id)
        if provider_class is None:
            raise ValueError(f"Unsupported blockchain: {chain_id}")
        
        # Create a new instance
        instance = provider_class(**kwargs)
        
        # Initialize the instance
        try:
            await instance.connect()
            logger.info(f"Connected to blockchain: {chain_id}")
        except Exception as e:
            logger.error(f"Failed to connect to blockchain {chain_id}: {e}")
            raise
        
        # Store the instance
        self._instances[chain_id] = instance
        
        return instance
    
    async def close_connection(self, chain_id: str) -> None:
        """
        Close a blockchain connection.
        
        Args:
            chain_id: The chain identifier
        """
        if chain_id in self._instances:
            try:
                await self._instances[chain_id].disconnect()
                logger.info(f"Disconnected from blockchain: {chain_id}")
            except Exception as e:
                logger.error(f"Error disconnecting from blockchain {chain_id}: {e}")
            
            del self._instances[chain_id]


def register_default_providers() -> None:
    """Register default blockchain providers."""
    registry = BlockchainRegistry()
    
    try:
        # Import and register Ethereum provider
        from llamachain.blockchain.ethereum import EthereumChain
        registry.register_provider("ethereum", EthereumChain)
        
        # Import and register Solana provider
        from llamachain.blockchain.solana import SolanaChain
        registry.register_provider("solana", SolanaChain)
        
        logger.info("Default blockchain providers registered")
    except ImportError as e:
        logger.error(f"Failed to import blockchain providers: {e}")


async def close_all_connections() -> None:
    """Close all blockchain connections."""
    registry = BlockchainRegistry()
    
    for chain_id in registry.get_available_chains():
        await registry.close_connection(chain_id) 