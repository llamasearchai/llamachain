"""
Base class for blockchain implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime


class BlockchainBase(ABC):
    """
    Abstract base class for blockchain implementations.
    
    This class defines the interface that all blockchain implementations must follow.
    """
    
    def __init__(self, rpc_url: Optional[str] = None, ws_url: Optional[str] = None):
        """
        Initialize the blockchain interface.
        
        Args:
            rpc_url: URL for the RPC endpoint
            ws_url: URL for the WebSocket endpoint (optional)
        """
        self.rpc_url = rpc_url
        self.ws_url = ws_url
        self.logger = logging.getLogger(f"blockchain.{self.__class__.__name__.lower()}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the blockchain node.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the blockchain node.
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if connected to the blockchain node.
        
        Returns:
            bool: True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_block(self, block_identifier: Union[int, str]) -> Dict[str, Any]:
        """
        Get block information.
        
        Args:
            block_identifier: Block number or hash
            
        Returns:
            Dict with block information
            
        Raises:
            ValueError: If block not found
        """
        pass
    
    @abstractmethod
    async def get_latest_block(self) -> Dict[str, Any]:
        """
        Get the latest block.
        
        Returns:
            Dict with block information
        """
        pass
    
    @abstractmethod
    async def get_latest_block_number(self) -> int:
        """
        Get the latest block number/height.
        
        Returns:
            Latest block number/height
        """
        pass
    
    @abstractmethod
    async def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """
        Get transaction information.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Dict with transaction information
            
        Raises:
            ValueError: If transaction not found
        """
        pass
    
    @abstractmethod
    async def get_transaction_receipt(self, tx_hash: str) -> Dict[str, Any]:
        """
        Get transaction receipt.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Dict with transaction receipt information
            
        Raises:
            ValueError: If transaction not found
        """
        pass
    
    @abstractmethod
    async def get_balance(self, address: str) -> int:
        """
        Get account balance.
        
        Args:
            address: Account address
            
        Returns:
            Account balance in the smallest unit (wei, lamports, etc.)
            
        Raises:
            ValueError: If address is invalid
        """
        pass
    
    @abstractmethod
    async def get_contract_code(self, address: str) -> str:
        """
        Get contract bytecode.
        
        Args:
            address: Contract address
            
        Returns:
            Contract bytecode as a hex string
            
        Raises:
            ValueError: If address is not a contract or is invalid
        """
        pass
    
    @abstractmethod
    async def get_chain_id(self) -> Union[int, str]:
        """
        Get chain ID.
        
        Returns:
            Chain ID (integer for Ethereum, string for Solana)
        """
        pass
    
    @abstractmethod
    async def get_chain_name(self) -> str:
        """
        Get chain name.
        
        Returns:
            Chain name (e.g., "Ethereum Mainnet", "Solana Mainnet")
        """
        pass
    
    @abstractmethod
    async def get_gas_price(self) -> int:
        """
        Get current gas price.
        
        Returns:
            Gas price in the smallest unit (wei, lamports, etc.)
        """
        pass
    
    @abstractmethod
    async def send_raw_transaction(self, signed_tx: str) -> str:
        """
        Send a raw transaction.
        
        Args:
            signed_tx: Signed transaction data as a hex string
            
        Returns:
            Transaction hash
            
        Raises:
            ValueError: If transaction is invalid
        """
        pass
    
    @abstractmethod
    async def estimate_gas(self, tx_params: Dict[str, Any]) -> int:
        """
        Estimate gas for a transaction.
        
        Args:
            tx_params: Transaction parameters
            
        Returns:
            Estimated gas amount
            
        Raises:
            ValueError: If transaction parameters are invalid
        """
        pass 