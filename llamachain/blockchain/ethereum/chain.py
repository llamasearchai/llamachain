from typing import Dict, List, Optional, Any, Union
import asyncio
from datetime import datetime
from web3 import Web3, AsyncWeb3
from web3.middleware import geth_poa_middleware
import json
from web3.exceptions import TransactionNotFound, BlockNotFound
from eth_typing import ChecksumAddress
import functools

from llamachain.blockchain.base import BlockchainBase
from llamachain.utils.config import settings


class EthereumChain(BlockchainBase):
    """Ethereum blockchain interface using web3.py"""
    
    def __init__(self, rpc_url: Optional[str] = None, ws_url: Optional[str] = None):
        """
        Initialize the Ethereum blockchain interface.
        
        Args:
            rpc_url: Ethereum HTTP RPC URL
            ws_url: Ethereum WebSocket URL (optional)
        """
        super().__init__(rpc_url, ws_url)
        
        self.rpc_url = rpc_url or settings.ETH_RPC_URL
        self.ws_url = ws_url or settings.ETH_WSS_URL
        
        self.web3 = None
        self.web3_ws = None
        self.is_poa = False
        self.connected = False
        
    async def connect(self) -> bool:
        """
        Connect to Ethereum nodes using HTTP and WebSocket (if available).
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Initialize AsyncWeb3 over HTTP
            self.web3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(self.rpc_url))
            
            # Try to detect if we're dealing with a PoA network
            # This middleware is important for networks like Polygon, Arbitrum, etc.
            try:
                # We need to check in a way that doesn't assume anything about the network
                # This needs to be executed in an async block
                is_connected = await self.web3.is_connected()
                
                if not is_connected:
                    self.logger.error(f"Failed to connect to Ethereum node at {self.rpc_url}")
                    return False
                
                # Try to get the latest block
                latest_block = await self.web3.eth.get_block('latest')
                
                # If extraData is present and has a specific size, it's likely a PoA network
                if 'extraData' in latest_block and len(latest_block['extraData']) >= 97:
                    self.logger.info("Detected PoA network, applying middleware")
                    # Apply PoA middleware for networks like Polygon, BSC, etc.
                    # Note: async doesn't have middleware manager yet, using sync version
                    self.web3 = Web3(Web3.HTTPProvider(self.rpc_url))
                    self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
                    self.is_poa = True
            except Exception as e:
                self.logger.warning(f"Error detecting PoA network: {e}")
            
            # Initialize WebSocket connection if available
            if self.ws_url:
                try:
                    self.web3_ws = AsyncWeb3(AsyncWeb3.AsyncWebsocketProvider(self.ws_url))
                    if await self.web3_ws.is_connected():
                        self.logger.info(f"Connected to Ethereum node via WebSocket: {self.ws_url}")
                    else:
                        self.logger.warning(f"Failed to connect to WebSocket: {self.ws_url}")
                        self.web3_ws = None
                except Exception as e:
                    self.logger.warning(f"Error connecting via WebSocket: {e}")
                    self.web3_ws = None
            
            # Check connection status
            if not self.is_poa:  # If not PoA, we keep using the async Web3
                is_connected = await self.web3.is_connected()
            else:  # If PoA, we switch to the sync Web3 with middleware
                is_connected = self.web3.is_connected()
            
            if is_connected:
                self.connected = True
                chain_id = await self.get_chain_id()
                latest_block = await self.get_latest_block_number()
                self.logger.info(f"Connected to Ethereum node (chain_id: {chain_id}, latest_block: {latest_block})")
                return True
            else:
                self.logger.error(f"Failed to connect to Ethereum node at {self.rpc_url}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to Ethereum node: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from Ethereum nodes.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            self.web3 = None
            self.web3_ws = None
            self.connected = False
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from Ethereum node: {e}")
            return False
    
    async def is_connected(self) -> bool:
        """
        Check if connected to the Ethereum node.
        
        Returns:
            True if connected, False otherwise
        """
        if self.web3 is None:
            return False
        
        try:
            if self.is_poa:
                return self.web3.is_connected()
            else:
                return await self.web3.is_connected()
        except Exception:
            return False
    
    async def get_block(self, block_identifier: Union[int, str]) -> Dict[str, Any]:
        """
        Retrieve a block by its number or hash.
        
        Args:
            block_identifier: Block number or hash
            
        Returns:
            Block data as a dictionary
        """
        try:
            if self.is_poa:
                # For PoA networks, use sync version with middleware
                block = self.web3.eth.get_block(block_identifier, full_transactions=True)
                return self._convert_block_data(dict(block))
            else:
                # For regular networks, use async version
                block = await self.web3.eth.get_block(block_identifier, full_transactions=True)
                return self._convert_block_data(dict(block))
        except BlockNotFound:
            self.logger.warning(f"Block not found: {block_identifier}")
            return {}
        except Exception as e:
            self.logger.error(f"Error retrieving block {block_identifier}: {e}")
            return {}
    
    async def get_latest_block(self) -> Dict[str, Any]:
        """
        Retrieve the latest block.
        
        Returns:
            Latest block data as a dictionary
        """
        return await self.get_block('latest')
    
    async def get_latest_block_number(self) -> int:
        """
        Get the latest block number.
        
        Returns:
            Latest block number
        """
        try:
            if self.is_poa:
                return self.web3.eth.block_number
            else:
                return await self.web3.eth.block_number
        except Exception as e:
            self.logger.error(f"Error retrieving latest block number: {e}")
            return 0
    
    async def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """
        Retrieve a transaction by its hash.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction data as a dictionary
        """
        try:
            if self.is_poa:
                tx = self.web3.eth.get_transaction(tx_hash)
                return self._convert_transaction_data(dict(tx))
            else:
                tx = await self.web3.eth.get_transaction(tx_hash)
                return self._convert_transaction_data(dict(tx))
        except TransactionNotFound:
            self.logger.warning(f"Transaction not found: {tx_hash}")
            return {}
        except Exception as e:
            self.logger.error(f"Error retrieving transaction {tx_hash}: {e}")
            return {}
    
    async def get_balance(self, address: str) -> int:
        """
        Get the balance of an address.
        
        Args:
            address: Ethereum address
            
        Returns:
            Balance in wei
        """
        try:
            # Ensure the address is checksummed
            checksummed_address = Web3.to_checksum_address(address)
            
            if self.is_poa:
                balance = self.web3.eth.get_balance(checksummed_address)
            else:
                balance = await self.web3.eth.get_balance(checksummed_address)
            
            return balance
        except Exception as e:
            self.logger.error(f"Error retrieving balance for {address}: {e}")
            return 0
    
    async def get_transaction_receipt(self, tx_hash: str) -> Dict[str, Any]:
        """
        Retrieve a transaction receipt.
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction receipt data as a dictionary
        """
        try:
            if self.is_poa:
                receipt = self.web3.eth.get_transaction_receipt(tx_hash)
                return self._convert_receipt_data(dict(receipt))
            else:
                receipt = await self.web3.eth.get_transaction_receipt(tx_hash)
                return self._convert_receipt_data(dict(receipt))
        except TransactionNotFound:
            self.logger.warning(f"Transaction receipt not found: {tx_hash}")
            return {}
        except Exception as e:
            self.logger.error(f"Error retrieving transaction receipt {tx_hash}: {e}")
            return {}
    
    async def get_contract_code(self, address: str) -> str:
        """
        Get the bytecode of a smart contract.
        
        Args:
            address: Contract address
            
        Returns:
            Contract bytecode as a hex string
        """
        try:
            # Ensure the address is checksummed
            checksummed_address = Web3.to_checksum_address(address)
            
            if self.is_poa:
                code = self.web3.eth.get_code(checksummed_address)
            else:
                code = await self.web3.eth.get_code(checksummed_address)
            
            return code.hex()
        except Exception as e:
            self.logger.error(f"Error retrieving contract code for {address}: {e}")
            return ""
    
    async def get_chain_id(self) -> int:
        """
        Get the chain ID.
        
        Returns:
            Chain ID
        """
        try:
            if self.is_poa:
                return self.web3.eth.chain_id
            else:
                return await self.web3.eth.chain_id
        except Exception as e:
            self.logger.error(f"Error retrieving chain ID: {e}")
            return 0
    
    async def get_gas_price(self) -> int:
        """
        Get the current gas price.
        
        Returns:
            Gas price in wei
        """
        try:
            if self.is_poa:
                return self.web3.eth.gas_price
            else:
                return await self.web3.eth.gas_price
        except Exception as e:
            self.logger.error(f"Error retrieving gas price: {e}")
            return 0
    
    async def send_raw_transaction(self, tx_hex: str) -> str:
        """
        Send a raw transaction.
        
        Args:
            tx_hex: Raw transaction hex
            
        Returns:
            Transaction hash
        """
        try:
            if self.is_poa:
                tx_hash = self.web3.eth.send_raw_transaction(tx_hex)
            else:
                tx_hash = await self.web3.eth.send_raw_transaction(tx_hex)
            
            return tx_hash.hex()
        except Exception as e:
            self.logger.error(f"Error sending raw transaction: {e}")
            return ""
    
    async def estimate_gas(self, tx: Dict[str, Any]) -> int:
        """
        Estimate gas for a transaction.
        
        Args:
            tx: Transaction data
            
        Returns:
            Estimated gas
        """
        try:
            if self.is_poa:
                return self.web3.eth.estimate_gas(tx)
            else:
                return await self.web3.eth.estimate_gas(tx)
        except Exception as e:
            self.logger.error(f"Error estimating gas: {e}")
            return 0
    
    async def call_contract(self, address: str, function_signature: str, args: List[Any]) -> Any:
        """
        Call a contract function.
        
        Args:
            address: Contract address
            function_signature: Function signature (e.g., "balanceOf(address)")
            args: Function arguments
            
        Returns:
            Function result
        """
        try:
            # Ensure the address is checksummed
            checksummed_address = Web3.to_checksum_address(address)
            
            # Create function selector
            function_selector = Web3.keccak(text=function_signature)[:4].hex()
            
            # Encode arguments (simplified - in real world you'd use proper ABI encoding)
            encoded_args = ""  # This is simplified, actual encoding depends on argument types
            
            # Create call data
            call_data = function_selector + encoded_args
            
            # Create transaction object
            tx = {
                'to': checksummed_address,
                'data': call_data,
            }
            
            # Call the contract
            if self.is_poa:
                result = self.web3.eth.call(tx)
            else:
                result = await self.web3.eth.call(tx)
            
            return result.hex()
        except Exception as e:
            self.logger.error(f"Error calling contract {address}.{function_signature}: {e}")
            return None
    
    def _convert_block_data(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """Convert web3.py block data to standardized format"""
        if not block:
            return {}
        
        # Convert binary types to hex strings
        for key in ['hash', 'parentHash', 'sha3Uncles', 'stateRoot', 'transactionsRoot', 
                   'receiptsRoot', 'logsBloom', 'extraData', 'mixHash', 'nonce']:
            if key in block and block[key] and hasattr(block[key], 'hex'):
                block[key] = block[key].hex()
        
        # Convert transactions to standardized format if they are full transactions
        if 'transactions' in block:
            if block['transactions'] and isinstance(block['transactions'][0], dict):
                block['transactions'] = [self._convert_transaction_data(tx) for tx in block['transactions']]
            else:
                # Only transaction hashes, convert to hex strings
                block['transactions'] = [tx.hex() if hasattr(tx, 'hex') else tx for tx in block['transactions']]
        
        return block
    
    def _convert_transaction_data(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Convert web3.py transaction data to standardized format"""
        if not tx:
            return {}
        
        # Convert binary types to hex strings
        for key in ['hash', 'blockHash', 'r', 's', 'v']:
            if key in tx and tx[key] and hasattr(tx[key], 'hex'):
                tx[key] = tx[key].hex()
        
        return tx
    
    def _convert_receipt_data(self, receipt: Dict[str, Any]) -> Dict[str, Any]:
        """Convert web3.py receipt data to standardized format"""
        if not receipt:
            return {}
        
        # Convert binary types to hex strings
        for key in ['transactionHash', 'blockHash', 'logsBloom']:
            if key in receipt and receipt[key] and hasattr(receipt[key], 'hex'):
                receipt[key] = receipt[key].hex()
        
        # Convert logs to standardized format
        if 'logs' in receipt and receipt['logs']:
            standardized_logs = []
            for log in receipt['logs']:
                log_dict = dict(log)
                for key in ['transactionHash', 'blockHash', 'data']:
                    if key in log_dict and log_dict[key] and hasattr(log_dict[key], 'hex'):
                        log_dict[key] = log_dict[key].hex()
                
                # Convert topics to hex strings
                if 'topics' in log_dict and log_dict['topics']:
                    log_dict['topics'] = [t.hex() if hasattr(t, 'hex') else t for t in log_dict['topics']]
                
                standardized_logs.append(log_dict)
            
            receipt['logs'] = standardized_logs
        
        return receipt 