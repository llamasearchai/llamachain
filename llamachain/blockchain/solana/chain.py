from typing import Dict, List, Optional, Any, Union
import asyncio
import base64
from datetime import datetime
import aiohttp
import json
import logging

from llamachain.blockchain.base import BlockchainBase
from llamachain.utils.config import settings


class SolanaChain(BlockchainBase):
    """Solana blockchain interface using JSON RPC"""
    
    def __init__(self, rpc_url: Optional[str] = None, ws_url: Optional[str] = None):
        """
        Initialize the Solana blockchain interface.
        
        Args:
            rpc_url: Solana HTTP RPC URL
            ws_url: Solana WebSocket URL (optional)
        """
        super().__init__(rpc_url, ws_url)
        
        self.rpc_url = rpc_url or settings.SOL_RPC_URL
        self.ws_url = ws_url
        
        self.session = None
        self.connected = False
        self.commitment = "confirmed"  # Default commitment level
    
    async def connect(self) -> bool:
        """
        Connect to Solana node.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.session = aiohttp.ClientSession()
            
            # Test the connection by requesting the version
            version = await self._send_request("getVersion", [])
            
            if version and "result" in version:
                self.connected = True
                self.logger.info(f"Connected to Solana node: {version['result']}")
                return True
            else:
                self.logger.error(f"Failed to connect to Solana node at {self.rpc_url}")
                return False
        except Exception as e:
            self.logger.error(f"Error connecting to Solana node: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from Solana node.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.connected = False
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from Solana node: {e}")
            return False
    
    async def is_connected(self) -> bool:
        """
        Check if connected to the Solana node.
        
        Returns:
            True if connected, False otherwise
        """
        if not self.session:
            return False
        
        try:
            # Test the connection by requesting the version
            version = await self._send_request("getVersion", [])
            return version is not None and "result" in version
        except Exception:
            return False
    
    async def get_block(self, block_identifier: Union[int, str]) -> Dict[str, Any]:
        """
        Retrieve a block by its number (slot).
        
        Args:
            block_identifier: Block number (slot)
            
        Returns:
            Block data as a dictionary
        """
        try:
            # For Solana, block_identifier is the slot number
            slot = int(block_identifier) if isinstance(block_identifier, str) else block_identifier
            
            # Get the block data
            params = [
                slot,
                {
                    "encoding": "json",
                    "transactionDetails": "full",
                    "rewards": True,
                    "commitment": self.commitment
                }
            ]
            
            response = await self._send_request("getBlock", params)
            
            if response and "result" in response:
                return response["result"]
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                self.logger.warning(f"Block not found or error: {error_msg}")
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
        try:
            # Get the latest slot
            latest_slot = await self.get_latest_slot()
            
            # Get the block data for the latest slot
            return await self.get_block(latest_slot)
        except Exception as e:
            self.logger.error(f"Error retrieving latest block: {e}")
            return {}
    
    async def get_latest_block_number(self) -> int:
        """
        Get the latest block number (slot).
        
        Returns:
            Latest block number (slot)
        """
        return await self.get_latest_slot()
    
    async def get_latest_slot(self) -> int:
        """
        Get the latest slot.
        
        Returns:
            Latest slot
        """
        try:
            params = [{"commitment": self.commitment}]
            response = await self._send_request("getSlot", params)
            
            if response and "result" in response:
                return response["result"]
            else:
                self.logger.warning("Error getting latest slot")
                return 0
        except Exception as e:
            self.logger.error(f"Error retrieving latest slot: {e}")
            return 0
    
    async def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """
        Retrieve a transaction by its hash (signature).
        
        Args:
            tx_hash: Transaction hash (signature)
            
        Returns:
            Transaction data as a dictionary
        """
        try:
            params = [
                tx_hash,
                {
                    "encoding": "json",
                    "commitment": self.commitment
                }
            ]
            
            response = await self._send_request("getTransaction", params)
            
            if response and "result" in response:
                return response["result"]
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                self.logger.warning(f"Transaction not found or error: {error_msg}")
                return {}
        except Exception as e:
            self.logger.error(f"Error retrieving transaction {tx_hash}: {e}")
            return {}
    
    async def get_balance(self, address: str) -> int:
        """
        Get the balance of an address.
        
        Args:
            address: Solana address
            
        Returns:
            Balance in lamports
        """
        try:
            params = [
                address,
                {
                    "commitment": self.commitment
                }
            ]
            
            response = await self._send_request("getBalance", params)
            
            if response and "result" in response:
                return response["result"]["value"]
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                self.logger.warning(f"Error getting balance: {error_msg}")
                return 0
        except Exception as e:
            self.logger.error(f"Error retrieving balance for {address}: {e}")
            return 0
    
    async def get_transaction_receipt(self, tx_hash: str) -> Dict[str, Any]:
        """
        Retrieve a transaction receipt. For Solana, this is the same as getTransaction.
        
        Args:
            tx_hash: Transaction hash (signature)
            
        Returns:
            Transaction receipt data as a dictionary
        """
        return await self.get_transaction(tx_hash)
    
    async def get_account_info(self, address: str) -> Dict[str, Any]:
        """
        Get information about an account.
        
        Args:
            address: Solana address
            
        Returns:
            Account information as a dictionary
        """
        try:
            params = [
                address,
                {
                    "encoding": "jsonParsed",
                    "commitment": self.commitment
                }
            ]
            
            response = await self._send_request("getAccountInfo", params)
            
            if response and "result" in response:
                return response["result"]["value"]
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                self.logger.warning(f"Error getting account info: {error_msg}")
                return {}
        except Exception as e:
            self.logger.error(f"Error retrieving account info for {address}: {e}")
            return {}
    
    async def get_contract_code(self, address: str) -> str:
        """
        Get the bytecode of a smart contract (program).
        For Solana, this is the account data.
        
        Args:
            address: Program address
            
        Returns:
            Contract bytecode as a base64 string
        """
        try:
            params = [
                address,
                {
                    "encoding": "base64",
                    "commitment": self.commitment
                }
            ]
            
            response = await self._send_request("getAccountInfo", params)
            
            if response and "result" in response and "value" in response["result"]:
                data = response["result"]["value"].get("data", ["", ""])
                if isinstance(data, list) and len(data) >= 2:
                    return data[0]
                else:
                    return data
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                self.logger.warning(f"Error getting program data: {error_msg}")
                return ""
        except Exception as e:
            self.logger.error(f"Error retrieving program data for {address}: {e}")
            return ""
    
    async def get_chain_id(self) -> int:
        """
        Get the chain ID.
        For Solana, this is a constant value depending on the network.
        
        Returns:
            Chain ID: 101 for mainnet, 102 for testnet, 103 for devnet
        """
        try:
            # For Solana, we use the genesis hash to identify the network
            params = []
            response = await self._send_request("getGenesisHash", params)
            
            if response and "result" in response:
                # Map genesis hash to network ID
                genesis_hash = response["result"]
                
                # These values are examples, use actual genesis hashes
                if genesis_hash == "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d":
                    return 101  # mainnet
                elif genesis_hash == "4uhcVJyU9pJkvQyS88uRDiswHXSCkY3zQawwpjk2NsNY":
                    return 102  # testnet
                else:
                    return 103  # devnet
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                self.logger.warning(f"Error getting genesis hash: {error_msg}")
                return 0
        except Exception as e:
            self.logger.error(f"Error retrieving chain ID: {e}")
            return 0
    
    async def get_gas_price(self) -> int:
        """
        Get the current gas price (minimum rent-exempt balance).
        
        Returns:
            Gas price in lamports
        """
        try:
            # For Solana, we return the minimum rent-exempt balance for a small account
            params = [
                100,  # Size of 100 bytes
                {
                    "commitment": self.commitment
                }
            ]
            
            response = await self._send_request("getMinimumBalanceForRentExemption", params)
            
            if response and "result" in response:
                return response["result"]
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                self.logger.warning(f"Error getting minimum rent: {error_msg}")
                return 0
        except Exception as e:
            self.logger.error(f"Error retrieving gas price: {e}")
            return 0
    
    async def send_raw_transaction(self, tx_hex: str) -> str:
        """
        Send a raw transaction.
        
        Args:
            tx_hex: Raw transaction as a base64-encoded string
            
        Returns:
            Transaction signature (hash)
        """
        try:
            params = [
                tx_hex,
                {
                    "encoding": "base64",
                    "commitment": self.commitment
                }
            ]
            
            response = await self._send_request("sendTransaction", params)
            
            if response and "result" in response:
                return response["result"]
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                self.logger.warning(f"Error sending transaction: {error_msg}")
                return ""
        except Exception as e:
            self.logger.error(f"Error sending raw transaction: {e}")
            return ""
    
    async def estimate_gas(self, tx: Dict[str, Any]) -> int:
        """
        Estimate gas for a transaction.
        For Solana, we just return a fixed value for now.
        
        Args:
            tx: Transaction data
            
        Returns:
            Estimated gas (fees in lamports)
        """
        return 5000  # Default fee for Solana transactions
    
    async def call_contract(self, address: str, function_signature: str, args: List[Any]) -> Any:
        """
        Call a contract function.
        For Solana, this is a program invocation.
        
        Args:
            address: Program address
            function_signature: Function signature
            args: Function arguments
            
        Returns:
            Function result
        """
        try:
            # For Solana, this needs to be implemented with specific program calls
            # This is a minimal implementation
            self.logger.warning("Contract call not fully implemented for Solana")
            return None
        except Exception as e:
            self.logger.error(f"Error calling program {address}.{function_signature}: {e}")
            return None
    
    async def get_token_accounts(self, address: str) -> List[Dict[str, Any]]:
        """
        Get token accounts owned by an address.
        
        Args:
            address: Solana address
            
        Returns:
            List of token accounts
        """
        try:
            params = [
                address,
                {
                    "encoding": "jsonParsed",
                    "commitment": self.commitment
                }
            ]
            
            response = await self._send_request("getTokenAccountsByOwner", params)
            
            if response and "result" in response:
                return response["result"]["value"]
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                self.logger.warning(f"Error getting token accounts: {error_msg}")
                return []
        except Exception as e:
            self.logger.error(f"Error retrieving token accounts for {address}: {e}")
            return []
    
    async def get_recent_block_hash(self) -> str:
        """
        Get a recent block hash for use in transactions.
        
        Returns:
            Recent block hash
        """
        try:
            params = [{"commitment": self.commitment}]
            response = await self._send_request("getRecentBlockhash", params)
            
            if response and "result" in response:
                return response["result"]["value"]["blockhash"]
            else:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                self.logger.warning(f"Error getting recent block hash: {error_msg}")
                return ""
        except Exception as e:
            self.logger.error(f"Error retrieving recent block hash: {e}")
            return ""
    
    async def _send_request(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request to the Solana node.
        
        Args:
            method: JSON-RPC method
            params: Method parameters
            
        Returns:
            JSON-RPC response
        """
        if not self.session:
            self.logger.error("No active session. Call connect() first.")
            return {}
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params
            }
            
            async with self.session.post(self.rpc_url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"HTTP error {response.status}: {await response.text()}")
                    return {}
        except Exception as e:
            self.logger.error(f"Error sending request {method}: {e}")
            return {} 