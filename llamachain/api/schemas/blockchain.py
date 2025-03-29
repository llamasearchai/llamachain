from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime


class BlockBasic(BaseModel):
    """Basic block information common across chains"""
    block_number: int = Field(..., description="Block number")
    block_hash: str = Field(..., description="Block hash")
    timestamp: Union[int, float] = Field(..., description="Block timestamp")
    transaction_count: int = Field(..., description="Number of transactions in the block")


class EthereumBlockInfo(BlockBasic):
    """Ethereum-specific block information"""
    gas_used: int = Field(..., description="Gas used by the block")
    gas_limit: int = Field(..., description="Gas limit for the block")
    difficulty: int = Field(..., description="Block difficulty")
    nonce: str = Field(..., description="Block nonce")
    miner: str = Field(..., description="Miner address")
    size: int = Field(..., description="Block size in bytes")
    total_difficulty: Optional[int] = Field(None, description="Total difficulty")
    base_fee_per_gas: Optional[int] = Field(None, description="Base fee per gas (EIP-1559)")


class SolanaBlockInfo(BlockBasic):
    """Solana-specific block information"""
    slot: int = Field(..., description="Block slot")
    parent_slot: Optional[int] = Field(None, description="Parent slot")
    blockhash: str = Field(..., description="Block hash")
    previous_blockhash: Optional[str] = Field(None, description="Previous block hash")
    rewards: Optional[List[Dict[str, Any]]] = Field(None, description="Block rewards")


class TransactionBasic(BaseModel):
    """Basic transaction information common across chains"""
    tx_hash: str = Field(..., description="Transaction hash")
    block_number: Optional[int] = Field(None, description="Block number containing the transaction")
    timestamp: Optional[Union[int, float]] = Field(None, description="Transaction timestamp")


class EthereumTransactionInfo(TransactionBasic):
    """Ethereum-specific transaction information"""
    from_address: str = Field(..., description="Sender address")
    to_address: Optional[str] = Field(None, description="Recipient address")
    value: int = Field(..., description="Transaction value in wei")
    gas: int = Field(..., description="Gas limit")
    gas_price: int = Field(..., description="Gas price in wei")
    nonce: int = Field(..., description="Transaction nonce")
    input: Optional[str] = Field(None, description="Transaction input data")
    transaction_index: Optional[int] = Field(None, description="Transaction index in block")


class SolanaTransactionInfo(TransactionBasic):
    """Solana-specific transaction information"""
    slot: int = Field(..., description="Block slot")
    signature: str = Field(..., description="Transaction signature")
    status: str = Field(..., description="Transaction status")
    fee: int = Field(..., description="Transaction fee in lamports")
    recent_blockhash: Optional[str] = Field(None, description="Recent blockhash")
    signatures: List[str] = Field(..., description="Transaction signatures")


class BlockResponse(BaseModel):
    """Response model for block information"""
    chain: str = Field(..., description="Blockchain name")
    block_number: int = Field(..., description="Block number")
    block_hash: str = Field(..., description="Block hash")
    parent_hash: str = Field(..., description="Parent block hash")
    timestamp: Union[int, float] = Field(..., description="Block timestamp")
    transaction_count: int = Field(..., description="Number of transactions in the block")
    is_cached: bool = Field(..., description="Whether the block is cached in the database")
    
    # Chain-specific fields - optional depending on the chain
    # Ethereum fields
    gas_used: Optional[int] = Field(None, description="Gas used by the block")
    gas_limit: Optional[int] = Field(None, description="Gas limit for the block")
    difficulty: Optional[int] = Field(None, description="Block difficulty")
    nonce: Optional[str] = Field(None, description="Block nonce")
    miner: Optional[str] = Field(None, description="Miner address")
    size: Optional[int] = Field(None, description="Block size in bytes")
    total_difficulty: Optional[int] = Field(None, description="Total difficulty")
    base_fee_per_gas: Optional[int] = Field(None, description="Base fee per gas (EIP-1559)")
    
    # Solana fields
    slot: Optional[int] = Field(None, description="Block slot")
    parent_slot: Optional[int] = Field(None, description="Parent slot")
    blockhash: Optional[str] = Field(None, description="Block hash (Solana)")
    previous_blockhash: Optional[str] = Field(None, description="Previous block hash")
    rewards: Optional[List[Dict[str, Any]]] = Field(None, description="Block rewards")
    
    # Transactions - optional, depending on query parameters
    transactions: Optional[List[Dict[str, Any]]] = Field(None, description="Transactions in the block")


class BlockListResponse(BaseModel):
    """Response model for a list of blocks"""
    chain: str = Field(..., description="Blockchain name")
    blocks: List[Dict[str, Any]] = Field(..., description="List of blocks")
    count: int = Field(..., description="Number of blocks")
    start_block: int = Field(..., description="Starting block number")
    end_block: Optional[int] = Field(None, description="Ending block number")


class TransactionResponse(BaseModel):
    """Response model for transaction information"""
    chain: str = Field(..., description="Blockchain name")
    tx_hash: str = Field(..., description="Transaction hash")
    
    # Chain-specific fields - optional depending on the chain
    # Ethereum fields
    block_number: Optional[int] = Field(None, description="Block number")
    block_hash: Optional[str] = Field(None, description="Block hash")
    from_address: Optional[str] = Field(None, description="Sender address")
    to_address: Optional[str] = Field(None, description="Recipient address")
    value: Optional[int] = Field(None, description="Transaction value in wei")
    gas: Optional[int] = Field(None, description="Gas limit")
    gas_price: Optional[int] = Field(None, description="Gas price in wei")
    nonce: Optional[int] = Field(None, description="Transaction nonce")
    input: Optional[str] = Field(None, description="Transaction input data")
    transaction_index: Optional[int] = Field(None, description="Transaction index in block")
    
    # Solana fields
    slot: Optional[int] = Field(None, description="Block slot")
    signature: Optional[str] = Field(None, description="Transaction signature")
    status: Optional[str] = Field(None, description="Transaction status")
    fee: Optional[int] = Field(None, description="Transaction fee in lamports")
    recent_blockhash: Optional[str] = Field(None, description="Recent blockhash")
    signatures: Optional[List[str]] = Field(None, description="Transaction signatures")
    
    # Receipt - optional, depending on query parameters
    receipt: Optional[Dict[str, Any]] = Field(None, description="Transaction receipt")


class TransactionListResponse(BaseModel):
    """Response model for a list of transactions"""
    chain: str = Field(..., description="Blockchain name")
    address: str = Field(..., description="Address for which transactions are listed")
    transactions: List[Dict[str, Any]] = Field(..., description="List of transactions")
    count: int = Field(..., description="Number of transactions")


class BalanceResponse(BaseModel):
    """Response model for address balance"""
    chain: str = Field(..., description="Blockchain name")
    address: str = Field(..., description="Address")
    balance: int = Field(..., description="Balance in the smallest denomination")
    balance_formatted: str = Field(..., description="Formatted balance with symbol")
    denomination: str = Field(..., description="Denomination name (wei, lamports, etc.)")
    symbol: str = Field(..., description="Currency symbol (ETH, SOL, etc.)")


class GasPriceResponse(BaseModel):
    """Response model for gas price information"""
    chain: str = Field(..., description="Blockchain name")
    gas_price: int = Field(..., description="Gas price in the smallest denomination")
    gas_price_formatted: str = Field(..., description="Formatted gas price")
    denomination: str = Field(..., description="Denomination name (wei, lamports, etc.)")
    
    # Chain-specific fields
    recommended_speeds: Optional[Dict[str, int]] = Field(None, description="Recommended gas prices for different speeds")
    fixed_cost: Optional[bool] = Field(None, description="Whether the chain has fixed transaction costs")
    prioritization_fees: Optional[bool] = Field(None, description="Whether the chain supports prioritization fees")


class ChainInfoResponse(BaseModel):
    """Response model for blockchain information"""
    chain: str = Field(..., description="Blockchain name")
    chain_id: int = Field(..., description="Chain ID")
    latest_block_number: int = Field(..., description="Latest block number")
    gas_price: int = Field(..., description="Current gas price")
    latest_block_hash: str = Field(..., description="Latest block hash")
    latest_block_timestamp: Union[int, float] = Field(..., description="Latest block timestamp")
    connected: bool = Field(..., description="Whether connected to the blockchain node") 