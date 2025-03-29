from fastapi import APIRouter, Depends, Query, Path, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
import logging

from llamachain.db.session import get_db
from llamachain.api.schemas.blockchain import (
    BlockResponse,
    TransactionResponse,
    BalanceResponse,
    ChainInfoResponse,
    GasPriceResponse,
    BlockListResponse,
    TransactionListResponse
)
from llamachain.db import models
from llamachain.db.models import BlockchainEnum


router = APIRouter()
logger = logging.getLogger(__name__)


# Helper function to get appropriate blockchain instance
async def get_blockchain(chain: str, request: Request):
    """Get the appropriate blockchain instance based on the chain parameter"""
    chain = chain.lower()
    
    if chain == "ethereum":
        return request.app.state.eth_chain
    elif chain == "solana":
        return request.app.state.sol_chain
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported blockchain: {chain}")


@router.get("/info/{chain}", response_model=ChainInfoResponse)
async def get_chain_info(
    chain: str = Path(..., description="Blockchain to query (ethereum, solana)"),
    request: Request = None
):
    """
    Get information about a blockchain network.
    
    Retrieves chain ID, latest block number, and other network information.
    """
    blockchain = await get_blockchain(chain, request)
    
    try:
        # Get chain info
        chain_id = await blockchain.get_chain_id()
        latest_block_number = await blockchain.get_latest_block_number()
        gas_price = await blockchain.get_gas_price()
        
        # Get latest block
        latest_block = await blockchain.get_latest_block()
        
        return {
            "chain": chain,
            "chain_id": chain_id,
            "latest_block_number": latest_block_number,
            "gas_price": gas_price,
            "latest_block_hash": latest_block.get("hash", ""),
            "latest_block_timestamp": latest_block.get("timestamp", 0),
            "connected": await blockchain.is_connected()
        }
    except Exception as e:
        logger.error(f"Error getting chain info for {chain}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chain info: {str(e)}")


@router.get("/blocks/{chain}/{block_number}", response_model=BlockResponse)
async def get_block(
    chain: str = Path(..., description="Blockchain to query (ethereum, solana)"),
    block_number: int = Path(..., description="Block number to retrieve"),
    request: Request = None,
    include_transactions: bool = Query(False, description="Include transaction details")
):
    """
    Get information about a specific block.
    
    Retrieves block details by block number, with optional transaction details.
    """
    blockchain = await get_blockchain(chain, request)
    
    try:
        # Get block data
        block = await blockchain.get_block(block_number)
        
        if not block:
            raise HTTPException(status_code=404, detail=f"Block {block_number} not found")
        
        # Check if this block is already in the database
        db = await anext(get_db())
        query = (
            db.query(models.Block)
            .filter(models.Block.blockchain == BlockchainEnum(chain.lower()))
            .filter(models.Block.block_number == block_number)
        )
        db_block = await db.execute(query)
        db_block = db_block.scalar_one_or_none()
        
        # Create standardized response
        response = {
            "chain": chain,
            "block_number": block_number,
            "block_hash": block.get("hash", ""),
            "parent_hash": block.get("parentHash", ""),
            "timestamp": block.get("timestamp", 0),
            "transaction_count": len(block.get("transactions", [])),
            "is_cached": db_block is not None
        }
        
        # Add chain-specific fields
        if chain == "ethereum":
            response.update({
                "gas_used": block.get("gasUsed", 0),
                "gas_limit": block.get("gasLimit", 0),
                "difficulty": block.get("difficulty", 0),
                "nonce": block.get("nonce", ""),
                "miner": block.get("miner", ""),
                "size": block.get("size", 0),
                "total_difficulty": block.get("totalDifficulty", 0),
                "base_fee_per_gas": block.get("baseFeePerGas", 0)
            })
        elif chain == "solana":
            response.update({
                "slot": block.get("slot", 0),
                "parent_slot": block.get("parentSlot", 0),
                "blockhash": block.get("blockhash", ""),
                "previous_blockhash": block.get("previousBlockhash", ""),
                "rewards": block.get("rewards", [])
            })
        
        # Include transactions if requested
        if include_transactions:
            if chain == "ethereum":
                response["transactions"] = [
                    {
                        "hash": tx.get("hash", ""),
                        "from_address": tx.get("from", ""),
                        "to_address": tx.get("to", ""),
                        "value": tx.get("value", 0),
                        "gas": tx.get("gas", 0),
                        "gas_price": tx.get("gasPrice", 0),
                        "nonce": tx.get("nonce", 0)
                    }
                    for tx in block.get("transactions", [])
                ]
            elif chain == "solana":
                response["transactions"] = [
                    {
                        "signature": tx.get("transaction", {}).get("signatures", [""])[0],
                        "status": "success" if tx.get("meta", {}).get("err") is None else "failed",
                        "fee": tx.get("meta", {}).get("fee", 0),
                        "slot": block.get("slot", 0)
                    }
                    for tx in block.get("transactions", [])
                ]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting block {block_number} for {chain}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting block: {str(e)}")


@router.get("/blocks/{chain}", response_model=BlockListResponse)
async def get_blocks(
    chain: str = Path(..., description="Blockchain to query (ethereum, solana)"),
    request: Request = None,
    start_block: Optional[int] = Query(None, description="Starting block number"),
    limit: int = Query(10, ge=1, le=100, description="Number of blocks to retrieve")
):
    """
    Get a list of blocks.
    
    Retrieves a list of blocks starting from the specified block number.
    If start_block is not provided, it starts from the latest block.
    """
    blockchain = await get_blockchain(chain, request)
    
    try:
        # Get latest block number if start_block is not provided
        if start_block is None:
            start_block = await blockchain.get_latest_block_number()
        
        blocks = []
        for i in range(limit):
            block_number = start_block - i
            if block_number < 0:
                break
                
            block = await blockchain.get_block(block_number)
            if not block:
                continue
                
            # Create standardized block summary
            block_summary = {
                "block_number": block_number,
                "block_hash": block.get("hash", ""),
                "timestamp": block.get("timestamp", 0),
                "transaction_count": len(block.get("transactions", []))
            }
            
            # Add chain-specific fields
            if chain == "ethereum":
                block_summary.update({
                    "gas_used": block.get("gasUsed", 0),
                    "miner": block.get("miner", "")
                })
            elif chain == "solana":
                block_summary.update({
                    "slot": block.get("slot", 0),
                    "blockhash": block.get("blockhash", "")
                })
                
            blocks.append(block_summary)
        
        return {
            "chain": chain,
            "blocks": blocks,
            "count": len(blocks),
            "start_block": start_block,
            "end_block": blocks[-1]["block_number"] if blocks else None
        }
        
    except Exception as e:
        logger.error(f"Error getting blocks for {chain}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting blocks: {str(e)}")


@router.get("/transactions/{chain}/{tx_hash}", response_model=TransactionResponse)
async def get_transaction(
    chain: str = Path(..., description="Blockchain to query (ethereum, solana)"),
    tx_hash: str = Path(..., description="Transaction hash to retrieve"),
    request: Request = None,
    include_receipt: bool = Query(False, description="Include transaction receipt")
):
    """
    Get information about a specific transaction.
    
    Retrieves transaction details by transaction hash, with optional receipt details.
    """
    blockchain = await get_blockchain(chain, request)
    
    try:
        # Get transaction data
        tx = await blockchain.get_transaction(tx_hash)
        
        if not tx:
            raise HTTPException(status_code=404, detail=f"Transaction {tx_hash} not found")
        
        # Create standardized response
        response = {
            "chain": chain,
            "tx_hash": tx_hash,
        }
        
        # Add chain-specific fields
        if chain == "ethereum":
            response.update({
                "block_number": tx.get("blockNumber", 0),
                "block_hash": tx.get("blockHash", ""),
                "from_address": tx.get("from", ""),
                "to_address": tx.get("to", ""),
                "value": tx.get("value", 0),
                "gas": tx.get("gas", 0),
                "gas_price": tx.get("gasPrice", 0),
                "nonce": tx.get("nonce", 0),
                "input": tx.get("input", ""),
                "transaction_index": tx.get("transactionIndex", 0)
            })
        elif chain == "solana":
            response.update({
                "slot": tx.get("slot", 0),
                "signature": tx_hash,
                "status": "success" if tx.get("meta", {}).get("err") is None else "failed",
                "fee": tx.get("meta", {}).get("fee", 0),
                "recent_blockhash": tx.get("transaction", {}).get("message", {}).get("recentBlockhash", ""),
                "signatures": tx.get("transaction", {}).get("signatures", []),
            })
        
        # Include receipt if requested
        if include_receipt and chain == "ethereum":
            receipt = await blockchain.get_transaction_receipt(tx_hash)
            
            if receipt:
                response["receipt"] = {
                    "status": receipt.get("status", 0),
                    "gas_used": receipt.get("gasUsed", 0),
                    "cumulative_gas_used": receipt.get("cumulativeGasUsed", 0),
                    "contract_address": receipt.get("contractAddress", None),
                    "logs_count": len(receipt.get("logs", []))
                }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transaction {tx_hash} for {chain}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting transaction: {str(e)}")


@router.get("/address/{chain}/{address}/balance", response_model=BalanceResponse)
async def get_address_balance(
    chain: str = Path(..., description="Blockchain to query (ethereum, solana)"),
    address: str = Path(..., description="Address to check balance"),
    request: Request = None
):
    """
    Get the balance of an address.
    
    Retrieves the native token balance of the specified address.
    """
    blockchain = await get_blockchain(chain, request)
    
    try:
        # Get balance
        balance = await blockchain.get_balance(address)
        
        # Convert to appropriate denomination based on chain
        if chain == "ethereum":
            # Convert wei to ether
            balance_in_eth = balance / 10**18
            return {
                "chain": chain,
                "address": address,
                "balance": balance,
                "balance_formatted": f"{balance_in_eth} ETH",
                "denomination": "wei",
                "symbol": "ETH"
            }
        elif chain == "solana":
            # Convert lamports to SOL
            balance_in_sol = balance / 10**9
            return {
                "chain": chain,
                "address": address,
                "balance": balance,
                "balance_formatted": f"{balance_in_sol} SOL",
                "denomination": "lamports",
                "symbol": "SOL"
            }
        
    except Exception as e:
        logger.error(f"Error getting balance for {address} on {chain}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting balance: {str(e)}")


@router.get("/gas-price/{chain}", response_model=GasPriceResponse)
async def get_gas_price(
    chain: str = Path(..., description="Blockchain to query (ethereum, solana)"),
    request: Request = None
):
    """
    Get the current gas price.
    
    Retrieves the current gas price/fee for transactions.
    """
    blockchain = await get_blockchain(chain, request)
    
    try:
        # Get gas price
        gas_price = await blockchain.get_gas_price()
        
        # Format response based on chain
        if chain == "ethereum":
            # Convert wei to gwei
            gas_price_gwei = gas_price / 10**9
            return {
                "chain": chain,
                "gas_price": gas_price,
                "gas_price_formatted": f"{gas_price_gwei} Gwei",
                "denomination": "wei",
                "recommended_speeds": {
                    "slow": int(gas_price * 0.8),
                    "standard": gas_price,
                    "fast": int(gas_price * 1.2),
                    "fastest": int(gas_price * 1.5)
                }
            }
        elif chain == "solana":
            # For Solana, return the rent exemption as the "gas price"
            return {
                "chain": chain,
                "gas_price": gas_price,
                "gas_price_formatted": f"{gas_price} lamports",
                "denomination": "lamports",
                "fixed_cost": True,
                "prioritization_fees": False
            }
        
    except Exception as e:
        logger.error(f"Error getting gas price for {chain}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting gas price: {str(e)}")


@router.get("/address/{chain}/{address}/transactions", response_model=TransactionListResponse)
async def get_address_transactions(
    chain: str = Path(..., description="Blockchain to query (ethereum, solana)"),
    address: str = Path(..., description="Address to check transactions"),
    request: Request = None,
    limit: int = Query(10, ge=1, le=100, description="Number of transactions to retrieve"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get transactions for an address.
    
    Retrieves transactions where the address is the sender or receiver.
    """
    try:
        # This is a simplified implementation that queries the database
        # A real implementation would likely include indexing services or blockchain API calls
        
        transactions = []
        
        # In our example, we'll query the database for cached transactions
        # In production, you'd typically use a blockchain indexer API
        stmt = (
            db.query(models.Transaction)
            .filter(models.Transaction.blockchain == BlockchainEnum(chain.lower()))
            .filter(
                (models.Transaction.from_address == address) | 
                (models.Transaction.to_address == address)
            )
            .order_by(models.Transaction.timestamp.desc())
            .limit(limit)
        )
        
        result = await db.execute(stmt)
        db_txs = result.scalars().all()
        
        for tx in db_txs:
            tx_data = {
                "tx_hash": tx.tx_hash,
                "block_number": tx.block_number,
                "timestamp": tx.timestamp.timestamp() if tx.timestamp else 0,
            }
            
            # Add chain-specific fields
            if chain == "ethereum":
                tx_data.update({
                    "from_address": tx.from_address,
                    "to_address": tx.to_address,
                    "value": tx.value,
                    "gas": tx.gas,
                    "gas_price": tx.gas_price,
                    "status": tx.status
                })
            elif chain == "solana":
                tx_data.update({
                    "fee": tx.fee,
                    "status": "success" if tx.status else "failed",
                    "signatures": tx.signers if tx.signers else []
                })
                
            transactions.append(tx_data)
        
        return {
            "chain": chain,
            "address": address,
            "transactions": transactions,
            "count": len(transactions)
        }
        
    except Exception as e:
        logger.error(f"Error getting transactions for {address} on {chain}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting transactions: {str(e)}")


@router.get("/contract/{chain}/{address}/code")
async def get_contract_code(
    chain: str = Path(..., description="Blockchain to query (ethereum, solana)"),
    address: str = Path(..., description="Contract address"),
    request: Request = None
):
    """
    Get the bytecode of a smart contract.
    
    Retrieves the bytecode of a smart contract at the specified address.
    """
    blockchain = await get_blockchain(chain, request)
    
    try:
        # Get contract code
        code = await blockchain.get_contract_code(address)
        
        if not code or code == "0x":
            raise HTTPException(status_code=404, detail=f"No contract found at address {address}")
        
        return {
            "chain": chain,
            "address": address,
            "bytecode": code,
            "bytecode_length": len(code) // 2 if code.startswith("0x") else len(code)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting contract code for {address} on {chain}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting contract code: {str(e)}")


@router.get("/supported-chains")
async def get_supported_chains():
    """
    Get a list of supported blockchain networks.
    
    Returns information about all supported blockchains.
    """
    return {
        "chains": [
            {
                "id": "ethereum",
                "name": "Ethereum",
                "chain_id": 1,
                "symbol": "ETH",
                "denomination": "wei",
                "decimals": 18
            },
            {
                "id": "solana",
                "name": "Solana",
                "chain_id": 101,
                "symbol": "SOL",
                "denomination": "lamports",
                "decimals": 9
            }
        ]
    } 