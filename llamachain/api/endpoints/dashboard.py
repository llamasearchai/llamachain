"""
Dashboard API endpoints for the LlamaChain platform.

This module provides API endpoints for retrieving dashboard data and visualizations.
"""

from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from llamachain.analytics.dashboard import (
    get_network_summary,
    get_recent_transactions,
    get_recent_blocks,
    get_security_alerts,
    get_transaction_volume_chart,
    get_gas_price_chart,
    get_platform_stats,
    get_chain_comparison,
)
from llamachain.db.session import get_db
from llamachain.log import get_logger

# Setup router
router = APIRouter()

# Setup logger
logger = get_logger("llamachain.api.endpoints.dashboard")


@router.get("/summary")
async def get_dashboard_summary():
    """
    Get a summary of key platform metrics for the dashboard.
    
    Returns:
        Dictionary with summary metrics
    """
    try:
        # This is a placeholder that would normally get data from various sources
        return {
            "active_chains": 2,
            "total_blocks_monitored": 12345,
            "total_transactions_monitored": 54321,
            "total_addresses_monitored": 789,
            "security_alerts": 5,
            "api_requests_today": 1234,
            "timestamp": "2023-10-12T15:30:45",
        }
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting dashboard summary: {str(e)}")


@router.get("/network/stats/{chain}")
async def get_network_stats(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    days: int = Query(7, description="Number of days of data to display"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get network statistics for a specific blockchain.
    
    Args:
        chain: Blockchain identifier (ethereum, solana)
        days: Number of days of data to include
        db: Database session
        
    Returns:
        Dictionary with network statistics
    """
    try:
        stats = await get_network_summary(chain, days, db)
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting network stats for {chain}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting network stats: {str(e)}")


@router.get("/transactions/recent/{chain}")
async def get_recent_tx(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    limit: int = Query(10, description="Number of transactions to retrieve"),
    include_details: bool = Query(False, description="Include transaction details"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recent transactions for a specific blockchain.
    
    Args:
        chain: Blockchain identifier (ethereum, solana)
        limit: Maximum number of transactions to return
        include_details: Whether to include transaction details
        db: Database session
        
    Returns:
        List of recent transactions
    """
    try:
        transactions = await get_recent_transactions(chain, limit, db)
        
        # If include_details is False, remove some fields for a lighter response
        if not include_details:
            for tx in transactions:
                # Remove detailed fields
                for field in ["gas_used", "gas_price", "status"]:
                    if field in tx:
                        del tx[field]
        
        return {"chain": chain, "transactions": transactions, "count": len(transactions)}
    except Exception as e:
        logger.error(f"Error getting recent transactions for {chain}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recent transactions: {str(e)}")


@router.get("/blocks/recent/{chain}")
async def get_recent_block(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    limit: int = Query(10, description="Number of blocks to retrieve"),
    include_transactions: bool = Query(False, description="Include block transactions"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get recent blocks for a specific blockchain.
    
    Args:
        chain: Blockchain identifier (ethereum, solana)
        limit: Maximum number of blocks to return
        include_transactions: Whether to include block transactions
        db: Database session
        
    Returns:
        List of recent blocks
    """
    try:
        blocks = await get_recent_blocks(chain, limit, db)
        
        # Include transactions if requested (this would require additional code)
        if include_transactions:
            logger.warning("Including transactions in blocks is not yet implemented")
            
        return {"chain": chain, "blocks": blocks, "count": len(blocks)}
    except Exception as e:
        logger.error(f"Error getting recent blocks for {chain}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recent blocks: {str(e)}")


@router.get("/security/alerts")
async def get_security_alert(
    limit: int = Query(10, description="Number of alerts to retrieve"),
    severity: Optional[str] = Query(None, description="Alert severity filter (low, medium, high, critical)"),
    chain: Optional[str] = Query(None, description="Blockchain filter (ethereum, solana)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get security alerts.
    
    Args:
        limit: Maximum number of alerts to return
        severity: Filter by severity level
        chain: Filter by blockchain
        db: Database session
        
    Returns:
        List of security alerts
    """
    try:
        alerts = await get_security_alerts(limit, severity, chain, db)
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        logger.error(f"Error getting security alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting security alerts: {str(e)}")


@router.get("/security/vulnerabilities")
async def get_vulnerability_summary(
    days: int = Query(30, description="Number of days of data to analyze"),
    chain: Optional[str] = Query(None, description="Blockchain filter (ethereum, solana)"),
    category: Optional[str] = Query(None, description="Vulnerability category filter"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a summary of detected vulnerabilities.
    
    Args:
        days: Number of days of data to include
        chain: Filter by blockchain
        category: Filter by vulnerability category
        db: Database session
        
    Returns:
        Summary of vulnerabilities
    """
    try:
        # This is a placeholder that would normally query vulnerability data
        return {
            "total_vulnerabilities": 42,
            "by_severity": {
                "critical": 5,
                "high": 10,
                "medium": 15,
                "low": 12,
            },
            "by_category": {
                "reentrancy": 8,
                "overflow": 6,
                "frontrunning": 4,
                "access_control": 12,
                "other": 12,
            },
            "time_period_days": days,
        }
    except Exception as e:
        logger.error(f"Error getting vulnerability summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting vulnerability summary: {str(e)}")


@router.get("/visualizations/chart-data")
async def get_chart_data(
    chart_type: str = Query(..., description="Chart type (transaction_volume, gas_price, active_addresses, etc.)"),
    chain: str = Query("ethereum", description="Blockchain (ethereum, solana)"),
    days: int = Query(7, description="Number of days of data to display"),
    interval: str = Query("1h", description="Data interval (1h, 4h, 1d)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get data for charts and visualizations.
    
    Args:
        chart_type: Type of chart data to retrieve
        chain: Blockchain identifier (ethereum, solana)
        days: Number of days of data to include
        interval: Time interval for data points
        db: Database session
        
    Returns:
        Chart data in the appropriate format
    """
    try:
        if chart_type == "transaction_volume":
            data = await get_transaction_volume_chart(chain, days, interval, db)
        elif chart_type == "gas_price":
            data = await get_gas_price_chart(chain, days, interval, db)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported chart type: {chart_type}")
        
        if "error" in data:
            raise HTTPException(status_code=500, detail=data["error"])
            
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chart data for {chart_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chart data: {str(e)}")


@router.get("/visualizations/network-data")
async def get_network_visualization_data(
    chain: str = Query("ethereum", description="Blockchain (ethereum, solana)"),
    scope: str = Query("global", description="Visualization scope (global, address, contract)"),
    address: Optional[str] = Query(None, description="Address for address-specific visualization"),
    depth: int = Query(1, description="Network depth for address visualization"),
    limit: int = Query(100, description="Maximum number of nodes to include"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get data for network visualizations.
    
    Args:
        chain: Blockchain identifier (ethereum, solana)
        scope: Visualization scope (global, address, contract)
        address: Address for address-specific visualization
        depth: Network depth for address visualization
        limit: Maximum number of nodes to include
        db: Database session
        
    Returns:
        Network visualization data
    """
    try:
        # This is a placeholder that would normally generate network visualization data
        if scope == "address" and not address:
            raise HTTPException(status_code=400, detail="Address required for address scope")
            
        # Placeholder response with minimal network data
        return {
            "chain": chain,
            "scope": scope,
            "nodes": [
                {"id": "node1", "label": "Node 1", "size": 10},
                {"id": "node2", "label": "Node 2", "size": 8},
                {"id": "node3", "label": "Node 3", "size": 6},
            ],
            "edges": [
                {"source": "node1", "target": "node2", "value": 5},
                {"source": "node2", "target": "node3", "value": 3},
                {"source": "node1", "target": "node3", "value": 2},
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting network visualization data: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting network visualization data: {str(e)}")


@router.get("/stats/platform")
async def get_platform_stat(
    db: AsyncSession = Depends(get_db)
):
    """
    Get platform usage statistics.
    
    Args:
        db: Database session
        
    Returns:
        Platform statistics
    """
    try:
        stats = await get_platform_stats(db)
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting platform stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting platform stats: {str(e)}")


@router.get("/stats/chains")
async def get_chain_compare(
    db: AsyncSession = Depends(get_db)
):
    """
    Get comparison data between different blockchains.
    
    Args:
        db: Database session
        
    Returns:
        Chain comparison data
    """
    try:
        comparison = await get_chain_comparison(db)
        if "error" in comparison:
            raise HTTPException(status_code=500, detail=comparison["error"])
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chain comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chain comparison: {str(e)}")


@router.get("/address/summary/{chain}/{address}")
async def get_address_summary(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    address: str = Path(..., description="Address to analyze"),
    days: int = Query(30, description="Number of days of data to analyze"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get summary information for an address.
    
    Args:
        chain: Blockchain identifier (ethereum, solana)
        address: Address to analyze
        days: Number of days of data to include
        db: Database session
        
    Returns:
        Address summary information
    """
    try:
        # This is a placeholder that would normally query address data
        return {
            "chain": chain,
            "address": address,
            "balance": 123456789,
            "transaction_count": 42,
            "first_seen": "2023-01-15T12:34:56",
            "last_seen": "2023-10-10T09:23:45",
            "is_contract": False,
            "tags": ["exchange", "high_volume"],
            "time_period_days": days,
        }
    except Exception as e:
        logger.error(f"Error getting address summary for {address}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting address summary: {str(e)}")


@router.get("/contract/summary/{chain}/{address}")
async def get_contract_summary(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    address: str = Path(..., description="Contract address to analyze"),
    days: int = Query(30, description="Number of days of data to analyze"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get summary information for a smart contract.
    
    Args:
        chain: Blockchain identifier (ethereum, solana)
        address: Contract address to analyze
        days: Number of days of data to include
        db: Database session
        
    Returns:
        Contract summary information
    """
    try:
        # This is a placeholder that would normally query contract data
        return {
            "chain": chain,
            "address": address,
            "contract_name": "ExampleToken",
            "deployment_date": "2023-01-15T12:34:56",
            "transaction_count": 9876,
            "verified": True,
            "audit_score": 85,
            "vulnerabilities": 2,
            "time_period_days": days,
        }
    except Exception as e:
        logger.error(f"Error getting contract summary for {address}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting contract summary: {str(e)}") 