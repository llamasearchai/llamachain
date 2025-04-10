"""
Dashboard analytics for the LlamaChain platform.

This module provides functions for generating dashboard data and visualizations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from llamachain.blockchain import BlockchainRegistry
from llamachain.db.models import Block, Transaction, Alert, ApiUsage
from llamachain.log import get_logger

# Setup logger
logger = get_logger("llamachain.analytics.dashboard")


async def get_network_summary(
    chain_id: str, days: int = 7, db: AsyncSession = None
) -> Dict[str, Any]:
    """
    Get a summary of network statistics.
    
    Args:
        chain_id: Blockchain identifier (ethereum, solana)
        days: Number of days to include in the summary
        db: Database session
    
    Returns:
        Dictionary with network statistics
    """
    try:
        # Get blockchain instance
        registry = BlockchainRegistry()
        chain = await registry.get_chain(chain_id)
        
        # Get latest block
        latest_block = await chain.get_latest_block()
        latest_block_number = latest_block.get("number", 0)
        
        # Get gas price
        gas_price = await chain.get_gas_price()
        
        # Get chain ID
        chain_id_val = await chain.get_chain_id()
        
        # Calculate start time for data range
        start_time = datetime.now() - timedelta(days=days)
        
        # Get transactions from database if session provided
        tx_count = 0
        avg_gas_price = 0
        if db:
            # Get transaction count
            tx_query = select(func.count()).select_from(Transaction).where(
                Transaction.blockchain == chain_id,
                Transaction.timestamp >= start_time
            )
            tx_count = await db.scalar(tx_query) or 0
            
            # Get average gas price
            gas_query = select(func.avg(Transaction.gas_price)).select_from(Transaction).where(
                Transaction.blockchain == chain_id,
                Transaction.timestamp >= start_time
            )
            avg_gas_price = await db.scalar(gas_query) or 0
        
        # Return summary
        return {
            "chain": chain_id,
            "chain_id": chain_id_val,
            "latest_block": latest_block_number,
            "current_gas_price": gas_price,
            "transaction_count": tx_count,
            "average_gas_price": avg_gas_price,
            "time_period_days": days,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting network summary: {e}")
        return {
            "chain": chain_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def get_recent_transactions(
    chain_id: str, limit: int = 10, db: AsyncSession = None
) -> List[Dict[str, Any]]:
    """
    Get recent transactions.
    
    Args:
        chain_id: Blockchain identifier (ethereum, solana)
        limit: Maximum number of transactions to return
        db: Database session
    
    Returns:
        List of recent transactions
    """
    try:
        if not db:
            logger.warning("Database session required for recent transactions")
            return []
        
        # Query recent transactions
        query = select(Transaction).where(
            Transaction.blockchain == chain_id
        ).order_by(
            Transaction.timestamp.desc()
        ).limit(limit)
        
        result = await db.execute(query)
        transactions = result.scalars().all()
        
        # Convert to dictionaries
        tx_list = []
        for tx in transactions:
            tx_dict = {
                "hash": tx.hash,
                "block_number": tx.block_number,
                "timestamp": tx.timestamp.isoformat() if tx.timestamp else None,
                "from_address": tx.from_address,
                "to_address": tx.to_address,
                "value": tx.value,
                "gas_price": tx.gas_price,
                "gas_used": tx.gas_used,
                "status": tx.status,
            }
            tx_list.append(tx_dict)
        
        return tx_list
    except Exception as e:
        logger.error(f"Error getting recent transactions: {e}")
        return []


async def get_recent_blocks(
    chain_id: str, limit: int = 10, db: AsyncSession = None
) -> List[Dict[str, Any]]:
    """
    Get recent blocks.
    
    Args:
        chain_id: Blockchain identifier (ethereum, solana)
        limit: Maximum number of blocks to return
        db: Database session
    
    Returns:
        List of recent blocks
    """
    try:
        if not db:
            logger.warning("Database session required for recent blocks")
            return []
        
        # Query recent blocks
        query = select(Block).where(
            Block.blockchain == chain_id
        ).order_by(
            Block.number.desc()
        ).limit(limit)
        
        result = await db.execute(query)
        blocks = result.scalars().all()
        
        # Convert to dictionaries
        block_list = []
        for block in blocks:
            block_dict = {
                "number": block.number,
                "hash": block.hash,
                "timestamp": block.timestamp.isoformat() if block.timestamp else None,
                "transaction_count": block.transaction_count,
                "parent_hash": block.parent_hash,
            }
            block_list.append(block_dict)
        
        return block_list
    except Exception as e:
        logger.error(f"Error getting recent blocks: {e}")
        return []


async def get_security_alerts(
    limit: int = 10, severity: Optional[str] = None, chain_id: Optional[str] = None, db: AsyncSession = None
) -> List[Dict[str, Any]]:
    """
    Get recent security alerts.
    
    Args:
        limit: Maximum number of alerts to return
        severity: Filter by severity level
        chain_id: Filter by blockchain
        db: Database session
    
    Returns:
        List of recent security alerts
    """
    try:
        if not db:
            logger.warning("Database session required for security alerts")
            return []
        
        # Build query
        query = select(Alert).order_by(Alert.timestamp.desc())
        
        # Apply filters
        if severity:
            query = query.where(Alert.severity == severity)
        if chain_id:
            query = query.where(Alert.blockchain == chain_id)
        
        # Apply limit
        query = query.limit(limit)
        
        # Execute query
        result = await db.execute(query)
        alerts = result.scalars().all()
        
        # Convert to dictionaries
        alert_list = []
        for alert in alerts:
            alert_dict = {
                "id": alert.id,
                "blockchain": alert.blockchain,
                "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity,
                "category": alert.category,
                "address": alert.address,
                "transaction_hash": alert.transaction_hash,
                "is_resolved": alert.is_resolved,
            }
            alert_list.append(alert_dict)
        
        return alert_list
    except Exception as e:
        logger.error(f"Error getting security alerts: {e}")
        return []


async def get_transaction_volume_chart(
    chain_id: str, days: int = 7, interval: str = "1d", db: AsyncSession = None
) -> Dict[str, Any]:
    """
    Get transaction volume chart data.
    
    Args:
        chain_id: Blockchain identifier (ethereum, solana)
        days: Number of days to include
        interval: Time interval (1h, 1d)
        db: Database session
    
    Returns:
        Dictionary with chart data
    """
    try:
        if not db:
            logger.warning("Database session required for transaction volume chart")
            return {"error": "Database session required"}
        
        # Calculate start time
        start_time = datetime.now() - timedelta(days=days)
        
        # Get transactions
        query = select(Transaction).where(
            Transaction.blockchain == chain_id,
            Transaction.timestamp >= start_time
        )
        
        result = await db.execute(query)
        transactions = result.scalars().all()
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                "timestamp": tx.timestamp,
                "value": float(tx.value) if tx.value else 0,
            }
            for tx in transactions
        ])
        
        if df.empty:
            return {
                "chain": chain_id,
                "data": [],
                "time_period_days": days,
                "interval": interval,
            }
        
        # Resample by interval
        if interval == "1h":
            df["date"] = df["timestamp"].dt.floor("H")
        else:  # Default to daily
            df["date"] = df["timestamp"].dt.floor("D")
        
        # Group by date and sum values
        volume_by_date = df.groupby("date")["value"].sum().reset_index()
        
        # Convert to chart data
        data = [
            {
                "date": date.isoformat(),
                "volume": volume,
            }
            for date, volume in zip(volume_by_date["date"], volume_by_date["value"])
        ]
        
        return {
            "chain": chain_id,
            "data": data,
            "time_period_days": days,
            "interval": interval,
        }
    except Exception as e:
        logger.error(f"Error getting transaction volume chart: {e}")
        return {
            "chain": chain_id,
            "error": str(e),
            "time_period_days": days,
            "interval": interval,
        }


async def get_gas_price_chart(
    chain_id: str, days: int = 7, interval: str = "1h", db: AsyncSession = None
) -> Dict[str, Any]:
    """
    Get gas price chart data.
    
    Args:
        chain_id: Blockchain identifier (ethereum, solana)
        days: Number of days to include
        interval: Time interval (1h, 1d)
        db: Database session
    
    Returns:
        Dictionary with chart data
    """
    try:
        if not db:
            logger.warning("Database session required for gas price chart")
            return {"error": "Database session required"}
        
        # Calculate start time
        start_time = datetime.now() - timedelta(days=days)
        
        # Get transactions
        query = select(Transaction).where(
            Transaction.blockchain == chain_id,
            Transaction.timestamp >= start_time,
            Transaction.gas_price.isnot(None)
        )
        
        result = await db.execute(query)
        transactions = result.scalars().all()
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                "timestamp": tx.timestamp,
                "gas_price": float(tx.gas_price) if tx.gas_price else 0,
            }
            for tx in transactions
        ])
        
        if df.empty:
            return {
                "chain": chain_id,
                "data": [],
                "time_period_days": days,
                "interval": interval,
            }
        
        # Resample by interval
        if interval == "1h":
            df["date"] = df["timestamp"].dt.floor("H")
        else:  # Default to daily
            df["date"] = df["timestamp"].dt.floor("D")
        
        # Group by date and average gas prices
        gas_by_date = df.groupby("date")["gas_price"].mean().reset_index()
        
        # Convert to chart data
        data = [
            {
                "date": date.isoformat(),
                "gas_price": gas_price,
            }
            for date, gas_price in zip(gas_by_date["date"], gas_by_date["gas_price"])
        ]
        
        return {
            "chain": chain_id,
            "data": data,
            "time_period_days": days,
            "interval": interval,
        }
    except Exception as e:
        logger.error(f"Error getting gas price chart: {e}")
        return {
            "chain": chain_id,
            "error": str(e),
            "time_period_days": days,
            "interval": interval,
        }


async def get_platform_stats(db: AsyncSession = None) -> Dict[str, Any]:
    """
    Get platform usage statistics.
    
    Args:
        db: Database session
    
    Returns:
        Dictionary with platform statistics
    """
    try:
        if not db:
            logger.warning("Database session required for platform stats")
            return {"error": "Database session required"}
        
        # Calculate time ranges
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=7)
        
        # Get API usage count
        total_query = select(func.count()).select_from(ApiUsage)
        total_count = await db.scalar(total_query) or 0
        
        today_query = select(func.count()).select_from(ApiUsage).where(ApiUsage.timestamp >= today)
        today_count = await db.scalar(today_query) or 0
        
        yesterday_query = select(func.count()).select_from(ApiUsage).where(
            ApiUsage.timestamp >= yesterday,
            ApiUsage.timestamp < today
        )
        yesterday_count = await db.scalar(yesterday_query) or 0
        
        # Get average response time
        avg_time_query = select(func.avg(ApiUsage.response_time)).select_from(ApiUsage)
        avg_time = await db.scalar(avg_time_query) or 0
        
        # Get endpoint usage
        endpoint_query = select(
            ApiUsage.endpoint,
            func.count().label("count")
        ).group_by(
            ApiUsage.endpoint
        ).order_by(
            func.count().desc()
        ).limit(5)
        
        endpoint_result = await db.execute(endpoint_query)
        top_endpoints = [{"endpoint": endpoint, "count": count} for endpoint, count in endpoint_result]
        
        # Return stats
        return {
            "total_api_calls": total_count,
            "api_calls_today": today_count,
            "api_calls_yesterday": yesterday_count,
            "average_response_time": avg_time,
            "top_endpoints": top_endpoints,
            "timestamp": now.isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting platform stats: {e}")
        return {"error": str(e)}


async def get_chain_comparison(db: AsyncSession = None) -> Dict[str, Any]:
    """
    Get comparison data between different blockchains.
    
    Args:
        db: Database session
    
    Returns:
        Dictionary with chain comparison data
    """
    try:
        # Get blockchain registry
        registry = BlockchainRegistry()
        chains = registry.get_available_chains()
        
        # Collect data for each chain
        chain_data = []
        for chain_id in chains:
            try:
                chain = await registry.get_chain(chain_id)
                
                # Get basic info
                chain_name = await chain.get_chain_name()
                latest_block = await chain.get_latest_block_number()
                gas_price = await chain.get_gas_price()
                
                # Get transaction count from database if available
                tx_count = 0
                if db:
                    tx_query = select(func.count()).select_from(Transaction).where(
                        Transaction.blockchain == chain_id
                    )
                    tx_count = await db.scalar(tx_query) or 0
                
                chain_data.append({
                    "chain_id": chain_id,
                    "chain_name": chain_name,
                    "latest_block": latest_block,
                    "gas_price": gas_price,
                    "transaction_count": tx_count,
                })
            except Exception as e:
                logger.error(f"Error getting data for chain {chain_id}: {e}")
                chain_data.append({
                    "chain_id": chain_id,
                    "error": str(e)
                })
        
        return {
            "chains": chain_data,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting chain comparison: {e}")
        return {"error": str(e)}


async def generate_transaction_volume_plot(
    chain_id: str, days: int = 7, interval: str = "1d", db: AsyncSession = None
) -> Optional[go.Figure]:
    """
    Generate a Plotly figure for transaction volume.
    
    Args:
        chain_id: Blockchain identifier (ethereum, solana)
        days: Number of days to include
        interval: Time interval (1h, 1d)
        db: Database session
    
    Returns:
        Plotly figure or None if error
    """
    try:
        # Get chart data
        chart_data = await get_transaction_volume_chart(chain_id, days, interval, db)
        
        if "error" in chart_data or not chart_data.get("data"):
            logger.warning(f"No data available for transaction volume chart: {chart_data.get('error', 'No data')}")
            return None
        
        # Extract data
        dates = [item["date"] for item in chart_data["data"]]
        volumes = [item["volume"] for item in chart_data["data"]]
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=volumes,
            mode="lines+markers",
            name="Transaction Volume",
            line=dict(color="#3366CC", width=2),
            marker=dict(size=6)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Transaction Volume for {chain_id.capitalize()} (Last {days} days)",
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_white",
            height=500,
            width=800,
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error generating transaction volume plot: {e}")
        return None


async def generate_gas_price_plot(
    chain_id: str, days: int = 7, interval: str = "1h", db: AsyncSession = None
) -> Optional[go.Figure]:
    """
    Generate a Plotly figure for gas prices.
    
    Args:
        chain_id: Blockchain identifier (ethereum, solana)
        days: Number of days to include
        interval: Time interval (1h, 1d)
        db: Database session
    
    Returns:
        Plotly figure or None if error
    """
    try:
        # Get chart data
        chart_data = await get_gas_price_chart(chain_id, days, interval, db)
        
        if "error" in chart_data or not chart_data.get("data"):
            logger.warning(f"No data available for gas price chart: {chart_data.get('error', 'No data')}")
            return None
        
        # Extract data
        dates = [item["date"] for item in chart_data["data"]]
        gas_prices = [item["gas_price"] for item in chart_data["data"]]
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=gas_prices,
            mode="lines+markers",
            name="Gas Price",
            line=dict(color="#FF9900", width=2),
            marker=dict(size=6)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Gas Price for {chain_id.capitalize()} (Last {days} days)",
            xaxis_title="Date",
            yaxis_title="Gas Price",
            template="plotly_white",
            height=500,
            width=800,
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error generating gas price plot: {e}")
        return None 