from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import json

from llamachain.db.session import get_db
from llamachain.db import models
from llamachain.analytics.price import PriceAnalyzer
from llamachain.analytics.patterns import PatternDetector
from llamachain.analytics.transactions import TransactionAnalyzer
from llamachain.analytics.addresses import AddressAnalyzer
from llamachain.analytics.visualizations import VisualizationGenerator


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/price/history/{token}")
async def get_price_history(
    token: str = Path(..., description="Token symbol (e.g., ETH, SOL)"),
    days: int = Query(7, description="Number of days of price history"),
    interval: str = Query("1h", description="Price interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)")
):
    """
    Get historical price data for a token.
    
    Returns historical price data for the specified token over the given time period and interval.
    """
    try:
        # Initialize price analyzer
        price_analyzer = PriceAnalyzer()
        
        # Get price history
        result = await price_analyzer.get_price_history(token, days, interval)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting price history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting price history: {str(e)}")


@router.get("/price/correlation")
async def get_price_correlation(
    tokens: List[str] = Query(..., description="List of token symbols to analyze correlation"),
    days: int = Query(30, description="Number of days for correlation analysis")
):
    """
    Get price correlation between tokens.
    
    Analyzes price correlation between the specified tokens over the given time period.
    """
    try:
        # Initialize price analyzer
        price_analyzer = PriceAnalyzer()
        
        # Get price correlation
        result = await price_analyzer.get_price_correlation(tokens, days)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting price correlation: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting price correlation: {str(e)}")


@router.get("/patterns/detect/{chain}/{address}")
async def detect_address_patterns(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    address: str = Path(..., description="Address to analyze"),
    days: int = Query(30, description="Number of days of transaction history to analyze")
):
    """
    Detect patterns in address activity.
    
    Analyzes transaction patterns for the specified address over the given time period.
    """
    try:
        # Initialize pattern detector
        pattern_detector = PatternDetector()
        
        # Detect address patterns
        result = await pattern_detector.detect_address_patterns(chain, address, days)
        
        return result
        
    except Exception as e:
        logger.error(f"Error detecting address patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting address patterns: {str(e)}")


@router.get("/patterns/anomalies/{chain}")
async def detect_network_anomalies(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    hours: int = Query(24, description="Number of hours to analyze"),
    threshold: float = Query(2.0, description="Anomaly detection threshold (standard deviations)")
):
    """
    Detect anomalies in blockchain network activity.
    
    Analyzes network-wide anomalies over the given time period, including transaction volume,
    gas prices, contract creation, etc.
    """
    try:
        # Initialize pattern detector
        pattern_detector = PatternDetector()
        
        # Detect network anomalies
        result = await pattern_detector.detect_network_anomalies(chain, hours, threshold)
        
        return result
        
    except Exception as e:
        logger.error(f"Error detecting network anomalies: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting network anomalies: {str(e)}")


@router.get("/transactions/metrics/{chain}")
async def get_transaction_metrics(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    days: int = Query(7, description="Number of days to analyze"),
    interval: str = Query("1h", description="Interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)")
):
    """
    Get transaction metrics for a blockchain.
    
    Analyzes transaction metrics over the given time period, including transaction volume,
    gas prices, block times, etc.
    """
    try:
        # Initialize transaction analyzer
        tx_analyzer = TransactionAnalyzer()
        
        # Get transaction metrics
        result = await tx_analyzer.get_transaction_metrics(chain, days, interval)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting transaction metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting transaction metrics: {str(e)}")


@router.get("/addresses/metrics/{chain}/{address}")
async def get_address_metrics(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    address: str = Path(..., description="Address to analyze"),
    days: int = Query(30, description="Number of days to analyze")
):
    """
    Get metrics for a specific address.
    
    Analyzes address metrics over the given time period, including transaction volume,
    token holdings, contract interactions, etc.
    """
    try:
        # Initialize address analyzer
        address_analyzer = AddressAnalyzer()
        
        # Get address metrics
        result = await address_analyzer.get_address_metrics(chain, address, days)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting address metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting address metrics: {str(e)}")


@router.get("/addresses/classify/{chain}/{address}")
async def classify_address(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    address: str = Path(..., description="Address to classify")
):
    """
    Classify an address by type.
    
    Analyzes address behavior to classify it as an individual, contract, exchange, miner, etc.
    """
    try:
        # Initialize address analyzer
        address_analyzer = AddressAnalyzer()
        
        # Classify address
        result = await address_analyzer.classify_address(chain, address)
        
        return result
        
    except Exception as e:
        logger.error(f"Error classifying address: {e}")
        raise HTTPException(status_code=500, detail=f"Error classifying address: {str(e)}")


@router.get("/addresses/network/{chain}/{address}")
async def get_address_network(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    address: str = Path(..., description="Central address to analyze"),
    depth: int = Query(1, description="Network depth to analyze (1-3)"),
    tx_limit: int = Query(1000, description="Maximum number of transactions to analyze")
):
    """
    Get address network for visualization.
    
    Analyzes the transaction network around the specified address, including connected addresses
    and transaction flows.
    """
    try:
        # Initialize visualization generator
        viz_generator = VisualizationGenerator()
        
        # Get address network
        result = await viz_generator.get_address_network(chain, address, depth, tx_limit)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting address network: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting address network: {str(e)}")


@router.get("/visualizations/token-flow/{chain}/{token}")
async def get_token_flow_visualization(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    token: str = Path(..., description="Token address or symbol"),
    hours: int = Query(24, description="Number of hours to analyze"),
    top_n: int = Query(20, description="Number of top addresses to include")
):
    """
    Get token flow visualization data.
    
    Analyzes token flows between major addresses over the given time period for visualization.
    """
    try:
        # Initialize visualization generator
        viz_generator = VisualizationGenerator()
        
        # Get token flow visualization
        result = await viz_generator.get_token_flow(chain, token, hours, top_n)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting token flow visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting token flow visualization: {str(e)}")


@router.get("/visualizations/heatmap/{chain}")
async def get_activity_heatmap(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    metric: str = Query("transactions", description="Metric to visualize (transactions, gas, volume)"),
    days: int = Query(7, description="Number of days to analyze")
):
    """
    Get activity heatmap visualization data.
    
    Generates heatmap data for blockchain activity over time (by hour and day).
    """
    try:
        # Initialize visualization generator
        viz_generator = VisualizationGenerator()
        
        # Get activity heatmap
        result = await viz_generator.get_activity_heatmap(chain, metric, days)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting activity heatmap: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting activity heatmap: {str(e)}")


@router.get("/visualizations/gas-price/{chain}")
async def get_gas_price_visualization(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    days: int = Query(7, description="Number of days to analyze"),
    interval: str = Query("1h", description="Interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)")
):
    """
    Get gas price visualization data.
    
    Analyzes gas prices over time for visualization, including percentiles for low, average, and high prices.
    """
    try:
        # Initialize visualization generator
        viz_generator = VisualizationGenerator()
        
        # Get gas price visualization
        result = await viz_generator.get_gas_price_visualization(chain, days, interval)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting gas price visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting gas price visualization: {str(e)}")


@router.get("/visualizations/contract-activity/{chain}/{address}")
async def get_contract_activity_visualization(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    address: str = Path(..., description="Contract address"),
    days: int = Query(30, description="Number of days to analyze"),
    interval: str = Query("1d", description="Interval (1h, 4h, 1d, 1w)")
):
    """
    Get contract activity visualization data.
    
    Analyzes contract activity over time for visualization, including method calls, volume, and active users.
    """
    try:
        # Initialize visualization generator
        viz_generator = VisualizationGenerator()
        
        # Get contract activity visualization
        result = await viz_generator.get_contract_activity(chain, address, days, interval)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting contract activity visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting contract activity visualization: {str(e)}")


@router.get("/metrics/defi/{chain}")
async def get_defi_metrics(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    days: int = Query(30, description="Number of days to analyze")
):
    """
    Get DeFi metrics for a blockchain.
    
    Analyzes DeFi activity on the blockchain, including TVL, volume, and active users.
    """
    try:
        from llamachain.analytics.defi import DefiAnalyzer
        
        # Initialize DeFi analyzer
        defi_analyzer = DefiAnalyzer()
        
        # Get DeFi metrics
        result = await defi_analyzer.get_defi_metrics(chain, days)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting DeFi metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting DeFi metrics: {str(e)}")


@router.get("/metrics/nft/{chain}")
async def get_nft_metrics(
    chain: str = Path(..., description="Blockchain (ethereum, solana)"),
    days: int = Query(30, description="Number of days to analyze")
):
    """
    Get NFT metrics for a blockchain.
    
    Analyzes NFT activity on the blockchain, including volume, floor prices, and active collections.
    """
    try:
        from llamachain.analytics.nft import NftAnalyzer
        
        # Initialize NFT analyzer
        nft_analyzer = NftAnalyzer()
        
        # Get NFT metrics
        result = await nft_analyzer.get_nft_metrics(chain, days)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting NFT metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting NFT metrics: {str(e)}")


@router.post("/predict/price")
async def predict_token_price(
    token: str = Body(..., description="Token symbol (e.g., ETH, SOL)"),
    days: int = Body(7, description="Number of days to predict"),
    model_type: str = Body("prophet", description="Prediction model to use (prophet, lstm, arima)")
):
    """
    Predict token price.
    
    Generates price predictions for the specified token over the given time period using the selected model.
    """
    try:
        from llamachain.analytics.predictions import PricePredictor
        
        # Initialize price predictor
        price_predictor = PricePredictor()
        
        # Predict token price
        result = await price_predictor.predict_price(token, days, model_type)
        
        return result
        
    except Exception as e:
        logger.error(f"Error predicting token price: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting token price: {str(e)}")


@router.post("/predict/gas")
async def predict_gas_price(
    chain: str = Body("ethereum", description="Blockchain (currently only ethereum supported)"),
    hours: int = Body(24, description="Number of hours to predict"),
    model_type: str = Body("prophet", description="Prediction model to use (prophet, lstm, arima)")
):
    """
    Predict gas price.
    
    Generates gas price predictions for the specified blockchain over the given time period using the selected model.
    """
    try:
        from llamachain.analytics.predictions import GasPredictor
        
        # Initialize gas predictor
        gas_predictor = GasPredictor()
        
        # Predict gas price
        result = await gas_predictor.predict_gas(chain, hours, model_type)
        
        return result
        
    except Exception as e:
        logger.error(f"Error predicting gas price: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting gas price: {str(e)}") 