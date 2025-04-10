from fastapi import APIRouter, Depends, HTTPException, Body, Query, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import tempfile
import os
import json

from llamachain.db.session import get_db
from llamachain.db import models
from llamachain.nlp.interface import QueryParser
from llamachain.ml.models.vulnerability import VulnerabilityDetectionModel
from llamachain.ml.models.address_classifier import AddressClassifier
from llamachain.ml.models.price_predictor import TokenPricePredictor


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/nl/parse")
async def parse_natural_language_query(
    query: str = Body(..., description="Natural language query about blockchain data"),
    chain: Optional[str] = Body(None, description="Blockchain context (ethereum, solana)")
):
    """
    Parse a natural language query into a structured blockchain request.
    
    Takes a human language query and converts it to a structured request that can be executed
    against blockchain APIs.
    """
    try:
        # Initialize query parser
        parser = QueryParser()
        
        # Parse the query
        parsed_result = await parser.parse_query(query, chain)
        
        return parsed_result
        
    except Exception as e:
        logger.error(f"Error parsing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing query: {str(e)}")


@router.post("/nl/execute")
async def execute_natural_language_query(
    query: str = Body(..., description="Natural language query about blockchain data"),
    chain: Optional[str] = Body(None, description="Blockchain context (ethereum, solana)")
):
    """
    Execute a natural language query against blockchain data.
    
    Takes a human language query, parses it, executes it against the blockchain,
    and returns formatted results.
    """
    try:
        # Initialize query parser
        parser = QueryParser()
        
        # Parse and execute the query
        result = await parser.execute_query(query, chain)
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")


@router.post("/vulnerability/detect")
async def detect_vulnerabilities(
    contract_file: UploadFile = File(..., description="Smart contract file to analyze"),
    chain: str = Query("ethereum", description="Blockchain (ethereum, solana)"),
    confidence_threshold: float = Query(0.7, description="Confidence threshold for detection (0.0-1.0)")
):
    """
    Detect vulnerabilities in a smart contract using ML models.
    
    Analyzes a smart contract file to detect potential vulnerabilities using
    machine learning models trained on known vulnerability patterns.
    """
    try:
        # Create temporary file to save the uploaded contract
        with tempfile.NamedTemporaryFile(suffix=".sol" if chain == "ethereum" else ".rs", delete=False) as temp_file:
            # Write file content
            content = await contract_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Initialize vulnerability detection model
            model = VulnerabilityDetectionModel()
            
            # Analyze contract
            result = await model.detect_vulnerabilities(
                temp_file_path, 
                chain=chain, 
                confidence_threshold=confidence_threshold
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting vulnerabilities: {e}")
            raise HTTPException(status_code=500, detail=f"Error detecting vulnerabilities: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@router.post("/classify/address")
async def classify_address(
    chain: str = Body(..., description="Blockchain (ethereum, solana)"),
    address: str = Body(..., description="Address to classify"),
    include_features: bool = Body(False, description="Include classification features in response")
):
    """
    Classify an address using ML models.
    
    Analyzes an address's transaction history and behavior to classify it as an individual,
    contract, exchange, miner, etc. using machine learning models.
    """
    try:
        # Initialize address classifier
        classifier = AddressClassifier()
        
        # Classify address
        result = await classifier.classify_address(chain, address, include_features)
        
        return result
        
    except Exception as e:
        logger.error(f"Error classifying address: {e}")
        raise HTTPException(status_code=500, detail=f"Error classifying address: {str(e)}")


@router.post("/predict/price")
async def predict_token_price(
    token: str = Body(..., description="Token symbol (e.g., ETH, SOL)"),
    days: int = Body(7, description="Number of days to predict"),
    model_type: str = Body("lstm", description="Model type (lstm, transformer, ensemble)"),
    confidence_intervals: bool = Body(True, description="Include confidence intervals")
):
    """
    Predict token price using ML models.
    
    Forecasts future token prices using advanced machine learning models,
    with optional confidence intervals.
    """
    try:
        # Initialize price predictor
        predictor = TokenPricePredictor()
        
        # Predict price
        result = await predictor.predict_price(
            token, 
            days, 
            model_type=model_type, 
            include_confidence=confidence_intervals
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error predicting price: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting price: {str(e)}")


@router.post("/predict/transaction")
async def predict_transaction_confirmation(
    chain: str = Body("ethereum", description="Blockchain (ethereum)"),
    gas_price: int = Body(..., description="Gas price in Wei"),
    gas_limit: int = Body(..., description="Gas limit"),
    current_base_fee: Optional[int] = Body(None, description="Current base fee (for EIP-1559)"),
    priority_fee: Optional[int] = Body(None, description="Priority fee (for EIP-1559)"),
    congestion_level: Optional[str] = Body(None, description="Network congestion (low, medium, high)")
):
    """
    Predict transaction confirmation time.
    
    Forecasts the expected confirmation time for a transaction based on gas price,
    current network conditions, and historical confirmation data.
    """
    try:
        from llamachain.ml.models.transaction_predictor import TransactionPredictor
        
        # Initialize transaction predictor
        predictor = TransactionPredictor()
        
        # Predict confirmation time
        result = await predictor.predict_confirmation_time(
            chain, 
            gas_price, 
            gas_limit, 
            current_base_fee, 
            priority_fee, 
            congestion_level
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error predicting confirmation time: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting confirmation time: {str(e)}")


@router.post("/predict/address-behavior")
async def predict_address_behavior(
    chain: str = Body(..., description="Blockchain (ethereum, solana)"),
    address: str = Body(..., description="Address to analyze"),
    prediction_type: str = Body("activity", description="Prediction type (activity, volume, interactions)")
):
    """
    Predict future behavior of an address.
    
    Forecasts the future behavior of an address based on its historical transaction patterns,
    including activity levels, volume, and interaction patterns.
    """
    try:
        from llamachain.ml.models.behavior_predictor import AddressBehaviorPredictor
        
        # Initialize behavior predictor
        predictor = AddressBehaviorPredictor()
        
        # Predict address behavior
        result = await predictor.predict_behavior(chain, address, prediction_type)
        
        return result
        
    except Exception as e:
        logger.error(f"Error predicting address behavior: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting address behavior: {str(e)}")


@router.post("/sentiment/token")
async def analyze_token_sentiment(
    token: str = Body(..., description="Token symbol (e.g., ETH, SOL)"),
    sources: List[str] = Body(["twitter", "reddit", "news"], description="Data sources to analyze"),
    hours: int = Body(24, description="Number of hours of data to analyze")
):
    """
    Analyze sentiment for a token across social media and news.
    
    Analyzes sentiment about a token from social media platforms and news sources,
    providing sentiment scores and trend analysis.
    """
    try:
        from llamachain.ml.models.sentiment_analyzer import TokenSentimentAnalyzer
        
        # Initialize sentiment analyzer
        analyzer = TokenSentimentAnalyzer()
        
        # Analyze sentiment
        result = await analyzer.analyze_sentiment(token, sources, hours)
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")


@router.post("/anomaly/detect")
async def detect_transaction_anomalies(
    chain: str = Body(..., description="Blockchain (ethereum, solana)"),
    address: Optional[str] = Body(None, description="Address to analyze (optional)"),
    hours: int = Body(24, description="Number of hours of data to analyze"),
    sensitivity: float = Body(2.0, description="Anomaly detection sensitivity (standard deviations)")
):
    """
    Detect transaction anomalies.
    
    Analyzes transaction patterns to detect anomalies that might indicate suspicious activity,
    using machine learning models and statistical methods.
    """
    try:
        from llamachain.ml.models.anomaly_detector import TransactionAnomalyDetector
        
        # Initialize anomaly detector
        detector = TransactionAnomalyDetector()
        
        # Detect anomalies
        result = await detector.detect_anomalies(chain, address, hours, sensitivity)
        
        return result
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting anomalies: {str(e)}")


@router.post("/cluster/transactions")
async def cluster_transactions(
    chain: str = Body(..., description="Blockchain (ethereum, solana)"),
    address: Optional[str] = Body(None, description="Address to analyze (optional)"),
    days: int = Body(30, description="Number of days of data to analyze"),
    num_clusters: int = Body(5, description="Number of clusters to identify")
):
    """
    Cluster transactions by behavior patterns.
    
    Applies unsupervised learning to cluster transactions based on behavioral patterns,
    identifying distinct usage patterns and behaviors.
    """
    try:
        from llamachain.ml.models.transaction_clustering import TransactionClusterer
        
        # Initialize transaction clusterer
        clusterer = TransactionClusterer()
        
        # Cluster transactions
        result = await clusterer.cluster_transactions(chain, address, days, num_clusters)
        
        return result
        
    except Exception as e:
        logger.error(f"Error clustering transactions: {e}")
        raise HTTPException(status_code=500, detail=f"Error clustering transactions: {str(e)}")


@router.post("/embed/contract")
async def embed_contract_code(
    contract_file: UploadFile = File(..., description="Smart contract file to embed"),
    chain: str = Query("ethereum", description="Blockchain (ethereum, solana)")
):
    """
    Generate embeddings for smart contract code.
    
    Creates vector embeddings of smart contract code that can be used for similarity
    comparison, clustering, and other ML tasks.
    """
    try:
        # Create temporary file to save the uploaded contract
        with tempfile.NamedTemporaryFile(suffix=".sol" if chain == "ethereum" else ".rs", delete=False) as temp_file:
            # Write file content
            content = await contract_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            from llamachain.ml.embeddings import ContractEmbedder
            
            # Initialize contract embedder
            embedder = ContractEmbedder()
            
            # Generate embeddings
            result = await embedder.embed_contract(temp_file_path, chain)
            
            return result
            
        except Exception as e:
            logger.error(f"Error embedding contract: {e}")
            raise HTTPException(status_code=500, detail=f"Error embedding contract: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@router.post("/embed/address")
async def embed_address_activity(
    chain: str = Body(..., description="Blockchain (ethereum, solana)"),
    address: str = Body(..., description="Address to embed"),
    days: int = Body(90, description="Number of days of activity to analyze")
):
    """
    Generate embeddings for address activity.
    
    Creates vector embeddings of address activity patterns that can be used for similarity
    comparison, clustering, and other ML tasks.
    """
    try:
        from llamachain.ml.embeddings import AddressEmbedder
        
        # Initialize address embedder
        embedder = AddressEmbedder()
        
        # Generate embeddings
        result = await embedder.embed_address(chain, address, days)
        
        return result
        
    except Exception as e:
        logger.error(f"Error embedding address: {e}")
        raise HTTPException(status_code=500, detail=f"Error embedding address: {str(e)}") 