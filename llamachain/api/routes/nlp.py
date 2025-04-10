"""
API routes for NLP processing.

This module provides API endpoints for processing natural language queries related to blockchain data.
"""

import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from llamachain.log import get_logger
from llamachain.nlp.processor import NLPProcessor
from llamachain.nlp.intent import IntentClassifier, Intent
from llamachain.nlp.entity import EntityExtractor, Entity
from llamachain.nlp.generation import generate_response, generate_structured_query
from llamachain.core.exceptions import ValidationError

# Get logger
logger = get_logger("llamachain.api.routes.nlp")

# Create router
router = APIRouter(
    prefix="/nlp",
    tags=["nlp"],
    responses={404: {"description": "Not found"}},
)

# Create NLP processor and related instances
nlp_processor = NLPProcessor()
intent_classifier = IntentClassifier()
entity_extractor = EntityExtractor()


class NLPQueryRequest(BaseModel):
    """Request model for NLP query processing."""
    
    query: str = Field(..., description="Natural language query to process")
    generate_response: bool = Field(False, description="Whether to generate a natural language response")
    translate_to_query: bool = Field(False, description="Whether to translate to a structured query")


class NLPQueryResponse(BaseModel):
    """Response model for NLP query processing."""
    
    original_query: str = Field(..., description="Original query")
    processed_query: str = Field(..., description="Processed query")
    intent: Dict[str, Any] = Field(..., description="Detected intent")
    entities: list = Field(..., description="Extracted entities")
    response: Optional[str] = Field(None, description="Generated response")
    structured_query: Optional[Dict[str, Any]] = Field(None, description="Structured query")


class EntityModel(BaseModel):
    """Model for entity extraction."""
    
    type: str = Field(..., description="Entity type")
    value: str = Field(..., description="Entity value")
    confidence: float = Field(..., description="Confidence score")
    start: int = Field(..., description="Start position in the query")
    end: int = Field(..., description="End position in the query")


class IntentModel(BaseModel):
    """Model for intent classification."""
    
    type: str = Field(..., description="Intent type")
    description: str = Field(..., description="Intent description")
    confidence: float = Field(..., description="Confidence score")


@router.post("/process", response_model=NLPQueryResponse)
async def process_query(request: NLPQueryRequest) -> Dict[str, Any]:
    """
    Process a natural language query.
    
    Args:
        request: The query request
        
    Returns:
        Processed query information
    """
    try:
        # Process the query
        processed_query = await nlp_processor.process_query(request.query)
        
        # Create response (processor now includes all information by default)
        response = {
            "original_query": processed_query["original_query"],
            "processed_query": processed_query["cleaned_query"],
            "intent": processed_query["intent"],
            "entities": processed_query["entities"],
            "response": processed_query["response"] if request.generate_response else None,
            "structured_query": processed_query["structured_query"] if request.translate_to_query else None,
        }
        
        return response
    
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing query")


@router.get("/analyze", response_model=NLPQueryResponse)
async def analyze_query(
    query: str = Query(..., description="Natural language query to analyze"),
    generate_response: bool = Query(False, description="Whether to generate a natural language response"),
    translate_to_query: bool = Query(False, description="Whether to translate to a structured query"),
) -> Dict[str, Any]:
    """
    Analyze a natural language query.
    
    Args:
        query: The query to analyze
        generate_response: Whether to generate a natural language response
        translate_to_query: Whether to translate to a structured query
        
    Returns:
        Analyzed query information
    """
    request = NLPQueryRequest(
        query=query,
        generate_response=generate_response,
        translate_to_query=translate_to_query,
    )
    
    return await process_query(request)


@router.post("/intent", response_model=Dict[str, Any])
async def classify_intent_endpoint(query: str = Query(..., description="Natural language query to classify")) -> Dict[str, Any]:
    """
    Classify the intent of a natural language query.
    
    Args:
        query: The query to classify
        
    Returns:
        Classified intent information
    """
    try:
        # Process the query
        processed_query = await nlp_processor.process(query)
        
        # Classify intent
        intent, confidence = await intent_classifier.classify(processed_query)
        
        # Get confidence scores
        scores = intent_classifier.get_confidence_scores(processed_query)
        
        # Create response
        response = {
            "query": query,
            "processed_query": processed_query,
            "intent": intent.value,
            "description": intent.get_description(),
            "confidence": confidence,
            "confidence_scores": {intent_type.value: score for intent_type, score in scores.items()}
        }
        
        return response
    
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error classifying intent: {str(e)}")
        raise HTTPException(status_code=500, detail="Error classifying intent")


@router.post("/entities", response_model=Dict[str, Any])
async def extract_entities_endpoint(query: str = Query(..., description="Natural language query to extract entities from")) -> Dict[str, Any]:
    """
    Extract entities from a natural language query.
    
    Args:
        query: The query to extract entities from
        
    Returns:
        Extracted entities information
    """
    try:
        # Process the query
        processed_query = await nlp_processor.process(query)
        
        # Classify intent
        intent, _ = await intent_classifier.classify(processed_query)
        
        # Extract entities
        entities = await entity_extractor.extract(processed_query, intent)
        
        # Create response
        response = {
            "query": query,
            "processed_query": processed_query,
            "intent": intent.value,
            "entities": [entity.to_dict() for entity in entities]
        }
        
        return response
    
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        raise HTTPException(status_code=500, detail="Error extracting entities")


@router.post("/generate", response_model=Dict[str, Any])
async def generate_response_endpoint(query: str = Query(..., description="Natural language query to generate a response for")) -> Dict[str, Any]:
    """
    Generate a response to a natural language query.
    
    Args:
        query: The query to generate a response for
        
    Returns:
        Generated response information
    """
    try:
        # Process the query
        processed_query = await nlp_processor.process(query)
        
        # Classify intent
        intent, confidence = await intent_classifier.classify(processed_query)
        
        # Extract entities
        entities = await entity_extractor.extract(processed_query, intent)
        
        # Generate response
        response_text = await generate_response(intent, entities)
        
        # Create response
        response = {
            "query": query,
            "processed_query": processed_query,
            "intent": intent.value,
            "confidence": confidence,
            "entities": [entity.to_dict() for entity in entities],
            "response": response_text
        }
        
        return response
    
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating response")