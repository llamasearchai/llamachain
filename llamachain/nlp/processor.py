"""
Natural Language Processing for blockchain data.

This module provides NLP capabilities for analyzing blockchain data.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union

from llamachain.core.exceptions import ValidationError
from llamachain.log import get_logger
from llamachain.nlp.intent import IntentClassifier, Intent
from llamachain.nlp.entity import EntityExtractor, Entity
from llamachain.nlp.generation import generate_response, generate_structured_query

# Get logger
logger = get_logger("llamachain.nlp.processor")


class NLPProcessor:
    """Natural Language Processor for blockchain queries."""
    
    def __init__(self):
        """Initialize the NLP processor."""
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        
        # Check if spaCy is available
        try:
            import spacy  # type: ignore
            self.spacy_available = True
            
            # Try to load a spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                # Try to load a smaller model
                try:
                    self.nlp = spacy.load("en_core_web_md")
                    logger.info("Loaded spaCy model: en_core_web_md")
                except OSError:
                    self.nlp = None
                    logger.warning("No spaCy model available, some NLP features will be limited")
        except ImportError:
            self.spacy_available = False
            self.nlp = None
            logger.warning("spaCy not available, some NLP features will be limited")
        
        # Check if transformers is available
        try:
            import transformers  # type: ignore
            self.transformers_available = True
        except ImportError:
            self.transformers_available = False
            logger.warning("transformers not available, some NLP features will be limited")
        
        logger.info(f"NLP processor initialized (spaCy: {self.spacy_available}, transformers: {self.transformers_available})")    
    async def process(self, query: str) -> str:
        """
        Process a query string to prepare it for further NLP tasks.
        
        Args:
            query: The query to process
            
        Returns:
            Processed query string
        """
        if not query:
            raise ValidationError("Query cannot be empty")
        
        # Clean and normalize the query
        return self._clean_query(query)
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query.
        
        Args:
            query: The query to process
            
        Returns:
            Dictionary with processed query information
        """
        if not query:
            raise ValidationError("Query cannot be empty")
        
        # Clean and normalize the query
        cleaned_query = await self.process(query)
        
        # Classify intent
        intent, confidence = await self.intent_classifier.classify(cleaned_query)
        
        # Extract entities
        entities = await self.entity_extractor.extract(cleaned_query, intent)
        
        # Generate response
        response = await generate_response(intent, entities)
        
        # Generate structured query
        structured_query = await generate_structured_query(intent, entities)
        
        # Process with spaCy if available
        spacy_analysis = None
        if self.spacy_available and self.nlp:
            spacy_analysis = self._analyze_with_spacy(cleaned_query)
        
        # Construct result
        result = {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "intent": {
                "type": intent.value,
                "description": intent.get_description(),
                "confidence": confidence
            },
            "entities": [entity.to_dict() for entity in entities],
            "response": response,
            "structured_query": structured_query,
            "spacy_analysis": spacy_analysis,
        }
        
        return result
    
    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize a query.
        
        Args:
            query: The query to clean
            
        Returns:
            Cleaned query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Remove special characters except those commonly used in blockchain addresses
        query = re.sub(r'[^\w\s\-\.0-9a-fA-F]', ' ', query)
        
        return query
    
    def _analyze_with_spacy(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query with spaCy.
        
        Args:
            query: The query to analyze
            
        Returns:
            Dictionary with spaCy analysis results
        """
        doc = self.nlp(query)
        
        # Extract named entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        
        # Extract noun chunks
        noun_chunks = []
        for chunk in doc.noun_chunks:
            noun_chunks.append({
                "text": chunk.text,
                "root": chunk.root.text,
                "start": chunk.start_char,
                "end": chunk.end_char,
            })
        
        # Extract tokens
        tokens = []
        for token in doc:
            tokens.append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "is_stop": token.is_stop,
            })
        
        return {
            "entities": entities,
            "noun_chunks": noun_chunks,
            "tokens": tokens,
        }
    
    async def generate_response(self, processed_query: Dict[str, Any]) -> str:
        """
        Generate a response to a processed query.
        
        Args:
            processed_query: The processed query
            
        Returns:
            Response string
        """
        # For backward compatibility
        intent = Intent(processed_query["intent"]["type"])
        entities = [Entity(**entity) for entity in processed_query["entities"]]
        
        return await generate_response(intent, entities)
    
    async def translate_to_query(self, processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a processed query to a structured query.
        
        Args:
            processed_query: The processed query
            
        Returns:
            Structured query
        """
        # For backward compatibility
        intent = Intent(processed_query["intent"]["type"])
        entities = [Entity(**entity) for entity in processed_query["entities"]]
        
        return await generate_structured_query(intent, entities) 