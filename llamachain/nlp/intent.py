"""
Intent classification for blockchain queries.

This module provides intent classification capabilities for blockchain-related natural language queries.
"""

import re
import logging
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Tuple, Union

from llamachain.log import get_logger

# Get logger
logger = get_logger("llamachain.nlp.intent")


class Intent(Enum):
    """Enum representing different query intents."""
    
    # Blockchain data retrieval intents
    GET_TRANSACTION = "get_transaction"
    GET_BLOCK = "get_block"
    GET_BALANCE = "get_balance"
    GET_CONTRACT = "get_contract"
    GET_GAS_PRICE = "get_gas_price"
    GET_PRICE = "get_price"
    
    # Analysis intents
    ANALYZE_CONTRACT = "analyze_contract"
    ANALYZE_WALLET = "analyze_wallet"
    ANALYZE_TRANSACTION = "analyze_transaction"
    
    # Monitoring intents
    MONITOR_ADDRESS = "monitor_address"
    MONITOR_CONTRACT = "monitor_contract"
    MONITOR_PRICE = "monitor_price"
    
    # Unknown intent
    UNKNOWN = "unknown"
    
    def to_dict(self) -> Dict[str, str]:
        """
        Convert intent to dictionary.
        
        Returns:
            Dictionary representation of the intent
        """
        return {
            "type": self.value,
            "description": self.get_description(),
        }
    
    def get_description(self) -> str:
        """
        Get a human-readable description of the intent.
        
        Returns:
            Description string
        """
        descriptions = {
            Intent.GET_TRANSACTION: "Retrieve transaction information",
            Intent.GET_BLOCK: "Retrieve block information",
            Intent.GET_BALANCE: "Check account balance",
            Intent.GET_CONTRACT: "Retrieve contract information",
            Intent.GET_GAS_PRICE: "Check current gas price",
            Intent.GET_PRICE: "Check token price",
            Intent.ANALYZE_CONTRACT: "Analyze contract for security issues",
            Intent.ANALYZE_WALLET: "Analyze wallet activity",
            Intent.ANALYZE_TRANSACTION: "Analyze transaction details",
            Intent.MONITOR_ADDRESS: "Monitor address activity",
            Intent.MONITOR_CONTRACT: "Monitor contract events",
            Intent.MONITOR_PRICE: "Monitor token price",
            Intent.UNKNOWN: "Unknown intent",
        }
        
        return descriptions.get(self, "Unknown intent")


class IntentClassifier:
    """Classifier for determining the intent of blockchain-related queries."""
    
    def __init__(self):
        """Initialize the intent classifier."""
        # Define keyword patterns for each intent
        self.intent_patterns = {
            Intent.GET_TRANSACTION: [
                r"transaction",
                r"tx",
                r"txn",
                r"transaction hash",
                r"tx hash",
                r"transaction details",
            ],
            Intent.GET_BLOCK: [
                r"block",
                r"block number",
                r"block hash",
                r"block details",
                r"block info",
            ],
            Intent.GET_BALANCE: [
                r"balance",
                r"account balance",
                r"wallet balance",
                r"how much",
                r"holdings",
            ],
            Intent.GET_CONTRACT: [
                r"contract",
                r"smart contract",
                r"contract address",
                r"contract code",
                r"contract abi",
            ],
            Intent.GET_GAS_PRICE: [
                r"gas price",
                r"gas fee",
                r"gas cost",
                r"transaction fee",
            ],
            Intent.GET_PRICE: [
                r"price",
                r"token price",
                r"coin price",
                r"value",
                r"worth",
                r"exchange rate",
            ],
            Intent.ANALYZE_CONTRACT: [
                r"analyze contract",
                r"audit contract",
                r"check contract",
                r"contract security",
                r"contract vulnerability",
                r"contract analysis",
            ],
            Intent.ANALYZE_WALLET: [
                r"analyze wallet",
                r"wallet analysis",
                r"wallet activity",
                r"account analysis",
            ],
            Intent.ANALYZE_TRANSACTION: [
                r"analyze transaction",
                r"transaction analysis",
                r"transaction details",
            ],
            Intent.MONITOR_ADDRESS: [
                r"monitor address",
                r"watch address",
                r"track address",
                r"alert",
                r"notification",
            ],
            Intent.MONITOR_CONTRACT: [
                r"monitor contract",
                r"watch contract",
                r"track contract",
                r"contract events",
            ],
            Intent.MONITOR_PRICE: [
                r"monitor price",
                r"price alert",
                r"price notification",
                r"track price",
                r"watch price",
            ],
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for intent, patterns in self.intent_patterns.items():
            self.compiled_patterns[intent] = [
                re.compile(r"\b" + pattern + r"\b", re.IGNORECASE)
                for pattern in patterns
            ]
        
        # Check if transformers is available for more advanced classification
        try:
            import transformers
            self.transformers_available = True
            logger.info("Transformers library available for advanced intent classification")
        except ImportError:
            self.transformers_available = False
            logger.warning("Transformers library not available, using rule-based intent classification only")
    
    async def classify(self, query: str) -> Intent:
        """
        Classify the intent of a query.
        
        Args:
            query: The query to classify
            
        Returns:
            Classified intent
        """
        # First try rule-based classification
        intent = self._rule_based_classification(query)
        
        # If intent is unknown and transformers is available, try model-based classification
        if intent == Intent.UNKNOWN and self.transformers_available:
            try:
                intent = await self._model_based_classification(query)
            except Exception as e:
                logger.error(f"Error in model-based classification: {str(e)}")
        
        logger.debug(f"Classified query '{query}' as intent: {intent.value}")
        return intent
    
    def _rule_based_classification(self, query: str) -> Intent:
        """
        Classify intent using rule-based approach.
        
        Args:
            query: The query to classify
            
        Returns:
            Classified intent
        """
        # Calculate scores for each intent
        scores = {}
        for intent, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(query)
                score += len(matches)
            scores[intent] = score
        
        # Find the intent with the highest score
        max_score = 0
        max_intent = Intent.UNKNOWN
        
        for intent, score in scores.items():
            if score > max_score:
                max_score = score
                max_intent = intent
        
        # If no patterns matched, return UNKNOWN
        if max_score == 0:
            return Intent.UNKNOWN
        
        return max_intent
    
    async def _model_based_classification(self, query: str) -> Intent:
        """
        Classify intent using a transformer model.
        
        Args:
            query: The query to classify
            
        Returns:
            Classified intent
        """
        # This is a placeholder for future implementation
        # In a real implementation, this would use a pre-trained model
        # to classify the intent
        
        # For now, just return the result of rule-based classification
        return self._rule_based_classification(query)
    
    def get_confidence_scores(self, query: str) -> Dict[Intent, float]:
        """
        Get confidence scores for each intent.
        
        Args:
            query: The query to classify
            
        Returns:
            Dictionary mapping intents to confidence scores
        """
        # Calculate raw scores
        raw_scores = {}
        for intent, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(query)
                score += len(matches)
            raw_scores[intent] = score
        
        # Calculate total score
        total_score = sum(raw_scores.values())
        
        # Convert to confidence scores
        confidence_scores = {}
        if total_score > 0:
            for intent, score in raw_scores.items():
                confidence_scores[intent] = score / total_score
        else:
            # If no patterns matched, assign a small confidence to UNKNOWN
            for intent in self.intent_patterns.keys():
                confidence_scores[intent] = 0.0
            confidence_scores[Intent.UNKNOWN] = 1.0
        
        return confidence_scores 