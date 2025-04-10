"""
Entity extraction for blockchain queries.

This module provides entity extraction capabilities for blockchain-related natural language queries.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set, Tuple, Union, TYPE_CHECKING

from llamachain.log import get_logger
from llamachain.nlp.intent import Intent

# Handle spaCy import for type checking
if TYPE_CHECKING:
    import spacy
    from spacy.tokens import Doc

# Get logger
logger = get_logger("llamachain.nlp.entity")


@dataclass
class Entity:
    """Class representing an extracted entity."""
    
    type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entity to dictionary.
        
        Returns:
            Dictionary representation of the entity
        """
        return {
            "type": self.type,
            "value": self.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


class EntityExtractor:
    """Extractor for blockchain-related entities from natural language queries."""
    
    def __init__(self):
        """Initialize the entity extractor."""
        # Define regex patterns for different entity types
        self.patterns = {
            "address": [
                # Ethereum address pattern
                r"0x[a-fA-F0-9]{40}",
                # Solana address pattern
                r"[1-9A-HJ-NP-Za-km-z]{32,44}",
            ],
            "tx_hash": [
                # Ethereum transaction hash pattern
                r"0x[a-fA-F0-9]{64}",
                # Generic transaction hash reference
                r"transaction\s+(?:hash|id)?\s*:?\s*([a-zA-Z0-9]{64})",
            ],
            "block_number": [
                # Block number patterns
                r"block\s+(?:number|#)?\s*:?\s*(\d+)",
                r"block\s+(\d+)",
                r"#(\d+)",
            ],
            "block_hash": [
                # Block hash pattern
                r"block\s+hash\s*:?\s*([a-zA-Z0-9]{64})",
                r"block\s+hash\s*:?\s*(0x[a-fA-F0-9]{64})",
            ],
            "token": [
                # Common token symbols
                r"\b(ETH|BTC|SOL|USDT|USDC|DAI|LINK|UNI|AAVE|COMP|MKR|SNX|YFI|SUSHI|CRV|BAL|MATIC|DOT|ADA|XRP|LTC|BCH|XLM|EOS|TRX|XTZ|ATOM|ALGO|FIL|THETA|VET|DOGE|SHIB)\b",
                # Token with decimals
                r"(\d+(?:\.\d+)?)\s*(ETH|BTC|SOL|USDT|USDC|DAI|LINK|UNI|AAVE|COMP|MKR|SNX|YFI|SUSHI|CRV|BAL|MATIC|DOT|ADA|XRP|LTC|BCH|XLM|EOS|TRX|XTZ|ATOM|ALGO|FIL|THETA|VET|DOGE|SHIB)",
            ],
            "blockchain": [
                # Blockchain names
                r"\b(ethereum|eth|solana|sol|bitcoin|btc|polygon|matic|avalanche|avax|binance smart chain|bsc|fantom|ftm|arbitrum|optimism|harmony|one|cosmos|atom|polkadot|dot|cardano|ada|tron|trx|tezos|xtz|algorand|algo|filecoin|fil|theta|vechain|vet)\b",
            ],
            "contract_name": [
                # Contract name patterns
                r"contract\s+(?:name|called|named)?\s*:?\s*([a-zA-Z0-9_]+)",
                r"([a-zA-Z0-9_]+)\s+contract",
            ],
            "time_period": [
                # Time period patterns
                r"\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
                r"\b(today|yesterday|this week|this month|this year|last week|last month|last year)\b",
            ],
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for entity_type, patterns in self.patterns.items():
            self.compiled_patterns[entity_type] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in patterns
            ]
        
        # Check if spaCy is available for more advanced extraction
        try:
            import spacy
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
                    logger.warning("No spaCy model available, some entity extraction features will be limited")
        except ImportError:
            self.spacy_available = False
            self.nlp = None
            logger.warning("spaCy not available, some entity extraction features will be limited")
    
    async def extract(self, query: str, intent: Intent) -> List[Entity]:
        """
        Extract entities from a query.
        
        Args:
            query: The query to extract entities from
            intent: The classified intent of the query
            
        Returns:
            List of extracted entities
        """
        # Extract entities using regex patterns
        entities = self._extract_with_regex(query)
        
        # If spaCy is available, extract additional entities
        if self.spacy_available and self.nlp:
            spacy_entities = self._extract_with_spacy(query)
            entities.extend(spacy_entities)
        
        # Filter and prioritize entities based on intent
        entities = self._filter_entities_by_intent(entities, intent)
        
        # Remove overlapping entities
        entities = self._remove_overlapping_entities(entities)
        
        logger.debug(f"Extracted {len(entities)} entities from query: '{query}'")
        return entities
    
    def _extract_with_regex(self, query: str) -> List[Entity]:
        """
        Extract entities using regex patterns.
        
        Args:
            query: The query to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(query):
                    # If there are capturing groups, use the first group
                    if match.groups():
                        value = match.group(1)
                        start = match.start(1)
                        end = match.end(1)
                    else:
                        value = match.group(0)
                        start = match.start(0)
                        end = match.end(0)
                    
                    entity = Entity(
                        type=entity_type,
                        value=value,
                        start=start,
                        end=end,
                        confidence=0.9,  # High confidence for regex matches
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_with_spacy(self, query: str) -> List[Entity]:
        """
        Extract entities using spaCy.
        
        Args:
            query: The query to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # If spaCy is not available, return empty list
        if not self.spacy_available or not self.nlp:
            return entities
            
        doc = self.nlp(query)
        
        # Map spaCy entity types to our entity types
        spacy_to_entity_type = {
            "MONEY": "token",
            "ORG": "blockchain",
            "PRODUCT": "token",
            "DATE": "time_period",
            "TIME": "time_period",
            "CARDINAL": "block_number",
        }
        
        for ent in doc.ents:
            entity_type = spacy_to_entity_type.get(ent.label_, None)
            if entity_type:
                entity = Entity(
                    type=entity_type,
                    value=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.7,  # Lower confidence for spaCy matches
                )
                entities.append(entity)
        
        return entities
    
    def _filter_entities_by_intent(self, entities: List[Entity], intent: Intent) -> List[Entity]:
        """
        Filter and prioritize entities based on intent.
        
        Args:
            entities: List of extracted entities
            intent: The classified intent
            
        Returns:
            Filtered list of entities
        """
        # Define required entity types for each intent
        required_types = {
            Intent.GET_TRANSACTION: ["tx_hash"],
            Intent.GET_BLOCK: ["block_number", "block_hash"],
            Intent.GET_BALANCE: ["address", "token"],
            Intent.GET_CONTRACT: ["address", "contract_name"],
            Intent.GET_GAS_PRICE: ["blockchain"],
            Intent.GET_PRICE: ["token"],
            Intent.ANALYZE_CONTRACT: ["address", "contract_name"],
            Intent.ANALYZE_WALLET: ["address"],
            Intent.ANALYZE_TRANSACTION: ["tx_hash"],
            Intent.MONITOR_ADDRESS: ["address"],
            Intent.MONITOR_CONTRACT: ["address", "contract_name"],
            Intent.MONITOR_PRICE: ["token"],
        }
        
        # If intent is unknown, return all entities
        if intent == Intent.UNKNOWN:
            return entities
        
        # Get required entity types for the intent
        intent_required_types = required_types.get(intent, [])
        
        # Increase confidence for entities of required types
        for entity in entities:
            if entity.type in intent_required_types:
                entity.confidence = min(1.0, entity.confidence + 0.2)
        
        return entities
    
    def _remove_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove overlapping entities, keeping the one with higher confidence.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            List of non-overlapping entities
        """
        # Sort entities by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
        
        # Keep track of non-overlapping entities
        result = []
        covered_ranges = []
        
        for entity in sorted_entities:
            # Check if this entity overlaps with any existing covered range
            overlapping = False
            for start, end in covered_ranges:
                if not (entity.end <= start or entity.start >= end):
                    overlapping = True
                    break
            
            if not overlapping:
                result.append(entity)
                covered_ranges.append((entity.start, entity.end))
        
        return result 