"""
Natural Language Processing module for the LlamaChain platform.

This module provides NLP capabilities for analyzing blockchain data.
"""

from llamachain.nlp.processor import NLPProcessor
from llamachain.nlp.intent import IntentClassifier, Intent
from llamachain.nlp.entity import EntityExtractor, Entity
from llamachain.nlp.generation import (
    ResponseGenerator, 
    StructuredQueryGenerator,
    generate_response,
    generate_structured_query
)
