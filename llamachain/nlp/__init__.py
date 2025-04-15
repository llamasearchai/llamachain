"""
Natural Language Processing module for the LlamaChain platform.

This module provides NLP capabilities for analyzing blockchain data.
"""

from llamachain.nlp.entity import Entity, EntityExtractor
from llamachain.nlp.generation import (
    ResponseGenerator,
    StructuredQueryGenerator,
    generate_response,
    generate_structured_query,
)
from llamachain.nlp.intent import Intent, IntentClassifier
from llamachain.nlp.processor import NLPProcessor
