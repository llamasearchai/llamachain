"""
Language generation for blockchain queries.

This module provides language generation capabilities for producing natural language responses
to blockchain-related queries.
"""

import logging
import random
from typing import Dict, List, Any, Optional, Union, Tuple

from llamachain.log import get_logger
from llamachain.nlp.intent import Intent
from llamachain.nlp.entity import Entity
from llamachain.core.exceptions import ValidationError

# Get logger
logger = get_logger("llamachain.nlp.generation")


class ResponseTemplate:
    """
    Template for generating responses.
    
    This class represents a response template that can be filled with
    values extracted from the query.
    """
    
    def __init__(self, template: str, required_entities: Optional[List[str]] = None):
        """
        Initialize the response template.
        
        Args:
            template: The template string with placeholders
            required_entities: List of entity types required for this template
        """
        self.template = template
        self.required_entities = required_entities or []
    
    def can_use(self, entities: List[Entity]) -> bool:
        """
        Check if this template can be used with the given entities.
        
        Args:
            entities: List of entities extracted from the query
            
        Returns:
            True if all required entities are present, False otherwise
        """
        if not self.required_entities:
            return True
        
        entity_types = {entity.type for entity in entities}
        return all(req_type in entity_types for req_type in self.required_entities)
    
    def fill(self, entities: List[Entity], additional_values: Optional[Dict[str, str]] = None) -> str:
        """
        Fill the template with values from entities and additional values.
        
        Args:
            entities: List of entities extracted from the query
            additional_values: Additional values to use for filling
            
        Returns:
            Filled template string
        """
        # Create a mapping of entity types to values
        values = {}
        for entity in entities:
            values[entity.type] = entity.value
        
        # Add additional values
        if additional_values:
            values.update(additional_values)
        
        # Fill the template
        result = self.template
        for key, value in values.items():
            result = result.replace(f"{{{key}}}", value)
        
        return result


class ResponseGenerator:
    """
    Generator for natural language responses.
    
    This class generates natural language responses based on the intent and entities
    extracted from a query.
    """
    
    def __init__(self):
        """Initialize the response generator."""
        # Define response templates for different intents
        self.templates = {
            Intent.GET_TRANSACTION: [
                ResponseTemplate("I'll retrieve information for transaction {tx_hash}.", ["tx_hash"]),
                ResponseTemplate("Fetching details for transaction {tx_hash}.", ["tx_hash"]),
                ResponseTemplate("Looking up transaction {tx_hash} on the {blockchain} blockchain.", ["tx_hash", "blockchain"]),
                ResponseTemplate("I'll get that transaction information for you."),
            ],
            Intent.GET_BLOCK: [
                ResponseTemplate("I'll fetch block {block_number} for you.", ["block_number"]),
                ResponseTemplate("Retrieving block number {block_number} from the {blockchain} blockchain.", ["block_number", "blockchain"]),
                ResponseTemplate("Getting block {block_number} details.", ["block_number"]),
                ResponseTemplate("I'll fetch the block information you requested."),
            ],
            Intent.GET_BALANCE: [
                ResponseTemplate("Checking the balance of address {address}.", ["address"]),
                ResponseTemplate("I'll retrieve the balance for {address} on the {blockchain} blockchain.", ["address", "blockchain"]),
                ResponseTemplate("Looking up the balance of {address}.", ["address"]),
                ResponseTemplate("Checking the {token} balance for {address}.", ["address", "token"]),
                ResponseTemplate("I'll check the account balance for you."),
            ],
            Intent.GET_CONTRACT: [
                ResponseTemplate("Retrieving information for contract {address}.", ["address"]),
                ResponseTemplate("I'll fetch details for the contract at {address} on the {blockchain} blockchain.", ["address", "blockchain"]),
                ResponseTemplate("Getting contract data for {address}.", ["address"]),
                ResponseTemplate("I'll retrieve the contract information for you."),
            ],
            Intent.GET_GAS_PRICE: [
                ResponseTemplate("Checking the current gas price on the {blockchain} network.", ["blockchain"]),
                ResponseTemplate("I'll retrieve the current gas price for you."),
                ResponseTemplate("Getting the latest gas price information."),
            ],
            Intent.GET_PRICE: [
                ResponseTemplate("Checking the current price of {token}.", ["token"]),
                ResponseTemplate("I'll retrieve the latest price for {token}..", ["token"]),
                ResponseTemplate("Getting the current market value of {token}.", ["token"]),
                ResponseTemplate("I'll check the token price for you."),
            ],
            Intent.ANALYZE_CONTRACT: [
                ResponseTemplate("Analyzing the security of contract {address}.", ["address"]),
                ResponseTemplate("I'll audit the contract at {address} for security vulnerabilities.", ["address"]),
                ResponseTemplate("Starting security analysis for contract {address} on the {blockchain} blockchain.", ["address", "blockchain"]),
                ResponseTemplate("I'll analyze the contract for security vulnerabilities."),
            ],
            Intent.ANALYZE_WALLET: [
                ResponseTemplate("Analyzing transaction history for address {address}.", ["address"]),
                ResponseTemplate("I'll examine the activity of {address} on the {blockchain} blockchain.", ["address", "blockchain"]),
                ResponseTemplate("Reviewing the transaction patterns for {address}.", ["address"]),
                ResponseTemplate("I'll analyze the wallet activity for you."),
            ],
            Intent.ANALYZE_TRANSACTION: [
                ResponseTemplate("Analyzing transaction {tx_hash} in detail.", ["tx_hash"]),
                ResponseTemplate("I'll provide an in-depth analysis of transaction {tx_hash}.", ["tx_hash"]),
                ResponseTemplate("Examining the execution of transaction {tx_hash} on the {blockchain} blockchain.", ["tx_hash", "blockchain"]),
                ResponseTemplate("I'll analyze the transaction details for you."),
            ],
            Intent.MONITOR_ADDRESS: [
                ResponseTemplate("Setting up monitoring for address {address}.", ["address"]),
                ResponseTemplate("I'll notify you of any activity related to {address} on the {blockchain} blockchain.", ["address", "blockchain"]),
                ResponseTemplate("Starting to track transactions for {address}.", ["address"]),
                ResponseTemplate("I'll set up monitoring for the address."),
            ],
            Intent.MONITOR_CONTRACT: [
                ResponseTemplate("Setting up monitoring for contract {address}.", ["address"]),
                ResponseTemplate("I'll track events from the contract at {address} on the {blockchain} blockchain.", ["address", "blockchain"]),
                ResponseTemplate("Starting to monitor contract {address} for events.", ["address"]),
                ResponseTemplate("I'll monitor the contract for you."),
            ],
            Intent.MONITOR_PRICE: [
                ResponseTemplate("Setting up price alerts for {token}.", ["token"]),
                ResponseTemplate("I'll notify you when the price of {token} reaches your target.", ["token"]),
                ResponseTemplate("Starting to track the price of {token}.", ["token"]),
                ResponseTemplate("I'll monitor the token price for you."),
            ],
            Intent.UNKNOWN: [
                ResponseTemplate("I'm not sure what you're asking for. Could you please rephrase your question?"),
                ResponseTemplate("I don't understand what you're looking for. Can you provide more details?"),
                ResponseTemplate("I'm having trouble understanding your request. Could you try asking in a different way?"),
            ],
        }
    
    async def generate_response(self, intent: Intent, entities: List[Entity], 
                              additional_values: Optional[Dict[str, str]] = None) -> str:
        """
        Generate a natural language response.
        
        Args:
            intent: The intent of the query
            entities: List of entities extracted from the query
            additional_values: Additional values to use for filling templates
            
        Returns:
            Generated response string
        """
        # Get templates for the intent
        intent_templates = self.templates.get(intent, self.templates[Intent.UNKNOWN])
        
        # Filter templates that can be used with the given entities
        usable_templates = [t for t in intent_templates if t.can_use(entities)]
        
        # If no usable templates, fall back to a generic one
        if not usable_templates:
            if intent == Intent.UNKNOWN:
                return "I don't understand what you're asking for. Could you please provide more details?"
            else:
                return f"I'll process your {intent.value.replace('_', ' ')} request."
        
        # Choose a random template
        template = random.choice(usable_templates)
        
        # Fill the template
        return template.fill(entities, additional_values)


class StructuredQueryGenerator:
    """
    Generator for structured queries based on natural language.
    
    This class generates structured queries based on the intent and entities
    extracted from a natural language query.
    """
    
    def __init__(self):
        """Initialize the structured query generator."""
        pass
    
    async def generate_query(self, intent: Intent, entities: List[Entity]) -> Dict[str, Any]:
        """
        Generate a structured query.
        
        Args:
            intent: The intent of the query
            entities: List of entities extracted from the query
            
        Returns:
            Structured query as a dictionary
        """
        # Extract parameters from entities
        params = {}
        for entity in entities:
            if entity.type == "address":
                params["address"] = entity.value
            elif entity.type == "tx_hash":
                params["tx_hash"] = entity.value
            elif entity.type == "block_number":
                params["block_number"] = entity.value
            elif entity.type == "block_hash":
                params["block_hash"] = entity.value
            elif entity.type == "token":
                params["token"] = entity.value
            elif entity.type == "blockchain":
                params["blockchain"] = entity.value
            elif entity.type == "contract_name":
                params["contract_name"] = entity.value
            elif entity.type == "time_period":
                params["time_period"] = entity.value
        
        # Set default blockchain if not specified
        if "blockchain" not in params:
            params["blockchain"] = "ethereum"
        
        # Construct structured query
        query = {
            "type": intent.value,
            "params": params,
        }
        
        return query


# Create singleton instances
response_generator = ResponseGenerator()
structured_query_generator = StructuredQueryGenerator()


async def generate_response(intent: Intent, entities: List[Entity], 
                          additional_values: Optional[Dict[str, str]] = None) -> str:
    """
    Generate a natural language response.
    
    Args:
        intent: The intent of the query
        entities: List of entities extracted from the query
        additional_values: Additional values to use for filling templates
        
    Returns:
        Generated response string
    """
    return await response_generator.generate_response(intent, entities, additional_values)


async def generate_structured_query(intent: Intent, entities: List[Entity]) -> Dict[str, Any]:
    """
    Generate a structured query.
    
    Args:
        intent: The intent of the query
        entities: List of entities extracted from the query
        
    Returns:
        Structured query as a dictionary
    """
    return await structured_query_generator.generate_query(intent, entities) 