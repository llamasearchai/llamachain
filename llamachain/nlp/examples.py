"""
Example usage of the NLP module.

This script demonstrates the NLP pipeline with example queries.
"""

import asyncio
import json
from pprint import pprint
from typing import List, Dict, Any, Optional

from llamachain.nlp.processor import NLPProcessor
from llamachain.nlp.intent import IntentClassifier, Intent
from llamachain.nlp.entity import EntityExtractor, Entity
from llamachain.nlp.generation import generate_response, generate_structured_query


# Sample queries for different intents
EXAMPLE_QUERIES = {
    Intent.GET_TRANSACTION: [
        "Show me transaction 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "Get details for tx 0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        "What happened in transaction 0x7890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234",
    ],
    Intent.GET_BLOCK: [
        "Show me block 12345678",
        "Get details for block 15000000",
        "What's in the latest block on Ethereum?",
        "Show block 12345678 on Polygon",
    ],
    Intent.GET_BALANCE: [
        "What's the balance of 0x1234567890abcdef1234567890abcdef12345678?",
        "Check my ETH balance at 0xabcdef1234567890abcdef1234567890abcdef12",
        "How much USDC does 0x7890abcdef1234567890abcdef1234567890abcdef hold?",
    ],
    Intent.GET_CONTRACT: [
        "Show me contract 0x1234567890abcdef1234567890abcdef12345678",
        "What's in the contract at 0xabcdef1234567890abcdef1234567890abcdef12?",
        "Get the ABI for contract 0x7890abcdef1234567890abcdef1234567890abcdef",
    ],
    Intent.GET_GAS_PRICE: [
        "What's the current gas price?",
        "How much is gas on Ethereum right now?",
        "What are the current gas fees on Polygon?",
    ],
    Intent.GET_PRICE: [
        "What's the price of ETH?",
        "How much is Bitcoin worth right now?",
        "Check the price of USDC",
    ],
    Intent.ANALYZE_CONTRACT: [
        "Audit contract 0x1234567890abcdef1234567890abcdef12345678",
        "Check contract 0xabcdef1234567890abcdef1234567890abcdef12 for vulnerabilities",
        "Is the contract at 0x7890abcdef1234567890abcdef1234567890abcdef secure?",
    ],
    Intent.ANALYZE_WALLET: [
        "Analyze wallet 0x1234567890abcdef1234567890abcdef12345678",
        "Check transaction history for 0xabcdef1234567890abcdef1234567890abcdef12",
        "Show me the activity for 0x7890abcdef1234567890abcdef1234567890abcdef",
    ],
    Intent.ANALYZE_TRANSACTION: [
        "Analyze transaction 0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        "Check if tx 0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890 is suspicious",
        "Is 0x7890abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234 a normal transaction?",
    ],
    Intent.MONITOR_ADDRESS: [
        "Monitor address 0x1234567890abcdef1234567890abcdef12345678",
        "Alert me when 0xabcdef1234567890abcdef1234567890abcdef12 receives funds",
        "Set up notifications for 0x7890abcdef1234567890abcdef1234567890abcdef",
    ],
    Intent.MONITOR_CONTRACT: [
        "Monitor contract 0x1234567890abcdef1234567890abcdef12345678",
        "Alert me when 0xabcdef1234567890abcdef1234567890abcdef12 has new events",
        "Watch for events from 0x7890abcdef1234567890abcdef1234567890abcdef",
    ],
    Intent.MONITOR_PRICE: [
        "Monitor ETH price",
        "Alert me when Bitcoin goes above $50000",
        "Notify me if USDC depegs",
    ],
}

# Additional examples with mixed intent
MIXED_QUERIES = [
    "What's the gas price on Ethereum and get me the latest block?",
    "Show me the ETH balance for 0x1234567890abcdef1234567890abcdef12345678 and monitor it",
    "Check if contract 0xabcdef1234567890abcdef1234567890abcdef12 is secure and show me its ABI",
    "What happened in the most recent transactions on Ethereum?",
    "Is 0x7890abcdef1234567890abcdef1234567890abcdef a good address to send money to?",
]


async def process_query(query: str) -> Dict[str, Any]:
    """
    Process a query through the NLP pipeline.
    
    Args:
        query: The natural language query
        
    Returns:
        Dictionary with structured results from the NLP pipeline
    """
    # Initialize NLP components
    processor = NLPProcessor()
    intent_classifier = IntentClassifier()
    entity_extractor = EntityExtractor()
    
    # Process the query
    processed_query = await processor.process(query)
    
    # Classify intent
    intent, confidence = await intent_classifier.classify(processed_query)
    
    # Extract entities
    entities = await entity_extractor.extract(processed_query, intent)
    
    # Generate structured query
    structured_query = await generate_structured_query(intent, entities)
    
    # Generate natural language response
    response = await generate_response(intent, entities)
    
    # Return all results
    result = {
        "query": query,
        "processed_query": processed_query,
        "intent": intent.value,
        "confidence": confidence,
        "entities": [{"type": e.type, "value": e.value} for e in entities],
        "structured_query": structured_query,
        "response": response,
    }
    
    return result


async def run_all_examples() -> None:
    """Run all example queries and print the results."""
    print("Running examples for each intent type...")
    print("-" * 80)
    
    for intent, queries in EXAMPLE_QUERIES.items():
        print(f"\n## Examples for {intent.value} ##")
        # Take just the first example for each intent to keep output manageable
        query = queries[0]
        print(f"\nQuery: {query}")
        
        try:
            result = await process_query(query)
            print("\nResult:")
            print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
            print("  Entities:")
            for entity in result['entities']:
                print(f"    - {entity['type']}: {entity['value']}")
            print(f"  Response: {result['response']}")
            print("  Structured Query:")
            pprint(result['structured_query'], indent=4)
        except Exception as e:
            print(f"Error processing query: {e}")
    
    print("\n" + "-" * 80)
    print("\n## Mixed Query Examples ##")
    
    for query in MIXED_QUERIES[:2]:  # Just run a couple of mixed examples
        print(f"\nQuery: {query}")
        
        try:
            result = await process_query(query)
            print("\nResult:")
            print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
            print("  Entities:")
            for entity in result['entities']:
                print(f"    - {entity['type']}: {entity['value']}")
            print(f"  Response: {result['response']}")
            print("  Structured Query:")
            pprint(result['structured_query'], indent=4)
        except Exception as e:
            print(f"Error processing query: {e}")


async def interactive_mode() -> None:
    """Run an interactive session for testing NLP queries."""
    print("Interactive NLP Query Testing")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 80)
    
    while True:
        query = input("\nEnter a query: ").strip()
        
        if query.lower() in ('exit', 'quit'):
            break
            
        if not query:
            continue
            
        try:
            result = await process_query(query)
            print("\nResult:")
            print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
            print("  Entities:")
            for entity in result['entities']:
                print(f"    - {entity['type']}: {entity['value']}")
            print(f"  Response: {result['response']}")
            print("  Structured Query:")
            pprint(result['structured_query'], indent=4)
        except Exception as e:
            print(f"Error processing query: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the NLP pipeline with example queries")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    if args.interactive:
        asyncio.run(interactive_mode())
    else:
        asyncio.run(run_all_examples()) 