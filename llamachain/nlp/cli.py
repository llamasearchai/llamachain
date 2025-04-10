"""
Command-line interface for the NLP module.

This module provides a command-line interface for interacting with the NLP capabilities.
"""

import argparse
import asyncio
import json
import sys
from typing import Dict, List, Any, Optional

from llamachain.log import get_logger
from llamachain.nlp.processor import NLPProcessor
from llamachain.nlp.intent import IntentClassifier, Intent
from llamachain.nlp.entity import EntityExtractor, Entity
from llamachain.nlp.generation import generate_response, generate_structured_query

# Get logger
logger = get_logger("llamachain.nlp.cli")


async def process_query(args: argparse.Namespace) -> int:
    """
    Process a natural language query.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not args.query:
        print("Error: Query is required")
        return 1
    
    try:
        # Create processor
        processor = NLPProcessor()
        intent_classifier = IntentClassifier()
        entity_extractor = EntityExtractor()
        
        # Process query
        processed_query = await processor.process(args.query)
        
        # Classify intent
        intent, confidence = await intent_classifier.classify(processed_query)
        
        # Extract entities
        entities = await entity_extractor.extract(processed_query, intent)
        
        # Create result dictionary
        result = {
            "original_query": args.query,
            "processed_query": processed_query,
            "intent": {
                "type": intent.value,
                "description": intent.get_description(),
                "confidence": confidence
            },
            "entities": [entity.to_dict() for entity in entities]
        }
        
        # Add response if requested
        if args.response:
            response_text = await generate_response(intent, entities)
            result["response"] = response_text
        
        # Add structured query if requested
        if args.structured:
            structured_query = await generate_structured_query(intent, entities)
            result["structured_query"] = structured_query
        
        # Print results in the requested format
        if args.format == "json":
            if args.pretty:
                print(json.dumps(result, indent=2))
            else:
                print(json.dumps(result))
        else:
            print(f"Query: {result['original_query']}")
            print(f"Processed: {result['processed_query']}")
            print(f"Intent: {result['intent']['type']} ({result['intent']['description']})")
            print(f"Confidence: {result['intent']['confidence']:.2f}")
            
            print("\nEntities:")
            if result['entities']:
                for entity in result['entities']:
                    print(f"- {entity['type']}: {entity['value']} (confidence: {entity['confidence']:.2f})")
            else:
                print("No entities found")
            
            if args.response:
                print(f"\nResponse: {result['response']}")
            
            if args.structured:
                print("\nStructured Query:")
                print(f"Type: {result['structured_query']['type']}")
                print("Parameters:")
                for key, value in result['structured_query']['params'].items():
                    print(f"- {key}: {value}")
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Error processing query: {str(e)}")
        return 1


async def classify_intent(args: argparse.Namespace) -> int:
    """
    Classify the intent of a query.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not args.query:
        print("Error: Query is required")
        return 1
    
    try:
        # Create processor and classifier
        processor = NLPProcessor()
        classifier = IntentClassifier()
        
        # Process query
        processed_query = await processor.process(args.query)
        
        # Classify intent
        intent, confidence = await classifier.classify(processed_query)
        
        # Get confidence scores
        scores = classifier.get_confidence_scores(processed_query)
        
        # Print results in the requested format
        if args.format == "json":
            result = {
                "query": args.query,
                "processed_query": processed_query,
                "intent": intent.value,
                "description": intent.get_description(),
                "confidence": confidence,
                "confidence_scores": {intent_type.value: score for intent_type, score in scores.items()}
            }
            
            if args.pretty:
                print(json.dumps(result, indent=2))
            else:
                print(json.dumps(result))
        else:
            print(f"Query: {args.query}")
            print(f"Processed: {processed_query}")
            print(f"Intent: {intent.value} ({intent.get_description()})")
            print(f"Confidence: {confidence:.2f}")
            
            print("\nConfidence Scores:")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for intent_type, score in sorted_scores:
                if score > 0:
                    print(f"- {intent_type.value}: {score:.4f}")
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Error classifying intent: {str(e)}")
        return 1


async def extract_entities(args: argparse.Namespace) -> int:
    """
    Extract entities from a query.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not args.query:
        print("Error: Query is required")
        return 1
    
    try:
        # Create processor, classifier and extractor
        processor = NLPProcessor()
        classifier = IntentClassifier()
        extractor = EntityExtractor()
        
        # Process query
        processed_query = await processor.process(args.query)
        
        # Classify intent
        intent, _ = await classifier.classify(processed_query)
        
        # Extract entities
        entities = await extractor.extract(processed_query, intent)
        
        # Print results in the requested format
        if args.format == "json":
            result = {
                "query": args.query,
                "processed_query": processed_query,
                "intent": intent.value,
                "entities": [entity.to_dict() for entity in entities]
            }
            
            if args.pretty:
                print(json.dumps(result, indent=2))
            else:
                print(json.dumps(result))
        else:
            print(f"Query: {args.query}")
            print(f"Processed: {processed_query}")
            print(f"Intent: {intent.value} ({intent.get_description()})")
            
            print("\nExtracted Entities:")
            if entities:
                for entity in entities:
                    print(f"- {entity.type}: {entity.value} (confidence: {entity.confidence:.2f})")
            else:
                print("No entities found")
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Error extracting entities: {str(e)}")
        return 1


async def generate_nlp_response(args: argparse.Namespace) -> int:
    """
    Generate a natural language response to a query.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    if not args.query:
        print("Error: Query is required")
        return 1
    
    try:
        # Create processor, classifier and extractor
        processor = NLPProcessor()
        classifier = IntentClassifier()
        extractor = EntityExtractor()
        
        # Process query
        processed_query = await processor.process(args.query)
        
        # Classify intent
        intent, confidence = await classifier.classify(processed_query)
        
        # Extract entities
        entities = await extractor.extract(processed_query, intent)
        
        # Generate response
        response_text = await generate_response(intent, entities)
        
        # Print results in the requested format
        if args.format == "json":
            result = {
                "query": args.query,
                "processed_query": processed_query,
                "intent": intent.value,
                "confidence": confidence,
                "entities": [entity.to_dict() for entity in entities],
                "response": response_text
            }
            
            if args.pretty:
                print(json.dumps(result, indent=2))
            else:
                print(json.dumps(result))
        else:
            print(f"Query: {args.query}")
            print(f"Processed: {processed_query}")
            print(f"Intent: {intent.value} ({intent.get_description()})")
            print(f"Confidence: {confidence:.2f}")
            
            print("\nExtracted Entities:")
            if entities:
                for entity in entities:
                    print(f"- {entity.type}: {entity.value} (confidence: {entity.confidence:.2f})")
            else:
                print("No entities found")
            
            print(f"\nResponse: {response_text}")
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Error generating response: {str(e)}")
        return 1


def setup_parser() -> argparse.ArgumentParser:
    """
    Set up the argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(description="LlamaChain NLP Command-Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    common_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a natural language query", parents=[common_parser])
    process_parser.add_argument("--query", "-q", required=True, help="Query to process")
    process_parser.add_argument("--response", "-r", action="store_true", help="Generate a response")
    process_parser.add_argument("--structured", "-s", action="store_true", help="Generate a structured query")
    
    # Intent command
    intent_parser = subparsers.add_parser("intent", help="Classify intent of a query", parents=[common_parser])
    intent_parser.add_argument("--query", "-q", required=True, help="Query to classify")
    
    # Entity command
    entity_parser = subparsers.add_parser("entity", help="Extract entities from a query", parents=[common_parser])
    entity_parser.add_argument("--query", "-q", required=True, help="Query to extract entities from")
    
    # Response command
    response_parser = subparsers.add_parser("response", help="Generate a response for a query", parents=[common_parser])
    response_parser.add_argument("--query", "-q", required=True, help="Query to generate a response for")
    
    return parser


async def main_async() -> int:
    """
    Main async function.
    
    Returns:
        Exit code
    """
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    if args.command == "process":
        return await process_query(args)
    elif args.command == "intent":
        return await classify_intent(args)
    elif args.command == "entity":
        return await extract_entities(args)
    elif args.command == "response":
        return await generate_nlp_response(args)
    else:
        parser.print_help()
        return 0


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code
    """
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 