#!/usr/bin/env python
"""
HuggingFace sentiment analysis example with LlamaChain
"""

from llamachain.core import Pipeline
from llamachain.components import TextProcessor
from llamachain.ml import HuggingFaceModel, ModelInference


def main():
    """Run a HuggingFace sentiment analysis pipeline example"""
    print("LlamaChain HuggingFace Sentiment Analysis Example")
    print("===============================================\n")
    
    print("Note: This example requires the transformers and torch packages:")
    print("pip install transformers torch\n")
    
    # Create components
    text_processor = TextProcessor(normalize_whitespace=True)
    
    # Initialize the HuggingFace model (will download if not cached)
    print("Initializing HuggingFace model (may download if not cached)...")
    model = HuggingFaceModel(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        model_type="text-classification"
    )
    
    # Create inference component
    inference = ModelInference(model)
    
    # Create pipeline
    pipeline = Pipeline([
        text_processor,
        inference,
    ])
    
    # Sample texts to classify
    texts = [
        "I absolutely loved this movie, it was amazing!",
        "The service at this restaurant was terrible and the food was bland.",
        "The book was neither particularly good nor particularly bad.",
        "This product exceeded all my expectations and I would highly recommend it.",
        "I was very disappointed with the quality and performance.",
        "It's decent for the price, but there are better options out there."
    ]
    
    print("Running sentiment analysis on sample texts...\n")
    
    try:
        # Process each text and display results
        for i, text in enumerate(texts):
            print(f"Text {i+1}: {text}")
            
            # Catch any model errors
            try:
                result = pipeline.run(text)
                
                print(f"Predicted label: {result['label']}")
                print(f"Confidence: {result['score']:.4f}\n")
            except ImportError:
                print("Could not run model inference: required packages are not installed.")
                print("Please install: pip install transformers torch\n")
                break
            except Exception as e:
                print(f"Error processing text: {e}\n")
    
    except ImportError:
        print("\nThis example requires the transformers and torch packages:")
        print("pip install transformers torch")
        print("\nUsing mock results instead:\n")
        
        # Provide mock results if packages aren't installed
        mock_results = [
            {"label": "POSITIVE", "score": 0.9978},
            {"label": "NEGATIVE", "score": 0.9945},
            {"label": "NEUTRAL", "score": 0.5467},
            {"label": "POSITIVE", "score": 0.9992},
            {"label": "NEGATIVE", "score": 0.9834},
            {"label": "NEUTRAL", "score": 0.6231},
        ]
        
        for i, (text, result) in enumerate(zip(texts, mock_results)):
            print(f"Text {i+1}: {text}")
            print(f"Predicted label: {result['label']}")
            print(f"Confidence: {result['score']:.4f}\n")


if __name__ == "__main__":
    main() 