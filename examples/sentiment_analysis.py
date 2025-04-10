#!/usr/bin/env python
"""
Sentiment analysis example with LlamaChain
"""

from llamachain.core import Pipeline
from llamachain.components import TextProcessor
from llamachain.nlp import SimpleSentimentAnalyzer


def main():
    """Run a sentiment analysis pipeline example"""
    print("LlamaChain Sentiment Analysis Example")
    print("=====================================\n")
    
    # Create components
    text_processor = TextProcessor(lowercase=True, normalize_whitespace=True)
    sentiment_analyzer = SimpleSentimentAnalyzer()
    
    # Create pipeline
    pipeline = Pipeline([
        text_processor,
        sentiment_analyzer,
    ])
    
    # Sample reviews
    reviews = [
        "This product is amazing! I love everything about it. The quality is excellent.",
        "I'm disappointed with this purchase. It broke after just one week of use.",
        "It's okay, but not worth the price. Some features are good, others are mediocre.",
        "Absolutely fantastic service and the staff was very helpful and friendly.",
        "Terrible experience. Avoid at all costs. The customer service was unhelpful.",
        "Not bad, but not great either. It gets the job done but don't expect anything special.",
    ]
    
    # Process each review and display results
    for i, review in enumerate(reviews):
        print(f"Review {i+1}: {review}")
        
        result = pipeline.run(review)
        
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Score: {result['score']:.2f}")
        
        if result['positive_words']:
            print(f"Positive words: {', '.join(result['positive_words'])}")
        
        if result['negative_words']:
            print(f"Negative words: {', '.join(result['negative_words'])}")
        
        print()


if __name__ == "__main__":
    main() 