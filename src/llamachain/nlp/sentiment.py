"""
Sentiment analysis components for LlamaChain
"""

import re
from typing import Any, Dict, List, Optional, Union

from llamachain.core import Component


class SimpleSentimentAnalyzer(Component):
    """Component for simple lexicon-based sentiment analysis"""
    
    # Simple sentiment lexicons
    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "awesome", "outstanding", "superb", "brilliant", "terrific", "love",
        "happy", "joy", "positive", "perfect", "best", "superior", "exceptional",
        "remarkable", "impressive", "extraordinary", "phenomenal", "incredible"
    }
    
    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "poor", "disappointing",
        "mediocre", "dreadful", "subpar", "inadequate", "inferior", "worst",
        "hate", "dislike", "negative", "sad", "unhappy", "awful", "annoying",
        "frustrating", "pathetic", "useless", "worthless", "unacceptable"
    }
    
    INTENSIFIERS = {
        "very", "extremely", "really", "absolutely", "truly", "completely",
        "totally", "thoroughly", "entirely", "utterly", "highly", "especially",
        "particularly", "exceedingly", "immensely", "intensely"
    }
    
    NEGATIONS = {
        "not", "no", "never", "none", "nobody", "nothing", "neither",
        "nor", "nowhere", "hardly", "scarcely", "barely", "doesn't",
        "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
        "haven't", "hasn't", "hadn't", "won't", "wouldn't", "can't",
        "cannot", "couldn't", "shouldn't", "mightn't", "mustn't"
    }
    
    def __init__(
        self,
        positive_words: Optional[set] = None,
        negative_words: Optional[set] = None,
        intensifiers: Optional[set] = None,
        negations: Optional[set] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize SimpleSentimentAnalyzer
        
        Args:
            positive_words: Set of positive sentiment words (uses default if None)
            negative_words: Set of negative sentiment words (uses default if None)
            intensifiers: Set of intensifier words (uses default if None)
            negations: Set of negation words (uses default if None)
            name: Optional name for the component
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        self.positive_words = positive_words or self.POSITIVE_WORDS
        self.negative_words = negative_words or self.NEGATIVE_WORDS
        self.intensifiers = intensifiers or self.INTENSIFIERS
        self.negations = negations or self.NEGATIONS
    
    def process(self, input_text: str) -> Dict[str, Any]:
        """Analyze sentiment of input text
        
        Args:
            input_text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results:
            {
                "sentiment": "positive", "negative", or "neutral",
                "score": Sentiment score (-1.0 to 1.0),
                "positive_words": List of positive words found,
                "negative_words": List of negative words found
            }
        """
        if not isinstance(input_text, str):
            raise TypeError(f"Expected string, got {type(input_text).__name__}")
        
        # Tokenize text
        words = re.findall(r'\b[a-zA-Z]+\b', input_text.lower())
        
        # Track sentiment
        positive_count = 0
        negative_count = 0
        found_positive_words = []
        found_negative_words = []
        
        # Track negation and intensifiers
        is_negated = False
        intensifier = 1.0
        
        for i, word in enumerate(words):
            # Check for negation
            if word in self.negations:
                is_negated = True
                continue
            
            # Check for intensifiers
            if word in self.intensifiers:
                intensifier = 1.5
                continue
            
            # Check sentiment of word
            if word in self.positive_words:
                if is_negated:
                    negative_count += intensifier
                    found_negative_words.append(f"not {word}")
                else:
                    positive_count += intensifier
                    found_positive_words.append(word)
            
            elif word in self.negative_words:
                if is_negated:
                    positive_count += intensifier
                    found_positive_words.append(f"not {word}")
                else:
                    negative_count += intensifier
                    found_negative_words.append(word)
            
            # Reset negation and intensifier flags after sentiment word
            if word in self.positive_words or word in self.negative_words:
                is_negated = False
                intensifier = 1.0
            
            # Reset negation after a few words if not used
            if is_negated and i > 0 and (i % 3 == 0):
                is_negated = False
        
        # Calculate sentiment score (-1.0 to 1.0)
        total_count = positive_count + negative_count
        if total_count == 0:
            score = 0.0
        else:
            score = (positive_count - negative_count) / total_count
        
        # Determine sentiment label
        if score > 0.1:
            sentiment = "positive"
        elif score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": score,
            "positive_words": found_positive_words,
            "negative_words": found_negative_words
        } 