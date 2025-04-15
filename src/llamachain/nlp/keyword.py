"""
Keyword extraction components for LlamaChain
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Union

from llamachain.core import Component


class KeywordExtractor(Component):
    """Component for extracting important keywords from text"""

    # Default English stop words for keyword extraction
    DEFAULT_STOP_WORDS = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "because",
        "as",
        "what",
        "which",
        "this",
        "that",
        "these",
        "those",
        "then",
        "just",
        "so",
        "than",
        "such",
        "when",
        "who",
        "whom",
        "how",
        "where",
        "why",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "of",
        "to",
        "in",
        "for",
        "on",
        "by",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "from",
        "up",
        "down",
        "with",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "too",
        "very",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
    }

    def __init__(
        self,
        max_keywords: int = 10,
        min_word_length: int = 3,
        stop_words: Optional[Set[str]] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize KeywordExtractor

        Args:
            max_keywords: Maximum number of keywords to extract
            min_word_length: Minimum length of words to consider
            stop_words: Set of stop words to exclude (uses default if None)
            name: Optional name for the component
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        self.max_keywords = max_keywords
        self.min_word_length = min_word_length
        self.stop_words = stop_words or self.DEFAULT_STOP_WORDS

    def process(self, input_text: str) -> List[Dict[str, Any]]:
        """Extract important keywords from input text

        Args:
            input_text: Text to extract keywords from

        Returns:
            List of dictionaries for each keyword:
            [
                {"word": "keyword1", "count": 5, "score": 0.75},
                {"word": "keyword2", "count": 3, "score": 0.45},
                ...
            ]
        """
        if not isinstance(input_text, str):
            raise TypeError(f"Expected string, got {type(input_text).__name__}")

        # Tokenize text
        words = re.findall(r"\b[a-zA-Z]+\b", input_text.lower())

        # Filter words
        filtered_words = [
            word
            for word in words
            if len(word) >= self.min_word_length and word not in self.stop_words
        ]

        # Count word frequencies
        word_counts = Counter(filtered_words)

        # Calculate TF (Term Frequency) for each word
        total_words = len(filtered_words)
        word_tf = {word: count / total_words for word, count in word_counts.items()}

        # Calculate "scores" - in a real implementation, this would use TF-IDF
        # Here we'll use a simplified formula based on frequency and word length
        word_scores = {}
        max_count = max(word_counts.values()) if word_counts else 1

        for word, count in word_counts.items():
            # Score formula: normalized count * log(word length)
            normalized_count = count / max_count
            length_factor = math.log(len(word) + 1) / math.log(
                20
            )  # Normalize to 0-1 range
            word_scores[word] = normalized_count * (0.5 + 0.5 * length_factor)

        # Sort words by score and take top keywords
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        top_keywords = sorted_words[: self.max_keywords]

        # Format results
        results = [
            {"word": word, "count": word_counts[word], "score": score}
            for word, score in top_keywords
        ]

        return results
