"""
Text processing components for LlamaChain
"""

import re
import string
from typing import Any, Dict, List, Optional, Set

from llamachain.core import Component


class TextProcessor(Component):
    """Component for basic text processing"""

    def __init__(
        self,
        lowercase: bool = False,
        remove_punctuation: bool = False,
        normalize_whitespace: bool = False,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize TextProcessor

        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            normalize_whitespace: Whether to normalize whitespace
            name: Optional name for the component
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace

    def process(self, input_data: str) -> str:
        """Process the input text

        Args:
            input_data: The text to process

        Returns:
            The processed text
        """
        if not isinstance(input_data, str):
            raise TypeError(f"Expected string, got {type(input_data).__name__}")

        result = input_data

        # Apply transformations
        if self.lowercase:
            result = result.lower()

        if self.remove_punctuation:
            result = result.translate(str.maketrans("", "", string.punctuation))

        if self.normalize_whitespace:
            result = re.sub(r"\s+", " ", result).strip()

        return result


class TextTokenizer(Component):
    """Component for tokenizing text into words"""

    def __init__(
        self,
        pattern: str = r"\s+",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize TextTokenizer

        Args:
            pattern: Regex pattern for splitting text
            name: Optional name for the component
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        self.pattern = pattern

    def process(self, input_data: str) -> List[str]:
        """Tokenize the input text

        Args:
            input_data: The text to tokenize

        Returns:
            List of tokens
        """
        if not isinstance(input_data, str):
            raise TypeError(f"Expected string, got {type(input_data).__name__}")

        return re.split(self.pattern, input_data)


class StopWordRemover(Component):
    """Component for removing stop words from a list of tokens"""

    # Default English stop words
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
        stop_words: Optional[Set[str]] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize StopWordRemover

        Args:
            stop_words: Set of stop words to remove (uses default if None)
            name: Optional name for the component
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        self.stop_words = stop_words or self.DEFAULT_STOP_WORDS

    def process(self, input_data: List[str]) -> List[str]:
        """Remove stop words from the input tokens

        Args:
            input_data: List of tokens to filter

        Returns:
            List of tokens with stop words removed
        """
        if not isinstance(input_data, list):
            raise TypeError(f"Expected list, got {type(input_data).__name__}")

        return [token for token in input_data if token.lower() not in self.stop_words]
