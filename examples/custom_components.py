#!/usr/bin/env python
"""
Custom components example with LlamaChain
"""

from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from llamachain.core import Component, Pipeline


class CharacterCounter(Component):
    """Custom component to count occurrences of each character"""

    def process(self, input_data: str) -> Dict[str, int]:
        """Count occurrences of each character in the input text

        Args:
            input_data: The input text to process

        Returns:
            Dictionary with character counts
        """
        if not isinstance(input_data, str):
            raise TypeError(f"Expected string, got {type(input_data).__name__}")

        counts = {}
        for char in input_data:
            if char in counts:
                counts[char] += 1
            else:
                counts[char] = 1

        return counts


class DictionarySorter(Component):
    """Custom component to sort dictionary items by value"""

    def __init__(self, reverse=True, name=None, config=None):
        """Initialize DictionarySorter

        Args:
            reverse: Whether to sort in descending order (default) or ascending
            name: Optional name for the component
            config: Optional configuration dictionary
        """
        self.reverse = reverse
        super().__init__(name, config)

    def process(self, input_data: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Sort dictionary items by value

        Args:
            input_data: Dictionary to sort

        Returns:
            List of (key, value) tuples sorted by value
        """
        if not isinstance(input_data, dict):
            raise TypeError(f"Expected dictionary, got {type(input_data).__name__}")

        return sorted(input_data.items(), key=lambda x: x[1], reverse=self.reverse)


class TextStatistics(Component):
    """Custom component to generate statistics about text"""

    def __init__(self, name=None, config=None):
        """Initialize TextStatistics

        Args:
            name: Optional name for the component
            config: Optional configuration dictionary
        """
        super().__init__(name, config)

    def process(self, input_data: str) -> Dict[str, Any]:
        """Calculate statistics about the input text

        Args:
            input_data: The input text to analyze

        Returns:
            Dictionary with text statistics
        """
        if not isinstance(input_data, str):
            raise TypeError(f"Expected string, got {type(input_data).__name__}")

        # Count characters
        char_count = len(input_data)

        # Count words
        words = input_data.split()
        word_count = len(words)

        # Count unique words
        unique_words = set(word.lower() for word in words)
        unique_word_count = len(unique_words)

        # Calculate average word length
        avg_word_length = (
            sum(len(word) for word in words) / word_count if word_count > 0 else 0
        )

        # Count sentences (simple approach)
        sentence_count = (
            input_data.count(".") + input_data.count("!") + input_data.count("?")
        )

        # Word frequency
        word_freq = Counter(word.lower() for word in words)
        top_words = word_freq.most_common(5)

        return {
            "character_count": char_count,
            "word_count": word_count,
            "unique_word_count": unique_word_count,
            "average_word_length": avg_word_length,
            "sentence_count": sentence_count,
            "top_words": top_words,
        }


def main():
    """Run a custom components pipeline example"""
    print("LlamaChain Custom Components Example")
    print("===================================\n")

    # Example 1: Character counter and sorter
    print("Example 1: Character Counter and Sorter")
    print("--------------------------------------")

    # Create pipeline with custom components
    char_pipeline = Pipeline(
        [
            CharacterCounter(),
            DictionarySorter(reverse=True),
        ]
    )

    # Process text
    text = "hello world"
    result = char_pipeline.run(text)

    print(f"Input text: {text}")
    print("Character counts (most frequent first):")
    for char, count in result:
        print(f"'{char}': {count}")

    # Example 2: Text statistics
    print("\nExample 2: Text Statistics")
    print("------------------------")

    # Create text statistics component
    stats_component = TextStatistics()

    # Process text
    sample_text = "The quick brown fox jumps over the lazy dog. Fox is a swift animal. Dogs are loyal pets!"
    stats = stats_component.process(sample_text)

    print(f"Input text: {sample_text}")
    print(f"Character count: {stats['character_count']}")
    print(f"Word count: {stats['word_count']}")
    print(f"Unique word count: {stats['unique_word_count']}")
    print(f"Average word length: {stats['average_word_length']:.2f} characters")
    print(f"Sentence count: {stats['sentence_count']}")

    print("Top 5 words:")
    for word, count in stats["top_words"]:
        print(f"- '{word}': {count}")

    # Example 3: Combining with built-in components
    print("\nExample 3: Combining Custom and Built-in Components")
    print("------------------------------------------------")

    # Import built-in components
    from llamachain.components import TextProcessor, TextTokenizer

    # Create pipeline mixing custom and built-in components
    mixed_pipeline = Pipeline(
        [
            TextProcessor(lowercase=True, remove_punctuation=True),
            TextTokenizer(),
            # Custom component that takes a list of tokens and returns word counts
            Component(name="WordCounter", process=lambda tokens: Counter(tokens)),
            DictionarySorter(reverse=True),
        ]
    )

    # Process text
    complex_text = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence."
    mixed_result = mixed_pipeline.run(complex_text)

    print(f"Input text: {complex_text}")
    print("Word counts (most frequent first):")
    for word, count in mixed_result[:5]:  # Show top 5
        print(f"'{word}': {count}")


if __name__ == "__main__":
    main()
