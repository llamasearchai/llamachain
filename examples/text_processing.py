#!/usr/bin/env python
"""
Basic text processing example with LlamaChain
"""

from llamachain.components import StopWordRemover, TextProcessor, TextTokenizer
from llamachain.core import Pipeline


def main():
    """Run a basic text processing pipeline example"""
    print("LlamaChain Text Processing Example")
    print("==================================\n")

    # Create components
    text_processor = TextProcessor(lowercase=True, remove_punctuation=True)
    tokenizer = TextTokenizer()
    stop_word_remover = StopWordRemover()

    # Create pipeline
    pipeline = Pipeline(
        [
            text_processor,  # Convert to lowercase and remove punctuation
            tokenizer,  # Split text into tokens
            stop_word_remover,  # Remove common stop words
        ]
    )

    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming how we interact with technology.",
        "Machine learning algorithms improve with more data and better training.",
    ]

    # Process each text and display results
    for i, text in enumerate(texts):
        print(f"Example {i+1}:")
        print(f"Input: {text}")

        result = pipeline.run(text)

        print(f"Output: {result}")
        print()

    # Demonstrate pipeline steps
    print("Pipeline Step-by-Step:")
    print("----------------------")

    sample_text = "The quick brown fox jumps over the lazy dog."
    print(f"Original: {sample_text}")

    step1 = text_processor.process(sample_text)
    print(f"After TextProcessor: {step1}")

    step2 = tokenizer.process(step1)
    print(f"After TextTokenizer: {step2}")

    step3 = stop_word_remover.process(step2)
    print(f"After StopWordRemover: {step3}")


if __name__ == "__main__":
    main()
