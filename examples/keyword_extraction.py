#!/usr/bin/env python
"""
Keyword extraction example with LlamaChain
"""

from llamachain.components import TextProcessor
from llamachain.core import Pipeline
from llamachain.nlp import KeywordExtractor


def main():
    """Run a keyword extraction pipeline example"""
    print("LlamaChain Keyword Extraction Example")
    print("===================================\n")

    # Create components
    text_processor = TextProcessor(lowercase=True, normalize_whitespace=True)
    keyword_extractor = KeywordExtractor(max_keywords=5)

    # Create pipeline
    pipeline = Pipeline(
        [
            text_processor,
            keyword_extractor,
        ]
    )

    # Sample documents
    documents = [
        """
        Machine learning is a branch of artificial intelligence and computer science which 
        focuses on the use of data and algorithms to imitate the way that humans learn, 
        gradually improving its accuracy. Machine learning is an important component of the 
        growing field of data science.
        """,
        """
        Natural language processing (NLP) is a subfield of linguistics, computer science, and
        artificial intelligence concerned with the interactions between computers and human
        language, in particular how to program computers to process and analyze large amounts
        of natural language data.
        """,
        """
        Computer vision is an interdisciplinary scientific field that deals with how computers
        can gain high-level understanding from digital images or videos. From the perspective
        of engineering, it seeks to understand and automate tasks that the human visual system
        can do.
        """,
    ]

    # Process each document and display results
    for i, document in enumerate(documents):
        print(f"Document {i+1}:")
        print(f"{document.strip()}\n")

        keywords = pipeline.run(document)

        print("Top keywords:")
        for keyword in keywords:
            print(
                f"- {keyword['word']} (count: {keyword['count']}, score: {keyword['score']:.2f})"
            )

        print()

    # Demonstrate custom stop words
    print("Example with Custom Stop Words:")
    print("------------------------------")

    tech_stop_words = {
        "machine",
        "learning",
        "data",
        "computer",
        "artificial",
        "intelligence",
        "science",
        "field",
        "algorithm",
    }

    custom_extractor = KeywordExtractor(
        max_keywords=5,
        stop_words=tech_stop_words.union(KeywordExtractor.DEFAULT_STOP_WORDS),
    )

    custom_pipeline = Pipeline(
        [
            text_processor,
            custom_extractor,
        ]
    )

    print(f"Document: {documents[0].strip()}\n")
    print("Keywords with tech terms filtered out:")

    keywords = custom_pipeline.run(documents[0])

    for keyword in keywords:
        print(
            f"- {keyword['word']} (count: {keyword['count']}, score: {keyword['score']:.2f})"
        )


if __name__ == "__main__":
    main()
