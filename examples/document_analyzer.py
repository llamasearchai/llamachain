#!/usr/bin/env python
"""
Document analysis pipeline example with LlamaChain

This example demonstrates a comprehensive document analysis pipeline
that combines multiple components to process documents, extract keywords,
perform sentiment analysis, and generate a summary report.
"""

from datetime import datetime
from typing import Any, Dict, List

from llamachain.components import TextProcessor
from llamachain.core import Component, Pipeline
from llamachain.nlp import KeywordExtractor, SimpleSentimentAnalyzer


class DocumentSplitter(Component):
    """Custom component to split documents into paragraphs"""

    def __init__(self, min_length: int = 20, name=None, config=None):
        """Initialize the document splitter

        Args:
            min_length: Minimum length of paragraph to keep
            name: Optional component name
            config: Optional component configuration
        """
        self.min_length = min_length
        super().__init__(name, config)

    def process(self, input_data: str) -> List[str]:
        """Split document into paragraphs

        Args:
            input_data: Text document to split

        Returns:
            List of paragraphs
        """
        if not isinstance(input_data, str):
            raise TypeError(f"Expected string, got {type(input_data).__name__}")

        paragraphs = [p.strip() for p in input_data.split("\n\n")]
        return [p for p in paragraphs if len(p) >= self.min_length]


class SummaryGenerator(Component):
    """Custom component to generate a summary from analysis results"""

    def __init__(self, name=None, config=None):
        """Initialize the summary generator

        Args:
            name: Optional component name
            config: Optional component configuration
        """
        super().__init__(name, config)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary from sentiment and keyword analysis

        Args:
            input_data: Dictionary with analysis results

        Returns:
            Updated dictionary with added summary
        """
        if not isinstance(input_data, dict):
            raise TypeError(f"Expected dictionary, got {type(input_data).__name__}")

        # Extract sentiment statistics
        sentiments = input_data.get("sentiment_by_paragraph", {})
        pos_count = sum(1 for s in sentiments.values() if s["sentiment"] == "positive")
        neg_count = sum(1 for s in sentiments.values() if s["sentiment"] == "negative")
        neutral_count = sum(
            1 for s in sentiments.values() if s["sentiment"] == "neutral"
        )

        # Extract keyword information
        keywords = input_data.get("keywords", [])

        # Generate summary text
        summary = (
            f"Document Analysis Summary:\n"
            f"- Document contains {len(sentiments)} paragraphs\n"
            f"- Overall sentiment: {pos_count} positive, {neg_count} negative, {neutral_count} neutral paragraphs\n"
            f"- Primary sentiment: {max(['positive', 'negative', 'neutral'], key=lambda s: [pos_count, neg_count, neutral_count][['positive', 'negative', 'neutral'].index(s)])}\n"
            f"- Top keywords: {', '.join(kw['word'] for kw in keywords[:5]) if keywords else 'None found'}\n"
            f"- Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Return updated results with summary
        result = input_data.copy()
        result["summary"] = summary
        return result


class DocumentAnalyzer:
    """Helper class to analyze documents using LlamaChain pipelines"""

    def __init__(self):
        """Initialize the document analyzer with components and pipelines"""
        # Create text processing components
        self.text_processor = TextProcessor(lowercase=True, normalize_whitespace=True)
        self.document_splitter = DocumentSplitter(min_length=30)
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor(max_keywords=10)
        self.summary_generator = SummaryGenerator()

        # Create individual pipelines for different tasks
        self.preprocess_pipeline = Pipeline(
            [
                self.text_processor,
                self.document_splitter,
            ]
        )

        self.keyword_pipeline = Pipeline(
            [
                self.text_processor,
                self.keyword_extractor,
            ]
        )

    def analyze_document(self, document_text: str) -> Dict[str, Any]:
        """Analyze a document using multiple pipelines

        Args:
            document_text: The document to analyze

        Returns:
            Dictionary with analysis results and summary
        """
        # Process document
        paragraphs = self.preprocess_pipeline.run(document_text)

        # Extract keywords from entire document
        keywords = self.keyword_pipeline.run(document_text)

        # Analyze sentiment of each paragraph
        sentiment_by_paragraph = {}
        for i, paragraph in enumerate(paragraphs):
            sentiment = self.sentiment_analyzer.process(paragraph)
            sentiment_by_paragraph[f"paragraph_{i+1}"] = sentiment

        # Collect all results
        results = {
            "document_length": len(document_text),
            "paragraph_count": len(paragraphs),
            "paragraphs": paragraphs,
            "keywords": keywords,
            "sentiment_by_paragraph": sentiment_by_paragraph,
        }

        # Generate summary
        final_results = self.summary_generator.process(results)

        return final_results


def main():
    """Run a document analysis pipeline example"""
    print("LlamaChain Document Analysis Example")
    print("====================================\n")

    # Sample document about AI
    sample_document = """
Artificial Intelligence: Transforming the Future

Artificial intelligence (AI) has rapidly evolved over the past decade, transforming from a niche research field into a technology that touches almost every aspect of our daily lives.

The growth of machine learning, particularly deep learning, has been the driving force behind this AI revolution. Neural networks, inspired by the human brain, have demonstrated remarkable abilities in various domains from image recognition to natural language processing.

However, the rise of AI also brings challenges. Ethical concerns regarding privacy, bias, and job displacement have become increasingly important as these technologies become more pervasive.

Despite these challenges, the potential benefits of AI are enormous. In healthcare, AI systems can help diagnose diseases earlier and more accurately than human doctors in some cases. In transportation, autonomous vehicles promise to reduce accidents and traffic congestion while increasing mobility for those who cannot drive.
"""

    print("Analyzing sample document...\n")

    # Create analyzer and analyze document
    analyzer = DocumentAnalyzer()
    results = analyzer.analyze_document(sample_document)

    # Display results
    print(f"Document length: {results['document_length']} characters")
    print(f"Paragraph count: {results['paragraph_count']}")

    print("\nExtracted paragraphs:")
    for i, paragraph in enumerate(results["paragraphs"]):
        if i < 2:  # Only print first two paragraphs to keep output concise
            print(f"Paragraph {i+1}: {paragraph[:100]}...")

    print("\nTop keywords:")
    for keyword in results["keywords"][:5]:
        print(f"- {keyword['word']} (score: {keyword['score']:.2f})")

    print("\nSentiment analysis by paragraph:")
    for para_id, sentiment_data in results["sentiment_by_paragraph"].items():
        print(
            f"{para_id}: {sentiment_data['sentiment'].upper()} (score: {sentiment_data['score']:.2f})"
        )

    print("\nSummary Report:")
    print(results["summary"])


if __name__ == "__main__":
    main()
