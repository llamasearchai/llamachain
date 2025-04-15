"""
Tests for document analyzer components.
"""

from datetime import datetime

import pytest

from llamachain.components import TextProcessor
from llamachain.core import Component, Pipeline
from llamachain.nlp import KeywordExtractor, SimpleSentimentAnalyzer


class TestDocumentSplitter:
    """Tests for DocumentSplitter class"""

    def test_split_paragraphs(self):
        """Test splitting a document into paragraphs"""
        from master_repositories.llamachain.examples.document_analyzer import (
            DocumentSplitter,
        )

        # Create test data
        test_document = """First paragraph.

Second paragraph with more text.

Third paragraph that's a bit longer than the others and has multiple sentences.

Short."""

        # Create document splitter with min_length=10
        splitter = DocumentSplitter(min_length=10)

        # Process the document
        result = splitter.process(test_document)

        # Verify results
        assert len(result) == 3  # 'Short' should be excluded due to min_length
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph with more text."
        assert result[2].startswith("Third paragraph")

    def test_empty_document(self):
        """Test handling of an empty document"""
        from master_repositories.llamachain.examples.document_analyzer import (
            DocumentSplitter,
        )

        # Create document splitter
        splitter = DocumentSplitter()

        # Process an empty document
        result = splitter.process("")

        # Verify results
        assert result == []

    def test_input_validation(self):
        """Test that non-string inputs raise TypeError"""
        from master_repositories.llamachain.examples.document_analyzer import (
            DocumentSplitter,
        )

        # Create document splitter
        splitter = DocumentSplitter()

        # Test with non-string input
        with pytest.raises(TypeError):
            splitter.process(123)


class TestSummaryGenerator:
    """Tests for SummaryGenerator class"""

    def test_summary_generation(self):
        """Test summary generation from analysis results"""
        from master_repositories.llamachain.examples.document_analyzer import (
            SummaryGenerator,
        )

        # Create test data
        test_data = {
            "document_length": 500,
            "paragraph_count": 3,
            "paragraphs": ["para1", "para2", "para3"],
            "keywords": [
                {"word": "test", "count": 5, "score": 0.8},
                {"word": "example", "count": 3, "score": 0.6},
            ],
            "sentiment_by_paragraph": {
                "paragraph_1": {"sentiment": "positive", "score": 0.8},
                "paragraph_2": {"sentiment": "neutral", "score": 0.5},
                "paragraph_3": {"sentiment": "negative", "score": 0.7},
            },
        }

        # Create summary generator
        generator = SummaryGenerator()

        # Generate summary
        result = generator.process(test_data)

        # Verify results
        assert "summary" in result
        assert "Document Analysis Summary" in result["summary"]
        assert "contains 3 paragraphs" in result["summary"]
        assert "1 positive, 1 negative, 1 neutral" in result["summary"]
        assert "test, example" in result["summary"]

    def test_empty_input(self):
        """Test handling of empty input data"""
        from master_repositories.llamachain.examples.document_analyzer import (
            SummaryGenerator,
        )

        # Create summary generator
        generator = SummaryGenerator()

        # Generate summary from empty data
        result = generator.process({})

        # Verify results
        assert "summary" in result
        assert "Document Analysis Summary" in result["summary"]
        assert "contains 0 paragraphs" in result["summary"]
        assert "None found" in result["summary"]

    def test_input_validation(self):
        """Test that non-dict inputs raise TypeError"""
        from master_repositories.llamachain.examples.document_analyzer import (
            SummaryGenerator,
        )

        # Create summary generator
        generator = SummaryGenerator()

        # Test with non-dict input
        with pytest.raises(TypeError):
            generator.process("not a dict")


class TestDocumentAnalyzer:
    """Tests for DocumentAnalyzer class"""

    def test_analyze_document(self):
        """Test document analysis workflow"""
        from master_repositories.llamachain.examples.document_analyzer import (
            DocumentAnalyzer,
        )

        # Create test document
        test_document = """
        Artificial Intelligence
        
        AI is changing the world in many positive ways.
        
        However, there are some concerns about AI safety.
        """

        # Create analyzer
        analyzer = DocumentAnalyzer()

        # Analyze document
        result = analyzer.analyze_document(test_document)

        # Verify result structure
        assert "document_length" in result
        assert "paragraph_count" in result
        assert "paragraphs" in result
        assert "keywords" in result
        assert "sentiment_by_paragraph" in result
        assert "summary" in result

        # Verify paragraph extraction
        assert len(result["paragraphs"]) == 3

        # Verify keyword extraction
        assert len(result["keywords"]) > 0

        # Verify sentiment analysis
        assert len(result["sentiment_by_paragraph"]) == 3

        # Verify summary
        assert "Document Analysis Summary" in result["summary"]
