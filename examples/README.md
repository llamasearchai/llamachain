# LlamaChain Examples

This directory contains examples demonstrating the usage of LlamaChain for various tasks.

## Basic Text Processing

The `text_processing.py` example demonstrates a simple text processing pipeline that tokenizes text and removes stop words.

```bash
# Run the example
python text_processing.py
```

## Sentiment Analysis

The `sentiment_analysis.py` example demonstrates how to use NLP components for sentiment analysis.

```bash
# Run the example
python sentiment_analysis.py
```

## Keyword Extraction

The `keyword_extraction.py` example demonstrates how to extract important keywords from text.

```bash
# Run the example
python keyword_extraction.py
```

## Weather API Integration

The `weather_api.py` example demonstrates how to create a pipeline that fetches weather data from a public API and processes the results.

```bash
# Run the example
python weather_api.py
```

## HuggingFace Model Integration

The `huggingface_sentiment.py` example demonstrates how to use HuggingFace models for text classification.

```bash
# Run the example
python huggingface_sentiment.py
```

## Custom Components

The `custom_components.py` example demonstrates how to create and use custom components.

```bash
# Run the example
python custom_components.py
```

## Document Analysis Pipeline

The `document_analyzer.py` example demonstrates a comprehensive document analysis pipeline that combines multiple components to analyze documents, extract keywords, perform sentiment analysis, and generate summary reports.

```bash
# Run the example
python document_analyzer.py
```

## Data Transformation

The `data_transformation.py` example demonstrates how to create a pipeline for transforming structured data, including filtering, mapping fields, and aggregating values.

```bash
# Run the example
python data_transformation.py
```

## Data Visualization

The `visualization_example.py` example demonstrates how to create interactive visualizations from data using chart components and different renderers (HTML and terminal).

```bash
# Run the example
python visualization_example.py
```

## Prerequisites

Some examples require additional dependencies:

- For `huggingface_sentiment.py`: `pip install transformers torch`
- For full API functionality: `pip install requests`
- For visualization: `pip install chart.js`

You can install all dependencies with:

```bash
pip install "llamachain[all]"
``` 