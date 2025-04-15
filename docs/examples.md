# LlamaChain Examples

This page provides examples of using LlamaChain for various tasks. All these examples can be found in the [examples directory](https://github.com/llamasearchai/llamachain/tree/main/examples) of the repository.

## Basic Text Processing

This example demonstrates a simple text processing pipeline that tokenizes text and removes stop words.

```python
from llamachain.core import Pipeline
from llamachain.components import TextProcessor, TextTokenizer, StopWordRemover

# Create components
text_processor = TextProcessor(lowercase=True, remove_punctuation=True)
tokenizer = TextTokenizer()
stop_word_remover = StopWordRemover()

# Create pipeline
pipeline = Pipeline([
    text_processor,  # Convert to lowercase and remove punctuation
    tokenizer,       # Split text into tokens
    stop_word_remover  # Remove common stop words
])

# Run the pipeline
input_text = "The quick brown fox jumps over the lazy dog."
result = pipeline.run(input_text)

print(result)  # ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
```

## Weather API Integration

This example demonstrates how to create a pipeline that fetches weather data from a public API and processes the results.

```python
from llamachain.core import Pipeline, Component
from llamachain.api import APIRequest, JSONExtractor, RESTEndpoint

# Create an API endpoint
endpoint = RESTEndpoint(
    url="https://api.openweathermap.org/data/2.5/weather",
    params={"appid": "YOUR_API_KEY_HERE"},  # Replace with your API key
    headers={"Content-Type": "application/json"},
    timeout=5.0,
)

# Custom component to format weather data
class WeatherFormatter(Component):
    def process(self, input_data):
        # Convert temperature from Kelvin to Celsius
        if "main.temp" in input_data:
            temp_k = input_data["main.temp"]
            temp_c = temp_k - 273.15
            input_data["main.temp"] = f"{temp_c:.1f}°C"
        
        # Format weather description
        if "weather.0.description" in input_data:
            input_data["weather.0.description"] = input_data["weather.0.description"].capitalize()
        
        # Create a cleaned up result
        return {
            "location": input_data.get("name", "Unknown"),
            "description": input_data.get("weather.0.description", "Unknown"),
            "temperature": input_data.get("main.temp", "Unknown"),
            "humidity": f"{input_data.get('main.humidity', 'Unknown')}%",
            "wind_speed": f"{input_data.get('wind.speed', 'Unknown')} m/s",
        }

# Create pipeline
pipeline = Pipeline([
    APIRequest(endpoint, method="GET"),  # Make API request
    JSONExtractor(fields=["name", "weather.0.description", "main.temp", "main.humidity", "wind.speed"]),  # Extract fields
    WeatherFormatter(),  # Format the data
])

# For example purposes, we'll use dummy data
dummy_response = {
    "name": "London",
    "weather": [{"description": "clear sky"}],
    "main": {"temp": 293.15, "humidity": 70},
    "wind": {"speed": 3.5},
}

result = pipeline.run(dummy_response)
print(result)
# Output:
# {
#   'location': 'London',
#   'description': 'Clear sky',
#   'temperature': '20.0°C',
#   'humidity': '70%',
#   'wind_speed': '3.5 m/s'
# }
```

## Sentiment Analysis

This example demonstrates how to use NLP components for sentiment analysis.

```python
from llamachain.core import Pipeline
from llamachain.components import TextProcessor
from llamachain.nlp import SimpleSentimentAnalyzer

# Create components
text_processor = TextProcessor(lowercase=True, normalize_whitespace=True)
sentiment_analyzer = SimpleSentimentAnalyzer()

# Create pipeline
pipeline = Pipeline([
    text_processor,
    sentiment_analyzer,
])

# Analyze some reviews
reviews = [
    "This product is amazing! I love everything about it. The quality is excellent.",
    "I'm disappointed with this purchase. It broke after just one week of use.",
    "It's okay, but not worth the price. Some features are good, others are mediocre.",
]

for review in reviews:
    result = pipeline.run(review)
    print(f"Review: {review}")
    print(f"Sentiment: {result['sentiment'].upper()}")
    print(f"Score: {result['score']:.2f}")
    print()
```

## Keyword Extraction

This example demonstrates how to extract important keywords from text.

```python
from llamachain.core import Pipeline
from llamachain.components import TextProcessor
from llamachain.nlp import KeywordExtractor

# Create components
text_processor = TextProcessor(lowercase=True, normalize_whitespace=True)
keyword_extractor = KeywordExtractor(max_keywords=5)

# Create pipeline
pipeline = Pipeline([
    text_processor,
    keyword_extractor,
])

# Extract keywords from a document
document = """
Machine learning is a branch of artificial intelligence and computer science which 
focuses on the use of data and algorithms to imitate the way that humans learn, 
gradually improving its accuracy. Machine learning is an important component of the 
growing field of data science.
"""

keywords = pipeline.run(document)

print("Top keywords:")
for keyword in keywords:
    print(f"- {keyword['word']} (count: {keyword['count']}, score: {keyword['score']:.2f})")
```

## HuggingFace Model Integration

This example demonstrates how to use HuggingFace models for text classification.

```python
from llamachain.core import Pipeline
from llamachain.components import TextProcessor
from llamachain.ml import HuggingFaceModel, ModelInference

# Create components
text_processor = TextProcessor(normalize_whitespace=True)
model = HuggingFaceModel(
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
    model_type="text-classification"
)
inference = ModelInference(model)

# Create pipeline
pipeline = Pipeline([
    text_processor,
    inference,
])

# Classify text sentiment
texts = [
    "I absolutely loved this movie, it was amazing!",
    "The service at this restaurant was terrible and the food was bland.",
    "The book was neither particularly good nor particularly bad.",
]

for text in texts:
    result = pipeline.run(text)
    print(f"Text: {text}")
    print(f"Label: {result['label']}")
    print(f"Confidence: {result['score']:.4f}\n")
```

## Custom Components

This example demonstrates how to create and use custom components.

```python
from typing import Any, Dict, List, Set, Tuple
from collections import Counter

from llamachain.core import Component, Pipeline

# Custom component to count occurrences of each character
class CharacterCounter(Component):
    def process(self, input_data: str) -> Dict[str, int]:
        if not isinstance(input_data, str):
            raise TypeError(f"Expected string, got {type(input_data).__name__}")
        
        counts = {}
        for char in input_data:
            if char in counts:
                counts[char] += 1
            else:
                counts[char] = 1
                
        return counts

# Custom component to sort dictionary items by value
class DictionarySorter(Component):
    def __init__(self, reverse=True, name=None, config=None):
        self.reverse = reverse
        super().__init__(name, config)
    
    def process(self, input_data: Dict[str, Any]) -> List[Tuple[str, Any]]:
        if not isinstance(input_data, dict):
            raise TypeError(f"Expected dictionary, got {type(input_data).__name__}")
        
        return sorted(input_data.items(), key=lambda x: x[1], reverse=self.reverse)

# Create pipeline with custom components
pipeline = Pipeline([
    CharacterCounter(),
    DictionarySorter(reverse=True),
])

# Process text
text = "hello world"
result = pipeline.run(text)

print(f"Input text: {text}")
print("Character counts (most frequent first):")
for char, count in result:
    print(f"'{char}': {count}")
```

## Document Analysis Pipeline

This example demonstrates a comprehensive document analysis pipeline that combines multiple components to analyze documents. It uses custom components for splitting documents into paragraphs, extracting keywords, performing sentiment analysis, and generating a summary report.

```python
from typing import Dict, List, Any
from datetime import datetime

from llamachain.core import Pipeline, Component
from llamachain.components import TextProcessor
from llamachain.nlp import KeywordExtractor, SimpleSentimentAnalyzer


class DocumentSplitter(Component):
    """Custom component to split documents into paragraphs"""
    
    def __init__(self, min_length: int = 20):
        """Initialize the document splitter
        
        Args:
            min_length: Minimum length of paragraph to keep
        """
        self.min_length = min_length
        
    def process(self, input_data: str) -> List[str]:
        """Split document into paragraphs
        
        Args:
            input_data: Text document to split
            
        Returns:
            List of paragraphs
        """
        if not isinstance(input_data, str):
            raise TypeError(f"Expected string, got {type(input_data).__name__}")
            
        paragraphs = [p.strip() for p in input_data.split('\n\n')]
        return [p for p in paragraphs if len(p) >= self.min_length]


class SummaryGenerator(Component):
    """Custom component to generate a summary from analysis results"""
    
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
        sentiments = input_data.get('sentiment_by_paragraph', {})
        pos_count = sum(1 for s in sentiments.values() if s['sentiment'] == 'positive')
        neg_count = sum(1 for s in sentiments.values() if s['sentiment'] == 'negative')
        neutral_count = sum(1 for s in sentiments.values() if s['sentiment'] == 'neutral')
        
        # Extract keyword information
        keywords = input_data.get('keywords', [])
        
        # Generate summary text
        summary = (
            f"Document Analysis Summary:\n"
            f"- Document contains {len(sentiments)} paragraphs\n"
            f"- Overall sentiment: {pos_count} positive, {neg_count} negative, {neutral_count} neutral paragraphs\n"
            f"- Primary sentiment: {max(['positive', 'negative', 'neutral'], key=lambda s: [pos_count, neg_count, neutral_count][['positive', 'negative', 'neutral'].index(s)])}\n"
            f"- Top keywords: {', '.join(keywords[:5]) if keywords else 'None found'}\n"
            f"- Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # Return updated results with summary
        result = input_data.copy()
        result['summary'] = summary
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
        self.preprocess_pipeline = Pipeline([
            self.text_processor,
            self.document_splitter,
        ])
        
        self.keyword_pipeline = Pipeline([
            self.text_processor,
            self.keyword_extractor,
        ])
    
    def analyze_document(self, document_text: str) -> Dict[str, Any]:
        """Analyze a document using multiple pipelines
        
        Args:
            document_text: The document to analyze
            
        Returns:
            Dictionary with analysis results and summary
        """
        # Process document
        paragraphs = self.preprocess_pipeline.process(document_text)
        
        # Extract keywords from entire document
        keywords = self.keyword_pipeline.process(document_text)
        
        # Analyze sentiment of each paragraph
        sentiment_by_paragraph = {}
        for i, paragraph in enumerate(paragraphs):
            sentiment = self.sentiment_analyzer.process(paragraph)
            sentiment_by_paragraph[f"paragraph_{i+1}"] = sentiment
        
        # Collect all results
        results = {
            'document_length': len(document_text),
            'paragraph_count': len(paragraphs),
            'paragraphs': paragraphs,
            'keywords': keywords,
            'sentiment_by_paragraph': sentiment_by_paragraph,
        }
        
        # Generate summary
        final_results = self.summary_generator.process(results)
        
        return final_results


# Usage example
sample_document = """
Artificial Intelligence: Transforming the Future

Artificial intelligence (AI) has rapidly evolved over the past decade, transforming from a niche research field into a technology that touches almost every aspect of our daily lives.

The growth of machine learning, particularly deep learning, has been the driving force behind this AI revolution. Neural networks, inspired by the human brain, have demonstrated remarkable abilities in various domains from image recognition to natural language processing.

However, the rise of AI also brings challenges. Ethical concerns regarding privacy, bias, and job displacement have become increasingly important as these technologies become more pervasive.

Despite these challenges, the potential benefits of AI are enormous. In healthcare, AI systems can help diagnose diseases earlier and more accurately than human doctors in some cases. In transportation, autonomous vehicles promise to reduce accidents and traffic congestion while increasing mobility for those who cannot drive.
"""

analyzer = DocumentAnalyzer()
results = analyzer.analyze_document(sample_document)

print(f"Document length: {results['document_length']} characters")
print(f"Paragraph count: {results['paragraph_count']}")
print("\nTop keywords:")
for keyword in results['keywords'][:5]:
    print(f"- {keyword}")
    
print("\nSummary Report:")
print(results['summary'])
```

This example demonstrates more advanced techniques including:
- Creating specialized component classes for specific document processing tasks
- Building helper classes that combine multiple pipelines 
- Managing complex processing flows with multiple stages
- Generating summary reports from analyzed data

## Combining with Built-in Components

This example demonstrates how to combine custom components with built-in ones.

```python
from llamachain.core import Pipeline, Component
from llamachain.components import TextProcessor, TextTokenizer
from collections import Counter

# Create a pipeline mixing custom and built-in components
pipeline = Pipeline([
    TextProcessor(lowercase=True, remove_punctuation=True),
    TextTokenizer(),
    # Custom component that takes a list of tokens and returns word counts
    Component(name="WordCounter", process=lambda tokens: Counter(tokens)),
])

# Process text
text = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence."
result = pipeline.run(text)

print(f"Text: {text}")
print("Word counts:")
for word, count in result.most_common():
    print(f"- '{word}': {count}")
```

## Data Transformation

This example demonstrates how to create a data transformation pipeline that processes structured data (JSON) through multiple transformation steps.

```python
from typing import Dict, List, Any
import json

from llamachain.core import Pipeline, Component


class DataFilter(Component):
    """Component to filter data based on conditions"""
    
    def __init__(self, field: str, condition: callable, name=None, config=None):
        """Initialize the data filter
        
        Args:
            field: The field to filter on
            condition: Function that returns True for items to keep
            name: Optional component name
            config: Optional component configuration
        """
        self.field = field
        self.condition = condition
        super().__init__(name, config)
    
    def process(self, input_data: List[Dict]) -> List[Dict]:
        """Filter a list of dictionaries based on the condition"""
        if not isinstance(input_data, list):
            raise TypeError(f"Expected list, got {type(input_data).__name__}")
            
        return [item for item in input_data if self.field in item and self.condition(item[self.field])]


class DataMapper(Component):
    """Component to map data from one format to another"""
    
    def __init__(self, mapping: Dict[str, str], name=None, config=None):
        """Initialize the data mapper
        
        Args:
            mapping: Dictionary mapping source fields to target fields
        """
        self.mapping = mapping
        super().__init__(name, config)
    
    def process(self, input_data: List[Dict]) -> List[Dict]:
        """Map fields in a list of dictionaries"""
        if not isinstance(input_data, list):
            raise TypeError(f"Expected list, got {type(input_data).__name__}")
            
        result = []
        for item in input_data:
            new_item = {}
            for source_field, target_field in self.mapping.items():
                if source_field in item:
                    new_item[target_field] = item[source_field]
            result.append(new_item)
        
        return result


# Sample data
sample_data = [
    {"id": 1, "product": "Widget", "category": "A", "price": 10.0, "quantity": 5},
    {"id": 2, "product": "Gadget", "category": "B", "price": 20.0, "quantity": 3},
    {"id": 3, "product": "Tool", "category": "A", "price": 15.0, "quantity": 2},
]

# Create a pipeline to filter, transform, and calculate sales
pipeline = Pipeline([
    # Filter items in category A
    DataFilter(
        field="category",
        condition=lambda x: x == "A",
    ),
    
    # Map fields to new structure
    DataMapper(
        mapping={
            "product": "item_name",
            "price": "unit_price",
            "quantity": "units_sold"
        }
    ),
    
    # Calculate total sales amount
    Component(
        name="SalesCalculator",
        process=lambda items: [
            {**item, "total_sales": item["unit_price"] * item["units_sold"]} 
            for item in items
        ]
    ),
])

# Process the data
result = pipeline.run(sample_data)
print(json.dumps(result, indent=2))
# Output:
# [
#   {
#     "item_name": "Widget",
#     "unit_price": 10.0,
#     "units_sold": 5,
#     "total_sales": 50.0
#   },
#   {
#     "item_name": "Tool",
#     "unit_price": 15.0,
#     "units_sold": 2,
#     "total_sales": 30.0
#   }
# ]
```

This example showcases:
- Creating components for data filtering and field mapping
- Processing structured data (JSON) through transformation pipelines
- Using lambda functions with Component for simple transformations
- Building data processing workflows for business applications

## Data Visualization

This example demonstrates how to create visualizations from data using chart components and different renderers.

```python
from llamachain.core import Pipeline
from llamachain.visualization import BarChart, LineChart, PieChart, HTMLRenderer, TerminalRenderer

# Sample data - quarterly sales
sales_data = {
    "Q1": 120,
    "Q2": 180,
    "Q3": 240,
    "Q4": 160
}

# Create a pipeline with BarChart and HTML renderer
pipeline = Pipeline([
    # Transform the data into a bar chart
    BarChart(
        title="Quarterly Sales",
        x_label="Quarter",
        y_label="Sales ($1,000)",
        name="SalesBarChart"
    ),
    # Render the chart as HTML
    HTMLRenderer(
        output_file="sales_chart.html",
        name="HTMLOutput"
    )
])

# Process the data
result = pipeline.run(sales_data)
print(f"Bar chart rendered to HTML: {result}")

# Create a pipeline with BarChart and Terminal renderer for text-based visualization
terminal_pipeline = Pipeline([
    BarChart(
        title="Quarterly Sales",
        name="SalesBarChart"
    ),
    TerminalRenderer(
        width=60,
        height=15,
        name="TerminalOutput"
    )
])

# Process the data and display in terminal
terminal_result = terminal_pipeline.run(sales_data)
print("\nTerminal Bar Chart:")
print(terminal_result)
```

The visualization module provides several chart types:

1. **Bar Charts** for categorical data
   ```python
   bar_chart = BarChart(
       title="Quarterly Sales",
       x_label="Quarter", 
       y_label="Sales",
       horizontal=False  # Set to True for horizontal bars
   )
   ```

2. **Line Charts** for time series or sequential data
   ```python
   line_chart = LineChart(
       title="Monthly Revenue",
       x_label="Month",
       y_label="Revenue",
       show_points=True  # Whether to display points on the line
   )
   ```

3. **Pie Charts** for showing proportions
   ```python
   pie_chart = PieChart(
       title="Market Share",
       donut=False  # Set to True for donut chart
   )
   ```

Charts can be rendered in different formats:

1. **HTML Renderer** for web-based interactive charts
   ```python
   html_renderer = HTMLRenderer(
       output_file="chart.html",  # File to save the chart (optional)
       include_scripts=True       # Whether to include Chart.js scripts
   )
   ```

2. **Terminal Renderer** for text-based charts in the console
   ```python
   terminal_renderer = TerminalRenderer(
       width=80,       # Width in characters
       height=20,      # Height in characters
       character="█"   # Character to use for plotting
   )
   ```

This visualization system can be combined with data transformation components to create powerful data analysis and reporting pipelines.

## Running the Examples

You can run any of these examples by saving them to a file and executing with Python:

```bash
# Install LlamaChain first
pip install llamachain

# Run an example
python examples/text_processing.py
```

For more advanced examples and tutorials, check out the [examples directory](https://github.com/llamasearchai/llamachain/tree/main/examples) in the GitHub repository.
