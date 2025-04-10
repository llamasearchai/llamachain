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

## Advanced Example: Text Statistics Component

This example demonstrates a more complex custom component that generates statistics about text.

```python
from typing import Any, Dict
from collections import Counter
from llamachain.core import Component, Pipeline
from llamachain.components import TextProcessor

class TextStatistics(Component):
    def process(self, input_data: str) -> Dict[str, Any]:
        # Count characters
        char_count = len(input_data)
        
        # Count words
        words = input_data.split()
        word_count = len(words)
        
        # Count unique words
        unique_words = set(word.lower() for word in words)
        unique_word_count = len(unique_words)
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        # Count sentences (simple approach)
        sentence_count = input_data.count('.') + input_data.count('!') + input_data.count('?')
        
        # Word frequency
        word_freq = Counter(word.lower() for word in words)
        top_words = word_freq.most_common(5)
        
        return {
            "character_count": char_count,
            "word_count": word_count,
            "unique_word_count": unique_word_count,
            "average_word_length": avg_word_length,
            "sentence_count": sentence_count,
            "top_words": top_words
        }

# Create pipeline with TextProcessor and TextStatistics
pipeline = Pipeline([
    TextProcessor(lowercase=False, normalize_whitespace=True),
    TextStatistics(),
])

# Process a sample text
sample_text = "The quick brown fox jumps over the lazy dog. Fox is a swift animal. Dogs are loyal pets!"
stats = pipeline.run(sample_text)

print(f"Text: {sample_text}")
print(f"Character count: {stats['character_count']}")
print(f"Word count: {stats['word_count']}")
print(f"Unique word count: {stats['unique_word_count']}")
print(f"Average word length: {stats['average_word_length']:.2f} characters")
print(f"Sentence count: {stats['sentence_count']}")

print("Top 5 words:")
for word, count in stats['top_words']:
    print(f"- '{word}': {count}")
```

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

## Running the Examples

You can run any of these examples by saving them to a file and executing with Python:

```bash
# Install LlamaChain first
pip install llamachain

# Run an example
python examples/text_processing.py
```

For more advanced examples and tutorials, check out the [examples directory](https://github.com/llamasearchai/llamachain/tree/main/examples) in the GitHub repository.
