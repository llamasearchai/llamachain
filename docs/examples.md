# LlamaChain Examples

This page contains a collection of examples showing how to use LlamaChain for various tasks.

## Basic Data Processing

A simple example showing basic data loading, transformation, and export:

```python
from llamachain import Chain
from llamachain import components as c

# Create a chain for processing CSV data
pipeline = Chain([
    # Load data from CSV
    c.FileLoader("customer_data.csv"),
    
    # Clean the data
    c.DataCleaner(
        drop_nulls=True,
        fill_values={
            "age": 0,
            "income": 0,
            "rating": 2.5
        }
    ),
    
    # Select and transform columns
    c.ColumnSelector(["customer_id", "name", "age", "income", "rating"]),
    c.DataTransformer({
        "age": lambda x: x + 1,  # Birthday update
        "name": lambda x: x.str.title(),  # Proper capitalization
        "income": lambda x: x * 1.05  # 5% raise
    }),
    
    # Filter data
    c.DataFilter(lambda df: df["age"] >= 18),
    
    # Export to CSV
    c.FileWriter("processed_customers.csv")
])

# Run the pipeline
result = pipeline.run()
print(f"Processed {result.metadata['record_count']} customer records")
```

## Database Integration

An example showing how to load data from a database, process it, and save it back:

```python
from llamachain import Chain
from llamachain import components as c

# Database connection string
db_conn = "postgresql://user:password@localhost:5432/customer_db"

# Create a chain for database ETL
pipeline = Chain([
    # Load data from database
    c.DatabaseLoader(
        connection_string=db_conn,
        query="SELECT * FROM customers WHERE signup_date > '2023-01-01'"
    ),
    
    # Process data
    c.DataCleaner(drop_duplicates=True),
    c.DataTransformer({
        "email": lambda x: x.str.lower(),
        "phone": lambda x: x.str.replace(r'[^0-9]', '', regex=True)
    }),
    
    # Enrich data with API call
    c.APIEnricher(
        url="https://api.example.com/enrich",
        input_column="email",
        output_column="customer_segment",
        batch_size=100
    ),
    
    # Save back to database
    c.DatabaseWriter(
        connection_string=db_conn,
        table_name="enriched_customers",
        if_exists="replace"
    )
])

# Run the pipeline
result = pipeline.run()
print(f"Processed {result.metadata['record_count']} records")
print(f"API calls: {result.metadata['api_calls']}")
```

## Machine Learning Pipeline

An example showing a complete ML pipeline from data preparation to model training and evaluation:

```python
from llamachain import Chain
from llamachain.ml import MLChain
from llamachain import components as c

# Create a preprocessing chain
preprocessing = Chain([
    c.FileLoader("training_data.csv"),
    c.DataCleaner(drop_nulls=True, drop_duplicates=True),
    c.FeatureEngineer({
        "days_since_purchase": lambda df: (pd.Timestamp.now() - pd.to_datetime(df["last_purchase"])).dt.days,
        "total_value": lambda df: df["purchase_count"] * df["avg_purchase_value"]
    }),
    c.FeatureSelector([
        "customer_age", "days_since_purchase", "purchase_count", 
        "total_value", "website_visits", "is_subscriber"
    ])
])

# Create a ML pipeline
ml_pipeline = MLChain(
    # Preprocessing steps
    preprocessor=Chain([
        c.CategoryEncoder(
            columns=["is_subscriber"],
            encoding="onehot"
        ),
        c.Normalizer(
            columns=["customer_age", "days_since_purchase", "purchase_count", "total_value", "website_visits"],
            method="standard"
        )
    ]),
    
    # Model configuration
    model=c.Model(
        type="random_forest",
        target_column="churn",
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5
        }
    ),
    
    # Evaluation metrics
    evaluator=c.Evaluator(
        metrics=["accuracy", "precision", "recall", "f1"],
        cv_folds=5
    )
)

# Combine into a full pipeline
full_pipeline = Chain([
    preprocessing,
    ml_pipeline,
    c.ModelSaver("churn_prediction_model.pkl")
])

# Run the pipeline
result = full_pipeline.run()

# Print results
print("Model performance:")
for metric, value in result.metadata["metrics"].items():
    print(f"  {metric}: {value:.4f}")
```

## NLP Processing Pipeline

An example showing text processing and sentiment analysis:

```python
from llamachain import Chain
from llamachain import components as c

# Create an NLP processing pipeline
nlp_pipeline = Chain([
    # Load text data
    c.TextLoader("customer_reviews.txt", split_by="line"),
    
    # Preprocess text
    c.TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True
    ),
    
    # Analyze sentiment
    c.SentimentAnalyzer(
        model="vader",  # or "transformers" for deep learning based
        output_column="sentiment"
    ),
    
    # Extract entities
    c.EntityExtractor(
        model="spacy",
        entities=["PERSON", "ORG", "PRODUCT"],
        output_column="entities"
    ),
    
    # Categorize text
    c.TextCategorizer(
        categories=["complaint", "inquiry", "praise"],
        model="tfidf+svm",
        output_column="category"
    ),
    
    # Export results
    c.JSONWriter("processed_reviews.json")
])

# Run the pipeline
result = nlp_pipeline.run()
print(f"Processed {result.metadata['document_count']} reviews")
print(f"Average sentiment: {result.metadata['average_sentiment']:.2f}")
```

## Web API Integration

An example showing how to create a web API from a LlamaChain pipeline:

```python
from llamachain import Chain
from llamachain.web import create_api
from llamachain import components as c

# Create a processing pipeline
analysis_pipeline = Chain([
    # This will receive JSON input from API requests
    c.JSONInput(),
    
    # Extract text field from JSON
    c.FieldSelector("text"),
    
    # Process text
    c.TextPreprocessor(lowercase=True, remove_stopwords=True),
    c.SentimentAnalyzer(output_column="sentiment"),
    c.KeywordExtractor(top_n=5, output_column="keywords"),
    
    # Format response
    c.JSONFormatter(fields=["sentiment", "keywords"])
])

# Create a FastAPI app from the pipeline
app = create_api(
    pipeline=analysis_pipeline,
    title="Text Analysis API",
    description="API for sentiment analysis and keyword extraction",
    version="1.0.0"
)

# Run the API server (in production, use uvicorn directly)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Stream Processing

An example showing how to process streaming data:

```python
from llamachain import Chain
from llamachain import components as c

# Create a streaming pipeline
stream_pipeline = Chain([
    # Connect to Kafka stream
    c.KafkaConsumer(
        bootstrap_servers="localhost:9092",
        topics=["user-events"],
        group_id="llama-processor"
    ),
    
    # Parse JSON messages
    c.JSONParser(),
    
    # Process in windows
    c.WindowAggregator(
        window_size="1m",  # 1 minute windows
        group_by="user_id",
        aggregations={
            "event_count": ("event_id", "count"),
            "total_value": ("transaction_value", "sum")
        }
    ),
    
    # Apply business rules
    c.RuleEngine(
        rules=[
            ("high_activity", "event_count > 10"),
            ("high_value", "total_value > 1000"),
            ("suspicious", "event_count > 20 and total_value > 5000")
        ],
        output_column="flags"
    ),
    
    # Alert on suspicious activity
    c.Alerter(
        condition=lambda df: df["flags"].str.contains("suspicious"),
        alert_method="slack",
        alert_config={
            "webhook_url": "https://hooks.slack.com/services/XXX/YYY/ZZZ",
            "channel": "#security-alerts"
        }
    ),
    
    # Save aggregated data
    c.TimeseriesDB(
        connection_string="influxdb://localhost:8086",
        database="user_metrics",
        measurement="user_activity"
    )
])

# Run the streaming pipeline (runs indefinitely)
stream_pipeline.run()
```

## Creating Custom Components

An example showing how to create custom components:

```python
from llamachain import Component, Chain
import pandas as pd
import numpy as np

# Create a custom outlier detection component
class OutlierDetector(Component):
    def __init__(self, column, method="zscore", threshold=3.0, output_column=None):
        super().__init__()
        self.column = column
        self.method = method
        self.threshold = threshold
        self.output_column = output_column or f"{column}_is_outlier"
        
    def process(self, context):
        df = context.data
        
        if self.method == "zscore":
            z_scores = np.abs((df[self.column] - df[self.column].mean()) / df[self.column].std())
            outliers = z_scores > self.threshold
        elif self.method == "iqr":
            q1 = df[self.column].quantile(0.25)
            q3 = df[self.column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (self.threshold * iqr)
            upper_bound = q3 + (self.threshold * iqr)
            outliers = (df[self.column] < lower_bound) | (df[self.column] > upper_bound)
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        df[self.output_column] = outliers
        
        # Update metrics
        context.metadata["outlier_count"] = outliers.sum()
        context.metadata["outlier_percentage"] = (outliers.sum() / len(df)) * 100
        
        # Update context data
        context.update(data=df)
        return context

# Use the custom component in a chain
pipeline = Chain([
    c.FileLoader("sensor_data.csv"),
    c.DataCleaner(drop_nulls=True),
    OutlierDetector(column="temperature", method="iqr", threshold=1.5),
    c.DataFilter(lambda df: ~df["temperature_is_outlier"]),
    c.FileWriter("cleaned_sensor_data.csv")
])

result = pipeline.run()
print(f"Removed {result.metadata['outlier_count']} outliers ({result.metadata['outlier_percentage']:.2f}%)")
```

## Using Configuration Files

An example showing how to define and load a chain from a YAML configuration file:

```yaml
# pipeline_config.yaml
chain:
  name: "CustomerDataPipeline"
  components:
    - type: "FileLoader"
      params:
        file_path: "customer_data.csv"
        
    - type: "DataCleaner"
      params:
        drop_nulls: true
        drop_duplicates: true
        
    - type: "DataTransformer"
      params:
        transformations:
          age: "lambda x: x + 1"
          income: "lambda x: x * 1.05"
          
    - type: "FileWriter"
      params:
        file_path: "processed_data.csv"
        format: "csv"
```

```python
from llamachain import Chain

# Load the chain from configuration
pipeline = Chain.from_config("pipeline_config.yaml")

# Run the pipeline
result = pipeline.run()
print(f"Processed {result.metadata['record_count']} records")
```

These examples demonstrate the flexibility and power of LlamaChain for various data processing tasks. You can combine and adapt these examples to fit your specific needs. 