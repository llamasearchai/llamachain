# LlamaChain

<p align="center">
<img src="assets/llamachain-logo.png" alt="LlamaChain Logo" width="200"/>
</p>

**LlamaChain** is a powerful and flexible Python framework for building end-to-end AI/ML data processing pipelines and applications. It provides a modular approach to connecting various components together, making it easy to create robust and scalable systems.

## Key Features

- **Modular Architecture**: Build complex pipelines by connecting simple, reusable components
- **ML Integration**: Seamlessly incorporate machine learning models into your data workflows
- **NLP Components**: Process and analyze text data with built-in natural language processing capabilities
- **API Connectors**: Connect to various external services and APIs with minimal configuration
- **Web Support**: Build web applications and APIs on top of your data pipelines
- **Analytics**: Track performance and gather insights from your data processing systems
- **CLI Tools**: Manage and interact with your pipelines through an intuitive command-line interface
- **Security**: Built-in security features to protect your data and applications

## Installation

```bash
# Basic installation
pip install llamachain

# With ML components
pip install llamachain[ml]

# With NLP components
pip install llamachain[nlp]

# With web development tools
pip install llamachain[web]

# Full installation with all optional dependencies
pip install llamachain[full]

# For development
pip install llamachain[dev]
```

## Quick Example

```python
from llamachain import Chain, components as c

# Create a simple data processing pipeline
pipeline = Chain([
    c.FileLoader("data/*.csv"),
    c.DataCleaner(drop_nulls=True),
    c.FeatureExtractor(["price", "location", "size"]),
    c.MLPredictor(model_type="regression"),
    c.ResultsExporter("predictions.json")
])

# Run the pipeline
results = pipeline.run()
print(f"Processed {results.record_count} records with accuracy {results.metrics['accuracy']:.2f}")
```

## Web API Example

```python
from llamachain.web import create_api
from llamachain import Chain, components as c

# Create a processing pipeline
analysis_pipeline = Chain([
    c.TextLoader(),
    c.SentimentAnalyzer(),
    c.EntityExtractor(),
    c.JSONFormatter()
])

# Create a web API from the pipeline
app = create_api(analysis_pipeline)

# Run the API server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

## ML Pipeline Example

```python
from llamachain.ml import MLChain
from llamachain import components as c

# Create an ML training and inference pipeline
ml_pipeline = MLChain(
    preprocessor=c.DataPreprocessor(
        categorical_encoder="onehot",
        numerical_scaler="standard",
        feature_selection=["f1", "f2", "f3"]
    ),
    model=c.Model(
        type="xgboost",
        hyperparameters={
            "n_estimators": 100,
            "max_depth": 3
        }
    ),
    evaluator=c.Evaluator(
        metrics=["accuracy", "f1", "recall"]
    )
)

# Train the model
ml_pipeline.train("training_data.csv")

# Make predictions
predictions = ml_pipeline.predict("new_data.csv")
```

## Documentation

For full documentation, visit [https://llamasearch.github.io/llamachain](https://llamasearch.github.io/llamachain)

## License

LlamaChain is released under the MIT License. See the [LICENSE](https://github.com/llamasearch/llamachain/blob/main/LICENSE) file for details.

## Community

- GitHub: [https://github.com/llamasearch/llamachain](https://github.com/llamasearch/llamachain)
- Twitter: [@llamasearch](https://twitter.com/llamasearch)
- Discord: [Join our community](https://discord.gg/llamachain) 