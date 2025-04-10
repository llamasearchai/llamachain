# Machine Learning Components

LlamaChain provides specialized components for building end-to-end machine learning pipelines. This guide covers how to use these components to create, train, evaluate, and deploy ML models.

## Overview

The ML components in LlamaChain are designed to:

1. Streamline the creation of ML pipelines
2. Provide a consistent interface for different ML libraries
3. Automate common ML tasks
4. Integrate ML with other data processing steps

## MLChain

The `MLChain` class is a specialized chain designed specifically for machine learning workflows. It provides a higher-level interface than building ML pipelines from individual components.

```python
from llamachain.ml import MLChain
from llamachain import components as c

# Create an ML pipeline
ml_pipeline = MLChain(
    preprocessor=Chain([
        c.CategoryEncoder(columns=["color", "size"]),
        c.Normalizer(columns=["price", "weight"])
    ]),
    model=c.Model(
        type="random_forest",
        target_column="sales",
        hyperparameters={"n_estimators": 100}
    ),
    evaluator=c.Evaluator(
        metrics=["mse", "r2"],
        cv_folds=5
    )
)

# Train the model
ml_pipeline.train("training_data.csv")

# Make predictions
predictions = ml_pipeline.predict("new_data.csv")
```

## Data Preparation Components

### CategoryEncoder

Encodes categorical variables using various methods.

```python
from llamachain import Chain
from llamachain import components as c

encoder = c.CategoryEncoder(
    columns=["color", "brand", "category"],
    encoding="onehot",  # Options: "onehot", "label", "binary", "frequency", "target"
    handle_unknown="ignore"
)
```

### Normalizer

Normalizes numerical features using different methods.

```python
normalizer = c.Normalizer(
    columns=["price", "weight", "height", "width"],
    method="standard",  # Options: "standard", "minmax", "robust", "log"
    output_distribution="normal"
)
```

### FeatureSelector

Selects features based on importance or correlation.

```python
selector = c.FeatureSelector(
    method="importance",  # Options: "importance", "correlation", "variance", "rfe", "manual"
    n_features=10,
    target_column="target",
    threshold=0.05
)
```

### Imputer

Fills missing values in the dataset.

```python
imputer = c.Imputer(
    numerical_strategy="mean",  # Options: "mean", "median", "mode", "constant"
    categorical_strategy="most_frequent",
    fill_value=0  # Used with "constant" strategy
)
```

### FeatureEngineer

Creates new features from existing ones.

```python
engineer = c.FeatureEngineer(
    transformations={
        "price_per_unit": lambda df: df["price"] / df["quantity"],
        "is_premium": lambda df: df["price"] > 100,
        "days_since": lambda df: (pd.Timestamp.now() - pd.to_datetime(df["date"])).dt.days
    }
)
```

## Model Components

### Model

The main component for creating and training models.

```python
model = c.Model(
    type="random_forest",  # Various options available
    target_column="target",
    mode="classification",  # or "regression"
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10
    }
)
```

Supported model types:

- **Scikit-learn models**: "random_forest", "gradient_boosting", "logistic_regression", "svm", "knn", etc.
- **XGBoost**: "xgboost"
- **LightGBM**: "lightgbm"
- **TensorFlow/Keras**: "neural_network", "cnn", "rnn", "lstm"
- **PyTorch**: "torch_mlp", "torch_cnn", "torch_rnn"
- **Auto ML**: "auto"

### ModelLoader

Loads a pre-trained model.

```python
loader = c.ModelLoader(
    model_path="models/customer_churn_model.pkl",
    framework="sklearn"  # Options: "sklearn", "tensorflow", "pytorch", "xgboost"
)
```

### ModelSaver

Saves a trained model.

```python
saver = c.ModelSaver(
    save_path="models/trained_model.pkl",
    save_format="pickle",  # Options: "pickle", "joblib", "h5", "onnx"
    overwrite=True
)
```

### Predictor

Makes predictions using a trained model.

```python
predictor = c.Predictor(
    output_column="prediction",
    proba=True,  # Whether to include prediction probabilities
    batch_size=1000
)
```

## Evaluation Components

### Evaluator

Evaluates model performance using various metrics.

```python
evaluator = c.Evaluator(
    metrics=["accuracy", "precision", "recall", "f1", "auc"],  # for classification
    # metrics=["mse", "rmse", "mae", "r2"],  # for regression
    cv_folds=5,
    stratify=True
)
```

### CrossValidator

Performs cross-validation for model selection.

```python
validator = c.CrossValidator(
    models=[
        c.Model(type="random_forest", hyperparameters={"n_estimators": 100}),
        c.Model(type="gradient_boosting", hyperparameters={"n_estimators": 100}),
        c.Model(type="svm", hyperparameters={"C": 1.0})
    ],
    cv_folds=5,
    metrics=["accuracy", "f1"],
    target_column="target"
)
```

### HyperparameterTuner

Optimizes model hyperparameters.

```python
tuner = c.HyperparameterTuner(
    model=c.Model(type="random_forest"),
    param_grid={
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10]
    },
    method="grid",  # Options: "grid", "random", "bayesian"
    cv_folds=3,
    scoring="accuracy",
    n_iter=10  # For random and bayesian search
)
```

## Complete ML Pipeline Example

Here's a complete example of an ML pipeline for customer churn prediction:

```python
from llamachain import Chain
from llamachain.ml import MLChain
from llamachain import components as c

# Create a preprocessing chain
preprocessing = Chain([
    c.FileLoader("customer_data.csv"),
    c.DataCleaner(drop_nulls=True),
    
    # Feature engineering
    c.FeatureEngineer({
        "tenure_months": lambda df: df["days_as_customer"] / 30,
        "avg_monthly_spend": lambda df: df["total_spend"] / df["tenure_months"],
        "has_complained": lambda df: df["complaint_count"] > 0,
        "product_count": lambda df: df["products"].str.split(',').str.len()
    }),
    
    # Feature selection
    c.FeatureSelector([
        "tenure_months", "avg_monthly_spend", "has_complained", 
        "product_count", "age", "income_category", "region"
    ])
])

# Create the ML pipeline
ml_pipeline = MLChain(
    # Preprocessing specific to ML
    preprocessor=Chain([
        # Encode categorical variables
        c.CategoryEncoder(
            columns=["income_category", "region"],
            encoding="onehot"
        ),
        
        # Normalize numerical features
        c.Normalizer(
            columns=["tenure_months", "avg_monthly_spend", "product_count", "age"],
            method="standard"
        )
    ]),
    
    # Model configuration
    model=c.Model(
        type="gradient_boosting",
        target_column="churned",
        mode="classification",
        hyperparameters={
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5
        }
    ),
    
    # Evaluation configuration
    evaluator=c.Evaluator(
        metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
        cv_folds=5,
        stratify=True
    )
)

# Hyperparameter tuning
hyperparameter_tuning = c.HyperparameterTuner(
    model=ml_pipeline.model,
    param_grid={
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    method="grid",
    cv_folds=3,
    scoring="f1"
)

# Model explanation
explainer = c.ModelExplainer(
    method="shap",  # Options: "shap", "lime", "eli5"
    n_samples=100,
    output_format="html"
)

# Create the full pipeline
full_pipeline = Chain([
    preprocessing,
    hyperparameter_tuning,
    ml_pipeline,
    explainer,
    c.ModelSaver("churn_model.pkl"),
    c.ReportGenerator(
        template="ml_report.html",
        output_path="churn_model_report.html"
    )
])

# Run the pipeline
result = full_pipeline.run()

# Print results
print(f"Best hyperparameters: {result.metadata['best_params']}")
print("Model performance:")
for metric, value in result.metadata["metrics"].items():
    print(f"  {metric}: {value:.4f}")

# The model can now be used for predictions
prediction_pipeline = Chain([
    c.FileLoader("new_customers.csv"),
    preprocessing,
    c.ModelLoader("churn_model.pkl"),
    c.Predictor(output_column="churn_probability"),
    c.FileWriter("churn_predictions.csv")
])

prediction_pipeline.run()
```

## Integration with ML Frameworks

LlamaChain integrates with popular ML frameworks:

### Scikit-learn

```python
from llamachain.ml.frameworks import sklearn

# Use a custom sklearn pipeline
custom_pipeline = sklearn.Pipeline(
    steps=[
        ('preprocessing', sklearn.preprocessing.StandardScaler()),
        ('model', sklearn.ensemble.RandomForestClassifier())
    ]
)

# Create a LlamaChain component from the sklearn pipeline
sklearn_component = c.SklearnAdapter(
    pipeline=custom_pipeline,
    target_column="target"
)
```

### TensorFlow/Keras

```python
from llamachain.ml.frameworks import tensorflow

# Create a custom Keras model
def create_model():
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tensorflow.keras.layers.Dense(32, activation='relu'),
        tensorflow.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a LlamaChain component from the TF model
tf_component = c.TensorFlowAdapter(
    model_fn=create_model,
    target_column="target",
    batch_size=32,
    epochs=10
)
```

### PyTorch

```python
from llamachain.ml.frameworks import torch
import torch.nn as nn

# Create a custom PyTorch model
class CustomModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

# Create a LlamaChain component from the PyTorch model
torch_component = c.PyTorchAdapter(
    model_class=CustomModel,
    target_column="target",
    batch_size=32,
    epochs=10,
    learning_rate=0.001
)
```

## Advanced ML Techniques

### AutoML

LlamaChain provides AutoML capabilities for automated model selection and hyperparameter tuning:

```python
automl = c.AutoML(
    target_column="target",
    mode="classification",  # or "regression"
    time_budget=60,  # In minutes
    metric="accuracy",
    include_models=["random_forest", "xgboost", "lightgbm"],
    exclude_models=["svm"]
)
```

### Ensemble

Creates an ensemble of multiple models:

```python
ensemble = c.Ensemble(
    models=[
        c.Model(type="random_forest"),
        c.Model(type="gradient_boosting"),
        c.Model(type="logistic_regression")
    ],
    method="voting",  # Options: "voting", "stacking", "bagging", "boosting"
    weights=[0.5, 0.3, 0.2],  # For weighted voting
    target_column="target"
)
```

### FeatureImportance

Analyzes and visualizes feature importance:

```python
feature_importance = c.FeatureImportance(
    method="permutation",  # Options: "permutation", "shap", "built_in"
    n_repeats=10,
    target_column="target",
    plot=True,
    output_file="feature_importance.png"
)
```

## Deployment Components

### ModelAPI

Creates a REST API for a trained model:

```python
model_api = c.ModelAPI(
    model_path="trained_model.pkl",
    host="0.0.0.0",
    port=8000,
    input_schema={
        "feature1": "float",
        "feature2": "string",
        "feature3": "integer"
    }
)
```

### ModelMonitor

Monitors model performance in production:

```python
monitor = c.ModelMonitor(
    metrics=["accuracy", "drift"],
    alert_threshold=0.1,
    alert_method="email",
    alert_config={
        "recipients": ["alerts@example.com"],
        "subject": "Model Performance Alert"
    },
    storage="database",  # Options: "database", "file", "cloud"
    storage_config={
        "connection_string": "postgresql://user:password@localhost/monitoring"
    }
)
```

## Next Steps

With the ML components covered in this guide, you can build sophisticated machine learning pipelines within LlamaChain. To continue exploring:

1. Check out the [API Reference](../api/ml.md) for detailed documentation of all ML components
2. See the [Examples](../examples.md) for concrete ML pipeline implementations
3. Learn about [NLP Processing](nlp-processing.md) for text-based ML tasks 