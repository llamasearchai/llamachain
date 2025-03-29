# Machine Learning API Reference

This page documents the machine learning components available in the LlamaChain package.

## MLChain

The `MLChain` class is a specialized chain for machine learning workflows.

```python
from llamachain.ml import MLChain
```

### Constructor

```python
MLChain(preprocessor=None, model=None, evaluator=None, name=None, config=None)
```

**Parameters:**

- `preprocessor` (Chain, optional): Chain of components for preprocessing data before model training/inference. Default is None.
- `model` (Model, optional): The model component to use. Default is None.
- `evaluator` (Evaluator, optional): The evaluator component to use. Default is None.
- `name` (str, optional): Name of the chain. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### Methods

#### train

```python
train(data, target_column=None, test_size=0.2, random_state=None)
```

Trains the model on the provided data.

**Parameters:**

- `data` (pandas.DataFrame or str): The training data as a DataFrame or path to a CSV file.
- `target_column` (str, optional): The name of the target column. If None, uses the one specified in the model. Default is None.
- `test_size` (float, optional): Proportion of data to use for testing. Default is 0.2.
- `random_state` (int, optional): Random seed for reproducibility. Default is None.

**Returns:**

- `dict`: Dictionary containing training results and metrics.

#### predict

```python
predict(data, output_column=None)
```

Makes predictions using the trained model.

**Parameters:**

- `data` (pandas.DataFrame or str): The data to predict on as a DataFrame or path to a CSV file.
- `output_column` (str, optional): The name of the column to store predictions. Default is "prediction".

**Returns:**

- `pandas.DataFrame`: DataFrame containing the input data and predictions.

#### evaluate

```python
evaluate(data, target_column=None, metrics=None)
```

Evaluates the model on the provided data.

**Parameters:**

- `data` (pandas.DataFrame or str): The data to evaluate on as a DataFrame or path to a CSV file.
- `target_column` (str, optional): The name of the target column. If None, uses the one specified in the model. Default is None.
- `metrics` (list, optional): List of metric names to compute. If None, uses the ones specified in the evaluator. Default is None.

**Returns:**

- `dict`: Dictionary containing evaluation metrics.

#### save

```python
save(path)
```

Saves the trained model to a file.

**Parameters:**

- `path` (str): Path to save the model to.

#### load

```python
load(path)
```

Loads a trained model from a file.

**Parameters:**

- `path` (str): Path to load the model from.

## Data Preparation Components

### CategoryEncoder

Encodes categorical variables using various methods.

```python
from llamachain import components as c
```

#### Constructor

```python
c.CategoryEncoder(columns, encoding="onehot", handle_unknown="ignore", name=None, config=None)
```

**Parameters:**

- `columns` (list): List of column names to encode.
- `encoding` (str, optional): Encoding method to use. Options are "onehot", "label", "binary", "frequency", "target". Default is "onehot".
- `handle_unknown` (str, optional): Strategy for handling unknown categories. Options are "ignore" or "error". Default is "ignore".
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### Normalizer

Normalizes numerical features using different methods.

#### Constructor

```python
c.Normalizer(columns, method="standard", output_distribution="normal", name=None, config=None)
```

**Parameters:**

- `columns` (list): List of column names to normalize.
- `method` (str, optional): Normalization method to use. Options are "standard", "minmax", "robust", "log". Default is "standard".
- `output_distribution` (str, optional): For some methods, the target distribution. Options are "normal" or "uniform". Default is "normal".
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### FeatureSelector

Selects features based on importance or correlation.

#### Constructor

```python
c.FeatureSelector(features=None, method="manual", n_features=None, target_column=None, threshold=None, name=None, config=None)
```

**Parameters:**

- `features` (list, optional): List of features to select. Required for method="manual". Default is None.
- `method` (str, optional): Feature selection method. Options are "manual", "importance", "correlation", "variance", "rfe". Default is "manual".
- `n_features` (int, optional): Number of features to select. Required for non-manual methods if threshold is None. Default is None.
- `target_column` (str, optional): Name of the target column. Required for some methods. Default is None.
- `threshold` (float, optional): Threshold for feature selection. Default is None.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### Imputer

Fills missing values in the dataset.

#### Constructor

```python
c.Imputer(numerical_strategy="mean", categorical_strategy="most_frequent", fill_value=None, name=None, config=None)
```

**Parameters:**

- `numerical_strategy` (str, optional): Strategy for filling numerical missing values. Options are "mean", "median", "mode", "constant". Default is "mean".
- `categorical_strategy` (str, optional): Strategy for filling categorical missing values. Options are "most_frequent", "constant". Default is "most_frequent".
- `fill_value` (any, optional): Value to use with "constant" strategy. Default is None.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### FeatureEngineer

Creates new features from existing ones.

#### Constructor

```python
c.FeatureEngineer(transformations, name=None, config=None)
```

**Parameters:**

- `transformations` (dict): Dictionary mapping new feature names to transformation functions.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

## Model Components

### Model

Main component for creating and training models.

#### Constructor

```python
c.Model(type, target_column=None, mode="classification", hyperparameters=None, name=None, config=None)
```

**Parameters:**

- `type` (str): Type of model to create (e.g., "random_forest", "xgboost", "neural_network").
- `target_column` (str, optional): Name of the target column. Default is None.
- `mode` (str, optional): Mode of the model. Options are "classification" or "regression". Default is "classification".
- `hyperparameters` (dict, optional): Hyperparameters for the model. Default is None.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### ModelLoader

Loads a pre-trained model.

#### Constructor

```python
c.ModelLoader(model_path, framework="sklearn", name=None, config=None)
```

**Parameters:**

- `model_path` (str): Path to the model file.
- `framework` (str, optional): Framework of the model. Options are "sklearn", "tensorflow", "pytorch", "xgboost". Default is "sklearn".
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### ModelSaver

Saves a trained model.

#### Constructor

```python
c.ModelSaver(save_path, save_format="pickle", overwrite=False, name=None, config=None)
```

**Parameters:**

- `save_path` (str): Path to save the model to.
- `save_format` (str, optional): Format to save the model in. Options are "pickle", "joblib", "h5", "onnx". Default is "pickle".
- `overwrite` (bool, optional): Whether to overwrite existing file. Default is False.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### Predictor

Makes predictions using a trained model.

#### Constructor

```python
c.Predictor(output_column="prediction", proba=False, batch_size=None, name=None, config=None)
```

**Parameters:**

- `output_column` (str, optional): Name of the column to store predictions. Default is "prediction".
- `proba` (bool, optional): Whether to include prediction probabilities (for classification). Default is False.
- `batch_size` (int, optional): Batch size for prediction. Default is None.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

## Evaluation Components

### Evaluator

Evaluates model performance using various metrics.

#### Constructor

```python
c.Evaluator(metrics=None, cv_folds=None, stratify=True, name=None, config=None)
```

**Parameters:**

- `metrics` (list, optional): List of metric names to compute. Default is None (uses sensible defaults based on model mode).
- `cv_folds` (int, optional): Number of cross-validation folds. If None, no cross-validation is performed. Default is None.
- `stratify` (bool, optional): Whether to use stratified sampling for classification problems. Default is True.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### CrossValidator

Performs cross-validation for model selection.

#### Constructor

```python
c.CrossValidator(models, cv_folds=5, metrics=None, target_column=None, name=None, config=None)
```

**Parameters:**

- `models` (list): List of model components to evaluate.
- `cv_folds` (int, optional): Number of cross-validation folds. Default is 5.
- `metrics` (list, optional): List of metric names to compute. Default is None (uses sensible defaults based on model mode).
- `target_column` (str, optional): Name of the target column. Default is None.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### HyperparameterTuner

Optimizes model hyperparameters.

#### Constructor

```python
c.HyperparameterTuner(model, param_grid, method="grid", cv_folds=3, scoring=None, n_iter=10, name=None, config=None)
```

**Parameters:**

- `model` (Model): The model component to tune.
- `param_grid` (dict): Dictionary mapping parameter names to lists of values to try.
- `method` (str, optional): Tuning method. Options are "grid", "random", "bayesian". Default is "grid".
- `cv_folds` (int, optional): Number of cross-validation folds. Default is 3.
- `scoring` (str, optional): Scoring metric to use. Default is None (uses accuracy for classification, r2 for regression).
- `n_iter` (int, optional): Number of iterations for random or bayesian search. Default is 10.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

## Advanced ML Components

### AutoML

Provides AutoML capabilities for automated model selection and hyperparameter tuning.

#### Constructor

```python
c.AutoML(target_column, mode="classification", time_budget=60, metric=None, include_models=None, exclude_models=None, name=None, config=None)
```

**Parameters:**

- `target_column` (str): Name of the target column.
- `mode` (str, optional): Model mode. Options are "classification" or "regression". Default is "classification".
- `time_budget` (int, optional): Time budget in minutes. Default is 60.
- `metric` (str, optional): Metric to optimize. Default is None (uses accuracy for classification, r2 for regression).
- `include_models` (list, optional): List of model types to include. Default is None (includes all).
- `exclude_models` (list, optional): List of model types to exclude. Default is None.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### Ensemble

Creates an ensemble of multiple models.

#### Constructor

```python
c.Ensemble(models, method="voting", weights=None, target_column=None, name=None, config=None)
```

**Parameters:**

- `models` (list): List of model components.
- `method` (str, optional): Ensembling method. Options are "voting", "stacking", "bagging", "boosting". Default is "voting".
- `weights` (list, optional): List of weights for weighted voting. Default is None (equal weights).
- `target_column` (str, optional): Name of the target column. Default is None.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### FeatureImportance

Analyzes and visualizes feature importance.

#### Constructor

```python
c.FeatureImportance(method="permutation", n_repeats=10, target_column=None, plot=False, output_file=None, name=None, config=None)
```

**Parameters:**

- `method` (str, optional): Method for calculating feature importance. Options are "permutation", "shap", "built_in". Default is "permutation".
- `n_repeats` (int, optional): Number of repetitions for permutation method. Default is 10.
- `target_column` (str, optional): Name of the target column. Default is None.
- `plot` (bool, optional): Whether to generate a plot. Default is False.
- `output_file` (str, optional): Path to save the plot. Default is None.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### ModelExplainer

Explains model predictions.

#### Constructor

```python
c.ModelExplainer(method="shap", n_samples=100, output_format="html", name=None, config=None)
```

**Parameters:**

- `method` (str, optional): Explanation method. Options are "shap", "lime", "eli5". Default is "shap".
- `n_samples` (int, optional): Number of samples to use for explanation. Default is 100.
- `output_format` (str, optional): Format of the output. Options are "html", "json", "png". Default is "html".
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

## Deployment Components

### ModelAPI

Creates a REST API for a trained model.

#### Constructor

```python
c.ModelAPI(model_path, host="0.0.0.0", port=8000, input_schema=None, name=None, config=None)
```

**Parameters:**

- `model_path` (str): Path to the model file.
- `host` (str, optional): Host to bind the server to. Default is "0.0.0.0".
- `port` (int, optional): Port to bind the server to. Default is 8000.
- `input_schema` (dict, optional): Schema defining expected input fields and types. Default is None.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### ModelMonitor

Monitors model performance in production.

#### Constructor

```python
c.ModelMonitor(metrics, alert_threshold=0.1, alert_method="email", alert_config=None, storage="database", storage_config=None, name=None, config=None)
```

**Parameters:**

- `metrics` (list): List of metrics to monitor.
- `alert_threshold` (float, optional): Threshold for triggering alerts. Default is 0.1.
- `alert_method` (str, optional): Method for sending alerts. Options are "email", "slack", "webhook". Default is "email".
- `alert_config` (dict, optional): Configuration for alerts. Default is None.
- `storage` (str, optional): Storage method for monitoring data. Options are "database", "file", "cloud". Default is "database".
- `storage_config` (dict, optional): Configuration for storage. Default is None.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

## Framework Adapters

### SklearnAdapter

Adapts a scikit-learn pipeline to a LlamaChain component.

#### Constructor

```python
c.SklearnAdapter(pipeline, target_column=None, name=None, config=None)
```

**Parameters:**

- `pipeline` (sklearn.pipeline.Pipeline): The scikit-learn pipeline.
- `target_column` (str, optional): Name of the target column. Default is None.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### TensorFlowAdapter

Adapts a TensorFlow/Keras model to a LlamaChain component.

#### Constructor

```python
c.TensorFlowAdapter(model_fn=None, model=None, target_column=None, batch_size=32, epochs=10, name=None, config=None)
```

**Parameters:**

- `model_fn` (callable, optional): Function that returns a Keras model. Either model_fn or model must be provided. Default is None.
- `model` (tensorflow.keras.Model, optional): Pre-built Keras model. Default is None.
- `target_column` (str, optional): Name of the target column. Default is None.
- `batch_size` (int, optional): Batch size for training. Default is 32.
- `epochs` (int, optional): Number of epochs for training. Default is 10.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None.

### PyTorchAdapter

Adapts a PyTorch model to a LlamaChain component.

#### Constructor

```python
c.PyTorchAdapter(model_class=None, model=None, target_column=None, batch_size=32, epochs=10, learning_rate=0.001, name=None, config=None)
```

**Parameters:**

- `model_class` (type, optional): PyTorch model class. Either model_class or model must be provided. Default is None.
- `model` (torch.nn.Module, optional): Pre-built PyTorch model. Default is None.
- `target_column` (str, optional): Name of the target column. Default is None.
- `batch_size` (int, optional): Batch size for training. Default is 32.
- `epochs` (int, optional): Number of epochs for training. Default is 10.
- `learning_rate` (float, optional): Learning rate for optimizer. Default is 0.001.
- `name` (str, optional): Name of the component. Default is None.
- `config` (dict, optional): Configuration dictionary. Default is None. 