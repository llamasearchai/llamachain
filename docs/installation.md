# Installation Guide

This guide will help you get LlamaChain installed and set up on your system.

## Prerequisites

Before installing LlamaChain, make sure you have:

- Python 3.8 or newer
- pip (Python package installer)
- Optional: A virtual environment tool like `venv` or `conda`

## Basic Installation

You can install LlamaChain using pip:

```bash
pip install llamachain
```

This will install the core functionality of LlamaChain.

## Installation with Optional Components

LlamaChain has several optional components that you can install based on your needs:

### Machine Learning Components

If you plan to use machine learning features:

```bash
pip install llamachain[ml]
```

This includes dependencies like:
- scikit-learn
- xgboost
- tensorflow
- pytorch (CPU version)

### Natural Language Processing Components

For NLP functionality:

```bash
pip install llamachain[nlp]
```

This includes:
- nltk
- spacy
- transformers
- sentence-transformers

### Web Development Components

For building web apps and APIs:

```bash
pip install llamachain[web]
```

This includes:
- flask
- fastapi
- uvicorn
- jinja2

### Full Installation

To install all optional components:

```bash
pip install llamachain[full]
```

### Development Installation

If you're contributing to LlamaChain:

```bash
pip install llamachain[dev]
```

This includes all dependencies plus development tools like:
- pytest
- black
- isort
- mypy
- flake8

## Installation from Source

For the latest development version, you can install directly from the repository:

```bash
git clone https://github.com/llamasearch/llamachain.git
cd llamachain
pip install -e .
```

For development installations from source:

```bash
pip install -e ".[dev]"
```

## GPU Support

If you're using machine learning components and have a CUDA-compatible GPU:

```bash
pip install llamachain[ml-gpu]
```

This will install GPU-compatible versions of pytorch and tensorflow.

## Verification

To verify that LlamaChain was installed correctly:

```python
import llamachain
print(llamachain.__version__)

# Test a simple chain
from llamachain import Chain, components as c
test_chain = Chain([c.Echo("LlamaChain installation successful!")])
result = test_chain.run()
print(result)
```

## Troubleshooting

If you encounter issues during installation:

### Common Problems

1. **Missing dependencies**: Some optional dependencies may require system libraries. Check the error message for details.

2. **Version conflicts**: If you have conflicting packages, try installing in a fresh virtual environment:
   ```bash
   python -m venv llamachain-env
   source llamachain-env/bin/activate  # On Windows: llamachain-env\Scripts\activate
   pip install llamachain
   ```

3. **GPU support issues**: Make sure you have compatible CUDA drivers installed when using GPU components.

### Getting Help

If you continue to experience problems:

- Check the [GitHub issues](https://github.com/llamasearch/llamachain/issues) to see if others have encountered the same problem
- Ask a question on our [Discord community](https://discord.gg/llamachain)
- Open a new issue on GitHub with details about your environment and the error messages 