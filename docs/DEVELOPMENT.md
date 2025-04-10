# Development Guide for LlamaChain

This guide will help you set up your development environment for working on the LlamaChain project.

## Initial Setup

### Prerequisites

- Python 3.9+ installed
- Git installed
- VSCode with Python and Pylance extensions (recommended)

### Clone the Repository

```bash
git clone https://github.com/llamachain/llamachain.git
cd llamachain
```

### Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate
```

## Installation Options

LlamaChain has several optional dependencies that can be installed based on your development needs. Here are the different installation options:

### Option 1: Basic Installation (Core Features Only)

```bash
# Install core dependencies only
pip install -e .
```

### Option 2: Development Installation (Recommended for Contributors)

```bash
# Install with development tools (testing, linting, etc.)
pip install -e ".[dev]"
```

### Option 3: Feature-specific Installation

```bash
# Install for NLP development
pip install -e ".[nlp]"

# Install for security/zero-knowledge proof development
pip install -e ".[security]"

# Install everything
pip install -e ".[all]"
```

## Install spaCy Models (For NLP Development)

If you're working on NLP features, you'll need to install spaCy models:

```bash
# Make the script executable
chmod +x scripts/install_spacy_models.py

# Run the script to install models
./scripts/install_spacy_models.py

# Or use the entry point
llamachain-install-spacy
```

## IDE Setup

### Configuring VSCode

1. Open the LlamaChain project in VSCode
2. Make sure you have the Python and Pylance extensions installed
3. Select your virtual environment as the Python interpreter
4. The project includes VSCode settings in `.vscode/settings.json`

## Type Checking

LlamaChain uses type hints throughout the codebase and provides stub files for third-party libraries.

### Running Type Checks

```bash
# Run mypy for static type checking
mypy llamachain

# Or use pyright (installed with Pylance)
pyright
```

### Understanding Import Errors

Sometimes Pylance may report errors for imports that are handled at runtime. 
We handle these with:

1. **Type stub files** in the `stubs/` directory
2. **TYPE_CHECKING pattern** for optional dependencies
3. **Type comments** for fallback cases

See `docs/TYPE_CHECKING_GUIDE.md` for more details.

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_specific_module.py

# Run with coverage
pytest --cov=llamachain
```

## Linting and Formatting

```bash
# Format code
black llamachain

# Sort imports
isort llamachain

# Run flake8
flake8 llamachain
```

## Common Development Tasks

### Adding New Dependencies

1. Add the dependency to `requirements.txt` or in setup.py's `extras_require`
2. Update relevant documentation
3. If it's an optional dependency, use the `TYPE_CHECKING` pattern as described in our type checking guide

### Creating Type Stubs

If you're working with a third-party library without type information:

1. Create stub files in the `stubs/` directory following the module structure
2. Make sure the stub files have `.pyi` extension
3. Update `setup.py` to include the new stubs in `data_files`

## Documentation

### Installing Development Documentation Tools

```bash
pip install sphinx sphinx-rtd-theme
```

### Building Documentation

```bash
cd docs
make html
```

## Troubleshooting

### Import Issues

If you encounter import errors or "unresolved import" warnings:

1. Make sure you've installed the appropriate dependencies with the right extras
2. Check if you need to add type stubs for the library
3. Refer to our `docs/IDE_SETUP.md` for VSCode-specific configuration
4. Try restarting your IDE to refresh the language server

### spaCy Model Issues

If you're having trouble with spaCy models:

```bash
# Show all installed models
python -c "import spacy; print(spacy.util.get_installed_models())"

# Manually install a specific model
python -m spacy download en_core_web_sm
```

### Type Checking Issues

If you're having issues with type checking:

1. Make sure your IDE is configured to use Pylance
2. Check that the stubs are properly located in the stubs directory
3. Try adding `# type: ignore` comments for problematic lines as a temporary solution 