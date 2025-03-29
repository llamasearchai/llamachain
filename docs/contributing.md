# Contributing to LlamaChain

Thank you for your interest in contributing to LlamaChain! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read and follow our [Code of Conduct](https://github.com/llamasearch/llamachain/blob/main/CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- Python 3.8 or newer
- pip
- git

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```
   git clone https://github.com/your-username/llamachain.git
   cd llamachain
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```
   pip install -e ".[dev]"
   ```

### Development Workflow

1. Create a new branch for your feature or bugfix:
   ```
   git checkout -b feature/your-feature-name
   ```
   or
   ```
   git checkout -b fix/issue-number
   ```

2. Make your changes, following our coding standards

3. Add tests for your changes

4. Run the tests locally:
   ```
   pytest
   ```

5. Make sure your code passes the linting checks:
   ```
   flake8 llamachain
   black --check llamachain
   isort --check llamachain
   mypy llamachain
   ```

6. Commit your changes using conventional commit messages:
   ```
   git commit -m "feat: add new component for time series processing"
   ```

7. Push your branch to your fork:
   ```
   git push origin feature/your-feature-name
   ```

8. Submit a pull request to the main repository

## Coding Standards

We follow these coding standards:

- [PEP 8](https://www.python.org/dev/peps/pep-0008/) - Style Guide for Python Code
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Use [Type Hints](https://www.python.org/dev/peps/pep-0484/) for all functions and methods
- Format code with [Black](https://black.readthedocs.io/)
- Sort imports with [isort](https://pycqa.github.io/isort/)

### Docstrings

We use Google-style docstrings. Example:

```python
def function_with_types_in_docstring(param1: int, param2: str) -> bool:
    """Example function with docstring.

    This function demonstrates proper docstring format.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        True if successful, False otherwise.

    Raises:
        ValueError: If param1 is negative.
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative.")
    return True
```

## Adding New Components

When adding a new component to LlamaChain:

1. Place your component in the appropriate module in the `llamachain/components/` directory
2. Ensure your component inherits from the `Component` base class
3. Implement the required `process` method
4. Add comprehensive docstrings
5. Add unit tests in the `tests/components/` directory
6. Update the component registry if necessary
7. Add an example of using your component to the documentation

Example of a new component:

```python
from typing import Dict, Any, Optional

from llamachain import Component


class MyNewComponent(Component):
    """Component for performing a specific task.

    This component does something useful with data.

    Attributes:
        param1: Description of parameter 1.
        param2: Description of parameter 2.
    """

    def __init__(
        self, 
        param1: str, 
        param2: Optional[int] = None, 
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the component.

        Args:
            param1: Description of parameter 1.
            param2: Description of parameter 2. Default is None.
            name: Component name. Default is None.
            config: Additional configuration. Default is None.
        """
        super().__init__(name=name, config=config)
        self.param1 = param1
        self.param2 = param2 or 10  # Default value

    def process(self, context):
        """Process the context.

        Args:
            context: The context object containing data and metadata.

        Returns:
            The updated context.
        """
        # Implementation
        data = context.data
        
        # Process the data
        processed_data = self._process_data(data)
        
        # Update metrics
        context.metadata["processed_records"] = len(processed_data)
        
        # Update context with processed data
        context.update(data=processed_data)
        
        return context
        
    def _process_data(self, data):
        """Internal helper method to process data.

        Args:
            data: The data to process.

        Returns:
            The processed data.
        """
        # Implementation
        return data
```

## Testing

We use pytest for testing. All new code should have corresponding tests.

### Running Tests

Run all tests:
```
pytest
```

Run tests for a specific module:
```
pytest tests/components/test_my_component.py
```

Run tests with coverage:
```
pytest --cov=llamachain
```

### Writing Tests

- Place tests in the `tests/` directory, mirroring the package structure
- Name test files with a `test_` prefix
- Name test functions with a `test_` prefix
- Use fixtures where appropriate
- Write tests for both normal operation and edge cases

Example test:

```python
import pytest
import pandas as pd
from llamachain import Context
from llamachain.components import MyNewComponent


def test_my_new_component_basic():
    # Setup
    component = MyNewComponent(param1="test")
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    context = Context(data=data)
    
    # Execute
    result = component.process(context)
    
    # Assert
    assert result.data is not None
    assert len(result.data) == 3
    assert "processed_records" in result.metadata
    assert result.metadata["processed_records"] == 3


def test_my_new_component_empty_data():
    # Setup for edge case
    component = MyNewComponent(param1="test")
    data = pd.DataFrame()
    context = Context(data=data)
    
    # Execute
    result = component.process(context)
    
    # Assert
    assert result.data is not None
    assert len(result.data) == 0
    assert result.metadata["processed_records"] == 0
```

## Documentation

We use MkDocs with the Material theme for documentation.

### Building Documentation

Build and view the documentation locally:
```
mkdocs serve
```

### Writing Documentation

- Place documentation in the `docs/` directory
- Use Markdown for documentation files
- Include examples for new features
- Document public API changes
- Update the changelog for noteworthy changes

## Pull Request Process

1. Ensure your code follows our coding standards
2. Update the documentation if needed
3. Add or update tests as appropriate
4. Update the changelog if necessary
5. Submit your pull request with a clear description of the changes
6. Wait for the CI checks to pass
7. Address any feedback from the code review

## Releasing

The release process is handled by maintainers. If you're interested in the release process, see the [RELEASING.md](https://github.com/llamasearch/llamachain/blob/main/RELEASING.md) document.

## Getting Help

If you need help with your contribution:

- Ask questions on [GitHub Discussions](https://github.com/llamasearch/llamachain/discussions)
- Join our [Discord community](https://discord.gg/llamachain)
- Check the [documentation](https://llamasearch.github.io/llamachain)

## Thank You

Your contributions to open source, large or small, make projects like this possible. Thank you for taking the time to contribute. 