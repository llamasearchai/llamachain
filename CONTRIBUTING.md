# Contributing to LlamaChain

Thank you for your interest in contributing to LlamaChain! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

If you find a bug in the project, please create an issue on GitHub with the following information:

- A clear, descriptive title
- A detailed description of the issue
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Environment information (OS, Python version, etc.)

### Suggesting Enhancements

If you have an idea for an enhancement, please create an issue on GitHub with the following information:

- A clear, descriptive title
- A detailed description of the enhancement
- Any relevant examples or mockups
- Why this enhancement would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/your-bugfix-name`
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Commit your changes with a descriptive commit message
6. Push your branch to your fork
7. Submit a pull request to the main repository

## Development Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llamachain.git
   cd llamachain
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Code Style

We follow the [Black](https://black.readthedocs.io/en/stable/) code style. Please ensure your code is formatted with Black before submitting a pull request.

```
black llamachain
```

We also use [isort](https://pycqa.github.io/isort/) to sort imports:

```
isort llamachain
```

And [flake8](https://flake8.pycqa.org/en/latest/) for linting:

```
flake8 llamachain
```

## Testing

Please write tests for your code and ensure all tests pass before submitting a pull request:

```
pytest
```

## Documentation

Please update the documentation to reflect your changes. This includes:

- Code comments
- Docstrings
- README.md and other markdown files
- API documentation

## License

By contributing to LlamaChain, you agree that your contributions will be licensed under the project's [MIT License](LICENSE). 