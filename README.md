# LlamaChain

A flexible data processing pipeline framework for AI applications

[![GitHub](https://img.shields.io/github/license/llamasearchai/llamachain)](https://github.com/llamasearchai/llamachain/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/llamachain.svg)](https://pypi.org/project/llamachain/)

## Overview


A flexible data processing pipeline framework for AI applications. This library provides a comprehensive set of tools and utilities for
working with chain tasks in AI and data processing workflows.
It's designed to be easy to use while offering powerful capabilities for complex scenarios.


## Features


- **Component-Based Architecture**: Build processing pipelines by connecting reusable components
- **Flexible Pipeline Design**: Create custom data processing flows for any use case
- **Extensive Component Library**: Use pre-built components or create your own
- **Type Safety**: Comprehensive type hints ensure correct usage
- **Easy Extensibility**: Simple interfaces for creating custom components


## Installation

```bash
pip install llamachain
```

## Usage

```python

from llamachain import Pipeline, Component

# Create a pipeline with components
pipeline = Pipeline([
    Component1(),
    Component2(),
    Component3()
])

# Process data through the pipeline
result = pipeline.run(input_data)
print(result)

```

## Documentation

For more detailed documentation, see the [docs](docs/) directory.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
