# Core API Reference

This page documents the core classes and functions in the LlamaChain package.

## Chain

The `Chain` class is the main orchestration component in LlamaChain.

```python
from llamachain import Chain
```

### Constructor

```python
Chain(components, mode="sequential", max_workers=None, name=None, config=None)
```

**Parameters:**

- `components` (list): List of components to execute in sequence.
- `mode` (str): Execution mode - one of "sequential", "parallel", or "async". Default is "sequential".
- `max_workers` (int, optional): Maximum number of worker threads or processes for parallel execution. Default is None (uses CPU count).
- `name` (str, optional): Name of the chain. Default is None.
- `config` (dict, optional): Configuration dictionary to be passed to all components. Default is None.

### Methods

#### run

```python
run(initial_data=None, context=None)
```

Executes the chain on the provided data.

**Parameters:**

- `initial_data` (any, optional): Initial data to process. Default is None.
- `context` (Context, optional): Context object to use. If not provided, a new one will be created. Default is None.

**Returns:**

- `Context`: The final context after all components have been processed.

#### run_async

```python
async run_async(initial_data=None, context=None)
```

Executes the chain asynchronously.

**Parameters:**

- `initial_data` (any, optional): Initial data to process. Default is None.
- `context` (Context, optional): Context object to use. If not provided, a new one will be created. Default is None.

**Returns:**

- `Context`: The final context after all components have been processed.

#### save_config

```python
save_config(file_path)
```

Saves the chain configuration to a file.

**Parameters:**

- `file_path` (str): Path to save the configuration file.

#### from_config

```python
@classmethod
from_config(file_path)
```

Creates a new chain from a configuration file.

**Parameters:**

- `file_path` (str): Path to the configuration file.

**Returns:**

- `Chain`: A new Chain instance.

## Component

The `Component` class is the base class for all components in LlamaChain.

```python
from llamachain import Component
```

### Constructor

```python
Component(name=None, config=None)
```

**Parameters:**

- `name` (str, optional): Name of the component. Default is None (uses class name).
- `config` (dict, optional): Configuration dictionary. Default is None.

### Methods

#### process

```python
process(context)
```

Processes the context. This method should be overridden by subclasses.

**Parameters:**

- `context` (Context): The context to process.

**Returns:**

- `Context`: The processed context.

#### setup

```python
setup()
```

Performs setup operations before processing. Can be overridden by subclasses.

#### teardown

```python
teardown()
```

Performs cleanup operations after processing. Can be overridden by subclasses.

## Context

The `Context` class represents the data and metadata flowing through a chain.

```python
from llamachain import Context
```

### Constructor

```python
Context(data=None, metadata=None, config=None)
```

**Parameters:**

- `data` (any, optional): The data to be processed. Default is None.
- `metadata` (dict, optional): Metadata about the execution. Default is an empty dict.
- `config` (dict, optional): Configuration parameters. Default is an empty dict.

### Methods

#### update

```python
update(data=None, metadata=None, config=None)
```

Updates the context with new data, metadata, or config.

**Parameters:**

- `data` (any, optional): New data to use. If None, the current data is kept. Default is None.
- `metadata` (dict, optional): Metadata to update. If provided, it's merged with existing metadata. Default is None.
- `config` (dict, optional): Config to update. If provided, it's merged with existing config. Default is None.

**Returns:**

- `Context`: The updated context (self).

#### add_metric

```python
add_metric(name, value)
```

Adds a metric to the context.

**Parameters:**

- `name` (str): The name of the metric.
- `value` (any): The value of the metric.

#### copy

```python
copy()
```

Creates a deep copy of the context.

**Returns:**

- `Context`: A new Context object with copied data.

## Flow Control Components

### Branch

The `Branch` component provides conditional execution of sub-chains.

```python
from llamachain import Branch
```

#### Constructor

```python
Branch(condition, if_true, if_false=None)
```

**Parameters:**

- `condition` (callable): A function that takes a context and returns a boolean.
- `if_true` (Chain): The chain to execute if the condition is True.
- `if_false` (Chain, optional): The chain to execute if the condition is False. Default is None.

### Loop

The `Loop` component provides iterative execution of a sub-chain.

```python
from llamachain import Loop
```

#### Constructor

```python
Loop(condition, body, max_iterations=100)
```

**Parameters:**

- `condition` (callable): A function that takes a context and returns a boolean.
- `body` (Chain): The chain to execute on each iteration.
- `max_iterations` (int, optional): Maximum number of iterations to prevent infinite loops. Default is 100.

### TryCatch

The `TryCatch` component provides exception handling.

```python
from llamachain import TryCatch
```

#### Constructor

```python
TryCatch(try_block, catch_block=None, finally_block=None, exceptions=Exception)
```

**Parameters:**

- `try_block` (Chain): The chain to execute.
- `catch_block` (Chain, optional): The chain to execute if an exception occurs. Default is None.
- `finally_block` (Chain, optional): The chain to execute at the end, regardless of whether an exception occurred. Default is None.
- `exceptions` (Exception or tuple, optional): The type(s) of exceptions to catch. Default is Exception.

## Utility Functions

### load_component

```python
from llamachain.utils import load_component

load_component(name)
```

Loads a component by name from the component registry.

**Parameters:**

- `name` (str): Name of the component to load.

**Returns:**

- `Component`: The component class.

### register_component

```python
from llamachain.utils import register_component

register_component(name, component_class)
```

Registers a component in the component registry.

**Parameters:**

- `name` (str): Name to register the component under.
- `component_class` (class): The component class to register.

### configure_logging

```python
from llamachain.utils import configure_logging

configure_logging(level="INFO", format=None, output_file=None)
```

Configures the logging for LlamaChain.

**Parameters:**

- `level` (str, optional): Logging level. Default is "INFO".
- `format` (str, optional): Log format string. Default is None (uses a predefined format).
- `output_file` (str, optional): File to write logs to. Default is None (logs to console). 