# Core Concepts

This guide explains the fundamental concepts and architecture of LlamaChain. Understanding these concepts will help you build efficient and maintainable data processing pipelines.

## The Chain Model

LlamaChain is built around the concept of a data processing chain. This architecture is inspired by both the Unix pipeline philosophy and functional programming concepts:

- **Input → Process → Output**: Data flows through a sequence of operations
- **Composability**: Simple components can be combined to create complex systems
- **Single Responsibility**: Each component does one thing well
- **Immutability**: Components don't modify the original data, but create transformed versions

## Key Components

### Chain

A `Chain` is the fundamental orchestration unit in LlamaChain. It:

- Contains a sequence of components
- Manages the flow of data between components
- Handles the execution lifecycle
- Collects metrics and execution information

```python
from llamachain import Chain
from llamachain import components as c

my_chain = Chain([
    c.Component1(),
    c.Component2(),
    c.Component3(),
])

result = my_chain.run(initial_data=input_data)
```

### Component

A `Component` is a processing unit that performs a specific operation on data. Components:

- Have a well-defined interface (`process` method)
- Can transform data
- Can have configuration parameters
- Can maintain state (though stateless is preferred)
- Can have side effects (e.g., writing to files, calling APIs)

```python
from llamachain import Component

class MyCustomComponent(Component):
    def __init__(self, parameter1, parameter2):
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        
    def process(self, context):
        # Transform the data in context
        transformed_data = some_transformation(context.data, 
                                              self.parameter1,
                                              self.parameter2)
        
        # Update the context with transformed data
        context.update(data=transformed_data)
        
        # Return the updated context
        return context
```

### Context

The `Context` is an object that flows through the chain, carrying:

- The current data being processed
- Metadata about the execution
- Temporary state that might be needed between components
- Configuration that can affect component behavior

```python
# Context is usually created and managed by the Chain,
# but you can create and manipulate it directly:

from llamachain import Context

context = Context(
    data=your_initial_data,
    metadata={"source": "file.csv", "timestamp": "2023-07-21T12:00:00Z"},
    config={"verbose": True, "max_rows": 1000}
)

# Components can read from context
data = context.data
source = context.metadata["source"]

# Components can update context
context.update(data=transformed_data)
context.metadata["processed_by"] = "my_component"
```

## Flow Control

LlamaChain provides several ways to control the flow of data:

### Branching

`Branch` allows conditional execution of different sub-chains:

```python
from llamachain import Chain, Branch
from llamachain import components as c

chain = Chain([
    c.DataLoader("data.csv"),
    Branch(
        condition=lambda ctx: ctx.data["size"] > 1000,
        if_true=Chain([
            c.Logger("Processing large dataset"),
            c.BatchProcessor(batch_size=100)
        ]),
        if_false=Chain([
            c.Logger("Processing small dataset"),
            c.StandardProcessor()
        ])
    ),
    c.DataWriter("output.csv")
])
```

### Looping

`Loop` allows iterative processing of data:

```python
from llamachain import Chain, Loop
from llamachain import components as c

chain = Chain([
    c.DataLoader("data.csv"),
    Loop(
        condition=lambda ctx: ctx.iteration < 5 and ctx.metrics["error"] > 0.01,
        body=Chain([
            c.Optimizer(),
            c.ErrorCalculator()
        ])
    ),
    c.ModelSaver("model.pkl")
])
```

### Error Handling

`TryCatch` provides exception handling capabilities:

```python
from llamachain import Chain, TryCatch
from llamachain import components as c

chain = Chain([
    TryCatch(
        try_block=Chain([
            c.RiskyOperation()
        ]),
        catch_block=Chain([
            c.Logger("Error occurred"),
            c.FallbackOperation()
        ]),
        finally_block=Chain([
            c.Cleanup()
        ])
    )
])
```

## Pipeline Configuration

LlamaChain chains can be configured for different execution modes:

### Sequential Execution

The default execution mode processes each component in sequence:

```python
chain = Chain(
    components=[c1, c2, c3],
    mode="sequential"  # This is the default
)
```

### Parallel Execution

For components that can run independently:

```python
chain = Chain(
    components=[c1, c2, c3],
    mode="parallel",
    max_workers=4
)
```

### Asynchronous Execution

For non-blocking operations:

```python
chain = Chain(
    components=[c1, c2, c3],
    mode="async"
)

# Run asynchronously
await chain.run_async()
```

## Component Registry

LlamaChain maintains a registry of built-in components:

```python
from llamachain import component_registry

# List all available components
all_components = component_registry.list_components()

# Get a component by name
file_loader = component_registry.get_component("FileLoader")

# Register a custom component
component_registry.register_component("MyCustomComponent", MyCustomComponent)
```

## Serialization

Chains can be serialized to and loaded from configuration files:

```python
# Save a chain configuration
chain.save_config("my_pipeline.yaml")

# Load a chain from configuration
from llamachain import Chain
loaded_chain = Chain.from_config("my_pipeline.yaml")
```

## Next Steps

Now that you understand the core concepts:

1. Learn about different component categories in the [Components Guide](../api/core.md)
2. Explore how to build [ML Pipelines](ml-components.md) with LlamaChain
3. Learn about [Web API Integration](web-applications.md) for your chains 