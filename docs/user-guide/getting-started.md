# Getting Started with LlamaChain

This guide will walk you through the basics of using LlamaChain to create data processing pipelines.

## Core Concepts

LlamaChain is built around a few key concepts:

- **Chain**: A sequence of components that process data in order
- **Component**: An individual processing unit that performs a specific operation
- **Pipeline**: A configured chain that's ready to process data
- **Context**: Data and metadata that flows through the chain

## Your First Chain

Let's create a simple chain that loads data from a CSV file, processes it, and saves the results.

```python
from llamachain import Chain
from llamachain import components as c

# Create a chain
my_chain = Chain([
    c.FileLoader("data.csv"),
    c.DataCleaner(drop_nulls=True),
    c.ColumnSelector(["name", "age", "income"]),
    c.DataTransformer(transformations={
        "age": lambda x: x + 1,  # Add 1 to all ages
        "income": lambda x: x * 1.1  # 10% raise for everyone!
    }),
    c.FileWriter("processed_data.csv")
])

# Run the chain
result = my_chain.run()

print(f"Processed {result.record_count} records")
print(f"Time taken: {result.execution_time:.2f} seconds")
```

## Understanding Chain Execution

When you call `run()` on a chain:

1. Each component is executed in sequence
2. Data flows from one component to the next through a context object
3. Components can modify the data, add metadata, or trigger side effects
4. The final result contains the processed data and execution metadata

## Working with Components

LlamaChain comes with many built-in components for common tasks:

### Data Loading

```python
# Load from various sources
csv_loader = c.FileLoader("data.csv")
json_loader = c.JSONLoader("data.json")
db_loader = c.DatabaseLoader(
    connection_string="postgresql://user:password@localhost/db",
    query="SELECT * FROM users WHERE age > 18"
)
api_loader = c.APILoader(
    url="https://api.example.com/data",
    headers={"Authorization": "Bearer token"}
)
```

### Data Processing

```python
# Clean and transform data
cleaner = c.DataCleaner(
    drop_nulls=True,
    drop_duplicates=True,
    fill_values={"age": 0, "income": "unknown"}
)

transformer = c.DataTransformer(
    transformations={
        "name": lambda x: x.upper(),
        "joined_date": lambda x: pd.to_datetime(x)
    }
)

filter = c.DataFilter(
    conditions={"age": lambda x: x >= 18}
)
```

### Data Export

```python
# Save results in various formats
csv_writer = c.FileWriter("output.csv")
json_writer = c.JSONWriter("output.json")
db_writer = c.DatabaseWriter(
    connection_string="postgresql://user:password@localhost/db",
    table_name="processed_users",
    if_exists="replace"
)
```

## Adding Conditions and Branching

You can add conditional logic to your chains:

```python
from llamachain import Chain, Condition, Branch
from llamachain import components as c

# Create a chain with conditional processing
my_chain = Chain([
    c.FileLoader("data.csv"),
    c.DataCleaner(drop_nulls=True),
    # Branch based on a condition
    Branch(
        condition=lambda ctx: ctx.data["category"].iloc[0] == "premium",
        if_true=Chain([
            c.Logger("Processing premium data"),
            c.DataTransformer({"price": lambda x: x * 0.9})  # 10% discount
        ]),
        if_false=Chain([
            c.Logger("Processing standard data"),
            c.DataTransformer({"price": lambda x: x * 0.95})  # 5% discount
        ])
    ),
    c.FileWriter("processed_data.csv")
])
```

## Error Handling

LlamaChain provides mechanisms for handling errors:

```python
from llamachain import Chain, TryCatch
from llamachain import components as c

# Create a chain with error handling
my_chain = Chain([
    TryCatch(
        try_block=Chain([
            c.FileLoader("data.csv"),
            c.DataCleaner(drop_nulls=True),
        ]),
        catch_block=Chain([
            c.Logger("Error loading or cleaning data, using backup"),
            c.FileLoader("backup_data.csv")
        ]),
        finally_block=Chain([
            c.Logger("Processing complete")
        ])
    ),
    c.FileWriter("processed_data.csv")
])
```

## Next Steps

Now that you understand the basics:

1. Check out the [Core Concepts](core-concepts.md) guide for a deeper dive
2. Explore the [API Reference](../api/core.md) for details on all components
3. Try the [Examples](../examples.md) to see LlamaChain in action

Happy chaining! 