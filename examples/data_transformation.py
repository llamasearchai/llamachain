#!/usr/bin/env python
"""
Data transformation example with LlamaChain

This example demonstrates how to create a data transformation pipeline
that processes structured data (JSON) through multiple transformation steps.
"""

import json
from typing import Any, Dict, List

from llamachain.core import Component, Pipeline


class DataFilter(Component):
    """Component to filter data based on conditions"""

    def __init__(self, field: str, condition: callable, name=None, config=None):
        """Initialize the data filter

        Args:
            field: The field to filter on
            condition: Function that returns True for items to keep
            name: Optional component name
            config: Optional component configuration
        """
        self.field = field
        self.condition = condition
        super().__init__(name, config)

    def process(self, input_data: List[Dict]) -> List[Dict]:
        """Filter a list of dictionaries based on the condition

        Args:
            input_data: List of dictionaries to filter

        Returns:
            Filtered list of dictionaries
        """
        if not isinstance(input_data, list):
            raise TypeError(f"Expected list, got {type(input_data).__name__}")

        return [
            item
            for item in input_data
            if self.field in item and self.condition(item[self.field])
        ]


class DataMapper(Component):
    """Component to map data from one format to another"""

    def __init__(self, mapping: Dict[str, str], name=None, config=None):
        """Initialize the data mapper

        Args:
            mapping: Dictionary mapping source fields to target fields
            name: Optional component name
            config: Optional component configuration
        """
        self.mapping = mapping
        super().__init__(name, config)

    def process(self, input_data: List[Dict]) -> List[Dict]:
        """Map fields in a list of dictionaries

        Args:
            input_data: List of dictionaries to transform

        Returns:
            List of transformed dictionaries
        """
        if not isinstance(input_data, list):
            raise TypeError(f"Expected list, got {type(input_data).__name__}")

        result = []
        for item in input_data:
            new_item = {}
            for source_field, target_field in self.mapping.items():
                if source_field in item:
                    new_item[target_field] = item[source_field]
            result.append(new_item)

        return result


class DataAggregator(Component):
    """Component to aggregate data based on a key field"""

    def __init__(
        self,
        key_field: str,
        aggregate_field: str,
        operation: str = "sum",
        name=None,
        config=None,
    ):
        """Initialize the data aggregator

        Args:
            key_field: Field to group by
            aggregate_field: Field to aggregate
            operation: Aggregation operation (sum, avg, max, min, count)
            name: Optional component name
            config: Optional component configuration
        """
        self.key_field = key_field
        self.aggregate_field = aggregate_field
        self.operation = operation.lower()

        if self.operation not in ("sum", "avg", "max", "min", "count"):
            raise ValueError(f"Unsupported operation: {operation}")

        super().__init__(name, config)

    def process(self, input_data: List[Dict]) -> List[Dict]:
        """Aggregate data by key field

        Args:
            input_data: List of dictionaries to aggregate

        Returns:
            List of dictionaries with aggregated values
        """
        if not isinstance(input_data, list):
            raise TypeError(f"Expected list, got {type(input_data).__name__}")

        # Group by key field
        grouped = {}
        for item in input_data:
            if self.key_field not in item:
                continue

            key = item[self.key_field]
            if key not in grouped:
                grouped[key] = []

            if self.operation != "count" and self.aggregate_field not in item:
                continue

            grouped[key].append(item)

        # Aggregate values
        result = []
        for key, items in grouped.items():
            if self.operation == "sum":
                value = sum(item[self.aggregate_field] for item in items)
            elif self.operation == "avg":
                values = [item[self.aggregate_field] for item in items]
                value = sum(values) / len(values) if values else 0
            elif self.operation == "max":
                value = max(item[self.aggregate_field] for item in items)
            elif self.operation == "min":
                value = min(item[self.aggregate_field] for item in items)
            elif self.operation == "count":
                value = len(items)

            result.append(
                {self.key_field: key, f"{self.aggregate_field}_{self.operation}": value}
            )

        return result


def main():
    """Run a data transformation pipeline example"""
    print("LlamaChain Data Transformation Example")
    print("======================================\n")

    # Sample sales data
    sample_data = [
        {"id": 1, "product": "Widget", "category": "A", "price": 10.0, "quantity": 5},
        {"id": 2, "product": "Gadget", "category": "B", "price": 20.0, "quantity": 3},
        {"id": 3, "product": "Tool", "category": "A", "price": 15.0, "quantity": 2},
        {"id": 4, "product": "Device", "category": "C", "price": 30.0, "quantity": 1},
        {
            "id": 5,
            "product": "Widget Pro",
            "category": "A",
            "price": 25.0,
            "quantity": 4,
        },
    ]

    print("Original data:")
    print(json.dumps(sample_data, indent=2))
    print()

    # Create a pipeline to filter, transform, and aggregate data
    pipeline = Pipeline(
        [
            # Filter items in category A
            DataFilter(
                field="category", condition=lambda x: x == "A", name="CategoryFilter"
            ),
            # Map fields to new structure
            DataMapper(
                mapping={
                    "product": "item_name",
                    "price": "unit_price",
                    "quantity": "units_sold",
                    "category": "product_category",
                },
                name="FieldMapper",
            ),
            # Calculate total sales amount
            Component(
                name="SalesCalculator",
                process=lambda items: [
                    {**item, "total_sales": item["unit_price"] * item["units_sold"]}
                    for item in items
                ],
            ),
        ]
    )

    # Process the data
    transformed_data = pipeline.run(sample_data)

    print("Transformed data (Category A items with mapped fields and total sales):")
    print(json.dumps(transformed_data, indent=2))
    print()

    # Create an aggregation pipeline
    aggregation_pipeline = Pipeline(
        [
            # Aggregate sales by product category
            DataAggregator(
                key_field="product_category",
                aggregate_field="total_sales",
                operation="sum",
                name="SalesAggregator",
            )
        ]
    )

    # Aggregate the transformed data
    aggregated_data = aggregation_pipeline.run(transformed_data)

    print("Aggregated data (Total sales by category):")
    print(json.dumps(aggregated_data, indent=2))


if __name__ == "__main__":
    main()
