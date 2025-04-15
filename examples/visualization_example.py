#!/usr/bin/env python
"""
Visualization example with LlamaChain

This example demonstrates how to use visualization components to create charts
and render them in different formats.
"""

import json
import os
from typing import Any, Dict, List

from llamachain.core import Component, Pipeline
from llamachain.visualization import (
    BarChart,
    HTMLRenderer,
    LineChart,
    PieChart,
    TerminalRenderer,
)


def create_bar_chart():
    """Create and render a bar chart example"""
    print("\nBar Chart Example")
    print("================\n")

    # Sample data for categories and values
    sales_data = {"Q1": 120, "Q2": 180, "Q3": 240, "Q4": 160}

    # Create a pipeline with BarChart and HTML renderer
    pipeline = Pipeline(
        [
            # Transform the data into a bar chart
            BarChart(
                title="Quarterly Sales",
                x_label="Quarter",
                y_label="Sales ($1,000)",
                name="SalesBarChart",
            ),
            # Render the chart as HTML
            HTMLRenderer(output_file="sales_chart.html", name="HTMLOutput"),
        ]
    )

    # Process the data
    result = pipeline.run(sales_data)
    print(f"Bar chart rendered to HTML: {result}")

    # Create a pipeline with BarChart and Terminal renderer
    terminal_pipeline = Pipeline(
        [
            BarChart(title="Quarterly Sales", name="SalesBarChart"),
            TerminalRenderer(width=60, height=15, name="TerminalOutput"),
        ]
    )

    # Process the data and display in terminal
    terminal_result = terminal_pipeline.run(sales_data)
    print("\nTerminal Bar Chart:\n")
    print(terminal_result)


def create_line_chart():
    """Create and render a line chart example"""
    print("\nLine Chart Example")
    print("=================\n")

    # Sample time series data
    time_series_data = {
        "series": [
            {
                "name": "Product A",
                "data": [
                    {"x": 1, "y": 10},
                    {"x": 2, "y": 15},
                    {"x": 3, "y": 13},
                    {"x": 4, "y": 17},
                    {"x": 5, "y": 20},
                    {"x": 6, "y": 18},
                    {"x": 7, "y": 22},
                ],
            },
            {
                "name": "Product B",
                "data": [
                    {"x": 1, "y": 5},
                    {"x": 2, "y": 7},
                    {"x": 3, "y": 10},
                    {"x": 4, "y": 11},
                    {"x": 5, "y": 12},
                    {"x": 6, "y": 15},
                    {"x": 7, "y": 16},
                ],
            },
        ]
    }

    # Create a pipeline with LineChart and HTML renderer
    pipeline = Pipeline(
        [
            # Transform the data into a line chart
            LineChart(
                title="Weekly Product Sales",
                x_label="Week",
                y_label="Units Sold",
                name="SalesLineChart",
            ),
            # Render the chart as HTML
            HTMLRenderer(output_file="line_chart.html", name="HTMLOutput"),
        ]
    )

    # Process the data
    result = pipeline.run(time_series_data)
    print(f"Line chart rendered to HTML: {result}")

    # Create a pipeline with LineChart and Terminal renderer
    terminal_pipeline = Pipeline(
        [
            LineChart(title="Weekly Product Sales", name="SalesLineChart"),
            TerminalRenderer(width=80, height=25, name="TerminalOutput"),
        ]
    )

    # Process the data and display in terminal
    terminal_result = terminal_pipeline.run(time_series_data)
    print("\nTerminal Line Chart:\n")
    print(terminal_result)


def create_pie_chart():
    """Create and render a pie chart example"""
    print("\nPie Chart Example")
    print("================\n")

    # Sample data for market share
    market_share = {
        "Company A": 35,
        "Company B": 25,
        "Company C": 20,
        "Company D": 15,
        "Others": 5,
    }

    # Create a pipeline with PieChart and HTML renderer
    pipeline = Pipeline(
        [
            # Transform the data into a pie chart
            PieChart(title="Market Share Distribution", name="MarketSharePieChart"),
            # Render the chart as HTML
            HTMLRenderer(output_file="pie_chart.html", name="HTMLOutput"),
        ]
    )

    # Process the data
    result = pipeline.run(market_share)
    print(f"Pie chart rendered to HTML: {result}")

    # Create a pipeline with PieChart and Terminal renderer
    terminal_pipeline = Pipeline(
        [
            PieChart(title="Market Share Distribution", name="MarketSharePieChart"),
            TerminalRenderer(width=70, height=15, name="TerminalOutput"),
        ]
    )

    # Process the data and display in terminal
    terminal_result = terminal_pipeline.run(market_share)
    print("\nTerminal Pie Chart:\n")
    print(terminal_result)


def create_advanced_visualization():
    """Create an advanced visualization with data transformation"""
    print("\nAdvanced Visualization Example")
    print("============================\n")

    # Sample sales data
    sales_data = [
        {"region": "North", "product": "Widget", "sales": 100, "profit": 20},
        {"region": "North", "product": "Gadget", "sales": 200, "profit": 40},
        {"region": "South", "product": "Widget", "sales": 150, "profit": 30},
        {"region": "South", "product": "Gadget", "sales": 100, "profit": 20},
        {"region": "East", "product": "Widget", "sales": 120, "profit": 24},
        {"region": "East", "product": "Gadget", "sales": 180, "profit": 36},
        {"region": "West", "product": "Widget", "sales": 90, "profit": 18},
        {"region": "West", "product": "Gadget", "sales": 220, "profit": 44},
    ]

    # Create a custom component to aggregate sales by region
    class SalesAggregator(Component):
        def process(self, input_data: List[Dict[str, Any]]) -> Dict[str, int]:
            """Aggregate sales by region

            Args:
                input_data: List of sales records

            Returns:
                Dictionary with regions and total sales
            """
            result = {}
            for record in input_data:
                region = record["region"]
                sales = record["sales"]

                if region in result:
                    result[region] += sales
                else:
                    result[region] = sales

            return result

    # Create a pipeline for sales by region
    region_pipeline = Pipeline(
        [
            # Aggregate sales by region
            SalesAggregator(name="RegionAggregator"),
            # Create a bar chart
            BarChart(
                title="Sales by Region",
                x_label="Region",
                y_label="Sales",
                horizontal=True,
                name="RegionBarChart",
            ),
            # Render the chart as HTML
            HTMLRenderer(output_file="region_sales_chart.html", name="HTMLOutput"),
        ]
    )

    # Process the data
    result = region_pipeline.run(sales_data)
    print(f"Region sales chart rendered to HTML: {result}")

    # Create a custom component to aggregate sales by product
    class ProductProfitAggregator(Component):
        def process(
            self, input_data: List[Dict[str, Any]]
        ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
            """Aggregate profit by product and region

            Args:
                input_data: List of sales records

            Returns:
                Dictionary with product profit series
            """
            # Group by product and region
            product_data = {}
            for record in input_data:
                product = record["product"]
                region = record["region"]
                profit = record["profit"]

                if product not in product_data:
                    product_data[product] = []

                product_data[product].append({"x": region, "y": profit})

            # Create series data structure
            result = {
                "series": [
                    {"name": product, "data": data}
                    for product, data in product_data.items()
                ]
            }

            return result

    # Create a pipeline for profit by product and region
    product_pipeline = Pipeline(
        [
            # Aggregate profit by product and region
            ProductProfitAggregator(name="ProductAggregator"),
            # Create a line chart
            LineChart(
                title="Profit by Product and Region",
                x_label="Region",
                y_label="Profit",
                name="ProductLineChart",
            ),
            # Render the chart as HTML
            HTMLRenderer(output_file="product_profit_chart.html", name="HTMLOutput"),
        ]
    )

    # Process the data
    result = product_pipeline.run(sales_data)
    print(f"Product profit chart rendered to HTML: {result}")


def main():
    """Run a visualization example pipeline"""
    print("LlamaChain Visualization Example")
    print("================================\n")

    print(
        "This example demonstrates how to use visualization components to create different types of charts."
    )
    print(
        "Charts will be rendered in the terminal and as HTML files in the current directory."
    )

    create_bar_chart()
    create_line_chart()
    create_pie_chart()
    create_advanced_visualization()

    print("\nAll visualizations have been created successfully!")
    print("HTML chart files have been saved to the current directory.")


if __name__ == "__main__":
    main()
