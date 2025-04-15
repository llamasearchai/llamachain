"""
Renderers for visualizing LlamaChain pipeline results.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

from ..core import Component


class BaseRenderer(Component):
    """Base class for visualization renderers"""

    def _validate_chart_data(self, input_data: Dict[str, Any]) -> bool:
        """Validate chart data format

        Args:
            input_data: Chart data to validate

        Returns:
            True if data is valid
        """
        if not isinstance(input_data, dict):
            return False

        required_fields = ["chart_type", "title", "width", "height", "data"]
        return all(field in input_data for field in required_fields)

    def process(self, input_data: Dict[str, Any]) -> Any:
        """Process chart data and generate visualization

        Args:
            input_data: Chart data to visualize

        Returns:
            Visualization result
        """
        if not self._validate_chart_data(input_data):
            raise ValueError("Invalid chart data format")

        return self._render(input_data)

    def _render(self, chart_data: Dict[str, Any]) -> Any:
        """Render the chart

        Args:
            chart_data: Chart data to render

        Returns:
            Rendered visualization
        """
        raise NotImplementedError("Subclasses must implement this method")


class HTMLRenderer(BaseRenderer):
    """HTML renderer for visualizing charts in web browsers"""

    def __init__(
        self,
        output_file: Optional[str] = None,
        template: Optional[str] = None,
        include_scripts: bool = True,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the HTML renderer

        Args:
            output_file: Path to save the HTML output (None for string return)
            template: Custom HTML template to use (None for default)
            include_scripts: Whether to include Chart.js scripts in the output
            name: Optional component name
            config: Optional component configuration
        """
        self.output_file = output_file
        self.template = template
        self.include_scripts = include_scripts
        super().__init__(name, config)

    def _get_default_template(self) -> str:
        """Get the default HTML template

        Returns:
            Default HTML template
        """
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{title}}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    {{scripts}}
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .chart-container {
            margin-bottom: 30px;
        }
    </style>
</head>
<body>
    <div class="chart-container" style="width: {{width}}px; height: {{height}}px;">
        <canvas id="chart"></canvas>
    </div>
    
    <script>
        {{chart_script}}
    </script>
</body>
</html>"""

    def _get_chart_js_scripts(self) -> str:
        """Get Chart.js library script tags

        Returns:
            HTML script tags for Chart.js
        """
        if self.include_scripts:
            return '<script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>'
        else:
            return ""

    def _generate_chart_script(self, chart_data: Dict[str, Any]) -> str:
        """Generate JavaScript to create the chart

        Args:
            chart_data: Chart data to visualize

        Returns:
            JavaScript code to create the chart
        """
        chart_type = chart_data["chart_type"]
        chart_config = {
            "type": chart_type.lower().replace("chart", ""),
            "data": {},
            "options": {
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": bool(chart_data["title"]),
                        "text": chart_data["title"],
                    },
                    "legend": {"display": True},
                },
            },
        }

        # Add axis labels if provided
        if chart_data.get("x_label") or chart_data.get("y_label"):
            chart_config["options"]["scales"] = {
                "x": {
                    "display": True,
                    "title": {
                        "display": bool(chart_data.get("x_label")),
                        "text": chart_data.get("x_label", ""),
                    },
                },
                "y": {
                    "display": True,
                    "title": {
                        "display": bool(chart_data.get("y_label")),
                        "text": chart_data.get("y_label", ""),
                    },
                },
            }

        # Configure chart data based on chart type
        data = chart_data["data"]

        if chart_type == "BarChart":
            # Set horizontal bar chart if specified
            if data.get("horizontal", False):
                chart_config["type"] = "horizontalBar"

            chart_config["data"] = {
                "labels": data["labels"],
                "datasets": [
                    {
                        "label": chart_data["title"],
                        "data": data["values"],
                        "backgroundColor": "rgba(54, 162, 235, 0.5)",
                        "borderColor": "rgba(54, 162, 235, 1)",
                        "borderWidth": 1,
                    }
                ],
            }

        elif chart_type == "LineChart":
            datasets = []
            for i, series in enumerate(data["series"]):
                # Use predefined colors
                colors = [
                    "rgba(54, 162, 235, 1)",  # Blue
                    "rgba(255, 99, 132, 1)",  # Red
                    "rgba(75, 192, 192, 1)",  # Green
                    "rgba(255, 159, 64, 1)",  # Orange
                    "rgba(153, 102, 255, 1)",  # Purple
                ]
                color = colors[i % len(colors)]

                datasets.append(
                    {
                        "label": series["name"],
                        "data": series["data"],
                        "fill": False,
                        "tension": 0.1,
                        "borderColor": color,
                        "backgroundColor": color,
                        "pointRadius": 4 if data.get("show_points", True) else 0,
                    }
                )

            chart_config["data"] = {"datasets": datasets}

        elif chart_type == "PieChart":
            # Configure as doughnut chart if specified
            if data.get("donut", False):
                chart_config["type"] = "doughnut"

            labels = [segment["label"] for segment in data["segments"]]
            values = [segment["value"] for segment in data["segments"]]

            # Use predefined colors
            colors = [
                "rgba(54, 162, 235, 0.7)",  # Blue
                "rgba(255, 99, 132, 0.7)",  # Red
                "rgba(75, 192, 192, 0.7)",  # Green
                "rgba(255, 159, 64, 0.7)",  # Orange
                "rgba(153, 102, 255, 0.7)",  # Purple
                "rgba(255, 205, 86, 0.7)",  # Yellow
                "rgba(201, 203, 207, 0.7)",  # Grey
            ]
            # Repeat colors if needed
            while len(colors) < len(labels):
                colors.extend(colors)

            chart_config["data"] = {
                "labels": labels,
                "datasets": [
                    {
                        "data": values,
                        "backgroundColor": colors[: len(labels)],
                        "borderWidth": 1,
                    }
                ],
            }

            # Add tooltips to show percentages
            chart_config["options"]["plugins"]["tooltip"] = {
                "callbacks": {
                    "label": "function(context) { return context.label + ': ' + context.formattedValue + ' (' + context.parsed + '%)'; }"
                }
            }

        # Convert to JavaScript code
        js_chart_config = json.dumps(chart_config, indent=2)
        js_chart_config = js_chart_config.replace('"function(', "function(")
        js_chart_config = js_chart_config.replace(');"', ");")

        script = f"const ctx = document.getElementById('chart').getContext('2d');\n"
        script += f"const chart = new Chart(ctx, {js_chart_config});\n"

        return script

    def _render(self, chart_data: Dict[str, Any]) -> str:
        """Render the chart as HTML

        Args:
            chart_data: Chart data to render

        Returns:
            HTML string or file path if output_file is specified
        """
        template = self.template or self._get_default_template()
        scripts = self._get_chart_js_scripts()
        chart_script = self._generate_chart_script(chart_data)

        # Replace template placeholders
        html = template.replace("{{title}}", chart_data["title"])
        html = html.replace("{{width}}", str(chart_data["width"]))
        html = html.replace("{{height}}", str(chart_data["height"]))
        html = html.replace("{{scripts}}", scripts)
        html = html.replace("{{chart_script}}", chart_script)

        # Save to file or return as string
        if self.output_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(html)
            return self.output_file
        else:
            return html


class TerminalRenderer(BaseRenderer):
    """Terminal renderer for visualizing charts in text mode"""

    def __init__(
        self,
        width: int = 80,
        height: int = 20,
        character: str = "█",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the terminal renderer

        Args:
            width: Width of the chart in characters
            height: Height of the chart in characters
            character: Character to use for plotting
            name: Optional component name
            config: Optional component configuration
        """
        self.width = width
        self.height = height
        self.character = character
        super().__init__(name, config)

    def _render_bar_chart(self, chart_data: Dict[str, Any]) -> str:
        """Render a bar chart in ASCII

        Args:
            chart_data: Bar chart data

        Returns:
            ASCII representation of the bar chart
        """
        data = chart_data["data"]
        labels = data["labels"]
        values = data["values"]
        horizontal = data.get("horizontal", False)

        # Find the maximum value for scaling
        max_value = max(values) if values else 0

        result = [chart_data["title"]]
        result.append("=" * len(chart_data["title"]))
        result.append("")

        if horizontal:
            # Calculate the maximum label length
            label_width = max(len(str(label)) for label in labels) if labels else 0
            # Calculate the maximum bar width
            max_bar_width = self.width - label_width - 4

            for i, (label, value) in enumerate(zip(labels, values)):
                # Scale the bar width
                bar_width = (
                    int((value / max_value) * max_bar_width) if max_value > 0 else 0
                )
                # Format the label and value
                label_str = str(label).ljust(label_width)
                # Draw the bar
                bar = self.character * bar_width
                # Format the line
                result.append(f"{label_str} | {bar} {value}")
        else:
            # Calculate column width based on available width and number of bars
            col_width = self.width // len(values) if values else self.width
            # Calculate the maximum bar height
            max_bar_height = self.height - 3  # Reserve space for labels and values

            # Create a matrix to represent the chart
            chart_matrix = [
                [" " for _ in range(self.width)] for _ in range(self.height)
            ]

            for i, (label, value) in enumerate(zip(labels, values)):
                # Scale the bar height
                bar_height = (
                    int((value / max_value) * max_bar_height) if max_value > 0 else 0
                )
                # Calculate column start position
                col_start = i * col_width
                # Draw the bar from bottom to top
                for h in range(bar_height):
                    row = self.height - 3 - h
                    for c in range(col_width - 2):
                        if 0 <= row < self.height and 0 <= col_start + c < self.width:
                            chart_matrix[row][col_start + c] = self.character

                # Add value at the top of the bar
                value_str = str(value)
                value_pos = col_start + (col_width - len(value_str)) // 2
                for c, char in enumerate(value_str):
                    if 0 <= value_pos + c < self.width:
                        chart_matrix[self.height - 3 - bar_height - 1][
                            value_pos + c
                        ] = char

                # Add label at the bottom
                label_str = str(label)
                if len(label_str) > col_width - 2:
                    label_str = label_str[: col_width - 5] + "..."
                label_pos = col_start + (col_width - len(label_str)) // 2
                for c, char in enumerate(label_str):
                    if (
                        0 <= label_pos + c < self.width
                        and self.height - 2 < self.height
                    ):
                        chart_matrix[self.height - 2][label_pos + c] = char

            # Convert matrix to strings
            for row in chart_matrix:
                result.append("".join(row))

        return "\n".join(result)

    def _render_pie_chart(self, chart_data: Dict[str, Any]) -> str:
        """Render a pie chart in ASCII

        Args:
            chart_data: Pie chart data

        Returns:
            ASCII representation of the pie chart
        """
        data = chart_data["data"]
        segments = data["segments"]

        result = [chart_data["title"]]
        result.append("=" * len(chart_data["title"]))
        result.append("")

        # Calculate the total value
        total = sum(segment["value"] for segment in segments)

        # Create a table showing segments with percentages
        max_label_width = (
            max(len(segment["label"]) for segment in segments) if segments else 0
        )
        for segment in segments:
            label = segment["label"].ljust(max_label_width)
            value = segment["value"]
            percentage = (value / total * 100) if total > 0 else 0

            # Draw a bar representing the percentage
            bar_width = int((self.width - max_label_width - 20) * percentage / 100)
            bar = self.character * bar_width

            result.append(f"{label} | {bar} {value} ({percentage:.1f}%)")

        return "\n".join(result)

    def _render_line_chart(self, chart_data: Dict[str, Any]) -> str:
        """Render a line chart in ASCII

        Args:
            chart_data: Line chart data

        Returns:
            ASCII representation of the line chart
        """
        data = chart_data["data"]
        series_list = data["series"]

        result = [chart_data["title"]]
        result.append("=" * len(chart_data["title"]))
        result.append("")

        # Find the min and max values across all series
        all_x = []
        all_y = []
        for series in series_list:
            for point in series["data"]:
                all_x.append(point["x"])
                all_y.append(point["y"])

        min_x = min(all_x) if all_x else 0
        max_x = max(all_x) if all_x else 0
        min_y = min(all_y) if all_y else 0
        max_y = max(all_y) if all_y else 0

        # Create the chart grid
        chart_width = self.width - 10  # Reserve space for y-axis labels
        chart_height = self.height - 5  # Reserve space for x-axis labels and title

        # Create a matrix to represent the chart
        chart_matrix = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Draw the axes
        for y in range(chart_height):
            chart_matrix[y][9] = "│"
        for x in range(chart_width):
            chart_matrix[chart_height - 1][x + 10] = "─"
        chart_matrix[chart_height - 1][9] = "└"

        # Draw y-axis labels
        if max_y > min_y:
            for i in range(5):
                value = max_y - i * (max_y - min_y) / 4
                label = f"{value:.1f}"
                y_pos = i * (chart_height - 1) // 4
                for c, char in enumerate(label):
                    if c < 8:
                        chart_matrix[y_pos][c] = char

        # Draw x-axis labels
        if max_x > min_x:
            for i in range(5):
                value = min_x + i * (max_x - min_x) / 4
                label = f"{value:.1f}"
                x_pos = 10 + i * (chart_width - 1) // 4
                for c, char in enumerate(label):
                    if 0 <= x_pos + c < self.width and chart_height < self.height:
                        chart_matrix[chart_height][x_pos + c - len(label) // 2] = char

        # Draw each series
        series_symbols = ["*", "+", "o", "x", "#"]
        for i, series in enumerate(series_list):
            symbol = series_symbols[i % len(series_symbols)]
            points = []

            # Transform data points to chart coordinates
            for point in series["data"]:
                x = point["x"]
                y = point["y"]

                # Scale to chart coordinates
                x_chart = (
                    10 + int((x - min_x) / (max_x - min_x) * (chart_width - 1))
                    if max_x > min_x
                    else 10
                )
                y_chart = (
                    chart_height
                    - 1
                    - int((y - min_y) / (max_y - min_y) * (chart_height - 1))
                    if max_y > min_y
                    else chart_height - 1
                )

                if 0 <= x_chart < self.width and 0 <= y_chart < self.height:
                    points.append((x_chart, y_chart))

            # Draw points and lines
            for j, (x, y) in enumerate(points):
                chart_matrix[y][x] = symbol

                # Draw lines between points
                if j > 0 and data.get("show_points", True):
                    prev_x, prev_y = points[j - 1]
                    # Simple line drawing algorithm (Bresenham's)
                    if prev_x != x or prev_y != y:
                        dx = abs(x - prev_x)
                        dy = abs(y - prev_y)
                        sx = 1 if prev_x < x else -1
                        sy = 1 if prev_y < y else -1
                        err = dx - dy

                        cx, cy = prev_x, prev_y
                        while cx != x or cy != y:
                            e2 = 2 * err
                            if e2 > -dy:
                                err -= dy
                                cx += sx
                            if e2 < dx:
                                err += dx
                                cy += sy

                            if (
                                0 <= cx < self.width
                                and 0 <= cy < self.height
                                and chart_matrix[cy][cx] == " "
                            ):
                                chart_matrix[cy][cx] = "·"

        # Add series legend
        for i, series in enumerate(series_list):
            legend = f"{series_symbols[i % len(series_symbols)]} {series['name']}"
            y_pos = chart_height + 2 + i
            if y_pos < self.height:
                for c, char in enumerate(legend):
                    if c < self.width:
                        chart_matrix[y_pos][c] = char

        # Convert matrix to strings
        for row in chart_matrix:
            result.append("".join(row))

        return "\n".join(result)

    def _render(self, chart_data: Dict[str, Any]) -> str:
        """Render the chart in text mode

        Args:
            chart_data: Chart data to render

        Returns:
            ASCII representation of the chart
        """
        chart_type = chart_data["chart_type"]

        if chart_type == "BarChart":
            return self._render_bar_chart(chart_data)
        elif chart_type == "PieChart":
            return self._render_pie_chart(chart_data)
        elif chart_type == "LineChart":
            return self._render_line_chart(chart_data)
        else:
            return f"Chart type {chart_type} not supported by TerminalRenderer"
