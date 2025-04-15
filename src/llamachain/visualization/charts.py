"""
Chart components for visualizing LlamaChain pipeline results.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core import Component


class BaseChart(Component):
    """Base class for chart components"""

    def __init__(
        self,
        title: str = "",
        width: int = 800,
        height: int = 400,
        x_label: str = "",
        y_label: str = "",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the chart component

        Args:
            title: Chart title
            width: Chart width in pixels
            height: Chart height in pixels
            x_label: Label for x-axis
            y_label: Label for y-axis
            name: Optional component name
            config: Optional component configuration
        """
        self.title = title
        self.width = width
        self.height = height
        self.x_label = x_label
        self.y_label = y_label
        self.chart_data = {}
        super().__init__(name, config)

    def _validate_data(self, input_data: Any) -> bool:
        """Validate input data format

        Args:
            input_data: Data to validate

        Returns:
            True if data is valid
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _prepare_chart_data(self, input_data: Any) -> Dict[str, Any]:
        """Transform input data into chart data format

        Args:
            input_data: Data to transform

        Returns:
            Data formatted for charting
        """
        raise NotImplementedError("Subclasses must implement this method")

    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data and generate chart data

        Args:
            input_data: Data to visualize

        Returns:
            Chart configuration and data
        """
        if not self._validate_data(input_data):
            raise ValueError(f"Invalid data format for {self.__class__.__name__}")

        self.chart_data = self._prepare_chart_data(input_data)

        # Add chart metadata
        result = {
            "chart_type": self.__class__.__name__,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "data": self.chart_data,
        }

        return result

    def to_json(self) -> str:
        """Convert chart data to JSON

        Returns:
            JSON representation of chart data
        """
        return json.dumps(self.chart_data, indent=2)


class BarChart(BaseChart):
    """Bar chart component for visualizing categorical data"""

    def __init__(
        self,
        title: str = "Bar Chart",
        width: int = 800,
        height: int = 400,
        x_label: str = "Categories",
        y_label: str = "Values",
        horizontal: bool = False,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the bar chart component

        Args:
            title: Chart title
            width: Chart width in pixels
            height: Chart height in pixels
            x_label: Label for x-axis
            y_label: Label for y-axis
            horizontal: Whether to display bars horizontally
            name: Optional component name
            config: Optional component configuration
        """
        super().__init__(title, width, height, x_label, y_label, name, config)
        self.horizontal = horizontal

    def _validate_data(self, input_data: Any) -> bool:
        """Validate input data for bar chart

        Args:
            input_data: Data to validate

        Returns:
            True if data is valid
        """
        # Check if data is a dict or list of tuples
        if isinstance(input_data, dict):
            return all(isinstance(v, (int, float)) for v in input_data.values())
        elif isinstance(input_data, list):
            if all(isinstance(item, tuple) and len(item) == 2 for item in input_data):
                return all(isinstance(item[1], (int, float)) for item in input_data)
            elif all(
                isinstance(item, dict) and "label" in item and "value" in item
                for item in input_data
            ):
                return all(
                    isinstance(item["value"], (int, float)) for item in input_data
                )

        return False

    def _prepare_chart_data(self, input_data: Any) -> Dict[str, Any]:
        """Transform input data into bar chart format

        Args:
            input_data: Data to transform

        Returns:
            Data formatted for bar chart
        """
        labels = []
        values = []

        if isinstance(input_data, dict):
            labels = list(input_data.keys())
            values = list(input_data.values())
        elif isinstance(input_data, list):
            if all(isinstance(item, tuple) and len(item) == 2 for item in input_data):
                labels = [item[0] for item in input_data]
                values = [item[1] for item in input_data]
            elif all(
                isinstance(item, dict) and "label" in item and "value" in item
                for item in input_data
            ):
                labels = [item["label"] for item in input_data]
                values = [item["value"] for item in input_data]

        return {
            "labels": labels,
            "values": values,
            "horizontal": self.horizontal,
        }


class LineChart(BaseChart):
    """Line chart component for visualizing time series or sequential data"""

    def __init__(
        self,
        title: str = "Line Chart",
        width: int = 800,
        height: int = 400,
        x_label: str = "X Axis",
        y_label: str = "Y Axis",
        show_points: bool = True,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the line chart component

        Args:
            title: Chart title
            width: Chart width in pixels
            height: Chart height in pixels
            x_label: Label for x-axis
            y_label: Label for y-axis
            show_points: Whether to display points on the line
            name: Optional component name
            config: Optional component configuration
        """
        super().__init__(title, width, height, x_label, y_label, name, config)
        self.show_points = show_points

    def _validate_data(self, input_data: Any) -> bool:
        """Validate input data for line chart

        Args:
            input_data: Data to validate

        Returns:
            True if data is valid
        """
        # Handle single series data
        if isinstance(input_data, list):
            # Check if it's a list of points
            if all(isinstance(item, tuple) and len(item) == 2 for item in input_data):
                return all(isinstance(item[1], (int, float)) for item in input_data)
            # Check if it's a list of values (implicit x-axis)
            elif all(isinstance(item, (int, float)) for item in input_data):
                return True
            # Check if it's a list of dicts with x and y keys
            elif all(
                isinstance(item, dict) and "x" in item and "y" in item
                for item in input_data
            ):
                return all(isinstance(item["y"], (int, float)) for item in input_data)

        # Handle multiple series data
        elif isinstance(input_data, dict) and "series" in input_data:
            series_list = input_data["series"]
            if not isinstance(series_list, list):
                return False

            for series in series_list:
                if (
                    not isinstance(series, dict)
                    or "name" not in series
                    or "data" not in series
                ):
                    return False

                data = series["data"]
                if not isinstance(data, list):
                    return False

                # Check data format
                if all(isinstance(item, tuple) and len(item) == 2 for item in data):
                    if not all(isinstance(item[1], (int, float)) for item in data):
                        return False
                elif all(isinstance(item, (int, float)) for item in data):
                    pass  # Valid
                elif all(
                    isinstance(item, dict) and "x" in item and "y" in item
                    for item in data
                ):
                    if not all(isinstance(item["y"], (int, float)) for item in data):
                        return False
                else:
                    return False

            return True

        return False

    def _prepare_chart_data(self, input_data: Any) -> Dict[str, Any]:
        """Transform input data into line chart format

        Args:
            input_data: Data to transform

        Returns:
            Data formatted for line chart
        """
        result = {
            "series": [],
            "show_points": self.show_points,
        }

        # Handle single series data
        if isinstance(input_data, list):
            series = {"name": "Series 1", "data": []}

            # List of tuples (x, y)
            if all(isinstance(item, tuple) and len(item) == 2 for item in input_data):
                series["data"] = [{"x": item[0], "y": item[1]} for item in input_data]
            # List of values (implicit x-axis)
            elif all(isinstance(item, (int, float)) for item in input_data):
                series["data"] = [
                    {"x": i, "y": value} for i, value in enumerate(input_data)
                ]
            # List of dicts with x and y keys
            elif all(
                isinstance(item, dict) and "x" in item and "y" in item
                for item in input_data
            ):
                series["data"] = input_data

            result["series"].append(series)

        # Handle multiple series data
        elif isinstance(input_data, dict) and "series" in input_data:
            for series_data in input_data["series"]:
                series = {"name": series_data["name"], "data": []}
                data = series_data["data"]

                # List of tuples (x, y)
                if all(isinstance(item, tuple) and len(item) == 2 for item in data):
                    series["data"] = [{"x": item[0], "y": item[1]} for item in data]
                # List of values (implicit x-axis)
                elif all(isinstance(item, (int, float)) for item in data):
                    series["data"] = [
                        {"x": i, "y": value} for i, value in enumerate(data)
                    ]
                # List of dicts with x and y keys
                elif all(
                    isinstance(item, dict) and "x" in item and "y" in item
                    for item in data
                ):
                    series["data"] = data

                result["series"].append(series)

        return result


class PieChart(BaseChart):
    """Pie chart component for visualizing proportions"""

    def __init__(
        self,
        title: str = "Pie Chart",
        width: int = 600,
        height: int = 600,
        donut: bool = False,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the pie chart component

        Args:
            title: Chart title
            width: Chart width in pixels
            height: Chart height in pixels
            donut: Whether to display as a donut chart
            name: Optional component name
            config: Optional component configuration
        """
        super().__init__(title, width, height, "", "", name, config)
        self.donut = donut

    def _validate_data(self, input_data: Any) -> bool:
        """Validate input data for pie chart

        Args:
            input_data: Data to validate

        Returns:
            True if data is valid
        """
        # Check if data is a dict
        if isinstance(input_data, dict):
            return all(
                isinstance(v, (int, float)) and v >= 0 for v in input_data.values()
            )
        # Check if data is a list of tuples
        elif isinstance(input_data, list):
            if all(isinstance(item, tuple) and len(item) == 2 for item in input_data):
                return all(
                    isinstance(item[1], (int, float)) and item[1] >= 0
                    for item in input_data
                )
            # Check if data is a list of dicts with label and value keys
            elif all(
                isinstance(item, dict) and "label" in item and "value" in item
                for item in input_data
            ):
                return all(
                    isinstance(item["value"], (int, float)) and item["value"] >= 0
                    for item in input_data
                )

        return False

    def _prepare_chart_data(self, input_data: Any) -> Dict[str, Any]:
        """Transform input data into pie chart format

        Args:
            input_data: Data to transform

        Returns:
            Data formatted for pie chart
        """
        segments = []

        # Transform dict to segments
        if isinstance(input_data, dict):
            segments = [
                {"label": key, "value": value} for key, value in input_data.items()
            ]
        # Transform list of tuples to segments
        elif isinstance(input_data, list):
            if all(isinstance(item, tuple) and len(item) == 2 for item in input_data):
                segments = [{"label": item[0], "value": item[1]} for item in input_data]
            # Use list of dicts directly
            elif all(
                isinstance(item, dict) and "label" in item and "value" in item
                for item in input_data
            ):
                segments = input_data

        # Calculate percentages
        total = sum(segment["value"] for segment in segments)
        if total > 0:
            for segment in segments:
                segment["percentage"] = (segment["value"] / total) * 100

        return {
            "segments": segments,
            "donut": self.donut,
        }
