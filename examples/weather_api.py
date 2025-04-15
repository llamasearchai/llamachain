#!/usr/bin/env python
"""
Weather API integration example with LlamaChain
"""

from llamachain.api import APIRequest, JSONExtractor, RESTEndpoint
from llamachain.core import Component, Pipeline


class WeatherFormatter(Component):
    """Custom component to format weather data"""

    def process(self, input_data):
        """Format weather data for display

        Args:
            input_data: Dictionary with extracted weather data

        Returns:
            Formatted weather data dictionary
        """
        # Convert temperature from Kelvin to Celsius
        if "main.temp" in input_data:
            temp_k = input_data["main.temp"]
            temp_c = temp_k - 273.15
            input_data["main.temp"] = f"{temp_c:.1f}°C"

        # Format weather description
        if "weather.0.description" in input_data:
            input_data["weather.0.description"] = input_data[
                "weather.0.description"
            ].capitalize()

        # Create a cleaned up result
        return {
            "location": input_data.get("name", "Unknown"),
            "description": input_data.get("weather.0.description", "Unknown"),
            "temperature": input_data.get("main.temp", "Unknown"),
            "humidity": f"{input_data.get('main.humidity', 'Unknown')}%",
            "wind_speed": f"{input_data.get('wind.speed', 'Unknown')} m/s",
        }


def main():
    """Run a weather API integration pipeline example"""
    print("LlamaChain Weather API Example")
    print("==============================\n")

    # Create endpoint configuration
    # Note: In a real application, you would need to provide your API key
    endpoint = RESTEndpoint(
        url="https://api.openweathermap.org/data/2.5/weather",
        params={"appid": "YOUR_API_KEY_HERE"},  # Replace with your API key
        headers={"Content-Type": "application/json"},
        timeout=5.0,
    )

    # Create request component
    request = APIRequest(endpoint, method="GET")

    # Create extractor component
    extractor = JSONExtractor(
        fields=[
            "name",
            "weather.0.description",
            "main.temp",
            "main.humidity",
            "wind.speed",
        ]
    )

    # Create formatter component
    formatter = WeatherFormatter()

    # Create pipeline
    pipeline = Pipeline(
        [
            request,  # Make API request
            extractor,  # Extract specific fields
            formatter,  # Format the data
        ]
    )

    # For example purposes, we'll use dummy data instead of making a real API call
    print("Using dummy data for demonstration purposes.\n")

    # Sample locations
    locations = [
        {"q": "London,uk"},
        {"q": "New York,us"},
        {"q": "Tokyo,jp"},
    ]

    # Sample responses (dummy data)
    dummy_responses = [
        {
            "name": "London",
            "weather": [{"description": "clear sky"}],
            "main": {"temp": 293.15, "humidity": 70},
            "wind": {"speed": 3.5},
        },
        {
            "name": "New York",
            "weather": [{"description": "light rain"}],
            "main": {"temp": 283.25, "humidity": 85},
            "wind": {"speed": 4.8},
        },
        {
            "name": "Tokyo",
            "weather": [{"description": "scattered clouds"}],
            "main": {"temp": 301.05, "humidity": 62},
            "wind": {"speed": 2.1},
        },
    ]

    # Process each location
    for i, (location, dummy_response) in enumerate(zip(locations, dummy_responses)):
        print(f"Location: {location['q']}")

        # In a real example, we would pass the location to the pipeline
        # and the API request would be made with those parameters.
        # result = pipeline.run(location)

        # Instead, we'll pass the dummy response directly to the extractor
        # Normally the request component would run first, but we're bypassing it
        result = Pipeline([extractor, formatter]).run(dummy_response)

        print(f"Weather: {result['description']}")
        print(f"Temperature: {result['temperature']}")
        print(f"Humidity: {result['humidity']}")
        print(f"Wind Speed: {result['wind_speed']}")
        print()


if __name__ == "__main__":
    main()
