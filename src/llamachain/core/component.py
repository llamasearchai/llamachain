"""
Component implementation for LlamaChain

A Component is the basic building block of the LlamaChain framework.
It represents a unit of processing that takes an input, processes it,
and returns an output.
"""

from typing import Any, Dict, Optional


class Component:
    """Base component class for LlamaChain pipeline elements"""

    def __init__(
        self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ):
        """Initialize a Component

        Args:
            name: Optional name for the component
            config: Optional configuration dictionary
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}

    def process(self, input_data: Any) -> Any:
        """Process the input data

        This method should be overridden by subclasses.

        Args:
            input_data: The data to process

        Returns:
            The processed data
        """
        raise NotImplementedError("Subclasses must implement process method")

    def __call__(self, input_data: Any) -> Any:
        """Make the component callable

        Args:
            input_data: The data to process

        Returns:
            The processed data
        """
        return self.process(input_data)
