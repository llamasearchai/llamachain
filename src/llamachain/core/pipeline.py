"""
Pipeline implementation for LlamaChain

A Pipeline is a sequence of Components that are executed in order,
with the output of each Component being passed as input to the next.
"""

from typing import Any, List, Optional, Union

from .component import Component


class Pipeline:
    """Pipeline for chaining components together"""
    
    def __init__(self, components: List[Component], name: Optional[str] = None):
        """Initialize a Pipeline
        
        Args:
            components: List of components to execute in sequence
            name: Optional name for the pipeline
        """
        self.components = components
        self.name = name or "Pipeline"
        
        # Validate that all items in the list are Components
        for component in components:
            if not isinstance(component, Component):
                raise TypeError(f"Expected Component, got {type(component).__name__}")
    
    def run(self, input_data: Any) -> Any:
        """Run the pipeline
        
        Args:
            input_data: The initial input data
            
        Returns:
            The final output after passing through all components
        """
        result = input_data
        
        for component in self.components:
            result = component.process(result)
            
        return result
    
    def add(self, component: Component) -> "Pipeline":
        """Add a component to the pipeline
        
        Args:
            component: The component to add
            
        Returns:
            The pipeline instance for chaining
        """
        if not isinstance(component, Component):
            raise TypeError(f"Expected Component, got {type(component).__name__}")
            
        self.components.append(component)
        return self
    
    def __call__(self, input_data: Any) -> Any:
        """Make the pipeline callable
        
        Args:
            input_data: The initial input data
            
        Returns:
            The final output after passing through all components
        """
        return self.run(input_data) 