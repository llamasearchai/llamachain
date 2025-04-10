"""
JSON data extraction components for LlamaChain
"""

from typing import Any, Dict, List, Optional, Union

from llamachain.core import Component


class JSONExtractor(Component):
    """Component for extracting specific fields from JSON data"""
    
    def __init__(
        self,
        fields: List[str],
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize JSONExtractor
        
        Args:
            fields: List of fields to extract, can use dot notation for nested fields
            name: Optional name for the component
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        self.fields = fields
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields from input JSON data
        
        Args:
            input_data: JSON data to extract fields from
            
        Returns:
            Dictionary with extracted fields
        """
        if not isinstance(input_data, dict):
            raise TypeError(f"Expected dictionary, got {type(input_data).__name__}")
        
        result = {}
        
        for field in self.fields:
            # Handle nested fields with dot notation
            if "." in field:
                parts = field.split(".")
                value = input_data
                
                try:
                    for part in parts:
                        if isinstance(value, dict):
                            value = value.get(part)
                        elif isinstance(value, list) and part.isdigit():
                            index = int(part)
                            if 0 <= index < len(value):
                                value = value[index]
                            else:
                                value = None
                        else:
                            value = None
                            break
                except (KeyError, TypeError, IndexError):
                    value = None
                
                result[field] = value
            else:
                # Simple field
                result[field] = input_data.get(field)
        
        return result 