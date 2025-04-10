"""
API request components for LlamaChain
"""

import json
import requests
from typing import Any, Dict, List, Optional, Union

from llamachain.core import Component


class RESTEndpoint:
    """REST API endpoint configuration"""
    
    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ):
        """Initialize a REST endpoint
        
        Args:
            url: The base URL for the endpoint
            headers: Optional headers to include in requests
            params: Optional query parameters to include in requests
            timeout: Request timeout in seconds
        """
        self.url = url
        self.headers = headers or {}
        self.params = params or {}
        self.timeout = timeout


class APIRequest(Component):
    """Component for making API requests"""
    
    def __init__(
        self,
        endpoint: RESTEndpoint,
        method: str = "GET",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize APIRequest
        
        Args:
            endpoint: The REST endpoint configuration
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            name: Optional name for the component
            config: Optional configuration dictionary
        """
        super().__init__(name, config)
        self.endpoint = endpoint
        self.method = method.upper()
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Make an API request
        
        Args:
            input_data: The input data for the request
                - If a dictionary, used as request body for POST/PUT
                - If a string, used as path suffix for the URL
                - If a list or tuple, first element is path, second is body
                - If None, makes request with no body
                
        Returns:
            The JSON response as a dictionary
        """
        url = self.endpoint.url
        headers = self.endpoint.headers.copy()
        params = self.endpoint.params.copy()
        data = None
        json_data = None
        
        # Process input data based on type
        if isinstance(input_data, dict):
            # Use as JSON body
            json_data = input_data
        elif isinstance(input_data, str):
            # Use as URL path
            url = url.rstrip('/') + '/' + input_data.lstrip('/')
        elif isinstance(input_data, (list, tuple)) and len(input_data) >= 2:
            # First element is path, second is body
            url = url.rstrip('/') + '/' + str(input_data[0]).lstrip('/')
            json_data = input_data[1]
        
        # Make the request
        try:
            response = requests.request(
                method=self.method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                data=data,
                timeout=self.endpoint.timeout
            )
            
            # Raise for HTTP errors
            response.raise_for_status()
            
            # Parse JSON response
            if response.text:
                return response.json()
            else:
                return {}
                
        except requests.exceptions.RequestException as e:
            error_data = {
                "error": str(e),
                "status_code": getattr(e.response, "status_code", None) if hasattr(e, "response") else None,
                "url": url
            }
            
            if hasattr(e, "response") and hasattr(e.response, "text"):
                try:
                    error_data["response"] = json.loads(e.response.text)
                except:
                    error_data["response"] = e.response.text
            
            return error_data 