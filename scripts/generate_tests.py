#!/usr/bin/env python3
"""
Script to generate unit tests for LlamaChain components.
This will analyze the codebase and create test files for components that don't have tests yet.

Usage:
    python generate_tests.py
"""

import os
import re
import inspect
import importlib
import pkgutil
from pathlib import Path
from typing import List, Dict, Set, Any, Type

# Root path of the repository
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_PATH, "src")
TESTS_PATH = os.path.join(REPO_PATH, "tests")

# Add the src directory to Python path
import sys
sys.path.insert(0, SRC_PATH)

# Template for test files
TEST_FILE_TEMPLATE = '''"""
Tests for {module_name} module
"""

import pytest
{imports}

{test_classes}
'''

# Template for test class
TEST_CLASS_TEMPLATE = '''
class Test{class_name}:
    """Tests for {class_name}"""
    
    def test_initialization(self):
        """Test that {class_name} initializes correctly"""
        {initialization_test}
    
    def test_process(self):
        """Test the process method"""
        {process_test}
    
    {additional_tests}
'''

def find_component_classes() -> Dict[str, List[Type]]:
    """Find all Component subclasses in the codebase"""
    components_by_module = {}
    
    try:
        # Import the Component base class
        from llamachain.core import Component
        
        # Walk through the llamachain package
        llamachain_pkg = importlib.import_module("llamachain")
        for _, module_name, is_pkg in pkgutil.walk_packages(llamachain_pkg.__path__, llamachain_pkg.__name__ + '.'):
            if is_pkg:
                continue
                
            try:
                module = importlib.import_module(module_name)
                
                # Look for Component subclasses in the module
                components = []
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, Component) and 
                        obj is not Component and
                        obj.__module__ == module_name):
                        components.append(obj)
                
                if components:
                    components_by_module[module_name] = components
            except (ImportError, AttributeError) as e:
                print(f"Error importing {module_name}: {e}")
    
    except ImportError:
        print("Could not import Component class. Make sure the repository is properly installed.")
    
    return components_by_module

def get_existing_test_files() -> Set[str]:
    """Get existing test files"""
    test_files = set()
    for root, _, files in os.walk(TESTS_PATH):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                test_files.add(file)
    return test_files

def module_needs_test(module_name: str, existing_test_files: Set[str]) -> bool:
    """Check if a module needs test files"""
    module_base = module_name.split('.')[-1]
    test_file = f"test_{module_base}.py"
    return test_file not in existing_test_files

def generate_imports(module_name: str, component_classes: List[Type]) -> str:
    """Generate import statements for the test file"""
    imports = [f"from {module_name} import {cls.__name__}" for cls in component_classes]
    additional_imports = []
    
    # Add additional imports if needed
    for cls in component_classes:
        # Check if component likely handles JSON data
        if "json" in cls.__name__.lower() or "json" in module_name.lower():
            additional_imports.append("import json")
        
        # Check if component likely handles API requests
        if "api" in cls.__name__.lower() or "api" in module_name.lower():
            additional_imports.append("import requests")
            additional_imports.append("from unittest.mock import patch, MagicMock")
    
    additional_imports = list(set(additional_imports))  # Remove duplicates
    return "\n".join(imports + additional_imports)

def generate_initialization_test(cls: Type) -> str:
    """Generate initialization test for a component"""
    params = inspect.signature(cls.__init__).parameters
    
    # Skip self and *args, **kwargs
    init_args = []
    for name, param in params.items():
        if name == 'self':
            continue
        
        if name == 'name':
            init_args.append(f'{name}="test_{cls.__name__}"')
        elif name == 'config':
            init_args.append(f'{name}{{"test_key": "test_value"}}')
        elif param.default is not inspect.Parameter.empty:
            continue  # Skip params with default values for simple test
        elif "list" in str(param.annotation).lower():
            init_args.append(f'{name}=[]')
        elif "dict" in str(param.annotation).lower():
            init_args.append(f'{name}={{}}')
        elif "str" in str(param.annotation).lower():
            init_args.append(f'{name}="test"')
        elif "int" in str(param.annotation).lower():
            init_args.append(f'{name}=1')
        elif "float" in str(param.annotation).lower():
            init_args.append(f'{name}=1.0')
        elif "bool" in str(param.annotation).lower():
            init_args.append(f'{name}=True')
    
    init_args_str = ", ".join(init_args)
    
    return f"""
        component = {cls.__name__}({init_args_str})
        assert component is not None
        assert component.name == "test_{cls.__name__}" if "name" in {init_args_str} else component.name == "{cls.__name__}"
        assert isinstance(component.config, dict)
        """

def generate_process_test(cls: Type) -> str:
    """Generate process test for a component"""
    # Try to determine the input and output types based on method signature
    try:
        signature = inspect.signature(cls.process)
        input_param = list(signature.parameters.values())[1]  # Skip self
        return_annotation = signature.return_annotation
        
        input_type = "Any"
        if input_param.annotation != inspect.Parameter.empty:
            input_type = input_param.annotation.__name__ if hasattr(input_param.annotation, "__name__") else str(input_param.annotation)
        
        output_type = "Any"
        if return_annotation != inspect.Parameter.empty:
            output_type = return_annotation.__name__ if hasattr(return_annotation, "__name__") else str(return_annotation)
        
        # Look for clues in the class name and module
        cls_name_lower = cls.__name__.lower()
        input_data = None
        
        if "text" in cls_name_lower or "token" in cls_name_lower:
            input_data = '"This is a test string for text processing."'
        elif "json" in cls_name_lower:
            input_data = '{"key1": "value1", "key2": 123, "nested": {"inner": "value"}}'
        elif "api" in cls_name_lower or "request" in cls_name_lower:
            return f"""
        # Mock the requests.request method
        with patch("requests.request") as mock_request:
            # Configure the mock to return a response with JSON data
            mock_response = MagicMock()
            mock_response.json.return_value = {{"result": "success", "data": "test"}}
            mock_response.text = '{{"result": "success", "data": "test"}}'
            mock_response.status_code = 200
            mock_request.return_value = mock_response
            
            # Initialize component with mocked endpoint
            from llamachain.api import RESTEndpoint
            endpoint = RESTEndpoint(url="https://test.api")
            component = {cls.__name__}(endpoint)
            
            # Test with different input types
            result1 = component.process({{"param": "value"}})
            assert result1 is not None
            
            result2 = component.process("test/endpoint")
            assert result2 is not None
        """
        elif "model" in cls_name_lower or "inference" in cls_name_lower:
            return f"""
        # This is a placeholder test - real implementation would mock ML model
        component = {cls.__name__}(model=MagicMock())
        
        # The mock model's predict method should return a dictionary
        component.model.predict.return_value = [{{"label": "test", "score": 0.95}}]
        
        result = component.process("Test text for model inference")
        assert result is not None
        """
        elif "dict" in input_type.lower():
            input_data = '{"key": "value"}'
        elif "list" in input_type.lower():
            input_data = '["item1", "item2"]'
        elif "str" in input_type.lower():
            input_data = '"test input"'
        else:
            input_data = '"default test input"'
        
        if input_data:
            return f"""
        component = {cls.__name__}()
        result = component.process({input_data})
        assert result is not None
        """
    
    except (AttributeError, IndexError):
        # Default fallback test
        return f"""
        component = {cls.__name__}()
        # This is a placeholder test - implement with appropriate input
        with pytest.raises(NotImplementedError):
            component.process("test input")
        """

def generate_additional_tests(cls: Type) -> str:
    """Generate additional tests based on component type"""
    cls_name_lower = cls.__name__.lower()
    additional_tests = []
    
    # Text processing components
    if "text" in cls_name_lower or "token" in cls_name_lower or "word" in cls_name_lower:
        additional_tests.append("""
    def test_with_empty_input(self):
        """Test behavior with empty input"""
        component = {class_name}()
        result = component.process("")
        assert result is not None
        
    def test_with_nonstring_input(self):
        """Test behavior with non-string input"""
        component = {class_name}()
        with pytest.raises(TypeError):
            component.process(123)
        """.format(class_name=cls.__name__))
    
    # JSON components
    elif "json" in cls_name_lower or "extract" in cls_name_lower:
        additional_tests.append("""
    def test_with_nested_fields(self):
        """Test extracting nested fields"""
        component = {class_name}(fields=["nested.field", "simple"])
        result = component.process({{"simple": "value", "nested": {{"field": "nested_value"}}}})
        assert result["nested.field"] == "nested_value"
        assert result["simple"] == "value"
        """.format(class_name=cls.__name__))
    
    # API components
    elif "api" in cls_name_lower or "request" in cls_name_lower:
        additional_tests.append("""
    def test_error_handling(self):
        """Test handling of API errors"""
        with patch("requests.request") as mock_request:
            # Configure the mock to raise an exception
            mock_request.side_effect = requests.exceptions.RequestException("Test error")
            
            # Initialize component with mocked endpoint
            from llamachain.api import RESTEndpoint
            endpoint = RESTEndpoint(url="https://test.api")
            component = {class_name}(endpoint)
            
            # Test error handling
            result = component.process({{}})
            assert "error" in result
            assert "Test error" in result["error"]
        """.format(class_name=cls.__name__))
    
    # Sentiment analysis components
    elif "sentiment" in cls_name_lower:
        additional_tests.append("""
    def test_positive_sentiment(self):
        """Test detection of positive sentiment"""
        component = {class_name}()
        result = component.process("This is great! I love it.")
        assert result["sentiment"] == "positive"
        assert result["score"] > 0
        
    def test_negative_sentiment(self):
        """Test detection of negative sentiment"""
        component = {class_name}()
        result = component.process("This is terrible! I hate it.")
        assert result["sentiment"] == "negative"
        assert result["score"] < 0
        """.format(class_name=cls.__name__))
    
    # Default - no additional tests
    return "\n".join(additional_tests)

def generate_test_class(cls: Type) -> str:
    """Generate test class for a component"""
    initialization_test = generate_initialization_test(cls)
    process_test = generate_process_test(cls)
    additional_tests = generate_additional_tests(cls)
    
    return TEST_CLASS_TEMPLATE.format(
        class_name=cls.__name__,
        initialization_test=initialization_test,
        process_test=process_test,
        additional_tests=additional_tests
    )

def generate_test_file(module_name: str, component_classes: List[Type]) -> str:
    """Generate test file for a module"""
    imports = generate_imports(module_name, component_classes)
    test_classes = "\n".join(generate_test_class(cls) for cls in component_classes)
    
    return TEST_FILE_TEMPLATE.format(
        module_name=module_name,
        imports=imports,
        test_classes=test_classes
    )

def save_test_file(module_name: str, content: str) -> str:
    """Save test file and return the file path"""
    module_base = module_name.split('.')[-1]
    test_file_name = f"test_{module_base}.py"
    test_file_path = os.path.join(TESTS_PATH, test_file_name)
    
    # Ensure tests directory exists
    os.makedirs(TESTS_PATH, exist_ok=True)
    
    with open(test_file_path, "w") as f:
        f.write(content)
    
    return test_file_path

def main():
    """Main function"""
    print("Generating tests for LlamaChain components...")
    
    # Find component classes
    component_classes = find_component_classes()
    if not component_classes:
        print("No component classes found. Make sure the code is properly installed.")
        return
    
    print(f"Found {sum(len(classes) for classes in component_classes.values())} component classes in {len(component_classes)} modules")
    
    # Get existing test files
    existing_test_files = get_existing_test_files()
    print(f"Found {len(existing_test_files)} existing test files")
    
    # Generate and save test files
    generated_files = []
    for module_name, classes in component_classes.items():
        if module_needs_test(module_name, existing_test_files):
            print(f"Generating tests for {module_name} ({len(classes)} components)")
            
            test_content = generate_test_file(module_name, classes)
            test_file = save_test_file(module_name, test_content)
            
            generated_files.append(test_file)
            print(f"Generated test file: {test_file}")
    
    print(f"\nGenerated {len(generated_files)} new test files")
    if not generated_files:
        print("No new test files needed - all components seem to be covered")

if __name__ == "__main__":
    main() 