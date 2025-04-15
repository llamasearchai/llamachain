"""
Tests for LlamaChain core functionality
"""

import pytest

from llamachain.core import Component, Pipeline


class TestComponent(Component):
    """Test component that doubles its input"""

    def process(self, input_data):
        return input_data * 2


class TestComponent2(Component):
    """Test component that adds 10 to its input"""

    def process(self, input_data):
        return input_data + 10


def test_component_initialization():
    """Test that components initialize properly"""
    # Test with default name
    component = TestComponent()
    assert component.name == "TestComponent"
    assert component.config == {}

    # Test with custom name and config
    component = TestComponent(name="CustomName", config={"key": "value"})
    assert component.name == "CustomName"
    assert component.config == {"key": "value"}


def test_component_process():
    """Test component processing"""
    component = TestComponent()
    result = component.process(5)
    assert result == 10

    # Test callable interface
    result = component(5)
    assert result == 10


def test_pipeline_initialization():
    """Test that pipelines initialize properly"""
    component1 = TestComponent()
    component2 = TestComponent2()

    # Test with list of components
    pipeline = Pipeline([component1, component2])
    assert pipeline.name == "Pipeline"
    assert len(pipeline.components) == 2
    assert pipeline.components[0] == component1
    assert pipeline.components[1] == component2

    # Test with custom name
    pipeline = Pipeline([component1, component2], name="CustomPipeline")
    assert pipeline.name == "CustomPipeline"

    # Test with non-component
    with pytest.raises(TypeError):
        Pipeline([component1, "not a component"])


def test_pipeline_run():
    """Test pipeline execution"""
    component1 = TestComponent()
    component2 = TestComponent2()

    pipeline = Pipeline([component1, component2])

    # Input goes through component1 (doubles to 10) then component2 (adds 10)
    result = pipeline.run(5)
    assert result == 20

    # Test callable interface
    result = pipeline(5)
    assert result == 20


def test_pipeline_add():
    """Test adding components to a pipeline"""
    component1 = TestComponent()
    pipeline = Pipeline([component1])

    assert len(pipeline.components) == 1

    # Add a new component
    component2 = TestComponent2()
    pipeline.add(component2)

    assert len(pipeline.components) == 2
    assert pipeline.components[1] == component2

    # Test method chaining
    component3 = TestComponent()
    result = pipeline.add(component3)

    assert result == pipeline
    assert len(pipeline.components) == 3
    assert pipeline.components[2] == component3

    # Test with non-component
    with pytest.raises(TypeError):
        pipeline.add("not a component")
