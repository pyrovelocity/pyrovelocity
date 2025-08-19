"""
Tests for PyroVelocity parameter metadata system.

This module contains tests for the parameter metadata components, including:

- test_poisson_metadata_creation: Test Poisson metadata component creation
- test_poisson_metadata_registry: Test Poisson metadata in registry
- test_parameter_inference: Test automatic parameter inference from parameter sets
- test_dynamic_parameter_detection: Test dynamic parameter detection in plotting functions
"""

import pytest

from pyrovelocity.models.metadata import (
    get_parameter_metadata,
    get_parameter_short_labels,
    get_parameter_display_names,
    ComponentParameterMetadata,
    ParameterMetadata,
)
from pyrovelocity.models.metadata.components import (
    create_poisson_prior_metadata,
    PARAMETER_METADATA_REGISTRY,
)
from pyrovelocity.plots.parameter_metadata import (
    infer_component_name_from_parameters,
    get_parameter_label,
)


def test_poisson_metadata_creation():
    """Test that Poisson metadata component is created correctly."""
    metadata = create_poisson_prior_metadata()
    
    # Check component structure
    assert isinstance(metadata, ComponentParameterMetadata)
    assert metadata.component_name == "poisson_prior"
    assert metadata.component_type == "prior"
    assert "Poisson-only" in metadata.description
    
    # Check expected parameters
    expected_params = {"lambda_j", "t_star", "U_0i", "S_0i"}
    assert set(metadata.parameters.keys()) == expected_params
    
    # Check specific parameter metadata
    lambda_j_meta = metadata.parameters["lambda_j"]
    assert isinstance(lambda_j_meta, ParameterMetadata)
    assert lambda_j_meta.name == "lambda_j"
    assert lambda_j_meta.display_name == r"$\lambda_j$"
    assert lambda_j_meta.short_label == "Library Scale"
    assert lambda_j_meta.plot_order == 1
    
    # Check U_0i parameter
    u0i_meta = metadata.parameters["U_0i"]
    assert u0i_meta.name == "U_0i"
    assert u0i_meta.display_name == r"$U_{0i}$"
    assert u0i_meta.short_label == "Unspliced Scale"
    assert u0i_meta.plot_order == 3
    
    # Check S_0i parameter
    s0i_meta = metadata.parameters["S_0i"]
    assert s0i_meta.name == "S_0i"
    assert s0i_meta.display_name == r"$S_{0i}$"
    assert s0i_meta.short_label == "Spliced Scale"
    assert s0i_meta.plot_order == 4


def test_poisson_metadata_registry():
    """Test that Poisson metadata is properly registered."""
    # Check that poisson_prior is in the registry
    assert "poisson_prior" in PARAMETER_METADATA_REGISTRY
    
    # Test get_parameter_metadata function
    poisson_metadata = get_parameter_metadata("poisson_prior")
    assert isinstance(poisson_metadata, ComponentParameterMetadata)
    assert poisson_metadata.component_name == "poisson_prior"
    
    # Test error handling for unknown component
    with pytest.raises(KeyError):
        get_parameter_metadata("nonexistent_component")


def test_parameter_labels_functions():
    """Test parameter label extraction functions."""
    # Test short labels
    short_labels = get_parameter_short_labels("poisson_prior")
    expected_short_labels = {
        "lambda_j": "Library Scale",
        "t_star": "Time Coord",
        "U_0i": "Unspliced Scale",
        "S_0i": "Spliced Scale"
    }
    assert short_labels == expected_short_labels
    
    # Test display names
    display_names = get_parameter_display_names("poisson_prior")
    expected_display_names = {
        "lambda_j": r"$\lambda_j$",
        "t_star": r"$t^*$", 
        "U_0i": r"$U_{0i}$",
        "S_0i": r"$S_{0i}$"
    }
    assert display_names == expected_display_names


def test_parameter_inference():
    """Test automatic parameter inference from parameter sets."""
    # Test Poisson parameter detection
    poisson_params = {"lambda_j": None, "t_star": None, "U_0i": None, "S_0i": None}
    detected = infer_component_name_from_parameters(poisson_params)
    assert detected == "poisson_prior"
    
    # Test partial Poisson parameter detection (missing S_0i)
    partial_poisson_params = {"lambda_j": None, "t_star": None, "U_0i": None}
    detected = infer_component_name_from_parameters(partial_poisson_params)
    assert detected == "poisson_prior"
    
    # Test piecewise activation parameter detection
    piecewise_params = {
        "alpha_off": None, "alpha_on": None, "gamma_star": None, 
        "t_star": None, "U_0i": None, "lambda_j": None
    }
    detected = infer_component_name_from_parameters(piecewise_params)
    assert detected == "piecewise_activation_prior"
    
    # Test mixed parameters (should NOT detect as Poisson due to piecewise-specific params)
    mixed_params = {
        "lambda_j": None, "t_star": None, "U_0i": None, "S_0i": None,
        "alpha_off": None  # This should prevent Poisson detection
    }
    detected = infer_component_name_from_parameters(mixed_params)
    assert detected == "piecewise_activation_prior"
    
    # Test unknown parameters
    unknown_params = {"foo": None, "bar": None}
    detected = infer_component_name_from_parameters(unknown_params)
    assert detected is None


def test_get_parameter_label():
    """Test parameter label retrieval function."""
    # Test with component name
    label = get_parameter_label(
        param_name="lambda_j",
        label_type="short",
        component_name="poisson_prior"
    )
    assert label == "Library Scale"
    
    # Test display name
    label = get_parameter_label(
        param_name="U_0i",
        label_type="display",
        component_name="poisson_prior"
    )
    assert label == r"$U_{0i}$"
    
    # Test fallback to parameter name
    label = get_parameter_label(
        param_name="unknown_param",
        label_type="short",
        component_name="poisson_prior"
    )
    assert label == "unknown_param"
    
    # Test with unknown component (should fall back to parameter name)
    label = get_parameter_label(
        param_name="lambda_j",
        label_type="short",
        component_name="unknown_component"
    )
    assert label == "lambda_j"


def test_parameter_ordering():
    """Test that parameters are properly ordered by plot_order."""
    metadata = get_parameter_metadata("poisson_prior")
    params = list(metadata.parameters.keys())
    plot_orders = [metadata.parameters[p].plot_order for p in params]
    
    # Check that plot_orders are in ascending order
    assert plot_orders == sorted(plot_orders)
    
    # Check specific ordering
    expected_order = ["lambda_j", "t_star", "U_0i", "S_0i"]
    assert params == expected_order


def test_metadata_completeness():
    """Test that all metadata fields are properly filled."""
    metadata = get_parameter_metadata("poisson_prior")
    
    for param_name, param_meta in metadata.parameters.items():
        # Check required fields
        assert param_meta.name == param_name
        assert param_meta.display_name is not None
        assert param_meta.short_label is not None
        assert param_meta.description is not None
        assert param_meta.biological_interpretation is not None
        assert param_meta.plot_order is not None
        
        # Check that display names use LaTeX formatting
        assert "$" in param_meta.display_name
        
        # Check that plot_order is a positive integer
        assert isinstance(param_meta.plot_order, int)
        assert param_meta.plot_order > 0