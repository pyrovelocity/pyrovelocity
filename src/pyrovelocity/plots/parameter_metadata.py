"""
Parameter metadata utilities for PyroVelocity plotting functions.

This module provides utilities for accessing parameter metadata from PyroVelocity
models and using it to enhance plotting functions with meaningful parameter labels,
display names, and other metadata.
"""

from typing import Any, Dict, Optional, Union

from beartype import beartype

from pyrovelocity.models.metadata import (
    ComponentParameterMetadata,
    get_parameter_display_names,
    get_parameter_short_labels,
)
# ParameterMetadataProvider protocol removed - using static metadata registry instead


@beartype
def get_model_parameter_metadata(model: Any) -> Dict[str, ComponentParameterMetadata]:
    """
    Extract parameter metadata from a PyroVelocity model.
    
    This function examines the model's components and extracts parameter metadata
    from any components that implement the ParameterMetadataProvider protocol.
    
    Args:
        model: PyroVelocity model instance
        
    Returns:
        Dictionary mapping component names to their parameter metadata
    """
    metadata = {}
    
    # ParameterMetadataProvider protocol has been removed
    # Models should now expose component_name property for metadata lookup
    # This function returns empty dict until models are updated to use the static registry
    # The fallback mechanisms in get_parameter_label will handle missing metadata
    
    return metadata


@beartype
def get_parameter_short_labels_from_model(
    model: Any,
    fallback_to_legacy: bool = True
) -> Dict[str, str]:
    """
    Get short labels for all parameters in a PyroVelocity model.
    
    This function extracts short labels from model components that provide
    parameter metadata, with fallback to legacy formatting if needed.
    
    Args:
        model: PyroVelocity model instance
        fallback_to_legacy: Whether to fall back to legacy _format_parameter_name
                           function for parameters without metadata
        
    Returns:
        Dictionary mapping parameter names to short labels
    """
    short_labels = {}
    
    # Get metadata from model components
    model_metadata = get_model_parameter_metadata(model)
    
    # Extract short labels from all components
    for component_metadata in model_metadata.values():
        short_labels.update(component_metadata.get_short_labels())
    
    return short_labels


@beartype
def get_parameter_display_names_from_model(
    model: Any,
    fallback_to_legacy: bool = True
) -> Dict[str, str]:
    """
    Get LaTeX display names for all parameters in a PyroVelocity model.
    
    This function extracts display names from model components that provide
    parameter metadata, with fallback to legacy formatting if needed.
    
    Args:
        model: PyroVelocity model instance
        fallback_to_legacy: Whether to fall back to legacy _format_parameter_name
                           function for parameters without metadata
        
    Returns:
        Dictionary mapping parameter names to LaTeX display names
    """
    display_names = {}
    
    # Get metadata from model components
    model_metadata = get_model_parameter_metadata(model)
    
    # Extract display names from all components
    for component_metadata in model_metadata.values():
        display_names.update(component_metadata.get_display_names())
    
    return display_names


@beartype
def get_parameter_label(
    param_name: str,
    label_type: str = "short",
    model: Optional[Any] = None,
    component_name: Optional[str] = None,
    fallback_to_legacy: bool = True
) -> str:
    """
    Get a label for a specific parameter.
    
    This function provides a unified interface for getting parameter labels,
    with multiple fallback strategies to ensure robust behavior.
    
    Args:
        param_name: Name of the parameter
        label_type: Type of label to get ("short", "display", or "name")
        model: Optional PyroVelocity model instance to extract metadata from
        component_name: Optional component name to look up metadata directly
        fallback_to_legacy: Whether to fall back to legacy formatting
        
    Returns:
        Parameter label string
    """
    # Strategy 1: Get from model metadata
    if model is not None:
        # Check if model has a component_name attribute (e.g., MetadataAwareJAXModel)
        if hasattr(model, 'component_name') and component_name is None:
            component_name = model.component_name
        
        if label_type == "short":
            labels = get_parameter_short_labels_from_model(model, fallback_to_legacy=False)
        elif label_type == "display":
            labels = get_parameter_display_names_from_model(model, fallback_to_legacy=False)
        else:
            labels = {}
        
        if param_name in labels:
            return labels[param_name]
    
    # Strategy 2: Get from component metadata registry
    if component_name is not None:
        try:
            if label_type == "short":
                labels = get_parameter_short_labels(component_name)
            elif label_type == "display":
                labels = get_parameter_display_names(component_name)
            else:
                labels = {}
            
            if param_name in labels:
                return labels[param_name]
        except KeyError:
            pass  # Component not found in registry
    
    # Strategy 3: Fall back to legacy formatting
    if fallback_to_legacy and label_type == "display":
        from pyrovelocity.plots.predictive_checks import _format_parameter_name
        return _format_parameter_name(param_name)
    
    # Strategy 4: Return the parameter name as-is
    return param_name


@beartype
def get_parameter_labels_for_plotting(
    parameters: Dict[str, Any],
    label_type: str = "short",
    model: Optional[Any] = None,
    component_name: Optional[str] = None,
    fallback_to_legacy: bool = True
) -> Dict[str, str]:
    """
    Get labels for a dictionary of parameters for plotting.
    
    This is a convenience function that applies get_parameter_label to all
    parameters in a dictionary.
    
    Args:
        parameters: Dictionary of parameters (values are ignored, only keys used)
        label_type: Type of label to get ("short", "display", or "name")
        model: Optional PyroVelocity model instance to extract metadata from
        component_name: Optional component name to look up metadata directly
        fallback_to_legacy: Whether to fall back to legacy formatting
        
    Returns:
        Dictionary mapping parameter names to labels
    """
    return {
        param_name: get_parameter_label(
            param_name=param_name,
            label_type=label_type,
            model=model,
            component_name=component_name,
            fallback_to_legacy=fallback_to_legacy
        )
        for param_name in parameters.keys()
    }


@beartype
def infer_component_name_from_parameters(parameters: Dict[str, Any]) -> Optional[str]:
    """
    Infer the likely component name based on the parameters present.
    
    This function uses heuristics to guess which component type generated
    a set of parameters, which can be useful for automatic metadata lookup.
    
    Args:
        parameters: Dictionary of parameters
        
    Returns:
        Inferred component name, or None if no clear match
    """
    param_names = set(parameters.keys())
    
    # Check for piecewise activation parameters
    piecewise_params = {
        'alpha_off', 'alpha_on', 'gamma_star', 't_on_star', 'delta_star',
        'T_M_star', 't_loc', 't_scale', 't_star', 'U_0i', 'lambda_j'
    }
    if len(param_names.intersection(piecewise_params)) >= 3:
        return "piecewise_activation_prior"
    
    # Check for standard log-normal parameters
    lognormal_params = {'alpha', 'beta', 'gamma', 'scaling'}
    if len(param_names.intersection(lognormal_params)) >= 2:
        return "lognormal_prior"
    
    return None
