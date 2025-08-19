"""
Parameter metadata definitions for PyroVelocity model components.

This module contains parameter metadata definitions for different model components,
providing standardized information about parameter semantics, display formatting,
and biological interpretations that can be shared across implementations.
"""

from typing import Dict

from pyrovelocity.models.metadata.base import (
    ComponentParameterMetadata,
    ParameterMetadata,
)
from pyrovelocity.models.jax.components.poisson.metadata import create_poisson_prior_metadata
from pyrovelocity.models.jax.components.piecewise.metadata import create_piecewise_activation_prior_metadata


# Piecewise activation metadata moved to src/pyrovelocity/models/jax/components/piecewise/metadata.py


def create_lognormal_prior_metadata() -> ComponentParameterMetadata:
    """
    Create parameter metadata for the standard log-normal prior model.
    
    Returns:
        ComponentParameterMetadata for the log-normal prior model.
    """
    parameters = {
        "alpha": ParameterMetadata(
            name="alpha",
            display_name=r"$\alpha$",
            short_label="Transcription",
            description="Transcription rate parameter",
            units="rate",
            typical_range=(0.1, 10.0),
            biological_interpretation="Rate of pre-mRNA transcription",
            plot_order=1
        ),
        
        "beta": ParameterMetadata(
            name="beta",
            display_name=r"$\beta$",
            short_label="Splicing",
            description="Splicing rate parameter",
            units="rate",
            typical_range=(0.1, 10.0),
            biological_interpretation="Rate of pre-mRNA splicing to mature mRNA",
            plot_order=2
        ),
        
        "gamma": ParameterMetadata(
            name="gamma",
            display_name=r"$\gamma$",
            short_label="Degradation",
            description="mRNA degradation rate parameter",
            units="rate",
            typical_range=(0.1, 10.0),
            biological_interpretation="Rate of mature mRNA degradation",
            plot_order=3
        ),
        
        "scaling": ParameterMetadata(
            name="scaling",
            display_name=r"$s$",
            short_label="Scaling",
            description="Global scaling factor",
            units="scale factor",
            typical_range=(0.1, 10.0),
            biological_interpretation="Overall expression scale adjustment",
            plot_order=4
        ),
    }
    
    return ComponentParameterMetadata(
        component_name="lognormal_prior",
        component_type="prior",
        parameters=parameters,
        description="Log-normal prior distributions for standard RNA velocity parameters"
    )

# Poisson metadata moved to src/pyrovelocity/models/jax/components/poisson/metadata.py


# Registry of metadata providers - now importing from component-colocated locations
PARAMETER_METADATA_REGISTRY: Dict[str, ComponentParameterMetadata] = {
    "piecewise_activation_prior": create_piecewise_activation_prior_metadata(),
    "lognormal_prior": create_lognormal_prior_metadata(),
    "poisson_prior": create_poisson_prior_metadata(),
}


def get_parameter_metadata(component_name: str) -> ComponentParameterMetadata:
    """
    Get parameter metadata for a named component.
    
    Args:
        component_name: Name of the component to get metadata for
        
    Returns:
        ComponentParameterMetadata for the component, or None if not found
        
    Raises:
        KeyError: If component_name is not found in the registry
    """
    if component_name not in PARAMETER_METADATA_REGISTRY:
        raise KeyError(f"No parameter metadata found for component '{component_name}'")
    
    return PARAMETER_METADATA_REGISTRY[component_name]


def get_parameter_short_labels(component_name: str) -> Dict[str, str]:
    """
    Get short labels for parameters of a named component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        Dictionary mapping parameter names to short labels
    """
    try:
        metadata = get_parameter_metadata(component_name)
        return metadata.get_short_labels()
    except KeyError:
        return {}


def get_parameter_display_names(component_name: str) -> Dict[str, str]:
    """
    Get LaTeX display names for parameters of a named component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        Dictionary mapping parameter names to LaTeX display names
    """
    try:
        metadata = get_parameter_metadata(component_name)
        return metadata.get_display_names()
    except KeyError:
        return {}