"""
Parameter metadata for PyroVelocity modular components (legacy compatibility).

This module provides backward compatibility for modular components that expect
metadata to be available from this location. All functionality delegates to the
shared metadata package at pyrovelocity.models.metadata.
"""

from typing import Dict

from pyrovelocity.models.metadata import (
    ComponentParameterMetadata,
    ParameterMetadata,
    get_parameter_metadata as _get_parameter_metadata,
    get_parameter_display_names as _get_parameter_display_names,
    get_parameter_short_labels as _get_parameter_short_labels,
    PARAMETER_METADATA_REGISTRY,
)


# For backward compatibility, delegate to shared implementation
def create_piecewise_activation_prior_metadata() -> ComponentParameterMetadata:
    """Create parameter metadata for the piecewise activation prior model."""
    return PARAMETER_METADATA_REGISTRY["piecewise_activation_prior"]


def create_lognormal_prior_metadata() -> ComponentParameterMetadata:
    """Create parameter metadata for the log-normal prior model."""
    return PARAMETER_METADATA_REGISTRY["lognormal_prior"]


# Delegate to shared metadata implementation
def get_parameter_metadata(component_name: str) -> ComponentParameterMetadata:
    """Get parameter metadata for a named component."""
    return _get_parameter_metadata(component_name)


def get_parameter_short_labels(component_name: str) -> Dict[str, str]:
    """Get short labels for parameters of a named component."""
    return _get_parameter_short_labels(component_name)


def get_parameter_display_names(component_name: str) -> Dict[str, str]:
    """Get LaTeX display names for parameters of a named component."""
    return _get_parameter_display_names(component_name)