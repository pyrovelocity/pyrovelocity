"""
Shared parameter metadata for PyroVelocity models.

This package provides implementation-agnostic parameter metadata that can be used
by both modular (PyTorch/Pyro) and JAX (NumPyro) implementations, as well as 
shared plotting and analysis code.
"""

from pyrovelocity.models.metadata.base import (
    ComponentParameterMetadata,
    ParameterMetadata,
)
from pyrovelocity.models.metadata.components import (
    get_parameter_metadata,
    get_parameter_display_names,
    get_parameter_short_labels,
    PARAMETER_METADATA_REGISTRY,
)

__all__ = [
    "ComponentParameterMetadata",
    "ParameterMetadata", 
    "get_parameter_metadata",
    "get_parameter_display_names",
    "get_parameter_short_labels",
    "PARAMETER_METADATA_REGISTRY",
]