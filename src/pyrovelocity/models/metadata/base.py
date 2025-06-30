"""
Base interfaces and data structures for parameter metadata.

This module defines the core data structures for parameter metadata that are 
shared across all PyroVelocity implementations.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParameterMetadata:
    """
    Metadata for a single model parameter.
    
    This class contains semantic information about a parameter including its
    display formatting, biological interpretation, and typical value ranges.
    """
    name: str
    display_name: str
    short_label: str
    description: str
    units: str
    typical_range: Tuple[float, float]
    biological_interpretation: str
    plot_order: int


@dataclass
class ComponentParameterMetadata:
    """
    Parameter metadata for a model component.
    
    This class aggregates parameter metadata for all parameters in a model
    component (e.g., prior model, dynamics model, likelihood model).
    """
    component_name: str
    component_type: str
    parameters: Dict[str, ParameterMetadata]
    description: str
    
    def get_parameter_names(self) -> list[str]:
        """Get list of parameter names."""
        return list(self.parameters.keys())
    
    def get_display_names(self) -> Dict[str, str]:
        """Get mapping of parameter names to LaTeX display names."""
        return {name: param.display_name for name, param in self.parameters.items()}
    
    def get_short_labels(self) -> Dict[str, str]:
        """Get mapping of parameter names to short labels."""
        return {name: param.short_label for name, param in self.parameters.items()}
    
    def get_parameter_by_name(self, name: str) -> Optional[ParameterMetadata]:
        """Get parameter metadata by name."""
        return self.parameters.get(name)
    
    def get_ordered_parameters(self) -> list[ParameterMetadata]:
        """Get parameters ordered by plot_order."""
        return sorted(self.parameters.values(), key=lambda p: p.plot_order)