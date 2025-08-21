"""
Base interfaces and data structures for parameter metadata.

This module defines the core data structures for parameter metadata that are 
shared across all PyroVelocity implementations.
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


# Standard parameter category constants
PARAMETER_CATEGORIES = {
    "gene_expression": "Parameters controlling gene-specific expression dynamics",
    "technical_scaling": "Parameters for technical normalization and scaling", 
    "temporal_dynamics": "Parameters controlling temporal progression and timing"
}


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
    category: Optional[str] = None


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
    
    def get_parameters_by_category(self, category: str) -> List[str]:
        """Get parameter names filtered by category."""
        return [
            name for name, param in self.parameters.items()
            if param.category == category
        ]
    
    def get_parameter_categories(self) -> List[str]:
        """Get all unique parameter categories defined in this component."""
        categories = {param.category for param in self.parameters.values() if param.category is not None}
        return sorted(categories)