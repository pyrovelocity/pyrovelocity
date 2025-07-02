"""Data structures for decoupled plotting functions."""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from beartype import beartype


@dataclass
class PlotMetadata:
    """Generic metadata for plotting functions.
    
    This structure allows plotting functions to work with any model implementation
    without requiring specific model attributes or methods.
    """
    component_name: Optional[str] = None
    parameter_labels: Optional[Dict[str, str]] = None
    parameter_display_names: Optional[Dict[str, str]] = None
    inference_state: Optional[Any] = None
    
    @classmethod
    def from_component_name(cls, component_name: str) -> "PlotMetadata":
        """Create metadata with just a component name."""
        return cls(component_name=component_name)
    
    @classmethod
    def empty(cls) -> "PlotMetadata":
        """Create empty metadata."""
        return cls()