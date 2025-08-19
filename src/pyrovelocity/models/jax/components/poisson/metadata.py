"""
Parameter metadata for Poisson-only components.

This module provides parameter metadata definitions for the Poisson-only model components,
offering standardized information about parameter semantics, display formatting,
and biological interpretations.
"""

from typing import Dict

from pyrovelocity.models.metadata.base import (
    ComponentParameterMetadata,
    ParameterMetadata,
)


def create_poisson_prior_metadata() -> ComponentParameterMetadata:
    """
    Create parameter metadata for the Poisson-only prior model.
    
    This function defines metadata for all parameters in the Poisson-only
    prior model, which uses a simplified hierarchical structure focusing
    on library size normalization and basic scaling parameters without
    temporal dynamics.
    
    Returns:
        ComponentParameterMetadata for the Poisson-only prior model.
    """
    parameters = {
        # Cell-specific parameters
        "lambda_j": ParameterMetadata(
            name="lambda_j",
            display_name=r"$\lambda_j$",
            short_label="Library Scale",
            description="Cell-specific library size scaling parameter",
            units="scale factor",
            typical_range=(0.1, 5.0),
            biological_interpretation=(
                "Cell capture efficiency and library size normalization factor. "
                "Accounts for technical variation in RNA capture and sequencing depth "
                "across individual cells."
            ),
            plot_order=1
        ),
        
        "t_star": ParameterMetadata(
            name="t_star",
            display_name=r"$t^*$",
            short_label="Time Coord",
            description="Temporal coordinate (constant for non-temporal model)",
            units="dimensionless time",
            typical_range=(0.5, 0.5),
            biological_interpretation=(
                "Compatibility parameter set to constant value (0.5) for non-temporal "
                "Poisson model. Maintains consistency with temporal model interfaces "
                "without adding temporal structure."
            ),
            plot_order=2
        ),
        
        # Gene-specific parameters
        "U_0i": ParameterMetadata(
            name="U_0i",
            display_name=r"$U_{0i}$",
            short_label="Unspliced Scale",
            description="Gene-specific unspliced RNA concentration scale",
            units="concentration scale",
            typical_range=(0.1, 5.0),
            biological_interpretation=(
                "Gene-specific baseline unspliced RNA concentration scaling factor. "
                "Captures gene-specific transcriptional activity and unspliced RNA "
                "steady-state levels without temporal dynamics."
            ),
            plot_order=3
        ),
        
        "S_0i": ParameterMetadata(
            name="S_0i",
            display_name=r"$S_{0i}$",
            short_label="Spliced Scale",
            description="Gene-specific spliced RNA concentration scale",
            units="concentration scale",
            typical_range=(0.1, 5.0),
            biological_interpretation=(
                "Gene-specific baseline spliced RNA concentration scaling factor. "
                "Captures gene-specific mature mRNA steady-state levels and "
                "expression magnitude without temporal dynamics."
            ),
            plot_order=4
        ),
    }
    
    return ComponentParameterMetadata(
        component_name="poisson_prior",
        component_type="prior",
        parameters=parameters,
        description="Simplified prior distributions for Poisson-only RNA count model without temporal dynamics"
    )