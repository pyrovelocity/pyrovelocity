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
    on library size normalization and gene-specific rate multipliers without
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
            plot_order=1,
            category="technical_scaling"
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
            plot_order=2,
            category="temporal_dynamics"
        ),
        
        # Gene-specific parameters
        "U_0i": ParameterMetadata(
            name="U_0i",
            display_name=r"$U_{0i}$",
            short_label="Expression Capacity",
            description="Gene-specific expression capacity",
            units="concentration scale",
            typical_range=(1.0, 50.0),
            biological_interpretation=(
                "Gene-specific characteristic expression capacity. "
                "Represents the baseline scale of expression for each gene, "
                "combining transcriptional activity and RNA processing efficiency."
            ),
            plot_order=3,
            category="technical_scaling"
        ),
        
        "r_u_i": ParameterMetadata(
            name="r_u_i",
            display_name=r"$r_{u,i}$",
            short_label="Unspliced Rate",
            description="Gene-specific unspliced rate multiplier",
            units="dimensionless rate",
            typical_range=(0.1, 5.0),
            biological_interpretation=(
                "Gene-specific unspliced RNA rate multiplier. "
                "Represents the relative rate of unspliced RNA production "
                "that would derive from differential equation solutions in the full model."
            ),
            plot_order=4,
            category="gene_expression"
        ),
        
        "r_s_i": ParameterMetadata(
            name="r_s_i",
            display_name=r"$r_{s,i}$",
            short_label="Spliced Rate",
            description="Gene-specific spliced rate multiplier",
            units="dimensionless rate",
            typical_range=(0.1, 5.0),
            biological_interpretation=(
                "Gene-specific spliced RNA rate multiplier. "
                "Represents the relative rate of spliced RNA production "
                "that would derive from differential equation solutions in the full model."
            ),
            plot_order=5,
            category="gene_expression"
        ),
    }
    
    return ComponentParameterMetadata(
        component_name="poisson_prior",
        component_type="prior",
        parameters=parameters,
        description="Simplified prior distributions for Poisson-only RNA count model without temporal dynamics"
    )