"""
PyroVelocity Predictive Checks Package.

This package provides modular plotting functions for PyroVelocity model validation,
including prior and posterior predictive checks, parameter analysis, and
temporal dynamics visualization.

The package is organized into specialized modules:
- core: Core computation functions (MAE calculation, parameter processing)
- main: Primary plotting functions (plot_prior_predictive_checks, plot_posterior_predictive_checks)
- parameters: Parameter-related plotting functions
- temporal: Temporal dynamics plotting functions
- expression: Expression validation plotting functions
- training: Training analysis plotting functions
- utils: Utility functions (file management, text formatting)
- internal: Internal helper functions and rainbow plots
"""

# Core computation functions
from .core import compute_and_store_mae

# Main plotting functions
from .main import (
    plot_prior_predictive_checks,
    plot_posterior_predictive_checks,
)

# Specialized plotting functions
from .parameters import (
    plot_parameter_marginals,
    plot_parameter_relationships,
    plot_parameter_marginals_by_gene,
    plot_parameter_recovery_correlation,
)

from .temporal import (
    plot_temporal_dynamics,
    plot_temporal_trajectories,
    plot_temporal_coordinate_validation,
)

from .expression import (
    plot_expression_validation,
)

from .training import (
    plot_training_loss,
    plot_mae_vs_spliced_count,
)

# Utility functions
from .utils import (
    cleanup_numbered_files,
    combine_pdfs,
    _select_genes_by_mae,
    _format_parameter_name,
    _format_pattern_name,
    _latex_safe_text,
    _save_figure,
)

# Internal functions (for advanced usage)
from .internal import (
    _plot_umap_leiden_clusters,
    _plot_umap_time_coordinate,
    _plot_pattern_proportions,
    _plot_correlation_structure,
    extract_gene_suffix,
    sigmoid_score,
    _plot_gene_phase_portrait_rainbow,
    _plot_gene_spliced_dynamics_rainbow,
    _plot_gene_predictive_umap_rainbow,
    _plot_gene_observed_umap_rainbow,
    _plot_gene_marginal_histogram_rainbow,
)

# Public API
__all__ = [
    # Core functions
    "compute_and_store_mae",
    
    # Main plotting functions
    "plot_prior_predictive_checks",
    "plot_posterior_predictive_checks",
    
    # Parameter plotting
    "plot_parameter_marginals",
    "plot_parameter_relationships",
    "plot_parameter_marginals_by_gene",
    "plot_parameter_recovery_correlation",
    
    # Temporal plotting
    "plot_temporal_dynamics",
    "plot_temporal_trajectories",
    "plot_temporal_coordinate_validation",
    
    # Expression plotting
    "plot_expression_validation",
    
    # Training plotting
    "plot_training_loss",
    "plot_mae_vs_spliced_count",
    
    # Utilities
    "cleanup_numbered_files",
    "combine_pdfs",
    "_select_genes_by_mae",
    "_format_parameter_name",
    "_format_pattern_name",
    "_latex_safe_text",
    "_save_figure",
    
    # Internal functions (advanced usage)
    "_plot_umap_leiden_clusters",
    "_plot_umap_time_coordinate",
    "_plot_pattern_proportions",
    "_plot_correlation_structure",
    "extract_gene_suffix",
    "sigmoid_score",
    "_plot_gene_phase_portrait_rainbow",
    "_plot_gene_spliced_dynamics_rainbow",
    "_plot_gene_predictive_umap_rainbow",
    "_plot_gene_observed_umap_rainbow",
    "_plot_gene_marginal_histogram_rainbow",
]