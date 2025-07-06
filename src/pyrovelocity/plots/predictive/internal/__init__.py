"""
Internal utilities for predictive checks.

This package contains internal helper functions and specialized plotting
utilities that are used by the main plotting functions.
"""

from .helpers import (
    _plot_umap_leiden_clusters,
    _plot_umap_time_coordinate,
    _plot_pattern_proportions,
    _plot_correlation_structure,
    extract_gene_suffix,
    sigmoid_score,
)

from .rainbow import (
    _plot_gene_phase_portrait_rainbow,
    _plot_gene_spliced_dynamics_rainbow,
    _plot_gene_predictive_umap_rainbow,
    _plot_gene_observed_umap_rainbow,
    _plot_gene_marginal_histogram_rainbow,
)

__all__ = [
    # Helpers
    "_plot_umap_leiden_clusters",
    "_plot_umap_time_coordinate", 
    "_plot_pattern_proportions",
    "_plot_correlation_structure",
    "extract_gene_suffix",
    "sigmoid_score",
    # Rainbow plots
    "_plot_gene_phase_portrait_rainbow",
    "_plot_gene_spliced_dynamics_rainbow",
    "_plot_gene_predictive_umap_rainbow",
    "_plot_gene_observed_umap_rainbow",
    "_plot_gene_marginal_histogram_rainbow",
]