"""
Internal helper functions for predictive checks.

This module contains internal helper functions for complex plotting operations
that are used by the main plotting functions.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from numpy.typing import ArrayLike

# For UMAP functionality (conditional import)
try:
    import umap
except ImportError:
    umap = None

from pyrovelocity.plots.tensor_utils import framework_agnostic_sigmoid
from ..utils import _select_genes_by_mae, _format_pattern_name


def _plot_umap_leiden_clusters(adata: AnnData, ax: plt.Axes, check_type: str, default_fontsize: Union[int, float] = 8) -> None:
    """Plot UMAP embedding colored by Leiden clusters."""
    if 'X_umap' in adata.obsm and 'leiden' in adata.obs:
        umap_coords = adata.obsm['X_umap']
        clusters = adata.obs['leiden']

        # Get unique clusters and assign colors with consistent ordering
        # Sort cluster names to ensure consistent color assignment across plots
        unique_clusters = sorted(clusters.unique())
        colors = sns.color_palette("tab10", len(unique_clusters))

        # Create a consistent cluster-to-color mapping
        cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

        for cluster in unique_clusters:
            mask = clusters == cluster
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                      c=[cluster_color_map[cluster]], label=f'Cluster {cluster}',
                      edgecolors="none",
                      alpha=0.7, s=5,)

        ax.set_xlabel('UMAP 1', fontsize=default_fontsize)
        ax.set_ylabel('UMAP 2', fontsize=default_fontsize)
        ax.set_title(f'{check_type.title()} UMAP (Leiden Clusters)', fontsize=default_fontsize)
        ax.tick_params(labelsize=default_fontsize * 0.75)
    else:
        ax.text(0.5, 0.5, 'UMAP data not available\nor UMAP not installed',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} UMAP (Clusters)')


def _plot_umap_time_coordinate(adata: AnnData, ax: plt.Axes, check_type: str, model: Optional[Any] = None, default_fontsize: Union[int, float] = 8) -> None:
    """Plot UMAP embedding colored by time coordinate."""

    if 'X_umap' in adata.obsm:
        umap_coords = adata.obsm['X_umap']

        # Look for time coordinate in various possible locations
        time_coord = None
        time_label = 'Time'
        found_key = None

        # Check canonical parameter names first (prioritize metadata system)
        canonical_time_keys = ['t_star', 'cell_time']

        # Then check common time coordinate names for compatibility
        fallback_time_keys = ['latent_time', 'velocity_pseudotime', 'dpt_pseudotime',
                             'pseudotime', 'time', 't', 'shared_time']

        # Combine in priority order
        time_keys = canonical_time_keys + fallback_time_keys

        for key in time_keys:
            if key in adata.obs:
                time_coord = adata.obs[key].values
                found_key = key
                break

        # Get appropriate label using parameter metadata system
        if found_key is not None:
            try:
                from pyrovelocity.plots.parameter_metadata import (
                    get_parameter_label,
                )

                # Try to get label from metadata system first
                time_label = get_parameter_label(
                    param_name=found_key,
                    label_type="short",
                    model=model,
                    fallback_to_legacy=False
                )
            except ImportError:
                pass  # parameter_metadata not available

            # If metadata system doesn't have it, use fallback mapping
            if time_label == found_key:  # No metadata found, use fallback
                fallback_labels = {
                    'latent_time': 'Latent Time',
                    'velocity_pseudotime': 'Velocity Pseudotime',
                    'dpt_pseudotime': 'DPT Pseudotime',
                    'pseudotime': 'Pseudotime',
                    'time': 'Time',
                    't': 'Time',
                    'shared_time': 'Shared Time',
                    'cell_time': 'Cell Time'
                }
                time_label = fallback_labels.get(found_key, 'Time Coordinate')

        if time_coord is not None:
            # Create scatter plot colored by time
            scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                                edgecolors="none",
                                c=time_coord, cmap='viridis', alpha=0.7, s=5)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label(time_label, fontsize=default_fontsize)
            cbar.ax.tick_params(labelsize=default_fontsize * 0.75)

            ax.set_xlabel('UMAP 1', fontsize=default_fontsize)
            ax.set_ylabel('UMAP 2', fontsize=default_fontsize)
            ax.set_title(f'{check_type.title()} UMAP (Time Coordinate)', fontsize=default_fontsize)
            ax.tick_params(labelsize=default_fontsize * 0.75)

        else:
            # No time coordinate found, use a simple gradient based on position
            gradient = np.arange(len(umap_coords))
            scatter = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                               c=gradient, cmap='viridis', alpha=0.7, s=5)

            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Cell Index', fontsize=10)

            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f'{check_type.title()} UMAP (Cell Index)')

    else:
        if 'X_pca' in adata.obsm and umap is not None:
            try:
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                embedding = reducer.fit_transform(adata.obsm['X_pca'][:, :50])

                # Use cell index as pseudo-time
                gradient = np.arange(len(embedding))
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                                   c=gradient, cmap='viridis', alpha=0.7, s=5)

                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label('Cell Index', fontsize=10)

                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.set_title(f'{check_type.title()} UMAP (Computed)')

            except Exception as e:
                ax.text(0.5, 0.5, f'UMAP computation failed:\n{str(e)[:50]}...',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{check_type.title()} UMAP (Failed)')
        else:
            ax.text(0.5, 0.5, 'UMAP data not available\nor UMAP not installed',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{check_type.title()} UMAP (Time Coordinate)')


def _classify_patterns_from_parameters(parameters: Dict[str, ArrayLike]) -> Dict[str, int]:
    """
    Classify expression patterns based on parameters.
    
    This is a fallback function when pattern information is not directly available.
    """
    # Simple heuristic based on parameter presence
    pattern_counts = {}
    
    # Check for different parameter sets that might indicate patterns
    if 'alpha_on' in parameters and 'alpha_off' in parameters:
        pattern_counts['transient'] = 1
    elif 'alpha_on' in parameters:
        pattern_counts['sustained'] = 1
    else:
        pattern_counts['pre_activation'] = 1
    
    return pattern_counts


def _plot_pattern_proportions(
    adata: AnnData,
    parameters: Dict[str, ArrayLike],
    ax: plt.Axes,
    check_type: str,
    default_fontsize: Union[int, float] = 8
) -> None:
    """Plot proportion of expression patterns."""
    # Try to get pattern information from AnnData
    if 'pattern' in adata.uns:
        pattern = adata.uns['pattern']
        pattern_counts = {pattern: 1}
    else:
        # Classify patterns based on parameters if available
        pattern_counts = _classify_patterns_from_parameters(parameters)

    if pattern_counts:
        patterns = list(pattern_counts.keys())
        counts = list(pattern_counts.values())
        colors = sns.color_palette("Set2", len(patterns))

        # Format pattern names for display
        formatted_patterns = [_format_pattern_name(pattern) for pattern in patterns]

        ax.pie(counts, labels=formatted_patterns, colors=colors,
               autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{check_type.title()} Pattern Proportions', fontsize=default_fontsize)
    else:
        ax.text(0.5, 0.5, 'Pattern information\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Pattern Proportions')


def _plot_correlation_structure(
    adata: AnnData,
    ax: plt.Axes,
    check_type: str,
    default_fontsize: Union[int, float] = 8,
    observed_adata: Optional[AnnData] = None,
    num_genes: int = 10
) -> None:
    """
    Plot gene-gene correlation structure for lowest error genes.

    Uses the same gene selection logic as temporal dynamics plots to show
    correlations among the genes that PyroVelocity models most accurately.

    Args:
        adata: AnnData object with predicted data
        ax: Matplotlib axes to plot on
        check_type: Type of check ("prior" or "posterior")
        default_fontsize: Font size for plot elements
        observed_adata: Optional AnnData object with observed data for MAE-based gene selection
        num_genes: Number of genes to include in correlation matrix (default: 10)
    """
    if 'spliced' in adata.layers:
        # Calculate gene-gene correlations
        expr_data = adata.layers['spliced']

        # Select genes using same logic as temporal dynamics plots
        if observed_adata is not None and 'spliced' in observed_adata.layers:
            # Use MAE-based selection to get the same genes as temporal dynamics
            try:
                gene_indices, gene_names = _select_genes_by_mae(
                    observed_adata=observed_adata,
                    predicted_adata=adata,
                    num_genes=min(num_genes, adata.n_vars),
                    select_highest_error=False  # Use lowest error genes
                )
                selection_method = f"lowest MAE genes"
            except Exception as e:
                print(f"Warning: MAE-based gene selection failed ({e}), using random selection")
                # Fallback to random selection
                n_genes_plot = min(num_genes, adata.n_vars)
                gene_indices = np.random.choice(adata.n_vars, n_genes_plot, replace=False).tolist()
                gene_names = [adata.var_names[i] for i in gene_indices]
                selection_method = f"random genes"
        else:
            # Fallback to random selection when observed data not available
            n_genes_plot = min(num_genes, adata.n_vars)
            gene_indices = np.random.choice(adata.n_vars, n_genes_plot, replace=False).tolist()
            gene_names = [adata.var_names[i] for i in gene_indices]
            selection_method = f"random genes"

        # Extract expression data for selected genes
        expr_subset = expr_data[:, gene_indices]

        # Convert to dense array if sparse
        if hasattr(expr_subset, 'toarray'):
            expr_subset = expr_subset.toarray()

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(expr_subset.T)

        # Extract numeric suffixes from gene names for cleaner labels
        gene_labels = [extract_gene_suffix(name) for name in gene_names]

        # Create heatmap with numeric gene labels
        sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8},
                   xticklabels=gene_labels,
                   yticklabels=gene_labels)

        ax.set_title(f'{check_type.title()} Gene Correlations\n({selection_method})', fontsize=default_fontsize)
        ax.set_xlabel('Gene', fontsize=default_fontsize)
        ax.set_ylabel('Gene', fontsize=default_fontsize)
        ax.tick_params(labelsize=default_fontsize * 0.6)  # Smaller labels for gene names

        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)

        # Adjust colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=default_fontsize * 0.75)
    else:
        ax.text(0.5, 0.5, 'Expression data\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Gene Correlations')


def extract_gene_suffix(gene_name: str) -> str:
    """Extract numeric suffix from gene names like 'gene_23' -> '23'."""
    digits = "".join(filter(str.isdigit, gene_name))
    return digits if digits else gene_name[:6]  # Fallback to truncated name


def sigmoid_score(value: float, threshold: float, direction: str, steepness: float = 5.0) -> float:
    """Compute soft score using sigmoid function."""
    if direction == '>':
        result = framework_agnostic_sigmoid(steepness * (value - threshold))
        return float(result.item()) if hasattr(result, 'item') else float(result)
    else:  # direction == '<'
        result = framework_agnostic_sigmoid(steepness * (threshold - value))
        return float(result.item()) if hasattr(result, 'item') else float(result)