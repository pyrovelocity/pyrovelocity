"""
Rainbow plot functions for predictive checks.

This module contains specialized rainbow plot functions for temporal dynamics
visualization that maintain consistent styling and coloring across different
plot types.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy.stats import wasserstein_distance
from numpy.typing import ArrayLike
from beartype import beartype

from pyrovelocity.plots._common import set_colorbar
from pyrovelocity.plots.tensor_utils import ensure_numpy_parameters
from ..utils import _save_figure


def _plot_gene_phase_portrait_rainbow(
    observed_adata: AnnData,
    adata: AnnData,
    gene_idx: int,
    gene_name: str,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    time_key: str = "shared_time",
    cluster_key: str = "leiden",
    default_fontsize: float = 10
) -> None:
    """
    Plot phase portrait (u,s) for a single gene using rainbow plot style.
    
    Args:
        observed_adata: AnnData object with observed data
        adata: AnnData object with predicted data  
        gene_idx: Index of gene to plot
        gene_name: Name of gene to plot
        axes_dict: Dictionary mapping plot names to axes objects
        n: Gene number for axis key generation
        time_key: Key for time coordinate (e.g., 't_star', 'shared_time')
        cluster_key: Key for cluster information
        default_fontsize: Base font size for labels
    """
    ax_phase = axes_dict[f"phase_{n}"]
    
    # Get observed data
    if "unspliced" in observed_adata.layers:
        u_obs = observed_adata.layers["unspliced"][:, gene_idx]
        s_obs = observed_adata.layers["spliced"][:, gene_idx]
        
        # Convert sparse to dense if needed
        if hasattr(u_obs, 'toarray'):
            u_obs = u_obs.toarray().flatten()
            s_obs = s_obs.toarray().flatten()
        
        # Plot observed data behind predictive data
        ax_phase.scatter(u_obs, s_obs, c='lightgray', alpha=0.3, s=3, 
                        edgecolors='none', label='Observed')
    
    # Get predicted data
    u_pred = adata.layers["unspliced"][:, gene_idx]
    s_pred = adata.layers["spliced"][:, gene_idx]
    
    # Convert sparse to dense if needed  
    if hasattr(u_pred, 'toarray'):
        u_pred = u_pred.toarray().flatten()
        s_pred = s_pred.toarray().flatten()
    
    # Get time coordinate for coloring
    time_coord = None
    time_keys_priority = [time_key, 't_star', 'cell_time', 'latent_time', 'shared_time']
    
    for tk in time_keys_priority:
        if tk in adata.obs:
            time_coord = adata.obs[tk].values
            break
    
    if time_coord is not None:
        # Create scatter plot colored by time using viridis colormap
        scatter = ax_phase.scatter(u_pred, s_pred, c=time_coord, cmap='viridis', 
                                  alpha=0.7, s=5, edgecolors='none')
        
        # Add MAE display using stored values from gene selection
        if (observed_adata is not None and
            'mae_combined' in adata.var and
            gene_idx < len(adata.var['mae_combined'])):

            # Get the stored MAE score (positive value)
            mae_score = adata.var['mae_combined'].iloc[gene_idx]

            # Use positive value directly for display (higher = worse)
            display_mae = mae_score

            # Display MAE in top-left corner
            ax_phase.text(
                0.02, 0.98, f'MAE: {display_mae:.2f}',
                transform=ax_phase.transAxes,
                fontsize=7 * 0.7, va='top', ha='left',
                color='black', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
            )
    else:
        # Fallback to simple plot without time coloring
        ax_phase.scatter(u_pred, s_pred, c='blue', alpha=0.7, s=5, edgecolors='none')
    
    ax_phase.set_xlabel('Unspliced', fontsize=default_fontsize * 0.7)
    ax_phase.set_ylabel('Spliced', fontsize=default_fontsize * 0.7)
    ax_phase.set_title(f'{gene_name}', fontsize=default_fontsize * 0.8)
    ax_phase.tick_params(labelsize=default_fontsize * 0.6)


def _plot_gene_spliced_dynamics_rainbow(
    observed_adata: AnnData,
    adata: AnnData,
    gene_idx: int,
    gene_name: str,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    time_key: str = "shared_time",
    cluster_key: str = "leiden",
    default_fontsize: float = 10
) -> None:
    """
    Plot spliced expression dynamics over time using rainbow plot style.
    
    Args:
        observed_adata: AnnData object with observed data
        adata: AnnData object with predicted data
        gene_idx: Index of gene to plot
        gene_name: Name of gene to plot
        axes_dict: Dictionary mapping plot names to axes objects
        n: Gene number for axis key generation
        time_key: Key for time coordinate
        cluster_key: Key for cluster information
        default_fontsize: Base font size for labels
    """
    ax_dynamics = axes_dict[f"dynamics_{n}"]
    
    # Get time coordinate
    time_coord = None
    time_keys_priority = [time_key, 't_star', 'cell_time', 'latent_time', 'shared_time']
    
    for tk in time_keys_priority:
        if tk in adata.obs:
            time_coord = adata.obs[tk].values
            break
    
    if time_coord is None:
        # Fallback to cell index
        time_coord = np.arange(adata.n_obs)
    
    # Get predicted spliced expression
    s_pred = adata.layers["spliced"][:, gene_idx]
    if hasattr(s_pred, 'toarray'):
        s_pred = s_pred.toarray().flatten()
    
    # Sort by time for proper temporal ordering
    sort_idx = np.argsort(time_coord)
    time_sorted = time_coord[sort_idx]
    s_pred_sorted = s_pred[sort_idx]
    
    # Plot observed data if available
    if "spliced" in observed_adata.layers:
        s_obs = observed_adata.layers["spliced"][:, gene_idx]
        if hasattr(s_obs, 'toarray'):
            s_obs = s_obs.toarray().flatten()
        
        # Get time coordinate for observed data
        obs_time_coord = None
        for tk in time_keys_priority:
            if tk in observed_adata.obs:
                obs_time_coord = observed_adata.obs[tk].values
                break
        
        if obs_time_coord is None:
            obs_time_coord = np.arange(observed_adata.n_obs)
        
        obs_sort_idx = np.argsort(obs_time_coord)
        obs_time_sorted = obs_time_coord[obs_sort_idx]
        s_obs_sorted = s_obs[obs_sort_idx]
        
        ax_dynamics.scatter(obs_time_sorted, s_obs_sorted, c='lightgray', 
                           alpha=0.3, s=3, edgecolors='none', label='Observed')
    
    # Color by clusters if available
    cluster_col = None
    cluster_keys_priority = [cluster_key, 'leiden', 'clusters', 'louvain']
    
    for ck in cluster_keys_priority:
        if ck in adata.obs:
            cluster_col = adata.obs[ck].values
            break
    
    if cluster_col is not None:
        # Use tab10 colormap for discrete clusters
        unique_clusters = sorted(cluster_col.unique())
        colors = sns.color_palette("tab10", len(unique_clusters))
        cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
        
        for cluster in unique_clusters:
            mask = cluster_col == cluster
            cluster_sort_idx = sort_idx[mask[sort_idx]]
            if len(cluster_sort_idx) > 0:
                ax_dynamics.scatter(time_coord[cluster_sort_idx], s_pred[cluster_sort_idx],
                                   c=[cluster_color_map[cluster]], s=5, alpha=0.7,
                                   edgecolors='none', label=f'Cluster {cluster}')
    else:
        # Fallback to time-based coloring
        ax_dynamics.scatter(time_sorted, s_pred_sorted, c=time_sorted, 
                           cmap='viridis', alpha=0.7, s=5, edgecolors='none')
    
    ax_dynamics.set_xlabel('Time', fontsize=default_fontsize * 0.7)
    ax_dynamics.set_ylabel('Spliced', fontsize=default_fontsize * 0.7)
    ax_dynamics.set_title(f'{gene_name}', fontsize=default_fontsize * 0.8)
    ax_dynamics.tick_params(labelsize=default_fontsize * 0.6)


def _plot_gene_predictive_umap_rainbow(
    adata: AnnData,
    gene_idx: int,
    gene_name: str,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    default_fontsize: float = 10
) -> None:
    """
    Plot predictive spliced expression in UMAP space using rainbow plot style.
    
    Args:
        adata: AnnData object with predicted data
        gene_idx: Index of gene to plot
        gene_name: Name of gene to plot
        axes_dict: Dictionary mapping plot names to axes objects
        n: Gene number for axis key generation
        default_fontsize: Base font size for labels
    """
    ax_pred_umap = axes_dict[f"pred_umap_{n}"]
    
    if 'X_umap' in adata.obsm and 'spliced' in adata.layers:
        umap_coords = adata.obsm['X_umap']
        s_pred = adata.layers["spliced"][:, gene_idx]
        
        if hasattr(s_pred, 'toarray'):
            s_pred = s_pred.toarray().flatten()
        
        # Use log transformation for better visualization
        s_pred_log = np.log1p(np.maximum(s_pred, 0))
        
        scatter = ax_pred_umap.scatter(umap_coords[:, 0], umap_coords[:, 1],
                                      c=s_pred_log, cmap='cividis', alpha=0.7, s=5,
                                      edgecolors='none')
        
        # Add colorbar
        cbar = set_colorbar(scatter, ax_pred_umap, label='log(spliced+1)')
        cbar.ax.tick_params(labelsize=default_fontsize * 0.6)
        
        ax_pred_umap.set_xlabel('UMAP 1', fontsize=default_fontsize * 0.7)
        ax_pred_umap.set_ylabel('UMAP 2', fontsize=default_fontsize * 0.7)
        ax_pred_umap.set_title(f'{gene_name} (Predicted)', fontsize=default_fontsize * 0.8)
        ax_pred_umap.tick_params(labelsize=default_fontsize * 0.6)
    else:
        ax_pred_umap.text(0.5, 0.5, 'UMAP data\nnot available',
                         ha='center', va='center', transform=ax_pred_umap.transAxes)
        ax_pred_umap.set_title(f'{gene_name} (Predicted)')


def _plot_gene_observed_umap_rainbow(
    observed_adata: AnnData,
    gene_idx: int,
    gene_name: str,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    default_fontsize: float = 10
) -> None:
    """
    Plot observed spliced expression in UMAP space using rainbow plot style.
    
    Args:
        observed_adata: AnnData object with observed data
        gene_idx: Index of gene to plot
        gene_name: Name of gene to plot
        axes_dict: Dictionary mapping plot names to axes objects
        n: Gene number for axis key generation
        default_fontsize: Base font size for labels
    """
    ax_obs_umap = axes_dict[f"obs_umap_{n}"]
    
    if 'X_umap' in observed_adata.obsm and 'spliced' in observed_adata.layers:
        umap_coords = observed_adata.obsm['X_umap']
        s_obs = observed_adata.layers["spliced"][:, gene_idx]
        
        if hasattr(s_obs, 'toarray'):
            s_obs = s_obs.toarray().flatten()
        
        # Use log transformation for better visualization
        s_obs_log = np.log1p(np.maximum(s_obs, 0))
        
        scatter = ax_obs_umap.scatter(umap_coords[:, 0], umap_coords[:, 1],
                                     c=s_obs_log, cmap='cividis', alpha=0.7, s=5,
                                     edgecolors='none')
        
        # Add colorbar
        cbar = set_colorbar(scatter, ax_obs_umap, label='log(spliced+1)')
        cbar.ax.tick_params(labelsize=default_fontsize * 0.6)
        
        ax_obs_umap.set_xlabel('UMAP 1', fontsize=default_fontsize * 0.7)
        ax_obs_umap.set_ylabel('UMAP 2', fontsize=default_fontsize * 0.7)
        ax_obs_umap.set_title(f'{gene_name} (Observed)', fontsize=default_fontsize * 0.8)
        ax_obs_umap.tick_params(labelsize=default_fontsize * 0.6)
    else:
        ax_obs_umap.text(0.5, 0.5, 'UMAP data\nnot available',
                        ha='center', va='center', transform=ax_obs_umap.transAxes)
        ax_obs_umap.set_title(f'{gene_name} (Observed)')


def _plot_gene_marginal_histogram_rainbow(
    observed_adata: AnnData,
    adata: AnnData,
    gene_idx: int,
    gene_name: str,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    default_fontsize: float = 10
) -> None:
    """
    Plot marginal histogram comparison of predictive vs observed spliced expression.
    
    Args:
        observed_adata: AnnData object with observed data
        adata: AnnData object with predicted data
        gene_idx: Index of gene to plot
        gene_name: Name of gene to plot
        axes_dict: Dictionary mapping plot names to axes objects
        n: Gene number for axis key generation
        default_fontsize: Base font size for labels
    """
    ax_hist = axes_dict[f"hist_{n}"]
    
    # Get predicted and observed data
    s_pred = adata.layers["spliced"][:, gene_idx]
    if hasattr(s_pred, 'toarray'):
        s_pred = s_pred.toarray().flatten()
    
    if 'spliced' in observed_adata.layers:
        s_obs = observed_adata.layers["spliced"][:, gene_idx]
        if hasattr(s_obs, 'toarray'):
            s_obs = s_obs.toarray().flatten()
        
        # Calculate Wasserstein distance
        try:
            wasserstein_dist = wasserstein_distance(s_pred, s_obs)
        except Exception:
            wasserstein_dist = np.nan
        
        # Create histograms with relative frequency normalization
        bins = np.histogram_bin_edges(np.concatenate([s_pred, s_obs]), bins=20)
        
        # Plot histograms
        ax_hist.hist(s_obs, bins=bins, alpha=0.6, density=True, 
                    color='lightcoral', label='Observed', edgecolor='none')
        ax_hist.hist(s_pred, bins=bins, alpha=0.6, density=True,
                    color='skyblue', label='Predicted', edgecolor='none')
        
        # Add median lines
        ax_hist.axvline(np.median(s_obs), color='red', linestyle='--', alpha=0.8, linewidth=1)
        ax_hist.axvline(np.median(s_pred), color='blue', linestyle='--', alpha=0.8, linewidth=1)
        
        # Add Wasserstein distance
        if not np.isnan(wasserstein_dist):
            ax_hist.text(0.98, 0.98, f'WD: {wasserstein_dist:.2f}',
                        transform=ax_hist.transAxes, fontsize=default_fontsize * 0.6,
                        va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Add legend only for first row to save space
        if n == 1:
            ax_hist.legend(fontsize=default_fontsize * 0.6, loc='upper right')
    else:
        # Only predicted data available
        ax_hist.hist(s_pred, bins=20, alpha=0.6, density=True,
                    color='skyblue', label='Predicted', edgecolor='none')
        ax_hist.axvline(np.median(s_pred), color='blue', linestyle='--', alpha=0.8, linewidth=1)
    
    ax_hist.set_xlabel('Spliced Expression', fontsize=default_fontsize * 0.7)
    ax_hist.set_ylabel('Density', fontsize=default_fontsize * 0.7)
    ax_hist.set_title(f'{gene_name}', fontsize=default_fontsize * 0.8)
    ax_hist.tick_params(labelsize=default_fontsize * 0.6)


@beartype
def plot_pattern_analysis(
    adata: AnnData,
    parameters: Dict[str, ArrayLike],
    check_type: str = "prior",
    figsize: Tuple[Union[int, float], Union[int, float]] = (7.5, 3.75),  # Standard width, half height
    save_path: Optional[str] = None,
    file_prefix: str = "",
    default_fontsize: Union[int, float] = 8,
    observed_adata: Optional[AnnData] = None
) -> plt.Figure:
    """
    Plot pattern analysis: proportions and gene correlations.

    Args:
        adata: AnnData object with expression data
        parameters: Dictionary of parameter tensors
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height)
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        default_fontsize: Default font size for all text elements
        observed_adata: Optional AnnData object with observed data for MAE-based gene selection in correlations

    Returns:
        matplotlib Figure object
    """
    # Import here to avoid circular imports
    from ..internal.helpers import _plot_pattern_proportions, _plot_correlation_structure
    
    # Convert to numpy for plotting
    parameters = ensure_numpy_parameters(parameters)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pattern proportions
    _plot_pattern_proportions(adata, parameters, axes[0], check_type, default_fontsize)

    # Gene correlations - now uses lowest error genes when observed_adata is available
    _plot_correlation_structure(adata, axes[1], check_type, default_fontsize, observed_adata=observed_adata)

    plt.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_pattern_analysis")

    return fig