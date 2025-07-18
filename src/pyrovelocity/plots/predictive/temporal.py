"""
Temporal dynamics plotting functions for predictive checks.

This module contains functions for plotting temporal dynamics,
trajectories, and time-based validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from anndata import AnnData
from beartype import beartype
from matplotlib.gridspec import GridSpec
from numpy.typing import ArrayLike
from scipy.stats import linregress, pearsonr

from pyrovelocity.plots._common import set_colorbar, set_font_size
from pyrovelocity.plots.parameter_metadata import (
    get_parameter_label,
    infer_component_name_from_parameters,
)
from pyrovelocity.plots.tensor_utils import (
    convert_parameters_to_numpy,
    convert_to_numpy,
    ensure_numpy_parameters,
    framework_agnostic_exp,
    framework_agnostic_log2,
    framework_agnostic_sigmoid,
)
from .core import compute_and_store_mae
from .parameters import _save_figure, _select_genes_by_mae, _compute_adaptive_timing_thresholds

# TODO: These helper functions need to be extracted from the original file
# For now, they are placeholder stubs that will be replaced with actual implementations
def _create_temporal_dynamics_figure(
    number_of_genes: int,
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Create figure and axes dict using rainbow plot gridspec pattern."""
    from matplotlib.gridspec import GridSpec

    # Define number of horizontal panels
    horizontal_panels = 6  # gene_label, phase, dynamics, predictive, observed, marginal

    # Calculate figure size - expand width to accommodate new column
    if figsize is None:
        width = 9.0  # Expanded from 7.5 to accommodate marginal histogram column
        subplot_height = 0.9
        figsize = (width, subplot_height * number_of_genes)

    fig = plt.figure(figsize=figsize)

    # Create gridspec with proper width ratios and spacing like rainbow plot
    # Carefully balance spacing: phase/dynamics are tight, UMAP columns have more space
    gs = GridSpec(
        nrows=number_of_genes + 1,  # Add extra row for titles
        ncols=horizontal_panels,
        figure=fig,
        width_ratios=[
            0.21,  # Gene label column (same as rainbow plot)
            1.0,   # Phase portrait (keep tight)
            1.0,   # Dynamics (keep tight)
            0.85,  # Predictive UMAP (reduce from excess space)
            0.85,  # Observed UMAP (reduce from excess space)
            0.75,  # Marginal histogram (compact but readable)
        ],
        height_ratios=[0.15] + [1] * number_of_genes,  # Small title row + gene rows
        wspace=0.25,  # Reduced from 0.3 to accommodate new column
        hspace=0.25,  # Increased vertical spacing between gene rows
    )

    axes_dict = {}

    # Create title row
    titles = ['', r'$(u, s)$ phase space', 'Spliced dynamics', 'Predictive spliced', 'Observed spliced', 'Marginal histograms']
    for col, title in enumerate(titles):
        if col == 0:
            continue  # Skip gene label column for titles
        title_ax = fig.add_subplot(gs[0, col])
        title_ax.text(0.5, 0.5, title, ha='center', va='center',
                     transform=title_ax.transAxes, fontsize=8, weight='bold')
        title_ax.axis('off')

    # Create gene rows
    for n in range(number_of_genes):
        row = n + 1  # Offset by 1 for title row
        axes_dict[f"gene_{n}"] = fig.add_subplot(gs[row, 0])
        axes_dict[f"gene_{n}"].axis("off")
        axes_dict[f"phase_{n}"] = fig.add_subplot(gs[row, 1])
        axes_dict[f"dynamics_{n}"] = fig.add_subplot(gs[row, 2])
        axes_dict[f"predictive_{n}"] = fig.add_subplot(gs[row, 3])
        axes_dict[f"observed_{n}"] = fig.add_subplot(gs[row, 4])
        axes_dict[f"marginal_{n}"] = fig.add_subplot(gs[row, 5])

    return fig, axes_dict

def _plot_gene_phase_portrait_rainbow(
    adata: AnnData,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    gene_idx: int,
    gene_name: str,
    check_type: str,
    available_genes: int,
    observed_adata: Optional[AnnData] = None
) -> None:
    """Plot phase portrait (u,s) for a single gene using rainbow plot style."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        u_gene = adata.layers['unspliced'][:, gene_idx]
        s_gene = adata.layers['spliced'][:, gene_idx]

        # Plot observed data first (behind predictive data in z-order) if available
        if (observed_adata is not None and
            'unspliced' in observed_adata.layers and
            'spliced' in observed_adata.layers and
            gene_idx < observed_adata.n_vars):

            u_obs = observed_adata.layers['unspliced'][:, gene_idx]
            s_obs = observed_adata.layers['spliced'][:, gene_idx]

            # Plot observed data as highly transparent, small gray points
            axes_dict[f"phase_{n}"].scatter(
                s_obs, u_obs,
                alpha=0.3,  # Highly transparent
                s=1.5,      # Relatively small
                color='gray',
                edgecolors='none',
                zorder=1     # Behind predictive data
            )

        # Color by time coordinate if available, prioritizing canonical parameter names
        # This ensures consistency with the UMAP time coordinate plot
        time_col = None
        # Check canonical parameter names first (prioritize metadata system)
        for col in ['t_star', 'cell_time', 'latent_time']:
            if col in adata.obs:
                time_col = col
                break

        if time_col is not None:
            c = adata.obs[time_col]
            # Use viridis colormap to match UMAP time coordinate plot
            axes_dict[f"phase_{n}"].scatter(
                s_gene, u_gene, c=c, cmap='viridis',
                alpha=0.6, s=3, edgecolors='none',
                zorder=2  # In front of observed data
            )
        else:
            axes_dict[f"phase_{n}"].scatter(
                s_gene, u_gene, alpha=0.6, s=3,
                color='steelblue', edgecolors='none',
                zorder=2  # In front of observed data
            )

        # Add MAE display using stored values from gene selection
        if (observed_adata is not None and
            'mae_combined' in adata.var and
            gene_idx < len(adata.var['mae_combined'])):

            # Get the stored MAE score (positive value)
            mae_score = adata.var['mae_combined'].iloc[gene_idx]

            # Use positive value directly for display (higher = worse)
            display_mae = mae_score

            # Display MAE in top-left corner
            axes_dict[f"phase_{n}"].text(
                0.02, 0.98, f'MAE: {display_mae:.2f}',
                transform=axes_dict[f"phase_{n}"].transAxes,
                fontsize=7 * 0.7, va='top', ha='left',
                color='black', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
            )
    else:
        axes_dict[f"phase_{n}"].text(0.5, 0.5, 'Expression data\nnot available',
               ha='center', va='center', transform=axes_dict[f"phase_{n}"].transAxes)

    axes_dict[f"phase_{n}"].grid(True, alpha=0.3)

def _plot_gene_spliced_dynamics_rainbow(
    adata: AnnData,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    gene_idx: int,
    gene_name: str,
    check_type: str,
    available_genes: int,
    observed_adata: Optional[AnnData] = None
) -> None:
    """Plot spliced expression dynamics over time using rainbow plot style."""
    # Find available time column, prioritizing canonical parameter names
    time_col = None
    for col in ['t_star', 'cell_time', 'latent_time']:
        if col in adata.obs:
            time_col = col
            break

    if 'spliced' in adata.layers and time_col is not None:
        s_gene = adata.layers['spliced'][:, gene_idx]
        time = adata.obs[time_col]

        # Sort by time for better visualization
        sort_idx = np.argsort(time)
        time_sorted = time.iloc[sort_idx] if hasattr(time, 'iloc') else time[sort_idx]
        s_sorted = s_gene[sort_idx]

        # Plot observed data first (behind predictive data in z-order) if available
        if (observed_adata is not None and
            'spliced' in observed_adata.layers and
            gene_idx < observed_adata.n_vars):
            
            # Find time column in observed data - this is the true time coordinate
            obs_time_col = None
            for col in ['t_star', 'cell_time', 'latent_time', 'time']:
                if col in observed_adata.obs:
                    obs_time_col = col
                    break
            
            if obs_time_col is not None:
                s_obs = observed_adata.layers['spliced'][:, gene_idx]
                time_obs = observed_adata.obs[obs_time_col]
                
                # Sort observed data by time for better visualization
                obs_sort_idx = np.argsort(time_obs)
                time_obs_sorted = time_obs.iloc[obs_sort_idx] if hasattr(time_obs, 'iloc') else time_obs[obs_sort_idx]
                s_obs_sorted = s_obs[obs_sort_idx]
                
                # Plot observed data as highly transparent, small gray points
                axes_dict[f"dynamics_{n}"].scatter(
                    time_obs_sorted, s_obs_sorted,
                    alpha=0.3,  # Highly transparent
                    s=1.5,      # Relatively small
                    color='gray',
                    edgecolors='none',
                    zorder=1     # Behind predictive data
                )

        # Color by clusters if available to match UMAP cluster plot
        cluster_col = None
        for col in ['leiden', 'clusters', 'louvain']:
            if col in adata.obs:
                cluster_col = col
                break

        if cluster_col is not None:
            clusters = adata.obs[cluster_col].iloc[sort_idx] if hasattr(adata.obs[cluster_col], 'iloc') else adata.obs[cluster_col][sort_idx]
            # Sort cluster names to ensure consistent color assignment across plots
            unique_clusters = sorted(np.unique(clusters))
            # Use tab10 colormap to match UMAP cluster plot coloring
            colors = sns.color_palette("tab10", len(unique_clusters))

            # Create a consistent cluster-to-color mapping (same as UMAP plot)
            cluster_color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}

            for cluster in unique_clusters:
                mask = clusters == cluster
                axes_dict[f"dynamics_{n}"].scatter(time_sorted[mask], s_sorted[mask],
                          alpha=0.6, s=3, color=cluster_color_map[cluster], edgecolors='none', zorder=2)
        else:
            axes_dict[f"dynamics_{n}"].scatter(time_sorted, s_sorted, alpha=0.6, s=3, color='steelblue', edgecolors='none', zorder=2)
    else:
        axes_dict[f"dynamics_{n}"].text(0.5, 0.5, 'Time data\nnot available',
               ha='center', va='center', transform=axes_dict[f"dynamics_{n}"].transAxes)

    axes_dict[f"dynamics_{n}"].grid(True, alpha=0.3)

def _plot_gene_predictive_umap_rainbow(
    adata: AnnData,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    gene_idx: int,
    gene_name: str,
    check_type: str,
    basis: str = "umap"
) -> None:
    """Plot predictive spliced expression in UMAP space using rainbow plot style."""
    if f'X_{basis}' in adata.obsm and 'spliced' in adata.layers:
        coords = adata.obsm[f'X_{basis}']
        s_gene = adata.layers['spliced'][:, gene_idx]

        # Use log-transformed expression for predictive data (same scale as observed)
        s_gene_log = np.log1p(s_gene)  # log(1 + x) to handle zeros

        # Create scatter plot with log gene expression as color
        im = axes_dict[f"predictive_{n}"].scatter(
            coords[:, 0], coords[:, 1],
            c=s_gene_log, cmap='cividis',
            alpha=0.8, s=3, edgecolors='none'
        )

        # Add colorbar using rainbow plot style
        set_colorbar(
            im,
            axes_dict[f"predictive_{n}"],
            labelsize=5,
            fig=axes_dict[f"predictive_{n}"].figure,
            rainbow=True,
        )

        axes_dict[f"predictive_{n}"].axis('off')
    else:
        axes_dict[f"predictive_{n}"].text(0.5, 0.5, f'{basis.upper()} or\nexpression data\nnot available',
               ha='center', va='center', transform=axes_dict[f"predictive_{n}"].transAxes)
        axes_dict[f"predictive_{n}"].axis('off')

def _plot_gene_observed_umap_rainbow(
    adata: AnnData,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    gene_idx: int,
    gene_name: str,
    check_type: str,
    basis: str = "umap"
) -> None:
    """Plot observed spliced expression in UMAP space using rainbow plot style."""
    if f'X_{basis}' in adata.obsm and 'spliced' in adata.layers:
        coords = adata.obsm[f'X_{basis}']
        s_gene = adata.layers['spliced'][:, gene_idx]

        # Use log-transformed expression for observed data (same scale as predictive)
        s_gene_log = np.log1p(s_gene)  # log(1 + x) to handle zeros

        # Create scatter plot with log gene expression as color
        im = axes_dict[f"observed_{n}"].scatter(
            coords[:, 0], coords[:, 1],
            c=s_gene_log, cmap='cividis',
            alpha=0.8, s=3, edgecolors='none'
        )

        # Add colorbar using rainbow plot style
        set_colorbar(
            im,
            axes_dict[f"observed_{n}"],
            labelsize=5,
            fig=axes_dict[f"observed_{n}"].figure,
            rainbow=True,
        )

        axes_dict[f"observed_{n}"].axis('off')
    else:
        axes_dict[f"observed_{n}"].text(0.5, 0.5, f'{basis.upper()} or\nexpression data\nnot available',
               ha='center', va='center', transform=axes_dict[f"observed_{n}"].transAxes)
        axes_dict[f"observed_{n}"].axis('off')

def _plot_gene_marginal_histogram_rainbow(
    predicted_adata: AnnData,
    observed_adata: AnnData,
    axes_dict: Dict[str, plt.Axes],
    n: int,
    gene_idx: int,
    gene_name: str,
    check_type: str,
    default_fontsize: int = 7
) -> None:
    """Plot marginal histogram comparison of predictive vs observed spliced expression."""

    # Extract spliced expression data for this gene
    predicted_expr = None
    observed_expr = None

    if 'spliced' in predicted_adata.layers:
        predicted_expr = predicted_adata.layers['spliced'][:, gene_idx]

    if 'spliced' in observed_adata.layers:
        observed_expr = observed_adata.layers['spliced'][:, gene_idx]

    if predicted_expr is not None and observed_expr is not None:
        # Apply log1p transformation to handle zeros and improve visualization
        predicted_log = np.log1p(predicted_expr)
        observed_log = np.log1p(observed_expr)

        # Determine common bin range for fair comparison
        all_data = np.concatenate([predicted_log, observed_log])
        data_min, data_max = np.min(all_data), np.max(all_data)

        # Use 25 bins for good resolution without overcrowding
        bins = np.linspace(data_min, data_max, 26)  # 26 edges = 25 bins

        # Plot histograms with transparency for overlay using relative frequencies
        # Get histogram counts first to compute relative frequencies
        pred_counts, _ = np.histogram(predicted_log, bins=bins)
        obs_counts, _ = np.histogram(observed_log, bins=bins)

        # Convert to relative frequencies (sum to 1)
        pred_freq = pred_counts / np.sum(pred_counts) if np.sum(pred_counts) > 0 else pred_counts
        obs_freq = obs_counts / np.sum(obs_counts) if np.sum(obs_counts) > 0 else obs_counts

        # Plot as bar charts with relative frequencies
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]

        axes_dict[f"marginal_{n}"].bar(
            bin_centers, pred_freq, width=bin_width * 0.8, alpha=0.6,
            color='steelblue', label='Predictive', edgecolor='none'
        )
        axes_dict[f"marginal_{n}"].bar(
            bin_centers, obs_freq, width=bin_width * 0.8, alpha=0.5,
            color='gray', label='Observed', edgecolor='none'
        )

        # Add median lines for quick comparison
        pred_median = np.median(predicted_log)
        obs_median = np.median(observed_log)

        axes_dict[f"marginal_{n}"].axvline(
            pred_median, color='steelblue', linestyle='--', alpha=0.8, linewidth=1
        )
        axes_dict[f"marginal_{n}"].axvline(
            obs_median, color='gray', linestyle='--', alpha=0.8, linewidth=1
        )

        # Compute and display Wasserstein distance as error metric
        from scipy.stats import wasserstein_distance
        wd = wasserstein_distance(predicted_log, observed_log)
        axes_dict[f"marginal_{n}"].text(
            0.02, 0.98, f'WD: {wd:.2f}',
            transform=axes_dict[f"marginal_{n}"].transAxes,
            fontsize=default_fontsize * 0.7, va='top', ha='left',
            color='black', weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
        )

        # Add legend only for first row to save space
        if n == 0:
            axes_dict[f"marginal_{n}"].legend(
                fontsize=default_fontsize * 0.7, loc='upper right'
            )

        # Set labels and formatting (no y-label to save space)
        axes_dict[f"marginal_{n}"].tick_params(labelsize=default_fontsize * 0.6)
        axes_dict[f"marginal_{n}"].grid(True, alpha=0.3)

    else:
        # Handle missing data
        axes_dict[f"marginal_{n}"].text(
            0.5, 0.5, 'Expression data\nnot available',
            ha='center', va='center', transform=axes_dict[f"marginal_{n}"].transAxes,
            fontsize=default_fontsize * 0.8
        )
        axes_dict[f"marginal_{n}"].axis('off')

def _classify_parameters_into_patterns(
    parameters: Dict[str, ArrayLike],
    n_examples: int = 10
) -> Dict[str, List[Dict[str, ArrayLike]]]:
    """
    Classify parameter samples into patterns and select examples.

    Uses the same pattern classification logic as _classify_patterns_from_parameters
    to ensure consistency between pattern proportions and trajectory plots.

    Args:
        parameters: Dictionary of parameter tensors
        n_examples: Number of examples to select per pattern

    Returns:
        Dictionary mapping pattern names to lists of parameter dictionaries
    """
    # Check for required parameters - use independent absolute parameters only
    required_params = ['R_on', 't_on_star', 'delta_star', 'gamma_star']

    if not all(param in parameters for param in required_params):
        print(f"Warning: Missing required parameters for pattern classification")
        print(f"  Required params: {required_params}")
        return {}

    # Get parameter values (flatten to 1D if needed)
    R_on = parameters['R_on'].flatten()
    gamma_star = parameters['gamma_star'].flatten()
    t_on_star = parameters['t_on_star'].flatten()
    delta_star = parameters['delta_star'].flatten()

    # Ensure all parameters have the same length
    min_length = min(len(R_on), len(t_on_star), len(delta_star), len(gamma_star))
    R_on = R_on[:min_length]
    t_on_star = t_on_star[:min_length]
    delta_star = delta_star[:min_length]
    gamma_star = gamma_star[:min_length]

    # Compute alpha_off (fixed at 1.0) and alpha_on from R_on
    alpha_off = np.ones_like(R_on)
    alpha_on = R_on * alpha_off  # R_on = alpha_on / alpha_off

    pattern_examples = {
        'pre_activation': [],
        'transient': [],
        'sustained': []
    }

    # Use the same soft scoring approach as _classify_patterns_from_parameters
    def sigmoid_score(value: float, threshold: float, direction: str, steepness: float = 5.0) -> float:
        """Compute soft score using sigmoid function."""
        if direction == '>':
            result = framework_agnostic_sigmoid(steepness * (value - threshold))
            return float(result.item()) if hasattr(result, 'item') else float(result)
        else:  # direction == '<'
            result = framework_agnostic_sigmoid(steepness * (threshold - value))
            return float(result.item()) if hasattr(result, 'item') else float(result)

    # Compute pattern scores for each sample using independent absolute parameters
    pattern_scores = {pattern: np.zeros(min_length) for pattern in pattern_examples.keys()}

    for i in range(min_length):
        # Use independent absolute temporal parameters with mathematical specification thresholds
        
        # Pre-activation: negative onset time
        pattern_scores['pre_activation'][i] = np.prod(np.array([
            sigmoid_score(R_on[i].item(), 2.0, '>'),
            sigmoid_score(t_on_star[i].item(), 0.0, '<')
        ])) ** (1.0 / 2)

        # Transient: positive onset, early timing, short duration
        pattern_scores['transient'][i] = np.prod(np.array([
            sigmoid_score(R_on[i].item(), 2.0, '>'),
            sigmoid_score(t_on_star[i].item(), 0.0, '>'),
            sigmoid_score(t_on_star[i].item(), 1.5, '<'),  # Absolute early onset
            sigmoid_score(delta_star[i].item(), 2.0, '<')  # Absolute short duration
        ])) ** (1.0 / 4)

        # Sustained: positive onset, early timing, long duration
        pattern_scores['sustained'][i] = np.prod(np.array([
            sigmoid_score(R_on[i].item(), 2.0, '>'),
            sigmoid_score(t_on_star[i].item(), 0.0, '>'),
            sigmoid_score(t_on_star[i].item(), 1.5, '<'),  # Absolute early onset
            sigmoid_score(delta_star[i].item(), 2.5, '>')  # Absolute long duration
        ])) ** (1.0 / 4)

    # Assign each sample to the pattern with highest score and collect examples
    pattern_names = list(pattern_examples.keys())

    for i in range(min_length):
        scores = [pattern_scores[pattern][i] for pattern in pattern_names]
        best_pattern_idx = np.argmax(np.array(scores))
        best_pattern = pattern_names[best_pattern_idx]

        # Only add if we haven't reached the limit for this pattern
        if len(pattern_examples[best_pattern]) < n_examples:
            # Create parameter dictionary for this sample
            params = {
                'alpha_off': alpha_off[i],
                'alpha_on': alpha_on[i],
                'gamma_star': gamma_star[i],
                't_on_star': t_on_star[i],
                'delta_star': delta_star[i],
                'R_on': R_on[i]
            }

            # Add T_M_star if available (independent absolute parameterization)
            if 'T_M_star' in parameters:
                T_M_star = parameters['T_M_star'].flatten()
                if len(T_M_star) == 1:
                    params['T_M_star'] = T_M_star[0]
                elif len(T_M_star) >= i + 1:
                    params['T_M_star'] = T_M_star[i]
                else:
                    # Use first T_M_star value as it's a global parameter
                    params['T_M_star'] = T_M_star[0]

            pattern_examples[best_pattern].append(params)

    # Remove empty patterns
    pattern_examples = {k: v for k, v in pattern_examples.items() if v}

    return pattern_examples

def _compute_adaptive_time_range(
    pattern_examples: Dict[str, List[Dict[str, ArrayLike]]],
    buffer_factor: float = 1.2,
    n_points: int = 300,
    adata: Optional[AnnData] = None
) -> np.ndarray:
    """
    Compute adaptive time range based on realized t_star values or T_M_star fallback.

    This function prioritizes using the actual realized t_star values from the AnnData
    object to ensure temporal trajectories match the time range of cells shown in UMAP plots.
    Falls back to T_M_star-based computation when AnnData is not available.

    Args:
        pattern_examples: Dictionary with parameter sets for each pattern
        buffer_factor: Multiplicative buffer beyond realized time range (default: 1.2 for 20% buffer)
        n_points: Number of time points to generate (default: 300)
        adata: Optional AnnData object containing realized t_star values

    Returns:
        Time array for trajectory evaluation
    """
    # Priority 1: Use realized t_star values from AnnData if available
    if adata is not None:
        # Look for time coordinate in various possible locations
        time_coord = None
        canonical_time_keys = ['t_star', 'cell_time']
        fallback_time_keys = ['latent_time', 'velocity_pseudotime', 'dpt_pseudotime',
                             'pseudotime', 'time', 't', 'shared_time']
        time_keys = canonical_time_keys + fallback_time_keys

        for key in time_keys:
            if key in adata.obs:
                time_coord = adata.obs[key].values
                break

        if time_coord is not None and len(time_coord) > 0:
            # Use the actual realized time range from the data
            time_range_max = float(np.max(time_coord)) * buffer_factor
            print(f"  Adaptive time range: [0, {time_range_max:.1f}] based on realized {key} values (max: {np.max(time_coord):.1f})")
            return np.linspace(0, time_range_max, n_points)

    # Priority 2: Look for T_M_star values in pattern examples
    global_T_M_star = None

    for pattern_name, examples in pattern_examples.items():
        for params in examples:
            if 'T_M_star' in params:
                # T_M_star is a global parameter - should be the same across all examples
                global_T_M_star = params['T_M_star'].item()
                break
        if global_T_M_star is not None:
            break

    if global_T_M_star is not None:
        # Use the global T_M_star value with buffer
        # This ensures we capture complete activation-decay cycles within the global time scale
        time_range_max = global_T_M_star * buffer_factor
        print(f"  Adaptive time range: [0, {time_range_max:.1f}] based on global T*_M = {global_T_M_star:.1f}")
    else:
        # Fallback: estimate from parameter values if T_M_star not available
        max_time_scale = 0.0

        for pattern_name, examples in pattern_examples.items():
            for params in examples:
                # Estimate time scale from onset time and duration
                t_on = params['t_on_star'].item()
                delta = params['delta_star'].item()
                gamma = params['gamma_star'].item()

                # Estimate when trajectory returns to baseline (3 time constants)
                decay_time = 3.0 / gamma if gamma > 0 else 10.0

                # Total time scale includes onset, duration, and decay
                total_time = max(0, t_on) + delta + decay_time
                max_time_scale = max(max_time_scale, total_time)

        # Use buffer factor and ensure minimum reasonable range
        time_range_max = max(max_time_scale * buffer_factor, 10.0)
        print(f"  Fallback time range: [0, {time_range_max:.1f}] estimated from parameter values")

    return np.linspace(0, time_range_max, n_points)

def _format_pattern_name(pattern_name: str) -> str:
    """
    Format pattern names for display in legends and titles.

    Args:
        pattern_name: Raw pattern name (e.g., 'pre_activation', 'transient')

    Returns:
        Formatted pattern name suitable for LaTeX rendering
    """
    # Pattern name mappings for better display
    pattern_mappings = {
        'pre_activation': 'Pre-activation',
        'transient': 'Transient',
        'sustained': 'Sustained'
    }

    formatted = pattern_mappings.get(pattern_name, pattern_name.replace('_', ' ').title())
    return _latex_safe_text(formatted)

def _compute_time_course(
    t_star: ArrayLike,
    params: Dict[str, ArrayLike]
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute time course using piecewise activation dynamics.

    Args:
        t_star: Time points to evaluate
        params: Parameter dictionary

    Returns:
        Tuple of (u_star, s_star) time courses
    """
    # Extract parameters
    alpha_off = params['alpha_off']
    alpha_on = params['alpha_on']
    gamma_star = params['gamma_star']
    t_on_star = params['t_on_star']
    delta_star = params['delta_star']

    # Initialize output tensors
    u_star = np.zeros_like(t_star)
    s_star = np.zeros_like(t_star)

    # Phase 1: Off state (t* < t*_on)
    phase1_mask = t_star < t_on_star
    u_star[phase1_mask] = 1.0  # Fixed reference state
    s_star[phase1_mask] = 1.0 / gamma_star

    # Phase 2: On state (t*_on ≤ t* < t*_on + δ*)
    phase2_mask = (t_star >= t_on_star) & (t_star < t_on_star + delta_star)
    if phase2_mask.any():
        tau_on = t_star[phase2_mask] - t_on_star
        u_on, s_on = _compute_on_phase_solution(tau_on, alpha_on, gamma_star)
        u_star[phase2_mask] = u_on
        s_star[phase2_mask] = s_on

    # Phase 3: Return to off state (t* ≥ t*_on + δ*)
    phase3_mask = t_star >= t_on_star + delta_star
    if phase3_mask.any():
        tau_off = t_star[phase3_mask] - (t_on_star + delta_star)
        u_off, s_off = _compute_off_phase_solution(
            tau_off, alpha_off, alpha_on, gamma_star, delta_star
        )
        u_star[phase3_mask] = u_off
        s_star[phase3_mask] = s_off

    return u_star, s_star


def _compute_on_phase_solution(
    tau_on: ArrayLike,
    alpha_on: ArrayLike,
    gamma_star: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute analytical solution for ON phase."""
    # Initial conditions: u*_0 = 1.0, s*_0 = 1.0/γ*
    u_0 = 1.0
    s_0 = 1.0 / gamma_star

    # Analytical solution for ON phase
    exp_tau = framework_agnostic_exp(-tau_on)
    exp_gamma_tau = framework_agnostic_exp(-gamma_star * tau_on)

    u_on = alpha_on + (u_0 - alpha_on) * exp_tau

    s_on = (alpha_on / gamma_star +
            (s_0 - alpha_on / gamma_star) * exp_gamma_tau +
            (alpha_on - u_0) * (exp_tau - exp_gamma_tau) / (gamma_star - 1))

    return u_on, s_on


def _compute_off_phase_solution(
    tau_off: ArrayLike,
    alpha_off: ArrayLike,
    alpha_on: ArrayLike,
    gamma_star: ArrayLike,
    delta_star: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute analytical solution for return to OFF phase."""
    # Initial conditions: endpoint values from ON phase
    u_end, s_end = _compute_on_phase_solution(
        delta_star, alpha_on, gamma_star
    )

    # Analytical solution for OFF phase
    exp_tau = framework_agnostic_exp(-tau_off)
    exp_gamma_tau = framework_agnostic_exp(-gamma_star * tau_off)

    u_off = alpha_off + (u_end - alpha_off) * exp_tau

    s_off = (alpha_off / gamma_star +
            (s_end - alpha_off / gamma_star) * exp_gamma_tau +
            (alpha_off - u_end) * (exp_tau - exp_gamma_tau) / (gamma_star - 1))

    return u_off, s_off

def _latex_safe_text(text: str) -> str:
    """
    Make text safe for LaTeX rendering by escaping special characters.

    Args:
        text: Input text that may contain LaTeX special characters

    Returns:
        LaTeX-safe text with special characters escaped
    """
    # Escape underscores for LaTeX
    text = text.replace('_', r'\_')

    # Replace other problematic characters
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '^': r'\^{}',
        '~': r'\~{}',
        '{': r'\{',
        '}': r'\}',
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    return text

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
        if 'X_pca' in adata.obsm:
            try:
                import umap
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

def _compute_temporal_uncertainty(
    t_star_samples: ArrayLike,
    num_cells: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute temporal uncertainty metrics from posterior samples.

    Args:
        t_star_samples: Tensor of temporal coordinate samples
        num_cells: Expected number of cells

    Returns:
        Tuple of (standard_deviation, coefficient_of_variation) arrays
    """
    try:
        # Handle different tensor shapes
        if t_star_samples.ndim == 1:
            # Flattened: [num_samples * num_cells]
            # Try to infer num_samples
            total_length = len(t_star_samples)
            if total_length % num_cells == 0:
                num_samples = total_length // num_cells
                samples_reshaped = t_star_samples.reshape(num_samples, num_cells)
            else:
                # Cannot reshape properly
                return None, None
        elif t_star_samples.ndim == 2:
            # Already shaped: [num_samples, num_cells]
            samples_reshaped = t_star_samples
        else:
            # Higher dimensions: try to flatten and reshape
            samples_flat = t_star_samples.flatten()
            if len(samples_flat) % num_cells == 0:
                num_samples = len(samples_flat) // num_cells
                samples_reshaped = samples_flat.reshape(num_samples, num_cells)
            else:
                return None, None

        # Ensure we have the right number of cells
        if samples_reshaped.shape[1] != num_cells:
            return None, None

        # Compute statistics across samples (dim=0)
        samples_np = convert_to_numpy(samples_reshaped)
        mean_values = np.mean(samples_np, axis=0)
        std_values = np.std(samples_np, axis=0)

        # Compute coefficient of variation (CV = std/mean)
        # Avoid division by zero
        cv_values = np.where(mean_values > 1e-8, std_values / mean_values, 0.0)

        return std_values, cv_values

    except Exception as e:
        print(f"Warning: Could not compute temporal uncertainty: {e}")
        return None, None


def plot_temporal_dynamics(
    adata: AnnData,
    check_type: str = "prior",
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    save_path: Optional[str] = None,
    num_genes: int = 6,
    basis: str = "umap",
    default_fontsize: int = 7,
    file_prefix: str = "",
    observed_adata: Optional[AnnData] = None,
    gene_selection_method: str = "mae",
    select_highest_error: bool = False
) -> plt.Figure:
    """
    Plot temporal dynamics: multi-gene visualization with phase portraits,
    spliced dynamics, predictive and observed expression in UMAP space.

    Creates a rainbow-plot style visualization with one row per gene showing:
    - (u,s) phase space scatter plot
    - Spliced dynamics over time
    - Predictive spliced expression in UMAP space
    - Observed log spliced expression in UMAP space

    Uses the same gridspec layout as the rainbow plot for proper spacing and formatting.

    Args:
        adata: AnnData object with predictive samples (used for predictive columns)
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height). If None, auto-calculated based on num_genes
        save_path: Optional directory path to save figures
        num_genes: Number of genes to plot (default: 6)
        basis: Embedding basis to use (default: "umap")
        default_fontsize: Default font size for all text elements (default: 7)
        file_prefix: Prefix for saved file names
        observed_adata: Optional AnnData object with observed data (used for observed column).
                       If None, uses adata for both predictive and observed columns.
        gene_selection_method: Method for selecting genes to plot. Options:
                              "mae" - select genes by MAE (requires observed_adata)
                              "random" - randomly select genes
                              Default: "mae"
        select_highest_error: If True and gene_selection_method="mae", select genes with highest MAE
                             instead of lowest. Genes are always sorted from lowest to highest error
                             regardless of this setting (default: False)

    Returns:
        matplotlib Figure object
    """
    from matplotlib.gridspec import GridSpec

    from pyrovelocity.plots._common import set_colorbar, set_font_size

    # Set font size
    set_font_size(default_fontsize)

    # Select genes to plot
    available_genes = min(num_genes, adata.n_vars)
    if available_genes < num_genes:
        print(f"Warning: Only {available_genes} genes available, plotting all")

    # Select genes based on method
    if gene_selection_method == "mae" and observed_adata is not None:
        gene_indices, gene_names = _select_genes_by_mae(
            observed_adata=observed_adata,
            predicted_adata=adata,
            num_genes=available_genes,
            select_highest_error=select_highest_error
        )
        error_type = "highest" if select_highest_error else "lowest"
        print(f"Selected {len(gene_names)} genes with {error_type} MAE (sorted lowest to highest error): {gene_names}")
    else:
        if gene_selection_method == "mae" and observed_adata is None:
            print("Warning: MAE gene selection requested but no observed_adata provided. Using random selection.")
        gene_indices = np.random.choice(adata.n_vars, available_genes, replace=False)
        gene_names = [adata.var_names[i] for i in gene_indices]
        print(f"Selected {len(gene_names)} genes randomly: {gene_names}")

    # Create figure and gridspec using rainbow plot pattern
    # Use expanded width if figsize not provided to accommodate new marginal column
    if figsize is None:
        width = 9.0  # Expanded from 7.5 to accommodate marginal histogram column
        height = width * (available_genes / 5) * 0.6  # Adjusted ratio for 5 content columns
        figsize = (width, height)

    fig, axes_dict = _create_temporal_dynamics_figure(available_genes, figsize)

    # Plot each gene
    for n, (gene_idx, gene_name) in enumerate(zip(gene_indices, gene_names)):
        # Phase portrait
        _plot_gene_phase_portrait_rainbow(adata, axes_dict, n, gene_idx, gene_name, check_type, available_genes, observed_adata)

        # Spliced dynamics
        _plot_gene_spliced_dynamics_rainbow(adata, axes_dict, n, gene_idx, gene_name, check_type, available_genes, observed_adata)

        # Predictive spliced in UMAP
        _plot_gene_predictive_umap_rainbow(adata, axes_dict, n, gene_idx, gene_name, check_type, basis)

        # Observed spliced in UMAP (use observed_adata if provided, otherwise use adata)
        observed_data = observed_adata if observed_adata is not None else adata
        _plot_gene_observed_umap_rainbow(observed_data, axes_dict, n, gene_idx, gene_name, check_type, basis)

        # Marginal histogram comparison (predictive vs observed)
        _plot_gene_marginal_histogram_rainbow(
            predicted_adata=adata,
            observed_adata=observed_data,
            axes_dict=axes_dict,
            n=n,
            gene_idx=gene_idx,
            gene_name=gene_name,
            check_type=check_type,
            default_fontsize=default_fontsize
        )

        # Set labels and formatting
        _set_temporal_dynamics_labels(axes_dict, n, gene_name, available_genes, default_fontsize)

    # Set aspect ratios like rainbow plot
    _set_temporal_dynamics_aspect(axes_dict)

    fig.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_temporal_dynamics")

    return fig


@beartype
def plot_temporal_trajectories(
    parameters: Dict[str, ArrayLike],
    check_type: str = "prior",
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    save_path: Optional[str] = None,
    file_prefix: str = "",
    n_examples: int = 10,
    n_time_points: int = 300,
    buffer_factor: float = 1.2,
    adata: Optional[AnnData] = None,
    default_fontsize: Union[int, float] = 8
) -> plt.Figure:
    """
    Plot temporal trajectories showing underlying continuous dynamics.

    This function creates trajectory plots similar to those in prior-hyperparameter-calibration.py,
    showing the continuous temporal dynamics that underlie the discrete count observations.
    Plots are organized by pattern type (pre-activation, transient, sustained) with multiple
    examples per pattern.

    Args:
        parameters: Dictionary of parameter tensors from prior/posterior samples
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height). If None, auto-calculated
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        n_examples: Number of examples to show per pattern (default: 10)
        n_time_points: Number of time points for trajectory evaluation (default: 300)
        buffer_factor: Multiplicative buffer beyond realized time range (default: 1.2)
        adata: Optional AnnData object containing realized t_star values for adaptive time range

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy for plotting
    parameters = ensure_numpy_parameters(parameters)
    
    # Classify parameters into patterns and select examples
    pattern_examples = _classify_parameters_into_patterns(parameters, n_examples)

    if not pattern_examples:
        # Create empty figure if no patterns found
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.text(0.5, 0.5, 'No pattern examples available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Temporal Trajectories')
        return fig

    n_patterns = len(pattern_examples)

    # Auto-calculate figure size if not provided
    if figsize is None:
        width = 7.5  # Standard width for 8.5x11" with margins
        height = 2.5 * n_patterns  # Proportionally reduced height for narrower width
        figsize = (width, height)

    fig, axes = plt.subplots(n_patterns, 3, figsize=figsize)

    if n_patterns == 1:
        axes = axes.reshape(1, -1)

    # Compute adaptive time range based on realized t_star values or T_M_star fallback
    t_star = _compute_adaptive_time_range(pattern_examples, buffer_factor, n_time_points, adata)

    pattern_colors = ['red', 'blue', 'green', 'orange', 'purple']

    for pattern_idx, (pattern_name, examples) in enumerate(pattern_examples.items()):
        if not examples:
            continue

        color = pattern_colors[pattern_idx % len(pattern_colors)]
        formatted_pattern = _format_pattern_name(pattern_name)

        # Plot multiple examples for this pattern
        for example_idx, params in enumerate(examples):
            # Compute time courses using piecewise dynamics
            u_star, s_star = _compute_time_course(t_star, params)

            # Apply log2 transformation to show fold changes more clearly
            u_star_log2 = framework_agnostic_log2(u_star)
            s_star_log2 = framework_agnostic_log2(s_star)

            # Plot unspliced (log2 scale)
            axes[pattern_idx, 0].plot(
                convert_to_numpy(t_star), convert_to_numpy(u_star_log2),
                color=color, alpha=0.7, linewidth=2,
                label=f'Example {example_idx+1}' if example_idx < 3 else None
            )

            # Plot spliced (log2 scale)
            axes[pattern_idx, 1].plot(
                convert_to_numpy(t_star), convert_to_numpy(s_star_log2),
                color=color, alpha=0.7, linewidth=2,
                label=f'Example {example_idx+1}' if example_idx < 3 else None
            )

            # Plot phase portrait (both axes log2)
            axes[pattern_idx, 2].plot(
                convert_to_numpy(u_star_log2), convert_to_numpy(s_star_log2),
                color=color, alpha=0.7, linewidth=2,
                label=f'Example {example_idx+1}' if example_idx < 3 else None
            )

        # Get parameter labels using metadata system
        from pyrovelocity.plots.parameter_metadata import get_parameter_label

        # Get time coordinate label (use t_star as canonical parameter name)
        time_label = get_parameter_label(
            param_name="t_star",
            label_type="short",
            model=None,  # Will fall back to legacy formatting
            fallback_to_legacy=True
        )

        # Format axes with metadata-derived labels (log2 scale for fold changes)
        axes[pattern_idx, 0].set_xlabel(_latex_safe_text(f'{time_label}'), fontsize=default_fontsize * 0.9)
        axes[pattern_idx, 0].set_ylabel(r'$\log_2(u^*_{ij})$', fontsize=default_fontsize * 0.9)
        axes[pattern_idx, 0].set_title(f'{formatted_pattern}: Unspliced', fontsize=default_fontsize)
        axes[pattern_idx, 0].grid(True, alpha=0.3)
        axes[pattern_idx, 0].legend(fontsize=default_fontsize * 0.8)
        axes[pattern_idx, 0].tick_params(labelsize=default_fontsize * 0.75)

        axes[pattern_idx, 1].set_xlabel(_latex_safe_text(f'{time_label}'), fontsize=default_fontsize * 0.9)
        axes[pattern_idx, 1].set_ylabel(r'$\log_2(s^*_{ij})$', fontsize=default_fontsize * 0.9)
        axes[pattern_idx, 1].set_title(f'{formatted_pattern}: Spliced', fontsize=default_fontsize)
        axes[pattern_idx, 1].grid(True, alpha=0.3)
        axes[pattern_idx, 1].tick_params(labelsize=default_fontsize * 0.75)

        axes[pattern_idx, 2].set_xlabel(r'$\log_2(u^*_{ij})$', fontsize=default_fontsize * 0.9)
        axes[pattern_idx, 2].set_ylabel(r'$\log_2(s^*_{ij})$', fontsize=default_fontsize * 0.9)
        axes[pattern_idx, 2].set_title(f'{formatted_pattern}: Phase Portrait', fontsize=default_fontsize)
        axes[pattern_idx, 2].grid(True, alpha=0.3)
        axes[pattern_idx, 2].tick_params(labelsize=default_fontsize * 0.75)

    plt.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_temporal_trajectories")

    return fig


def plot_temporal_coordinate_validation(
    model: Any,
    adata: AnnData,
    parameters: Dict[str, ArrayLike],
    true_parameters_adata: Optional[AnnData] = None,
    figsize: Tuple[float, float] = (7.5, 2.5),  # Match parameter relationships aspect ratio
    save_path: Optional[str] = None,
    file_prefix: str = "",
    check_type: str = "prior",
    default_fontsize: Union[int, float] = 8,
    **kwargs
) -> plt.Figure:
    """
    Plot temporal coordinate validation with three panels: true vs estimated, UMAP, and uncertainty.

    This function validates temporal coordinate estimation quality and uncertainty quantification
    by creating a three-panel visualization:
    1. True vs Estimated Cell Time: Scatter plot with error bars and correlation metrics
    2. UMAP (Time Coordinate): UMAP embedding colored by temporal coordinates
    3. Temporal Uncertainty (CV): Coefficient of variation analysis

    Args:
        model: PyroVelocity model instance (for parameter metadata)
        adata: AnnData object with predictive samples and temporal coordinates
        parameters: Dictionary of parameter samples for uncertainty quantification
        true_parameters_adata: Optional AnnData object containing true parameters
                              in adata.uns['true_parameters'] for validation
        figsize: Figure size (width, height)
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        check_type: Type of check ("prior" or "posterior")
        default_fontsize: Default font size for all text elements
        **kwargs: Additional keyword arguments (for compatibility)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_temporal_coordinate_validation(
        ...     model=model,
        ...     adata=posterior_adata,
        ...     parameters=posterior_parameters,
        ...     true_parameters_adata=prior_predictive_adata,
        ...     save_path="reports/docs/posterior_predictive",
        ...     file_prefix="12",
        ...     check_type="posterior"
        ... ) # xdoctest: +SKIP
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Convert to numpy for plotting
    parameters = ensure_numpy_parameters(parameters)
    
    # Panel 1: True vs Estimated Cell Time
    _plot_true_vs_estimated_time(adata, parameters, true_parameters_adata, axes[0], check_type, default_fontsize)

    # Panel 2: UMAP (Time Coordinate) - reuse existing logic
    _plot_umap_time_coordinate(adata, axes[1], check_type, model, default_fontsize)

    # Panel 3: Temporal Uncertainty (CV)
    _plot_temporal_uncertainty(adata, parameters, axes[2], check_type, default_fontsize)

    plt.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}temporal_coordinate_validation")

    return fig


def _plot_true_vs_estimated_time(
    adata: AnnData,
    parameters: Dict[str, ArrayLike],
    true_parameters_adata: Optional[AnnData],
    ax: plt.Axes,
    check_type: str,
    default_fontsize: Union[int, float]
) -> None:
    """Plot true vs estimated temporal coordinates with error bars and correlation metrics."""

    # For prior predictive checks, "true" and "estimated" should be the same
    # since we're just sampling from the prior (no inference)
    if check_type == "prior":
        # For prior checks, use the same source for both true and estimated
        # This ensures perfect alignment as expected for prior predictive checks
        if true_parameters_adata is not None and 't_star' in true_parameters_adata.obs:
            # Use the same values for both true and estimated
            true_t_star = true_parameters_adata.obs['t_star'].values
            estimated_t_star = true_parameters_adata.obs['t_star'].values
        elif 'true_parameters' in true_parameters_adata.uns and 't_star' in true_parameters_adata.uns['true_parameters']:
            # Fallback to uns storage, but still use same values
            true_t_star = true_parameters_adata.uns['true_parameters']['t_star']
            estimated_t_star = true_t_star
        else:
            true_t_star = None
            estimated_t_star = None
    else:
        # For posterior predictive checks, use different sources as intended
        # Extract estimated temporal coordinates (posterior mean)
        estimated_t_star = adata.obs.get('t_star', None)

        # Extract true temporal coordinates for validation
        true_t_star = None
        if true_parameters_adata is not None:
            if 'true_parameters' in true_parameters_adata.uns:
                true_t_star = true_parameters_adata.uns['true_parameters'].get('t_star', None)
            elif 't_star' in true_parameters_adata.obs:
                true_t_star = true_parameters_adata.obs['t_star'].values

    # Extract posterior samples for uncertainty quantification
    t_star_samples = parameters.get('t_star', None)

    if estimated_t_star is not None and true_t_star is not None:
        # Ensure arrays are compatible
        true_t_star = convert_to_numpy(true_t_star)
        estimated_t_star = convert_to_numpy(estimated_t_star)

        # Flatten true_t_star if it has multiple dimensions
        if true_t_star.ndim > 1:
            true_t_star = true_t_star.flatten()

        # Ensure estimated_t_star is 1D
        if estimated_t_star.ndim > 1:
            estimated_t_star = estimated_t_star.flatten()

        # Handle length mismatch
        min_length = min(len(true_t_star), len(estimated_t_star))
        true_t_star = true_t_star[:min_length]
        estimated_t_star = estimated_t_star[:min_length]

        # Compute uncertainty if samples available
        uncertainty = None
        cv_values = None
        if t_star_samples is not None:
            # Use the original number of cells for uncertainty computation
            num_cells_original = len(estimated_t_star) if isinstance(estimated_t_star, np.ndarray) else estimated_t_star.shape[0]
            uncertainty, cv_values = _compute_temporal_uncertainty(t_star_samples, num_cells_original)

            # Trim uncertainty arrays to match the minimum length
            if uncertainty is not None:
                uncertainty = uncertainty[:min_length]
            if cv_values is not None:
                cv_values = cv_values[:min_length]

        # Create scatter plot
        if cv_values is not None and len(cv_values) == min_length:
            # Use log scale for CV coloring to better distinguish small values
            epsilon = 1e-6
            log_cv_values = np.log10(cv_values + epsilon)

            # Color points by log(coefficient of variation)
            scatter = ax.scatter(true_t_star, estimated_t_star, c=log_cv_values,
                               cmap='viridis', alpha=0.7, s=20, edgecolors='none')
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('log10(CV)', fontsize=default_fontsize * 0.9)
            cbar.ax.tick_params(labelsize=default_fontsize * 0.75)
        else:
            ax.scatter(true_t_star, estimated_t_star, alpha=0.7, s=20, color='steelblue')

        # Add error bars if uncertainty available
        if uncertainty is not None and len(uncertainty) == min_length:
            ax.errorbar(true_t_star, estimated_t_star, yerr=uncertainty,
                       fmt='none', alpha=0.3, color='gray', capsize=2)

        # Add perfect correlation line (y=x)
        min_val = min(np.min(true_t_star), np.min(estimated_t_star))
        max_val = max(np.max(true_t_star), np.max(estimated_t_star))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1.5, label='Perfect Recovery')

        # Compute and display correlation metrics
        correlation, _ = pearsonr(true_t_star, estimated_t_star)
        rmse = np.sqrt(np.mean((true_t_star - estimated_t_star) ** 2))

        # Add metrics as text
        metrics_text = f'r = {correlation:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=default_fontsize * 0.9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Get parameter labels using metadata system
        from pyrovelocity.plots.parameter_metadata import get_parameter_label
        time_label = get_parameter_label(
            param_name="t_star",
            label_type="display",
            model=None,
            fallback_to_legacy=True
        )

        ax.set_xlabel(f'True {time_label}', fontsize=default_fontsize)
        ax.set_ylabel(f'Estimated {time_label}', fontsize=default_fontsize)
        ax.set_title('True vs Estimated Cell Time', fontsize=default_fontsize)
        ax.legend(fontsize=default_fontsize * 0.8)

    else:
        # No validation data available
        ax.text(0.5, 0.5, f'True vs estimated validation\nnot available for {check_type} checks',
               ha='center', va='center', transform=ax.transAxes, fontsize=default_fontsize)
        ax.set_title('True vs Estimated Cell Time', fontsize=default_fontsize)

    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=default_fontsize * 0.75)


def _plot_temporal_uncertainty(
    adata: AnnData,
    parameters: Dict[str, ArrayLike],
    ax: plt.Axes,
    check_type: str,
    default_fontsize: Union[int, float]
) -> None:
    """Plot temporal uncertainty analysis using coefficient of variation."""

    # Extract temporal coordinates and samples
    estimated_t_star = adata.obs.get('t_star', None)
    t_star_samples = parameters.get('t_star', None)

    if estimated_t_star is not None and t_star_samples is not None:
        num_cells = len(estimated_t_star)
        uncertainty, cv_values = _compute_temporal_uncertainty(t_star_samples, num_cells)

        if cv_values is not None:
            # Use log scale for CV to better distinguish small values
            # Add small epsilon to avoid log(0) issues
            epsilon = 1e-6
            log_cv_values = np.log10(cv_values + epsilon)

            # Create scatter plot of log(CV) vs estimated time
            scatter = ax.scatter(estimated_t_star, log_cv_values, alpha=0.7, s=20, color='orange')

            # Add horizontal line for mean log(CV)
            mean_log_cv = np.mean(log_cv_values)
            mean_cv = np.mean(cv_values)
            ax.axhline(mean_log_cv, color='red', linestyle='--', alpha=0.8,
                      label=f'Mean CV = {mean_cv:.3f}')

            # Get parameter labels using metadata system
            from pyrovelocity.plots.parameter_metadata import (
                get_parameter_label,
            )
            time_label = get_parameter_label(
                param_name="t_star",
                label_type="short",
                model=None,
                fallback_to_legacy=True
            )

            ax.set_xlabel(f'Estimated {time_label}', fontsize=default_fontsize)
            ax.set_ylabel('log10(Coefficient of Variation)', fontsize=default_fontsize)
            ax.set_title('Temporal Uncertainty (log CV)', fontsize=default_fontsize)
            ax.legend(fontsize=default_fontsize * 0.8)

            # Add interpretation text with both linear and log scale info
            high_uncertainty_threshold = mean_cv + 2 * np.std(cv_values)
            high_uncertainty_cells = np.sum(cv_values > high_uncertainty_threshold)
            uncertainty_text = f'High uncertainty cells: {high_uncertainty_cells}/{len(cv_values)}\nCV range: [{np.min(cv_values):.3f}, {np.max(cv_values):.3f}]'
            ax.text(0.05, 0.95, uncertainty_text, transform=ax.transAxes,
                   fontsize=default_fontsize * 0.8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Cannot compute temporal\nuncertainty from samples',
                   ha='center', va='center', transform=ax.transAxes, fontsize=default_fontsize)
            ax.set_title('Temporal Uncertainty (CV)', fontsize=default_fontsize)
    else:
        ax.text(0.5, 0.5, f'Temporal uncertainty analysis\nnot available for {check_type} checks',
               ha='center', va='center', transform=ax.transAxes, fontsize=default_fontsize)
        ax.set_title('Temporal Uncertainty (CV)', fontsize=default_fontsize)

    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=default_fontsize * 0.75)


def _set_temporal_dynamics_labels(
    axes_dict: Dict[str, plt.Axes],
    n: int,
    gene_name: str,
    total_genes: int,
    default_fontsize: int
) -> None:
    """Set labels for temporal dynamics plot using rainbow plot style."""
    # Set gene name in the gene label column
    axes_dict[f"gene_{n}"].text(
        0.0, 0.5, gene_name[:7],
        transform=axes_dict[f"gene_{n}"].transAxes,
        rotation=0, va='center', ha='center',
        fontsize=default_fontsize, weight='normal'
    )

    # Set axis labels only for bottom row (like rainbow plot)
    if n == total_genes - 1:
        # Set x-axis labels
        axes_dict[f"phase_{n}"].set_xlabel(
            r'spliced, $\hat{\mu}(s)$',
            loc="left",
            labelpad=0.7,
            fontsize=default_fontsize
        )
        axes_dict[f"dynamics_{n}"].set_xlabel(
            r'shared time, $\hat{\mu}(t)$',
            loc="left",
            labelpad=0.7,
            fontsize=default_fontsize
        )
        axes_dict[f"marginal_{n}"].set_xlabel(
            r'log(1+spliced)',
            loc="center",
            labelpad=0.7,
            fontsize=default_fontsize * 0.9
        )

        # Set y-axis labels on the RIGHT side (like rainbow plot)
        axes_dict[f"phase_{n}"].set_ylabel(
            r'unspliced, $\hat{\mu}(u)$',
            loc="bottom",
            labelpad=0.7,
            fontsize=default_fontsize
        )
        axes_dict[f"phase_{n}"].yaxis.set_label_position("right")

        axes_dict[f"dynamics_{n}"].set_ylabel(
            r'spliced, $\hat{\mu}(s)$',
            loc="bottom",
            labelpad=0.7,
            fontsize=default_fontsize
        )
        axes_dict[f"dynamics_{n}"].yaxis.set_label_position("right")
    else:
        # Remove axis labels for non-bottom rows
        axes_dict[f"phase_{n}"].set_xlabel('')
        axes_dict[f"phase_{n}"].set_ylabel('')
        axes_dict[f"dynamics_{n}"].set_xlabel('')
        axes_dict[f"dynamics_{n}"].set_ylabel('')
        # Remove marginal labels for non-bottom rows
        if f"marginal_{n}" in axes_dict:
            axes_dict[f"marginal_{n}"].set_xlabel('')

    # Set tick parameters
    axes_dict[f"phase_{n}"].tick_params(labelsize=default_fontsize * 0.75)
    axes_dict[f"dynamics_{n}"].tick_params(labelsize=default_fontsize * 0.75)
    # Set tick parameters for marginal histogram
    if f"marginal_{n}" in axes_dict:
        axes_dict[f"marginal_{n}"].tick_params(labelsize=default_fontsize * 0.6)


def _set_temporal_dynamics_aspect(axes_dict: Dict[str, plt.Axes]) -> None:
    """Set aspect ratios for temporal dynamics plot using rainbow plot style."""
    # Set equal aspect for UMAP plots (like rainbow plot)
    for key in axes_dict.keys():
        if 'predictive_' in key or 'observed_' in key:
            axes_dict[key].set_aspect('equal')
        elif 'phase_' in key:
            # Phase plots can have auto aspect
            axes_dict[key].set_aspect('auto')
        elif 'dynamics_' in key:
            # Dynamics plots can have auto aspect
            axes_dict[key].set_aspect('auto')
        elif 'marginal_' in key:
            # Marginal histogram plots can have auto aspect
            axes_dict[key].set_aspect('auto')