"""
Expression validation plotting functions for predictive checks.

This module contains functions for validating expression data
and relationships.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from numpy.typing import ArrayLike
from scipy.stats import linregress, pearsonr

from pyrovelocity.plots.tensor_utils import (
    convert_parameters_to_numpy,
    convert_to_numpy,
    ensure_numpy_parameters,
    framework_agnostic_exp,
    framework_agnostic_log2,
    framework_agnostic_sigmoid,
)
from pyrovelocity.styles import configure_matplotlib_style

configure_matplotlib_style()


def _save_figure(
    fig: plt.Figure,
    save_path: str,
    figure_name: str,
    formats: List[str] = ["png", "pdf"]
) -> None:
    """
    Save figure in multiple formats with consistent naming.

    Args:
        fig: matplotlib Figure object
        save_path: Directory path to save figures
        figure_name: Base name for the figure files
        formats: List of file formats to save
    """
    if save_path is not None:
        output_dir = Path(save_path)
        os.makedirs(output_dir, exist_ok=True)

        for ext in formats:
            save_file = output_dir / f"{figure_name}.{ext}"
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {save_file}")


@beartype
def compute_and_store_mae(
    predicted_adata: AnnData,
    observed_adata: AnnData,
    store_in_var: bool = True
) -> np.ndarray:
    """
    Compute MAE and store in predicted_adata.var for reuse across plotting functions.
    
    Args:
        predicted_adata: AnnData with predicted counts
        observed_adata: AnnData with observed counts
        store_in_var: Whether to store MAE in predicted_adata.var
    
    Returns:
        mae_scores: Combined MAE scores (positive values, higher = worse)
    """
    from pyrovelocity.analysis.analyze import mae_per_gene
    
    # Get data arrays
    observed_u = observed_adata.layers["unspliced"]
    observed_s = observed_adata.layers["spliced"]
    predicted_u = predicted_adata.layers["unspliced"]
    predicted_s = predicted_adata.layers["spliced"]
    
    # Handle sparse matrices
    for arr_name, arr in [("observed_u", observed_u), ("observed_s", observed_s), 
                          ("predicted_u", predicted_u), ("predicted_s", predicted_s)]:
        if hasattr(arr, 'toarray'):
            locals()[arr_name] = arr.toarray()
    
    # Compute MAE for both spliced and unspliced
    mae_u = mae_per_gene(predicted_u, observed_u)
    mae_s = mae_per_gene(predicted_s, observed_s)
    mae_combined = (mae_u + mae_s) / 2
    
    # Convert to positive values (higher = worse performance)
    mae_scores_positive = -mae_combined
    
    if store_in_var:
        predicted_adata.var['mae_combined'] = mae_scores_positive
        predicted_adata.uns['mae_summary'] = {
            'mean_mae': float(mae_scores_positive.mean()),
            'std_mae': float(mae_scores_positive.std()),
            'median_mae': float(np.median(mae_scores_positive))
        }
    
    return mae_scores_positive


@beartype
def _select_genes_by_mae(
    observed_adata: AnnData,
    predicted_adata: AnnData,
    num_genes: int = 6,
    layer: str = "spliced",
    select_highest_error: bool = False
) -> Tuple[List[int], List[str]]:
    """
    Select genes by MAE for temporal dynamics plotting.
    
    Computes MAE using both unspliced and spliced data for comprehensive model evaluation.

    Args:
        observed_adata: AnnData object with observed data
        predicted_adata: AnnData object with predicted data
        num_genes: Number of genes to select
        layer: Layer to use for MAE computation (default: "spliced", kept for backward compatibility)
        select_highest_error: If True, select genes with highest MAE instead of lowest.
                             Genes are always sorted from lowest to highest error (default: False)

    Returns:
        Tuple of (gene_indices, gene_names) for selected genes, sorted from lowest to highest error
    """
    # Check if MAE scores are already computed, if not compute them
    if 'mae_combined' in predicted_adata.var.columns:
        # Use pre-computed positive MAE values
        mae_scores_positive = predicted_adata.var['mae_combined'].values
    else:
        # Compute MAE scores if not available
        mae_scores_positive = compute_and_store_mae(predicted_adata, observed_adata, store_in_var=True)

    # Sort all genes by MAE (lowest error to highest error)
    # Use positive MAE values - sort in ascending order for lowest to highest error
    sorted_indices = np.argsort(mae_scores_positive)

    if select_highest_error:
        # Select genes with highest error (from the end of the sorted list)
        # but maintain the lowest-to-highest error ordering
        selected_indices = sorted_indices[-num_genes:]
    else:
        # Select genes with lowest error (from the beginning of the sorted list)
        selected_indices = sorted_indices[:num_genes]

    # Ensure the selected genes are ordered from lowest to highest error
    # by sorting the selected indices by their MAE scores (ascending order for positive values)
    selected_mae_scores = mae_scores_positive[selected_indices]
    reorder_indices = np.argsort(selected_mae_scores)
    final_gene_indices = selected_indices[reorder_indices]
    final_gene_names = [predicted_adata.var_names[i] for i in final_gene_indices]

    return final_gene_indices.tolist(), final_gene_names


def _compute_adaptive_fold_change_thresholds(
    fold_change: np.ndarray
) -> Dict[str, float]:
    """
    Compute adaptive thresholds for fold-change classification.

    Based on PyroVelocity pattern classification logic:
    - Min threshold: R_on > 2.0 for meaningful activation
    - Transient threshold: R_on > 3.3 for transient patterns
    - Activation threshold: R_on > 7.5 for strong activation patterns

    Args:
        fold_change: Array of fold-change values (R_on)

    Returns:
        Dictionary with threshold values
    """
    # Use pattern classification thresholds as baseline
    min_threshold = 2.0
    transient_threshold = 3.3
    activation_threshold = 7.5

    # Adjust thresholds based on actual data distribution if needed
    # For now, use the established biological thresholds from pattern classification
    # but ensure they're within the data range
    max_fold_change = np.max(fold_change)

    # If data doesn't reach activation threshold, use a percentile-based approach
    if max_fold_change < activation_threshold:
        activation_threshold = np.percentile(fold_change, 90)

    if max_fold_change < transient_threshold:
        transient_threshold = np.percentile(fold_change, 75)

    return {
        'min_threshold': min_threshold,
        'transient_threshold': transient_threshold,
        'activation_threshold': activation_threshold
    }


def _compute_adaptive_timing_thresholds(
    parameters: Dict[str, ArrayLike]
) -> Dict[str, float]:
    """
    Compute thresholds for activation timing classification using independent absolute parameters.

    Based on PyroVelocity mathematical specification:
    - Transient/Sustained boundary: delta_star = 2.0 vs 2.5
    - Early/Late activation: t_on_star = 1.5

    Args:
        parameters: Dictionary of parameter tensors

    Returns:
        Dictionary with threshold values
    """
    # Use mathematical specification thresholds for independent absolute parameters
    return {
        'transient_sustained_boundary': 2.0,  # delta_star threshold
        'early_late_activation': 1.5,         # t_on_star threshold
        'sustained_boundary': 2.5,            # sustained delta_star threshold
        'use_relative': False
    }


@beartype
def plot_expression_validation(
    adata: AnnData,
    check_type: str = "prior",
    figsize: Tuple[Union[int, float], Union[int, float]] = (7.5, 7.5),  # Square aspect ratio with standard width
    save_path: Optional[str] = None,
    file_prefix: str = "",
    default_fontsize: Union[int, float] = 8
) -> plt.Figure:
    """
    Plot expression data validation: counts, relationships, library sizes, ranges.

    Args:
        adata: AnnData object with expression data
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height) - defaults to square aspect ratio
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved figure files
        default_fontsize: Default font size for all text elements

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Count distributions (top-left)
    _plot_count_distributions(adata, axes[0, 0], check_type, default_fontsize)

    # U vs S relationships (top-right)
    _plot_expression_relationships(adata, axes[0, 1], check_type, default_fontsize)

    # Library sizes (bottom-left)
    _plot_library_sizes(adata, axes[1, 0], check_type, default_fontsize)

    # Expression ranges (bottom-right)
    _plot_expression_ranges(adata, axes[1, 1], check_type, default_fontsize)

    plt.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_expression_validation")

    return fig


def _plot_count_distributions(adata: AnnData, ax: plt.Axes, check_type: str, default_fontsize: Union[int, float] = 8) -> None:
    """Plot unspliced and spliced count distributions."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        unspliced = adata.layers['unspliced'].flatten()
        spliced = adata.layers['spliced'].flatten()
        
        # Remove zeros for log scale
        unspliced_nz = unspliced[unspliced > 0]
        spliced_nz = spliced[spliced > 0]
        
        # Use relative frequency for consistency
        ax.hist(np.log1p(unspliced_nz), bins=50, alpha=0.6,
               label='Unspliced', color='red', density=False,
               weights=np.ones(len(unspliced_nz)) / len(unspliced_nz))
        ax.hist(np.log1p(spliced_nz), bins=50, alpha=0.6,
               label='Spliced', color='blue', density=False,
               weights=np.ones(len(spliced_nz)) / len(spliced_nz))

        ax.set_xlabel('log(count + 1)', fontsize=default_fontsize)
        ax.set_ylabel('Relative Frequency', fontsize=default_fontsize)
        ax.set_title(f'{check_type.title()} Count Distributions', fontsize=default_fontsize)
        ax.tick_params(labelsize=default_fontsize * 0.75)
        ax.legend(fontsize=default_fontsize * 0.75)
    else:
        ax.text(0.5, 0.5, 'Count data\nnot available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Count Distributions', fontsize=default_fontsize)
    
    ax.grid(True, alpha=0.3)


def _plot_expression_relationships(adata: AnnData, ax: plt.Axes, check_type: str, default_fontsize: Union[int, float] = 8) -> None:
    """Plot unspliced vs spliced expression relationships."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        # Sample subset for visualization
        n_sample = min(1000, adata.n_obs * adata.n_vars)
        unspliced = adata.layers['unspliced'].flatten()
        spliced = adata.layers['spliced'].flatten()
        
        # Random sample for plotting
        idx = np.random.choice(len(unspliced), n_sample, replace=False)
        u_sample = unspliced[idx]
        s_sample = spliced[idx]
        
        ax.scatter(np.log1p(s_sample), np.log1p(u_sample), 
                  alpha=0.5, s=1, color='purple')
        ax.set_xlabel('log(Spliced + 1)', fontsize=default_fontsize)
        ax.set_ylabel('log(Unspliced + 1)', fontsize=default_fontsize)
        ax.set_title(f'{check_type.title()} U vs S Relationship', fontsize=default_fontsize)
        
        # Add diagonal reference
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='U = S')
        ax.legend(fontsize=default_fontsize * 0.75)
        ax.tick_params(labelsize=default_fontsize * 0.75)
    else:
        ax.text(0.5, 0.5, 'Expression data\nnot available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} U vs S Relationship')
    
    ax.grid(True, alpha=0.3)


def _plot_library_sizes(adata: AnnData, ax: plt.Axes, check_type: str, default_fontsize: Union[int, float] = 8) -> None:
    """Plot library size distributions."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        total_counts = adata.layers['unspliced'].sum(axis=1) + adata.layers['spliced'].sum(axis=1)

        # Use relative frequency for consistency
        ax.hist(total_counts, bins=50, alpha=0.7, color='green', density=False,
               weights=np.ones(len(total_counts)) / len(total_counts))
        ax.axvline(total_counts.mean(), color='red', linestyle='--',
                  label=f'Mean: {total_counts.mean():.0f}')

        ax.set_xlabel('Total Counts per Cell', fontsize=default_fontsize)
        ax.set_ylabel('Relative Frequency', fontsize=default_fontsize)
        ax.set_title(f'{check_type.title()} Library Sizes', fontsize=default_fontsize)
        ax.legend(fontsize=default_fontsize * 0.75)
        ax.tick_params(labelsize=default_fontsize * 0.75)
    else:
        ax.text(0.5, 0.5, 'Count data\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Library Sizes')

    ax.grid(True, alpha=0.3)


def _plot_expression_ranges(adata: AnnData, ax: plt.Axes, check_type: str, default_fontsize: Union[int, float] = 8) -> None:
    """Plot expression range validation."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        # Calculate expression ranges per gene
        u_ranges = np.ptp(adata.layers['unspliced'], axis=0)  # peak-to-peak
        s_ranges = np.ptp(adata.layers['spliced'], axis=0)

        ax.scatter(s_ranges, u_ranges, alpha=0.7, s=5,
                   edgecolors="none",
                   color='orange')
        ax.set_xlabel('Spliced', fontsize=default_fontsize)
        ax.set_ylabel('Unspliced', fontsize=default_fontsize)
        ax.set_title(f'{check_type.title()} Expression Ranges', fontsize=default_fontsize)

        # Add diagonal reference
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='U = S')
        ax.legend(fontsize=default_fontsize * 0.75)
        ax.tick_params(labelsize=default_fontsize * 0.75)
    else:
        ax.text(0.5, 0.5, 'Expression data\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Expression Ranges')

    ax.grid(True, alpha=0.3)


def _plot_phase_portrait(adata: AnnData, ax: plt.Axes, check_type: str, default_fontsize: Union[int, float] = 8) -> None:
    """Plot phase portrait (unspliced vs spliced trajectories)."""
    if 'unspliced' in adata.layers and 'spliced' in adata.layers:
        # Sample genes for visualization
        n_genes_plot = min(3, adata.n_vars)
        gene_indices = np.random.choice(adata.n_vars, n_genes_plot, replace=False)

        colors = sns.color_palette("husl", n_genes_plot)

        for i, gene_idx in enumerate(gene_indices):
            u_gene = adata.layers['unspliced'][:, gene_idx]
            s_gene = adata.layers['spliced'][:, gene_idx]

            ax.scatter(s_gene, u_gene, alpha=0.6, s=5, color=colors[i],
                       edgecolors="none",
                       label=f'Gene {gene_idx}')

        ax.set_xlabel('Spliced', fontsize=default_fontsize)
        ax.set_ylabel('Unspliced', fontsize=default_fontsize)
        ax.set_title(f'{check_type.title()} Phase Portrait', fontsize=default_fontsize)
        # ax.legend(fontsize=6)
        ax.tick_params(labelsize=default_fontsize * 0.75)
    else:
        ax.text(0.5, 0.5, 'Expression data\nnot available',
               ha='center', va='center', transform=ax.transAxes, fontsize=default_fontsize * 0.9)
        ax.set_title(f'{check_type.title()} Phase Portrait', fontsize=default_fontsize)

    ax.grid(True, alpha=0.3)


def _plot_velocity_magnitudes(adata: AnnData, ax: plt.Axes, check_type: str, default_fontsize: Union[int, float] = 8) -> None:
    """Plot RNA velocity magnitude distributions."""
    # Check if velocity has been computed
    velocity_layers = [key for key in adata.layers.keys() if 'velocity' in key.lower()]

    if velocity_layers:
        # Use the first velocity layer found
        velocity = adata.layers[velocity_layers[0]]
        velocity_magnitudes = np.linalg.norm(velocity, axis=1)

        # Use relative frequency for consistency
        ax.hist(velocity_magnitudes, bins=50, alpha=0.7, color='teal', density=False,
               weights=np.ones(len(velocity_magnitudes)) / len(velocity_magnitudes))
        ax.axvline(velocity_magnitudes.mean(), color='red', linestyle='--',
                  label=f'Mean: {velocity_magnitudes.mean():.3f}')

        ax.set_xlabel('Velocity Magnitude', fontsize=default_fontsize)
        ax.set_ylabel('Relative Frequency', fontsize=default_fontsize)
        ax.set_title(f'{check_type.title()} Velocity Magnitudes', fontsize=default_fontsize)
        ax.legend(fontsize=default_fontsize * 0.75)
        ax.tick_params(labelsize=default_fontsize * 0.75)
    else:
        # Compute velocity using the correct piecewise activation model formula
        if 'unspliced' in adata.layers and 'spliced' in adata.layers:
            u = adata.layers['unspliced']
            s = adata.layers['spliced']

            # For the dimensionless piecewise activation model: ds*/dt* = u* - γ*s*
            # We need gamma_star values, but if not available, use a reasonable approximation
            if 'gamma_star' in adata.var:
                gamma_star = adata.var['gamma_star'].values
                # Compute velocity per gene: ds*/dt* = u* - γ*s*
                velocity_per_gene = u - gamma_star[np.newaxis, :] * s
            else:
                # Use a typical gamma_star value (~1.0) as approximation
                gamma_star_approx = 1.0
                velocity_per_gene = u - gamma_star_approx * s

            # Take mean velocity magnitude across genes for each cell
            velocity_magnitudes = np.mean(np.abs(velocity_per_gene), axis=1)

            # Use relative frequency for consistency
            ax.hist(velocity_magnitudes, bins=50, alpha=0.7, color='teal', density=False,
                   weights=np.ones(len(velocity_magnitudes)) / len(velocity_magnitudes))
            ax.axvline(velocity_magnitudes.mean(), color='red', linestyle='--',
                      label=f'Mean: {velocity_magnitudes.mean():.3f}')

            ax.set_xlabel('Velocity Magnitude', fontsize=default_fontsize)
            ax.set_ylabel('Relative Frequency', fontsize=default_fontsize)
            ax.set_title(f'{check_type.title()} Velocity Magnitudes', fontsize=default_fontsize)
            ax.legend(fontsize=default_fontsize * 0.75)
            ax.tick_params(labelsize=default_fontsize * 0.75)
        else:
            ax.text(0.5, 0.5, 'Velocity data\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{check_type.title()} Velocity Magnitudes')

    ax.grid(True, alpha=0.3)


def _plot_fold_change_distribution(
    parameters: Dict[str, ArrayLike],
    ax: plt.Axes,
    check_type: str,
    model: Optional[Any] = None,
    default_fontsize: Union[int, float] = 8
) -> None:
    """Plot fold-change distribution using R_on parameter."""
    from pyrovelocity.plots.parameter_metadata import (
        get_parameter_label,
        infer_component_name_from_parameters,
    )

    # Try to infer component name from all parameters if model not provided
    # Try to infer component name from all parameters for robust metadata lookup
    component_name = infer_component_name_from_parameters(parameters)

    # Use R_on directly (preferred) or fall back to alpha_on/alpha_off ratio
    if 'R_on' in parameters:
        fold_change = convert_to_numpy(parameters['R_on'].flatten())
        param_source = "R_on"
    elif 'alpha_off' in parameters and 'alpha_on' in parameters:
        alpha_off = parameters['alpha_off'].flatten()
        alpha_on = parameters['alpha_on'].flatten()
        fold_change = convert_to_numpy(alpha_on / alpha_off)
        param_source = "alpha_ratio"
    else:
        ax.text(0.5, 0.5, 'Fold-change parameters\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Fold-change Distribution', fontsize=default_fontsize)
        ax.grid(True, alpha=0.3)
        return

    # Compute adaptive thresholds
    thresholds = _compute_adaptive_fold_change_thresholds(fold_change)

    # Use relative frequency for consistency
    ax.hist(fold_change, bins=50, alpha=0.7, color='skyblue', density=False,
           weights=np.ones(len(fold_change)) / len(fold_change))
    ax.axvline(fold_change.mean(), color='red', linestyle='--',
              label=f'Mean: {fold_change.mean():.1f}')
    ax.axvline(thresholds['transient_threshold'], color='orange', linestyle=':',
              label=f'Min threshold: {thresholds["transient_threshold"]:.1f}')
    ax.axvline(thresholds['activation_threshold'], color='green', linestyle=':',
              label=f'Activation threshold: {thresholds["activation_threshold"]:.1f}')

    # Get parameter label using new metadata system
    if param_source == "R_on":
        param_label = get_parameter_label(
            param_name="R_on",
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )
        xlabel = f'Fold-change ({param_label})'
    else:
        # Legacy fallback for alpha_on/alpha_off ratio
        alpha_on_label = get_parameter_label(
            param_name="alpha_on",
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )
        alpha_off_label = get_parameter_label(
            param_name="alpha_off",
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )
        xlabel = f'Fold-change ({alpha_on_label} / {alpha_off_label})'

    ax.set_xlabel(xlabel, fontsize=default_fontsize)
    ax.set_ylabel('Relative Frequency', fontsize=default_fontsize)
    ax.set_title(f'{check_type.title()} Fold-change Distribution', fontsize=default_fontsize)
    ax.tick_params(labelsize=default_fontsize * 0.75)  # Reduce tick label size
    ax.legend(fontsize=4)  # Reduced legend font size to prevent overlap
    ax.set_xlim(0, min(100, fold_change.max()))
    ax.grid(True, alpha=0.3)


def _plot_activation_timing(
    parameters: Dict[str, ArrayLike],
    ax: plt.Axes,
    check_type: str,
    model: Optional[Any] = None,
    default_fontsize: Union[int, float] = 8,
    unprocessed_parameters: Optional[Dict[str, ArrayLike]] = None
) -> None:
    """Plot activation timing and duration distributions."""
    from pyrovelocity.plots.parameter_metadata import (
        get_parameter_label,
        infer_component_name_from_parameters,
    )

    # Try to infer component name from all parameters if model not provided
    # Try to infer component name from all parameters for robust metadata lookup
    component_name = infer_component_name_from_parameters(parameters)

    # Use independent absolute parameters only
    # CRITICAL FIX: Use unprocessed parameters when available for proper scatter plots
    source_params = unprocessed_parameters if unprocessed_parameters is not None else parameters
    
    if 't_on_star' in source_params and 'delta_star' in source_params:
        t_on_raw = convert_to_numpy(source_params['t_on_star'])
        delta_raw = convert_to_numpy(source_params['delta_star'])
        
        # Handle different parameter structures:
        # For posterior samples: shape is (n_samples, n_genes) → compute gene-level summaries  
        # For prior samples: shape is (n_genes,) → use as-is
        
        if t_on_raw.ndim == 2:  # Posterior samples: (n_samples, n_genes)
            # Compute posterior mean for each gene (one point per gene)
            t_on = np.mean(t_on_raw, axis=0)  # Shape: (n_genes,)
            delta = np.mean(delta_raw, axis=0)  # Shape: (n_genes,)
        else:  # Prior samples or already summarized: (n_genes,)
            t_on = t_on_raw.flatten()
            delta = delta_raw.flatten()

        # Get parameter labels
        t_on_label = get_parameter_label(
            param_name="t_on_star",
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )
        delta_label = get_parameter_label(
            param_name="delta_star",
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )
    else:
        ax.text(0.5, 0.5, 'Timing parameters\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Activation Timing', fontsize=default_fontsize)
        ax.grid(True, alpha=0.3)
        return

    # Compute adaptive thresholds
    thresholds = _compute_adaptive_timing_thresholds(parameters)

    ax.scatter(t_on, delta, alpha=0.6, s=5,
               edgecolors="none",
               color='purple')

    ax.set_xlabel(f'Activation Onset ({t_on_label})', fontsize=default_fontsize)
    ax.set_ylabel(f'Activation Duration ({delta_label})', fontsize=default_fontsize)
    ax.set_title(f'{check_type.title()} Activation Timing', fontsize=default_fontsize)
    ax.tick_params(labelsize=default_fontsize * 0.75)

    # Add adaptive pattern boundaries
    ax.axhline(thresholds['transient_sustained_boundary'], color='red', linestyle='--', alpha=0.7,
              label='Transient/Sustained boundary')
    ax.axvline(thresholds['early_late_activation'], color='orange', linestyle='--', alpha=0.7,
              label='Early/Late activation')
    ax.legend(fontsize=4)  # Reduced legend font size to prevent overlap

    ax.grid(True, alpha=0.3)