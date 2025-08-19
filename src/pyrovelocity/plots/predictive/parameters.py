"""
Parameter plotting functions for predictive checks.

This module contains functions for plotting parameter distributions,
relationships, and recovery validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from matplotlib.gridspec import GridSpec
from numpy.typing import ArrayLike
from scipy.stats import linregress, pearsonr

from pyrovelocity.analysis.analyze import mae_per_gene
from pyrovelocity.plots.parameter_metadata import (
    get_parameter_label,
    get_model_parameter_metadata,
    infer_component_name_from_parameters,
)
from pyrovelocity.plots.tensor_utils import (
    convert_to_numpy,
    ensure_numpy_parameters,
)
from .core import compute_and_store_mae


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


def _get_available_parameters(
    parameters: Dict[str, ArrayLike],
    exclude_params: Optional[List[str]] = None,
    model: Optional[Any] = None
) -> List[str]:
    """
    Get list of available parameters, optionally excluding specified ones.

    Parameters are ordered by plot_order from metadata if available, otherwise alphabetically.

    Args:
        parameters: Dictionary of parameter tensors
        exclude_params: Optional list of parameter names to exclude
        model: Optional PyroVelocity model instance for parameter metadata

    Returns:
        List of available parameter names in proper order
    """
    # Default exclusions for deprecated parameters in corrected parameterization
    default_exclude = ['alpha_off', 'alpha_on']  # alpha_off fixed at 1.0, alpha_on computed from R_on
    exclude_params = (exclude_params or []) + default_exclude

    available_params = [name for name in parameters.keys() if name not in exclude_params]

    # Try to order by metadata plot_order if available
    try:
        from pyrovelocity.plots.parameter_metadata import (
            get_model_parameter_metadata,
        )

        if model is not None:
            metadata = get_model_parameter_metadata(model)
            if metadata is not None:
                # Create ordering based on plot_order
                param_order = {}
                for param_name in available_params:
                    if param_name in metadata.parameters:
                        plot_order = metadata.parameters[param_name].plot_order
                        param_order[param_name] = plot_order if plot_order is not None else 999
                    else:
                        param_order[param_name] = 999  # Put unordered params at end

                # Sort by plot_order, then alphabetically for ties
                available_params.sort(key=lambda x: (param_order.get(x, 999), x))
                return available_params
    except Exception:
        pass  # Fall back to alphabetical ordering

    # Fall back to alphabetical ordering
    return sorted(available_params)


def _plot_temporal_coordinate_distribution(
    parameters: Dict[str, ArrayLike],
    ax: plt.Axes,
    check_type: str,
    model: Optional[Any] = None,
    default_fontsize: Union[int, float] = 8
) -> None:
    """Plot temporal coordinate distribution (t_star)."""
    from pyrovelocity.plots.parameter_metadata import (
        get_parameter_label,
        infer_component_name_from_parameters,
    )

    # Try to infer component name from all parameters for robust metadata lookup
    component_name = infer_component_name_from_parameters(parameters)

    if 't_star' in parameters:
        t_star = convert_to_numpy(parameters['t_star'].flatten())

        # Plot histogram of temporal coordinates
        ax.hist(t_star, bins=50, alpha=0.7, color='purple', density=True)

        # Get parameter label using metadata system
        t_star_label = get_parameter_label(
            param_name="t_star",
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )

        ax.set_xlabel(f'Temporal Coordinate ({t_star_label})', fontsize=default_fontsize)
        ax.set_ylabel('Density', fontsize=default_fontsize)
        ax.set_title(f'{check_type.title()} Temporal Coordinates', fontsize=default_fontsize)
        ax.tick_params(labelsize=default_fontsize * 0.75)

        # Add summary statistics
        t_star_mean = np.mean(t_star)
        t_star_median = np.median(t_star)
        ax.axvline(t_star_mean, color='red', linestyle='--', alpha=0.7,
                  label=f'Mean: {t_star_mean:.2f}')
        ax.axvline(t_star_median, color='orange', linestyle='--', alpha=0.7,
                  label=f'Median: {t_star_median:.2f}')
        ax.legend(fontsize=default_fontsize * 0.8)
    else:
        ax.text(0.5, 0.5, 'Temporal coordinates\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Temporal Coordinates', fontsize=default_fontsize)

    ax.grid(True, alpha=0.3)


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
    from pyrovelocity.analysis.analyze import mae_per_gene
    
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


@beartype
def plot_parameter_marginals(
    parameters: Dict[str, ArrayLike],
    check_type: str = "prior",
    exclude_params: Optional[List[str]] = None,
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    save_path: Optional[str] = None,
    file_prefix: str = "",
    model: Optional[Any] = None,
    default_fontsize: Union[int, float] = 8,
    true_parameters_adata: Optional[AnnData] = None
) -> plt.Figure:
    """
    Plot individual histograms for all parameter marginal distributions.

    Args:
        parameters: Dictionary of parameter tensors (PyTorch, JAX, or NumPy)
        check_type: Type of check ("prior" or "posterior")
        exclude_params: Optional list of parameter names to exclude
        figsize: Optional figure size (auto-calculated if None)
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        model: Optional PyroVelocity model instance for parameter metadata
        default_fontsize: Default font size for all text elements
        true_parameters_adata: Optional AnnData object containing true parameters
                              in adata.uns['true_parameters'] for validation

    Returns:
        matplotlib Figure object
    """
    # Convert all parameters to NumPy arrays for framework-agnostic plotting
    numpy_parameters = ensure_numpy_parameters(parameters)
    
    available_params = _get_available_parameters(numpy_parameters, exclude_params, model)

    if not available_params:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No parameters available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Parameter Marginals')
        return fig

    # Extract true parameters if provided for validation
    true_parameters = {}
    global_true_params = ['T_M_star']  # Global parameters that should appear on global marginals

    if true_parameters_adata is not None and 'true_parameters' in true_parameters_adata.uns:
        true_params_dict = true_parameters_adata.uns['true_parameters']
        for param_name in available_params:
            if param_name in true_params_dict and param_name in global_true_params:
                true_value = true_params_dict[param_name]
                # Convert to tensor if needed
                if not isinstance(true_value, np.ndarray):
                    true_value = convert_to_numpy(true_value)
                # Only store scalar global parameters
                if true_value.size == 1:
                    true_parameters[param_name] = true_value
        if true_parameters:
            print(f"Found {len(true_parameters)} global true parameters for validation: {list(true_parameters.keys())}")

    # Auto-calculate figure size based on number of parameters
    # Use 7.5" width to fit 8.5x11" page with 0.5" margins
    n_params = len(available_params)
    cols = min(4, n_params)
    rows = (n_params + cols - 1) // cols

    if figsize is None:
        width = 7.5  # Standard width for 8.5x11" with margins
        height = width * (rows / cols) * 0.75  # Maintain reasonable aspect ratio
        figsize = (width, height)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_params == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    colors = sns.color_palette("husl", n_params)

    for i, param_name in enumerate(available_params):
        ax = axes[i]
        values = numpy_parameters[param_name].flatten()

        # Use relative frequency instead of density for consistent y-axis interpretation
        counts, bins, _ = ax.hist(values, bins=30, alpha=0.7, color=colors[i], density=False)

        # Normalize to relative frequency (0-1 scale)
        total_count = len(values)
        relative_freq = counts / total_count

        # Clear and replot with relative frequency
        ax.clear()
        ax.bar(bins[:-1], relative_freq, width=np.diff(bins), alpha=0.7, color=colors[i],
               align='edge', edgecolor='none')

        # Get parameter label using the new metadata system
        from pyrovelocity.plots.parameter_metadata import (
            get_parameter_label,
            infer_component_name_from_parameters,
        )

        # Try to infer component name from all parameters for robust metadata lookup
        component_name = infer_component_name_from_parameters(numpy_parameters)

        # Get short label for x-axis and display name for title
        short_label = get_parameter_label(
            param_name=param_name,
            label_type="short",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )
        display_name = get_parameter_label(
            param_name=param_name,
            label_type="display",
            model=model,
            component_name=component_name,
            fallback_to_legacy=True
        )

        ax.set_xlabel(short_label, fontsize=default_fontsize * 0.9)
        ax.set_ylabel('Freq.', fontsize=default_fontsize * 0.9)
        ax.set_title(display_name, fontsize=default_fontsize)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(relative_freq) * 1.1)  # Add some headroom
        ax.tick_params(labelsize=default_fontsize * 0.75)  # Reduce tick label size

        # Add median line in light gray instead of mean line with legend
        median_val = np.median(values)
        median_line = ax.axvline(median_val, color='lightgray', linestyle='--', alpha=0.7)

        # Add vertical line for true value if available (for global parameters)
        true_line = None
        if param_name in true_parameters:
            true_val = float(true_parameters[param_name].item())

            # Extend x-axis range if true value is outside current range
            current_xlim = ax.get_xlim()
            if true_val < current_xlim[0] or true_val > current_xlim[1]:
                # Extend range to include true value with some padding
                range_padding = (current_xlim[1] - current_xlim[0]) * 0.1
                new_xlim = (
                    min(current_xlim[0], true_val - range_padding),
                    max(current_xlim[1], true_val + range_padding)
                )
                ax.set_xlim(new_xlim)

            true_line = ax.axvline(true_val, color='green', linestyle='-', alpha=0.8, linewidth=1.5)

        # Add legend for the first subplot if we have true values
        if i == 0 and true_line is not None:
            legend_elements = [
                plt.Line2D([0], [0], color='lightgray', linestyle='--', label='Posterior Median'),
                plt.Line2D([0], [0], color='green', linestyle='-', label='True Value')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=default_fontsize * 0.8)

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_parameter_marginals")

    return fig


@beartype
def plot_parameter_relationships(
    parameters: Dict[str, ArrayLike],
    check_type: str = "prior",
    figsize: Tuple[Union[int, float], Union[int, float]] = (7.5, 2.5),  # Standard width, appropriate height
    save_path: Optional[str] = None,
    file_prefix: str = "",
    model: Optional[Any] = None,
    default_fontsize: Union[int, float] = 8
) -> plt.Figure:
    """
    Plot parameter relationships: correlations, fold-change, and timing.

    Args:
        parameters: Dictionary of parameter arrays (framework-agnostic)
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height)
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        model: Optional PyroVelocity model instance for parameter metadata

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy for plotting
    parameters = ensure_numpy_parameters(parameters)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Temporal coordinate distribution
    _plot_temporal_coordinate_distribution(parameters, axes[0], check_type, model=model, default_fontsize=default_fontsize)

    # Fold-change distribution
    _plot_fold_change_distribution(parameters, axes[1], check_type, model=model, default_fontsize=default_fontsize)

    # Activation timing
    _plot_activation_timing(parameters, axes[2], check_type, model=model, default_fontsize=default_fontsize)

    plt.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_parameter_relationships")

    return fig


@beartype
def plot_parameter_marginals_by_gene(
    posterior_parameters: Dict[str, ArrayLike],
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    save_path: Optional[str] = None,
    num_genes: int = 6,
    gene_selection_method: str = "mae",
    select_highest_error: bool = False,
    parameters_to_show: List[str] = ["R_on", "gamma_star", "t_on_star", "delta_star"],
    default_fontsize: int = 7,
    file_prefix: str = "",
    observed_adata: Optional[AnnData] = None,
    predicted_adata: Optional[AnnData] = None,
    model: Optional[Any] = None,
    check_type: str = "posterior",
    true_parameters_adata: Optional[AnnData] = None
) -> plt.Figure:
    """
    Plot marginal histograms of gene-specific parameter posterior samples.

    This function creates a plot showing the posterior distributions of key
    gene-specific parameters for the same genes selected in temporal dynamics plots.
    Each row corresponds to a gene, and each column shows the marginal distribution
    of a different parameter. When true_parameters_adata is provided, green solid
    vertical lines show the true parameter values for parameter recovery validation.

    Args:
        posterior_parameters: Dictionary of posterior parameter samples
        figsize: Figure size (width, height). If None, auto-calculated
        save_path: Optional directory path to save figures
        num_genes: Number of genes to show
        gene_selection_method: Method for gene selection ("mae" or "random")
        select_highest_error: If True, select genes with highest MAE instead of lowest
        parameters_to_show: List of parameter names to display as columns
        default_fontsize: Default font size for all text elements
        file_prefix: Prefix for saved file names
        observed_adata: AnnData object with observed data for gene selection
        predicted_adata: AnnData object with predicted data for gene selection
        model: Optional PyroVelocity model instance for parameter metadata
        check_type: Type of check ("prior" or "posterior")
        true_parameters_adata: Optional AnnData object containing true parameters
                              in adata.uns['true_parameters'] for validation

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_parameter_marginals_by_gene(
        ...     posterior_parameters=params,
        ...     observed_adata=observed_data,
        ...     predicted_adata=predicted_data,
        ...     num_genes=6,
        ...     save_path="reports/docs/posterior_predictive",
        ...     file_prefix="05",
        ...     true_parameters_adata=prior_predictive_adata  # For validation
        ... ) # xdoctest: +SKIP
    """
    from matplotlib.gridspec import GridSpec

    from pyrovelocity.plots.parameter_metadata import get_parameter_label

    # Convert to numpy for plotting
    posterior_parameters = ensure_numpy_parameters(posterior_parameters)

    # Determine gene selection - use same logic as temporal dynamics
    if gene_selection_method == "mae" and observed_adata is not None and predicted_adata is not None:
        gene_indices, gene_names = _select_genes_by_mae(
            observed_adata=observed_adata,
            predicted_adata=predicted_adata,
            num_genes=num_genes,
            select_highest_error=select_highest_error
        )
        print(f"Selected {len(gene_names)} genes with {'highest' if select_highest_error else 'lowest'} MAE (sorted lowest to highest error): {gene_names}")
    else:
        # Fallback to first N genes if MAE selection not possible
        available_genes = min(num_genes, observed_adata.n_vars if observed_adata is not None else 100)
        gene_indices = list(range(available_genes))
        gene_names = [f"gene_{i}" for i in gene_indices]
        print(f"Using first {available_genes} genes (MAE selection not available)")

    available_genes = len(gene_names)

    # Use dynamic parameter detection if no specific parameters requested or if none are available
    if parameters_to_show == ["R_on", "gamma_star", "t_on_star", "delta_star"]:  # Default hardcoded list
        # Try to infer component name and get parameters dynamically
        from pyrovelocity.plots.parameter_metadata import infer_component_name_from_parameters
        from pyrovelocity.models.metadata import get_parameter_metadata
        
        component_name = infer_component_name_from_parameters(posterior_parameters)
        if component_name is not None:
            try:
                metadata = get_parameter_metadata(component_name)
                # Get parameters sorted by plot_order
                available_params = [
                    param_name for param_name in metadata.parameters.keys()
                    if param_name in posterior_parameters
                ]
                available_params.sort(key=lambda x: metadata.parameters[x].plot_order or 999)
                print(f"Using dynamic parameter detection for component '{component_name}': {available_params}")
            except KeyError:
                # Fall back to available parameters if metadata not found
                available_params = [p for p in parameters_to_show if p in posterior_parameters]
        else:
            # Fall back to manual filtering if component not detected
            available_params = [p for p in parameters_to_show if p in posterior_parameters]
    else:
        # Use user-specified parameters
        available_params = [p for p in parameters_to_show if p in posterior_parameters]
    
    if not available_params:
        raise ValueError(f"None of the requested parameters {parameters_to_show} found in posterior_parameters. Available: {list(posterior_parameters.keys())}")

    num_params = len(available_params)

    # Extract true parameters if provided for validation
    true_parameters = {}
    if true_parameters_adata is not None and 'true_parameters' in true_parameters_adata.uns:
        true_params_dict = true_parameters_adata.uns['true_parameters']
        for param_name in available_params:
            if param_name in true_params_dict:
                true_value = true_params_dict[param_name]
                # Convert to tensor if needed
                if not isinstance(true_value, np.ndarray):
                    true_value = convert_to_numpy(true_value)
                true_parameters[param_name] = true_value
        print(f"Found {len(true_parameters)} true parameters for validation: {list(true_parameters.keys())}")

    # Calculate figure size - add gene label column like temporal dynamics plot
    horizontal_panels = num_params + 1  # gene_label + parameter columns

    if figsize is None:
        width = 2.0 * num_params + 1.0  # 2 inches per parameter column + 1 for gene label
        height = 0.8 * available_genes + 0.5  # 0.8 inches per gene row + title space
        figsize = (width, height)

    # Create figure and gridspec with gene label column
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=available_genes + 1,  # Add extra row for titles
        ncols=horizontal_panels,
        figure=fig,
        width_ratios=[0.21] + [1.0] * num_params,  # Gene label column + parameter columns
        height_ratios=[0.15] + [1] * available_genes,  # Small title row + gene rows
        hspace=0.3,  # Vertical spacing between gene rows
        wspace=0.3,  # Horizontal spacing between parameter columns
    )

    # Add column titles (skip gene label column)
    for col, param_name in enumerate(available_params):
        ax_title = fig.add_subplot(gs[0, col + 1])  # +1 to skip gene label column
        ax_title.axis('off')

        # Get parameter label using metadata system
        param_label = get_parameter_label(
            param_name=param_name,
            label_type="display",
            model=model,
            fallback_to_legacy=True
        )

        ax_title.text(0.5, 0.5, param_label,
                     ha='center', va='center',
                     fontsize=default_fontsize + 1,
                     weight='bold',
                     transform=ax_title.transAxes)

    # Extract parameter samples for each gene
    # Assume parameters have shape [num_samples, num_genes] or [num_samples * num_genes]
    param_samples_by_gene = {}
    param_global_ranges = {}  # Store global min/max for each parameter

    for param_name in available_params:
        param_tensor = posterior_parameters[param_name]

        # Handle different tensor shapes
        if param_tensor.ndim == 1:
            # Flattened: [num_samples * num_genes]
            # Need to determine num_samples to reshape properly
            total_length = len(param_tensor)
            # Assume we can infer from other parameters or use reasonable default
            num_samples = 30  # Default from the script
            if total_length % num_samples == 0:
                num_genes_in_param = total_length // num_samples
                param_reshaped = param_tensor.reshape(num_samples, num_genes_in_param)
            else:
                # Fallback: treat as single sample per gene
                param_reshaped = np.expand_dims(param_tensor, axis=0)  # [1, num_genes]
        elif param_tensor.ndim == 2:
            # Already shaped: [num_samples, num_genes]
            param_reshaped = param_tensor
        else:
            # Higher dimensions: flatten and reshape
            param_reshaped = param_tensor.reshape(-1, param_tensor.shape[-1])

        param_samples_by_gene[param_name] = param_reshaped

        # Compute global range for this parameter across all genes
        param_min = float(param_reshaped.min())
        param_max = float(param_reshaped.max())

        # Include true parameter values in range calculation if available
        if param_name in true_parameters:
            true_param = true_parameters[param_name]
            if true_param.size == 1:
                # Scalar parameter - single value for all genes
                true_val = float(true_param.item())
                param_min = min(param_min, true_val)
                param_max = max(param_max, true_val)
            else:
                # Array parameter - gene-specific values
                true_vals = true_param.flatten()
                param_min = min(param_min, float(true_vals.min()))
                param_max = max(param_max, float(true_vals.max()))

        param_global_ranges[param_name] = {
            'min': param_min,
            'max': param_max
        }

    # Calculate global parameter ranges for consistent x-axis scaling
    param_ranges = {}
    for param_name in available_params:
        param_samples = param_samples_by_gene[param_name]
        all_values = convert_to_numpy(param_samples).flatten()
        # Use 5th and 95th percentiles to avoid extreme outliers
        param_ranges[param_name] = (np.percentile(all_values, 5), np.percentile(all_values, 95))

    # Plot histograms for each gene and parameter
    for row, (gene_idx, gene_name) in enumerate(zip(gene_indices, gene_names)):
        # Create gene label axis (first column)
        gene_ax = fig.add_subplot(gs[row + 1, 0])  # +1 to account for title row
        gene_ax.axis('off')
        gene_ax.text(0.0, 0.5, gene_name[:7],
                    transform=gene_ax.transAxes,
                    rotation=0, va='center', ha='center',
                    fontsize=default_fontsize, weight='normal')

        for col, param_name in enumerate(available_params):
            ax = fig.add_subplot(gs[row + 1, col + 1])  # +1 for title row, +1 for gene label column

            # Extract samples for this gene and parameter
            param_samples = param_samples_by_gene[param_name]

            # Handle gene indexing
            if gene_idx < param_samples.shape[1]:
                gene_param_samples = convert_to_numpy(param_samples[:, gene_idx])
            else:
                # Gene index out of range, skip this plot
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Set consistent x-axis range for this parameter using global range
            param_range = param_global_ranges[param_name]
            range_padding = (param_range['max'] - param_range['min']) * 0.05  # 5% padding
            x_min = param_range['min'] - range_padding
            x_max = param_range['max'] + range_padding
            ax.set_xlim(x_min, x_max)

            # Create histogram with fixed range and matching marginal histogram style
            ax.hist(gene_param_samples, bins=20, alpha=0.6, color='steelblue',
                   density=True, edgecolor='none', range=(x_min, x_max))

            # Add vertical line for median
            median_val = np.median(gene_param_samples)
            median_line = ax.axvline(median_val, color='red', linestyle='--', alpha=0.8, linewidth=1)

            # Add vertical line for true value if available
            true_line = None
            if param_name in true_parameters:
                true_param = true_parameters[param_name]
                if true_param.size == 1:
                    # Scalar parameter - same value for all genes
                    true_val = float(true_param.item())
                else:
                    # Array parameter - gene-specific value
                    # Fix: Use flattened tensor for proper indexing
                    true_param_flat = true_param.flatten()
                    if gene_idx < len(true_param_flat):
                        true_val = float(true_param_flat[gene_idx])
                    else:
                        true_val = None

                if true_val is not None:
                    true_line = ax.axvline(true_val, color='green', linestyle='-', alpha=0.8, linewidth=1)

            # Add legend only for the first subplot (top-left)
            if row == 0 and col == 0 and (median_line is not None or true_line is not None):
                legend_elements = []
                if median_line is not None:
                    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', label='Posterior Median'))
                if true_line is not None:
                    legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='-', label='True Value'))

                if legend_elements:
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=default_fontsize * 0.7)

            # Formatting
            ax.tick_params(labelsize=default_fontsize * 0.8)
            ax.grid(True, alpha=0.3)

            # Add x-axis labels only on bottom row using parameter metadata
            if row == available_genes - 1:
                # Get parameter display label using metadata system
                param_display_label = get_parameter_label(
                    param_name=param_name,
                    label_type="display",
                    model=model,
                    fallback_to_legacy=True
                )
                ax.set_xlabel(param_display_label, fontsize=default_fontsize)
            else:
                ax.set_xlabel('')

            # Remove y-axis labels for cleaner look
            ax.set_ylabel('')
            ax.set_yticklabels([])

    # Save figure if path provided
    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_parameter_marginals_by_gene")

    return fig


@beartype
def plot_parameter_recovery_correlation(
    posterior_parameters: Dict[str, ArrayLike],
    true_parameters_adata: AnnData,
    parameters_to_validate: List[str] = ["R_on", "gamma_star", "t_on_star", "delta_star"],
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    save_path: Optional[str] = None,
    file_prefix: str = "",
    model: Optional[Any] = None,
    default_fontsize: int = 8,
    summary_statistic: str = "median",
    observed_adata: Optional[AnnData] = None,
    color_by_spliced_count: bool = True,
    spliced_count_statistic: str = "median",
    predicted_adata: Optional[AnnData] = None,
    color_by_mae: bool = False
) -> Tuple[plt.Figure, Dict[str, Dict[str, float]]]:
    """
    Plot parameter recovery correlation analysis comparing posterior estimates to true values.

    Creates scatter plots showing true parameter values (x-axis) vs posterior estimates (y-axis)
    with correlation metrics, perfect recovery line (y=x), and best-fit line.

    Args:
        posterior_parameters: Dictionary of posterior parameter samples with shape [num_samples, num_genes]
        true_parameters_adata: AnnData object containing true parameters in adata.uns['true_parameters']
        parameters_to_validate: List of parameter names to include in correlation analysis
        figsize: Optional figure size (auto-calculated if None)
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        model: Optional PyroVelocity model instance for parameter metadata
        default_fontsize: Default font size for all text elements
        summary_statistic: Statistic to compute from posterior samples ("median" or "mean")
        observed_adata: Optional AnnData object with observed data for spliced count coloring
        color_by_spliced_count: Whether to color points by spliced count statistics (default: True)
        spliced_count_statistic: Statistic to use for spliced count coloring ("median" or "max")
        predicted_adata: Optional AnnData object with predicted data for MAE coloring
        color_by_mae: Whether to color points by MAE values (overrides color_by_spliced_count)

    Returns:
        Tuple of (matplotlib Figure object, recovery metrics dictionary)

    Recovery metrics dictionary structure:
        {
            'parameter_name': {
                'pearson_r': float,
                'pearson_p': float,
                'r_squared': float,
                'slope': float,
                'intercept': float,
                'n_genes': int
            },
            'summary': {
                'mean_pearson_r': float,
                'mean_r_squared': float,
                'overall_recovery_quality': str
            }
        }

    Example:
        >>> fig, metrics = plot_parameter_recovery_correlation(
        ...     posterior_parameters=posterior_samples,
        ...     true_parameters_adata=prior_predictive_adata,
        ...     parameters_to_validate=["R_on", "gamma_star", "t_on_star", "delta_star"],
        ...     save_path="reports/docs/posterior_predictive",
        ...     file_prefix="06"
        ... ) # xdoctest: +SKIP
        >>> print(f"Mean correlation: {metrics['summary']['mean_pearson_r']:.3f}") # xdoctest: +SKIP
    """
    from pyrovelocity.plots.parameter_metadata import get_parameter_label

    # Convert to numpy for plotting
    posterior_parameters = ensure_numpy_parameters(posterior_parameters)

    # Validate inputs
    if 'true_parameters' not in true_parameters_adata.uns:
        raise ValueError("true_parameters_adata must contain 'true_parameters' in adata.uns")

    true_params_dict = true_parameters_adata.uns['true_parameters']

    # Use dynamic parameter detection if default hardcoded list is used
    if parameters_to_validate == ["R_on", "gamma_star", "t_on_star", "delta_star"]:  # Default hardcoded list
        # Try to infer component name and get parameters dynamically
        from pyrovelocity.plots.parameter_metadata import infer_component_name_from_parameters
        from pyrovelocity.models.metadata import get_parameter_metadata
        
        component_name = infer_component_name_from_parameters(posterior_parameters)
        if component_name is not None:
            try:
                metadata = get_parameter_metadata(component_name)
                # Get parameters sorted by plot_order that are available in both posterior and true parameters
                available_params = [
                    param_name for param_name in metadata.parameters.keys()
                    if param_name in posterior_parameters and param_name in true_params_dict
                ]
                available_params.sort(key=lambda x: metadata.parameters[x].plot_order or 999)
                print(f"Using dynamic parameter detection for correlation analysis with component '{component_name}': {available_params}")
            except KeyError:
                # Fall back to manual filtering if metadata not found
                available_params = []
                for param_name in parameters_to_validate:
                    if param_name in posterior_parameters and param_name in true_params_dict:
                        available_params.append(param_name)
                    else:
                        print(f"Warning: Parameter '{param_name}' not found in both posterior and true parameters")
        else:
            # Fall back to manual filtering if component not detected
            available_params = []
            for param_name in parameters_to_validate:
                if param_name in posterior_parameters and param_name in true_params_dict:
                    available_params.append(param_name)
                else:
                    print(f"Warning: Parameter '{param_name}' not found in both posterior and true parameters")
    else:
        # Use user-specified parameters
        available_params = []
        for param_name in parameters_to_validate:
            if param_name in posterior_parameters and param_name in true_params_dict:
                available_params.append(param_name)
            else:
                print(f"Warning: Parameter '{param_name}' not found in both posterior and true parameters")

    if not available_params:
        raise ValueError("No valid parameters found for correlation analysis")

    # Determine color mapping source
    color_values = None
    color_label = None
    
    if color_by_mae and predicted_adata is not None:
        # Use MAE for coloring (highest priority)
        try:
            if 'mae_combined' in predicted_adata.var.columns:
                color_values = predicted_adata.var['mae_combined'].values
                color_label = 'Mean Absolute Error'
            else:
                # Compute MAE if not available
                if observed_adata is not None:
                    mae_scores = compute_and_store_mae(predicted_adata, observed_adata, store_in_var=True)
                    color_values = mae_scores
                    color_label = 'Mean Absolute Error'
                else:
                    print("Warning: Cannot compute MAE without observed_adata")
                    color_by_mae = False
        except Exception as e:
            print(f"Warning: Could not compute MAE for coloring: {e}")
            color_by_mae = False
    
    # Fall back to spliced count coloring if MAE not available
    if not color_by_mae and color_by_spliced_count and observed_adata is not None:
        try:
            spliced_counts = observed_adata.layers['spliced']
            if hasattr(spliced_counts, 'toarray'):
                spliced_counts = spliced_counts.toarray()
            
            if spliced_count_statistic == "median":
                color_values = np.median(spliced_counts, axis=0)
                color_label = 'Median Spliced Count'
            elif spliced_count_statistic == "max":
                color_values = np.max(spliced_counts, axis=0)
                color_label = 'Maximum Spliced Count'
            else:
                print(f"Warning: Unknown spliced_count_statistic '{spliced_count_statistic}', using median")
                color_values = np.median(spliced_counts, axis=0)
                color_label = 'Median Spliced Count'
        except Exception as e:
            print(f"Warning: Could not compute spliced count statistics: {e}")
            color_by_spliced_count = False

    # Parameter analysis info available in returned metrics dictionary

    # Calculate figure size
    n_params = len(available_params)
    cols = min(4, n_params)
    rows = (n_params + cols - 1) // cols

    if figsize is None:
        width = 2.5 * cols  # 2.5 inches per subplot
        height = 2.5 * rows
        figsize = (width, height)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_params == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Initialize recovery metrics
    recovery_metrics = {}

    for i, param_name in enumerate(available_params):
        ax = axes[i]

        # Extract true parameter values
        true_param = true_params_dict[param_name]
        if not isinstance(true_param, np.ndarray):
            true_param = convert_to_numpy(true_param)

        # Handle different tensor shapes and flatten properly
        true_param_flat = true_param.flatten()

        # Extract posterior parameter samples
        posterior_param = posterior_parameters[param_name]

        # Handle different posterior tensor shapes
        if posterior_param.ndim == 1:
            # Flattened: [num_samples * num_genes] - need to reshape
            total_length = len(posterior_param)
            num_genes = len(true_param_flat)
            if total_length % num_genes == 0:
                num_samples = total_length // num_genes
                posterior_reshaped = posterior_param.reshape(num_samples, num_genes)
            else:
                # Fallback: treat as single sample per gene
                posterior_reshaped = np.expand_dims(posterior_param, axis=0)
        elif posterior_param.ndim == 2:
            # Already shaped: [num_samples, num_genes]
            posterior_reshaped = posterior_param
        else:
            # Higher dimensions: flatten and reshape
            posterior_reshaped = posterior_param.reshape(-1, posterior_param.shape[-1])

        # Convert to numpy and compute summary statistic and standard deviation across samples
        posterior_np = convert_to_numpy(posterior_reshaped)
        if summary_statistic == "median":
            posterior_summary = np.median(posterior_np, axis=0)
            # For median, use MAD (median absolute deviation) scaled to approximate std
            mad = np.median(np.abs(posterior_np - posterior_summary[np.newaxis, :]), axis=0)
            posterior_std = 1.4826 * mad  # Scale factor to approximate std from MAD
        elif summary_statistic == "mean":
            posterior_summary = np.mean(posterior_np, axis=0)
            posterior_std = np.std(posterior_np, axis=0)
        else:
            raise ValueError(f"Unknown summary_statistic: {summary_statistic}")

        # Ensure we have the same number of genes
        n_genes = min(len(true_param_flat), len(posterior_summary))
        true_values = convert_to_numpy(true_param_flat[:n_genes])
        estimated_values = convert_to_numpy(posterior_summary[:n_genes])
        estimated_std = convert_to_numpy(posterior_std[:n_genes])

        # Compute correlation metrics
        try:
            pearson_r, pearson_p = pearsonr(true_values, estimated_values)
            slope, intercept, r_value, p_value, std_err = linregress(true_values, estimated_values)
            r_squared = r_value ** 2
        except Exception as e:
            print(f"Warning: Could not compute correlation for {param_name}: {e}")
            pearson_r = pearson_p = r_squared = slope = intercept = np.nan

        # Store metrics
        recovery_metrics[param_name] = {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'r_squared': float(r_squared),
            'slope': float(slope),
            'intercept': float(intercept),
            'n_genes': int(n_genes)
        }

        # Create scatter plot with optional color mapping
        if color_values is not None:
            # Get color values for this parameter's genes
            gene_color_values = color_values[:n_genes]
            
            # Ensure color values are numpy array and handle NaN/inf values
            gene_color_values = np.asarray(gene_color_values)
            gene_color_values = np.nan_to_num(gene_color_values, nan=0.0, posinf=np.nanmax(gene_color_values[np.isfinite(gene_color_values)]))
            
            # Ensure all arrays are 1D and same length
            true_values = np.asarray(true_values).flatten()
            estimated_values = np.asarray(estimated_values).flatten()
            gene_color_values = gene_color_values.flatten()
            
            # Ensure arrays have same length
            min_len = min(len(true_values), len(estimated_values), len(gene_color_values))
            true_values = true_values[:min_len]
            estimated_values = estimated_values[:min_len]
            gene_color_values = gene_color_values[:min_len]
            
            # Check if we have valid color data and multiple values
            if len(gene_color_values) > 1 and np.std(gene_color_values) > 0:
                # Create scatter plot with color mapping
                try:
                    scatter = ax.scatter(true_values, estimated_values, 
                                       c=gene_color_values, cmap='viridis', 
                                       alpha=0.7, s=25, edgecolors='none')
                    
                    # Add colorbar for the first subplot only
                    if i == 0 and color_label is not None:
                        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label(color_label, fontsize=default_fontsize * 0.8)
                        cbar.ax.tick_params(labelsize=default_fontsize * 0.7)
                except Exception as e:
                    # Fall back to default coloring on any error
                    ax.scatter(true_values, estimated_values, 
                              alpha=0.7, s=25, color='steelblue', 
                              edgecolors='none')
            else:
                # Fall back to default coloring if insufficient color variation
                ax.scatter(true_values, estimated_values, 
                          alpha=0.7, s=25, color='steelblue', 
                          edgecolors='none')
            
            # Add error bars separately (without markers)
            estimated_std = np.asarray(estimated_std).flatten()[:min_len]
            ax.errorbar(true_values, estimated_values, yerr=estimated_std,
                       fmt='none', alpha=0.3, ecolor='gray', 
                       elinewidth=0.5, capsize=1)
        else:
            # Default scatter plot without color mapping
            ax.errorbar(true_values, estimated_values, yerr=estimated_std,
                       fmt='o', alpha=0.6, markersize=4, color='steelblue',
                       ecolor='steelblue', elinewidth=0.5, capsize=2)

        # Add perfect recovery line (y=x)
        min_val = min(np.min(true_values), np.min(estimated_values - estimated_std))
        max_val = max(np.max(true_values), np.max(estimated_values + estimated_std))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1, label='Perfect recovery')

        # Add best-fit line
        if not np.isnan(slope):
            fit_x = np.array([min_val, max_val])
            fit_y = slope * fit_x + intercept
            ax.plot(fit_x, fit_y, 'r-', alpha=0.7, linewidth=1, label='Best fit')

        # Get parameter labels using metadata system
        x_label = get_parameter_label(
            param_name=param_name,
            label_type="display",
            model=model,
            fallback_to_legacy=True
        )

        # Set labels and title
        ax.set_xlabel(f'True {x_label}', fontsize=default_fontsize)
        ax.set_ylabel(f'Est. {x_label}', fontsize=default_fontsize)

        # Add correlation info to title
        if not np.isnan(pearson_r):
            title = f'{x_label}\n$r = {pearson_r:.3f}$'
        else:
            title = f'{x_label}\n$r = $ NaN'
        ax.set_title(title, fontsize=default_fontsize)

        # Add legend only to first subplot
        if i == 0:
            ax.legend(fontsize=default_fontsize * 0.8, loc='upper left')

        # Set equal aspect ratio and grid
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

        # Tick label size
        ax.tick_params(labelsize=default_fontsize * 0.8)

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    # Compute summary metrics
    valid_correlations = [metrics['pearson_r'] for metrics in recovery_metrics.values()
                         if not np.isnan(metrics['pearson_r'])]
    valid_r_squared = [metrics['r_squared'] for metrics in recovery_metrics.values()
                      if not np.isnan(metrics['r_squared'])]

    mean_pearson_r = np.mean(valid_correlations) if valid_correlations else np.nan
    mean_r_squared = np.mean(valid_r_squared) if valid_r_squared else np.nan

    # Assess overall recovery quality
    if np.isnan(mean_pearson_r):
        recovery_quality = "Failed"
    elif mean_pearson_r >= 0.9:
        recovery_quality = "Excellent"
    elif mean_pearson_r >= 0.7:
        recovery_quality = "Good"
    elif mean_pearson_r >= 0.5:
        recovery_quality = "Moderate"
    else:
        recovery_quality = "Poor"

    recovery_metrics['summary'] = {
        'mean_pearson_r': float(mean_pearson_r),
        'mean_r_squared': float(mean_r_squared),
        'overall_recovery_quality': recovery_quality,
        'n_valid_parameters': len(valid_correlations)
    }

    # No global title - will be provided in textual figure legend
    plt.tight_layout()

    # Save figure if path provided
    if save_path is not None:
        if file_prefix:
            figure_name = f"{file_prefix}_parameter_recovery_correlation"
        else:
            figure_name = "parameter_recovery_correlation"
        _save_figure(fig, save_path, figure_name)

    # Summary information is returned in metrics dictionary for textual figure legend

    return fig, recovery_metrics