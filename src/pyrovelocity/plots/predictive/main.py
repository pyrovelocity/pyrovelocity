"""
Main plotting functions for predictive checks.

This module contains the primary public API functions for PyroVelocity
predictive check plotting.
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
from matplotlib.gridspec import GridSpec
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

# Local imports
from .utils import cleanup_numbered_files, _save_figure, combine_pdfs
from .internal.helpers import (
    _plot_umap_leiden_clusters,
    _plot_umap_time_coordinate,
    _plot_pattern_proportions,
    _plot_correlation_structure,
)
from .core import _process_parameters_for_plotting
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
    _plot_count_distributions,
    _plot_expression_relationships,
    _plot_library_sizes,
    _plot_expression_ranges,
    _plot_phase_portrait,
    _plot_velocity_magnitudes,
    _plot_fold_change_distribution,
    _plot_activation_timing,
)
from .training import (
    plot_training_loss,
    plot_mae_vs_spliced_count,
)
from .internal.rainbow import plot_pattern_analysis


@beartype
def plot_posterior_predictive_checks(
    model: Any,
    posterior_adata: AnnData,
    posterior_parameters: Dict[str, ArrayLike],
    figsize: Tuple[Union[int, float], Union[int, float]] = (7.5, 5.0),  # Standard width for 8.5x11" with margins
    save_path: Optional[str] = None,
    figure_name: Optional[str] = None,
    create_individual_plots: bool = True,
    combine_individual_pdfs: bool = False,
    default_fontsize: Union[int, float] = 8,
    observed_adata: Optional[AnnData] = None,
    num_genes: int = 6,
    true_parameters_adata: Optional[AnnData] = None,
) -> plt.Figure:
    """
    Generate posterior predictive check plots.

    This is an alias for plot_prior_predictive_checks with check_type="posterior".

    Args:
        model: PyroVelocity model instance
        posterior_adata: AnnData object with posterior predictive samples
        posterior_parameters: Dictionary of posterior parameter samples
        figsize: Figure size (width, height)
        save_path: Optional directory path to save figures (creates if doesn't exist)
        figure_name: Optional figure name (defaults to "posterior_predictive_checks")
        create_individual_plots: Whether to create individual modular plots
        combine_individual_pdfs: Whether to combine individual PDF plots into a single file
        default_fontsize: Default font size for all text elements
        observed_adata: Optional AnnData object with observed data for comparison in temporal dynamics plots.
                       If None, uses posterior_adata for both predictive and observed columns.
        num_genes: Number of genes to include in temporal dynamics plots (default: 6)
        true_parameters_adata: Optional AnnData object containing true parameters
                              in adata.uns['true_parameters'] for parameter recovery validation

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_posterior_predictive_checks(
        ...     model=model,
        ...     posterior_adata=adata,
        ...     posterior_parameters=params,
        ...     save_path="reports/docs/posterior_predictive",
        ...     figure_name="piecewise_activation_posterior_checks",
        ...     observed_adata=original_adata,
        ...     num_genes=10
        ... ) # xdoctest: +SKIP
    """
    return plot_prior_predictive_checks(
        model=model,
        prior_adata=posterior_adata,
        prior_parameters=posterior_parameters,
        figsize=figsize,
        check_type="posterior",
        save_path=save_path,
        figure_name=figure_name,
        create_individual_plots=create_individual_plots,
        combine_individual_pdfs=combine_individual_pdfs,
        default_fontsize=default_fontsize,
        observed_adata=observed_adata,
        num_genes=num_genes,
        true_parameters_adata=true_parameters_adata,
    )


@beartype
def plot_prior_predictive_checks(
    model: Any,
    prior_adata: AnnData,
    prior_parameters: Dict[str, ArrayLike],
    figsize: Tuple[Union[int, float], Union[int, float]] = (7.5, 5.0),  # Standard width for 8.5x11" with margins
    check_type: str = "prior",
    save_path: Optional[str] = None,
    figure_name: Optional[str] = None,
    create_individual_plots: bool = True,
    combine_individual_pdfs: bool = False,
    default_fontsize: Union[int, float] = 8,
    observed_adata: Optional[AnnData] = None,
    num_genes: int = 6,
    true_parameters_adata: Optional[AnnData] = None,
    metadata: Optional[Any] = None,
) -> plt.Figure:
    """
    Generate comprehensive predictive check plots for PyroVelocity models.

    This function creates a multi-panel figure showing parameter distributions,
    expression data validation, temporal dynamics, and biological plausibility checks.
    Can be used for both prior and posterior predictive checks.

    Optionally creates individual modular plots in addition to the overview.

    Args:
        model: PyroVelocity model instance (deprecated, use metadata instead)
        prior_adata: AnnData object with predictive samples
        prior_parameters: Dictionary of parameter samples (PyTorch, JAX, or NumPy)
        figsize: Figure size (width, height)
        check_type: Type of check ("prior" or "posterior")
        save_path: Optional directory path to save figures (creates if doesn't exist)
        figure_name: Optional figure name (defaults to "{check_type}_predictive_checks")
        create_individual_plots: Whether to create individual modular plots
        combine_individual_pdfs: Whether to combine individual PDF plots into a single file
        default_fontsize: Default font size for all text elements (titles, labels, legends)
        observed_adata: Optional AnnData object with observed data for comparison in temporal dynamics plots.
                       If None, uses prior_adata for both predictive and observed columns.
        num_genes: Number of genes to include in temporal dynamics plots (default: 6)
        true_parameters_adata: Optional AnnData object containing true parameters
                              in adata.uns['true_parameters'] for parameter recovery validation
        metadata: Optional PlotMetadata object or dict with component_name for parameter labeling

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_prior_predictive_checks(
        ...     model=model,
        ...     prior_adata=adata,
        ...     prior_parameters=params,
        ...     save_path="reports/docs/prior_predictive",
        ...     figure_name="piecewise_activation_prior_checks",
        ...     combine_individual_pdfs=True,
        ...     default_fontsize=8,
        ...     num_genes=10
        ... ) # xdoctest: +SKIP
    """
    # Convert all parameters to NumPy arrays for framework-agnostic plotting
    numpy_prior_parameters = ensure_numpy_parameters(prior_parameters)
    # Create individual modular plots if requested
    if create_individual_plots and save_path is not None:
        # Clean up numbered files from previous executions before creating new plots
        print("🧹 Cleaning up numbered files from previous executions...")
        cleanup_numbered_files(save_path)

        # Process parameters for plotting compatibility (handle batch dimensions)
        processed_parameters = _process_parameters_for_plotting(numpy_prior_parameters)

        # Create plots in logical order with numbered prefixes for proper PDF combination ordering
        plot_parameter_marginals(processed_parameters, check_type, save_path=save_path, file_prefix="02", model=model, default_fontsize=default_fontsize, true_parameters_adata=true_parameters_adata)
        plot_parameter_relationships(numpy_prior_parameters, check_type, save_path=save_path, file_prefix="03", model=model, default_fontsize=default_fontsize)
        plot_temporal_trajectories(processed_parameters, check_type, save_path=save_path, file_prefix="04", adata=prior_adata, default_fontsize=default_fontsize)

        # Plot 05: Parameter marginals by gene - lowest error genes
        plot_parameter_marginals_by_gene(
            posterior_parameters=processed_parameters,
            observed_adata=observed_adata,
            predicted_adata=prior_adata,
            num_genes=num_genes,
            save_path=save_path,
            file_prefix="05",
            default_fontsize=default_fontsize,
            model=model,
            check_type=check_type,
            select_highest_error=False,
            true_parameters_adata=true_parameters_adata
        )

        # Plot 06: Parameter marginals by gene - highest error genes (NEW)
        plot_parameter_marginals_by_gene(
            posterior_parameters=processed_parameters,
            observed_adata=observed_adata,
            predicted_adata=prior_adata,
            num_genes=num_genes,
            save_path=save_path,
            file_prefix="06",
            default_fontsize=default_fontsize,
            model=model,
            check_type=check_type,
            select_highest_error=True,
            true_parameters_adata=true_parameters_adata
        )

        # Plot 07: Parameter recovery correlation analysis
        if true_parameters_adata is not None:
            plot_parameter_recovery_correlation(
                posterior_parameters=processed_parameters,
                true_parameters_adata=true_parameters_adata,
                parameters_to_validate=["R_on", "gamma_star", "t_on_star", "delta_star"],
                save_path=save_path,
                file_prefix="07",
                model=model,
                default_fontsize=default_fontsize,
                observed_adata=observed_adata,
                predicted_adata=prior_adata,
                color_by_mae=True,
                color_by_spliced_count=False,
                spliced_count_statistic="median"
            )

        # Shifted plots (previously 07-10, now 08-11)
        plot_temporal_dynamics(prior_adata, check_type, save_path=save_path, file_prefix="08", default_fontsize=default_fontsize, observed_adata=observed_adata, gene_selection_method="mae", num_genes=num_genes, select_highest_error=False)
        plot_temporal_dynamics(prior_adata, check_type, save_path=save_path, file_prefix="09", default_fontsize=default_fontsize, observed_adata=observed_adata, gene_selection_method="mae", num_genes=num_genes, select_highest_error=True)
        plot_expression_validation(prior_adata, check_type, save_path=save_path, file_prefix="10", default_fontsize=default_fontsize)
        plot_pattern_analysis(prior_adata, processed_parameters, check_type, save_path=save_path, file_prefix="11", default_fontsize=default_fontsize, observed_adata=observed_adata)

        # Plot 12: Temporal coordinate validation
        plot_temporal_coordinate_validation(
            model=model,
            adata=prior_adata,
            parameters=processed_parameters,
            true_parameters_adata=true_parameters_adata,
            save_path=save_path,
            file_prefix="12",
            check_type=check_type,
            default_fontsize=default_fontsize
        )

        # Plot 13: Gene and cell-specific parameter recovery correlation analysis (U_0i and lambda_j)
        if true_parameters_adata is not None:
            plot_parameter_recovery_correlation(
                posterior_parameters=processed_parameters,
                true_parameters_adata=true_parameters_adata,
                parameters_to_validate=["U_0i", "lambda_j"],
                save_path=save_path,
                file_prefix="13",
                model=model,
                default_fontsize=default_fontsize,
                observed_adata=observed_adata,
                predicted_adata=prior_adata,
                color_by_mae=True,
                color_by_spliced_count=False,
                spliced_count_statistic="median"
            )

        # Plot 14: MAE vs Spliced Count Analysis - NEW
        if observed_adata is not None:
            plot_mae_vs_spliced_count(
                predicted_adata=prior_adata,
                observed_adata=observed_adata,
                save_path=save_path,
                file_prefix="14",
                default_fontsize=default_fontsize,
                check_type=check_type
            )
        
        # Plot 15: Training loss (ELBO) - only for posterior checks when model has been trained
        if check_type == "posterior":
            try:
                plot_training_loss(
                    model=model,
                    save_path=save_path,
                    file_prefix="15",
                    default_fontsize=default_fontsize
                )
            except ValueError as e:
                print(f"Warning: Could not create training loss plot: {e}")
                print("This is expected for prior predictive checks or untrained models.")

    # Process parameters for plotting compatibility (handle batch dimensions)
    processed_parameters = _process_parameters_for_plotting(prior_parameters)

    # Create comprehensive overview plot
    fig = plt.figure(figsize=figsize)

    # Create subplot grid: 3 rows × 4 columns with optimized spacing for reduced overlap
    gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.4)

    # Row 1: UMAP and Parameter Distribution Plots
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_umap_leiden_clusters(prior_adata, ax1, check_type, default_fontsize)

    ax2 = fig.add_subplot(gs[0, 1])
    _plot_umap_time_coordinate(prior_adata, ax2, check_type, model=model, default_fontsize=default_fontsize)

    ax3 = fig.add_subplot(gs[0, 2])
    _plot_fold_change_distribution(processed_parameters, ax3, check_type, model=model, default_fontsize=default_fontsize)

    ax4 = fig.add_subplot(gs[0, 3])
    _plot_activation_timing(processed_parameters, ax4, check_type, model=model, default_fontsize=default_fontsize, unprocessed_parameters=prior_parameters)

    # Row 2: Expression Data Validation
    ax5 = fig.add_subplot(gs[1, 0])
    _plot_count_distributions(prior_adata, ax5, check_type, default_fontsize)

    ax6 = fig.add_subplot(gs[1, 1])
    _plot_expression_relationships(prior_adata, ax6, check_type, default_fontsize)

    ax7 = fig.add_subplot(gs[1, 2])
    _plot_library_sizes(prior_adata, ax7, check_type, default_fontsize)

    ax8 = fig.add_subplot(gs[1, 3])
    _plot_expression_ranges(prior_adata, ax8, check_type, default_fontsize)

    # Row 3: Temporal Dynamics and Biological Validation
    ax9 = fig.add_subplot(gs[2, 0])
    _plot_phase_portrait(prior_adata, ax9, check_type, default_fontsize)

    ax10 = fig.add_subplot(gs[2, 1])
    _plot_velocity_magnitudes(prior_adata, ax10, check_type, default_fontsize)

    ax11 = fig.add_subplot(gs[2, 2])
    _plot_pattern_proportions(prior_adata, processed_parameters, ax11, check_type, default_fontsize)

    ax12 = fig.add_subplot(gs[2, 3])
    _plot_correlation_structure(prior_adata, ax12, check_type, default_fontsize, observed_adata=observed_adata)

    # Save comprehensive figure if path is provided
    if save_path is not None:
        # Use provided name or default, with "01" prefix for proper ordering
        if figure_name is not None:
            name = f"01_{figure_name}"
        else:
            name = f"01_{check_type}_predictive_checks"
        _save_figure(fig, save_path, name)

        # Combine individual PDFs if requested
        if combine_individual_pdfs and create_individual_plots:
            try:
                # Extract seed from figure_name if present, otherwise use default filename
                if figure_name is not None and '_' in figure_name:
                    # Try to extract seed from figure_name (e.g., "piecewise_activation_prior_checks_42")
                    parts = figure_name.split('_')
                    if parts[-1].isdigit():
                        seed_suffix = f"_{parts[-1]}"
                    else:
                        seed_suffix = ""
                else:
                    seed_suffix = ""

                combine_pdfs(
                    pdf_directory=save_path,
                    output_filename=f"combined_{check_type}_predictive_checks{seed_suffix}.pdf",
                    exclude_patterns=["combined_*.pdf"]  # Don't include previous combined files
                )
            except Exception as e:
                print(f"Warning: Could not combine PDFs: {e}")
                print("Individual PDF files are still available in the directory.")

    return fig


def _create_temporal_dynamics_figure(
    num_genes: int,
    figsize_scale: float = 1.0,
    default_fontsize: float = 10
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:
    """
    Create figure layout for temporal dynamics plots.

    Args:
        num_genes: Number of genes to plot
        figsize_scale: Scale factor for figure size
        default_fontsize: Base font size

    Returns:
        Tuple of (figure, axes_dict) where axes_dict maps plot names to axes
    """
    # Create figure with GridSpec layout
    fig_width = 18 * figsize_scale
    fig_height = (2 + num_genes * 2) * figsize_scale
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create gridspec: title row + gene rows, 6 columns
    gs = GridSpec(1 + num_genes, 6, figure=fig, hspace=0.4, wspace=0.3)

    axes_dict = {}

    # Create title row axes
    title_axes = [
        fig.add_subplot(gs[0, 0]),  # Gene label
        fig.add_subplot(gs[0, 1]),  # Phase portrait
        fig.add_subplot(gs[0, 2]),  # Dynamics
        fig.add_subplot(gs[0, 3]),  # Predictive UMAP
        fig.add_subplot(gs[0, 4]),  # Observed UMAP
        fig.add_subplot(gs[0, 5]),  # Marginal histogram
    ]

    # Set title row labels
    titles = ['Gene', 'Phase Portrait', 'Dynamics', 'Predicted', 'Observed', 'Marginal']
    for ax, title in zip(title_axes, titles):
        ax.text(0.5, 0.5, title, ha='center', va='center',
                fontsize=default_fontsize, weight='bold', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Create gene-specific axes
    for n in range(1, num_genes + 1):
        row_idx = n  # 0 is title row
        axes_dict[f'gene_label_{n}'] = fig.add_subplot(gs[row_idx, 0])
        axes_dict[f'phase_{n}'] = fig.add_subplot(gs[row_idx, 1])
        axes_dict[f'dynamics_{n}'] = fig.add_subplot(gs[row_idx, 2])
        axes_dict[f'pred_umap_{n}'] = fig.add_subplot(gs[row_idx, 3])
        axes_dict[f'obs_umap_{n}'] = fig.add_subplot(gs[row_idx, 4])
        axes_dict[f'hist_{n}'] = fig.add_subplot(gs[row_idx, 5])

    return fig, axes_dict