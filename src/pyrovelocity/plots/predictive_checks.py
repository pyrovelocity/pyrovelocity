"""
Predictive check plotting functions for PyroVelocity models.

This module provides flexible plotting functions for both prior and posterior
predictive checks, designed to validate model behavior and biological plausibility.
The module is organized into modular plotting functions that can be used individually
or combined into comprehensive overview plots.
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
from scipy.stats import linregress, pearsonr

from pyrovelocity.styles import configure_matplotlib_style

# Try to import UMAP, fall back gracefully if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

configure_matplotlib_style()


@beartype
def cleanup_numbered_files(output_dir: str, patterns: Optional[List[str]] = None) -> None:
    """
    Remove numbered PDF and PNG files from previous executions.

    This function removes files matching patterns like:
    - 01_*.pdf, 01_*.png
    - 02_*.pdf, 02_*.png
    - ...
    - 09_*.pdf, 09_*.png

    This ensures that when combining PDFs, we don't pick up remnant files
    from previous executions that might have different seeds or configurations.

    Args:
        output_dir: Directory containing the files to clean up
        patterns: Optional list of file patterns to remove. If None, uses default numbered patterns.

    Example:
        >>> # Clean up default numbered files
        >>> cleanup_numbered_files("reports/docs/prior_predictive")
        >>>
        >>> # Clean up custom patterns
        >>> cleanup_numbered_files("reports/docs/validation", ["temp_*.pdf", "draft_*.png"])
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"📁 Output directory {output_dir} does not exist yet - nothing to clean")
        return

    # Use default numbered patterns if none provided
    if patterns is None:
        patterns = []
        for num in range(1, 10):  # 01-09 to be future-proof
            patterns.extend([
                f"{num:02d}_*.pdf",
                f"{num:02d}_*.png"
            ])

    files_removed = 0
    for pattern in patterns:
        matching_files = list(output_path.glob(pattern))
        for file_path in matching_files:
            try:
                file_path.unlink()
                print(f"🗑️  Removed: {file_path.name}")
                files_removed += 1
            except OSError as e:
                print(f"⚠️  Could not remove {file_path.name}: {e}")

    if files_removed > 0:
        print(f"✅ Cleaned up {files_removed} numbered files from previous executions")
    else:
        print("✨ No numbered files found to clean up")


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


def _format_parameter_name(param_name: str) -> str:
    """
    Format parameter names for LaTeX rendering in plots.

    Args:
        param_name: Raw parameter name (e.g., 'alpha_off', 't_on_star')

    Returns:
        LaTeX-formatted parameter name
    """
    # Greek letter replacements
    greek_replacements = {
        'alpha': r'\alpha',
        'beta': r'\beta',
        'gamma': r'\gamma',
        'delta': r'\delta',
        'epsilon': r'\epsilon',
        'theta': r'\theta',
        'lambda': r'\lambda',
        'mu': r'\mu',
        'sigma': r'\sigma',
        'tau': r'\tau',
        'phi': r'\phi',
        'psi': r'\psi',
        'omega': r'\omega'
    }

    # Subscript replacements for common suffixes
    subscript_replacements = {
        '_loc': r'_{loc}',
        '_scale': r'_{scl}',
        '_on': r'_{on}',
        '_off': r'_{off}',
        '_0i': r'_{0i}',
        '_star': r'^*',
        '_max': r'_{max}',
        '_min': r'_{min}',
        '_rate': r'_{rate}',
        '_conc': r'_{conc}',
        '_shape': r'_{shape}',
        '_mean': r'_{mean}',
        '_std': r'_{std}',
        '_var': r'_{var}',
        '_j': r'_j'
    }

    # Start with the original name
    formatted = param_name

    # Handle special cases first
    special_cases = {
        'T_M_star': r'T^*_M',
        't_star': r't^*',
        't_on_star': r't^*_{on}',
        'delta_star': r'\delta^*',
        'lambda_j': r'\lambda_j',
        'U_0i': r'U_{0i}',
        'latent_time': r't_{latent}',
        'velocity_pseudotime': r't_{velocity}',
        'dpt_pseudotime': r't_{dpt}',
        'pseudotime': r't_{pseudo}',
        'shared_time': r't_{shared}',
        # Latent RNA concentrations (true/unobserved values)
        'ut': r'u^*_{ij}',
        'st': r's^*_{ij}',
        # Observed RNA counts (measured values)
        'u_obs': r'u_{ij}',
        's_obs': r's_{ij}'
    }

    if param_name in special_cases:
        return f'${special_cases[param_name]}$'

    # Apply subscript/superscript replacements
    for suffix, latex_suffix in subscript_replacements.items():
        formatted = formatted.replace(suffix, latex_suffix)

    # Replace Greek letters at the beginning
    for greek, latex in greek_replacements.items():
        if formatted.startswith(greek):
            formatted = formatted.replace(greek, latex, 1)
            break

    # Wrap in math mode
    return f'${formatted}$'


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
def combine_pdfs(
    pdf_directory: str,
    output_filename: str = "combined_predictive_checks.pdf",
    pdf_pattern: str = "*.pdf",
    exclude_patterns: Optional[List[str]] = None
) -> None:
    """
    Combine multiple PDF files into a single PDF using only Python libraries.

    This function uses PyPDF2/pypdf to merge PDF files without requiring system calls.
    It's designed to work with the output from plot_prior_predictive_checks.

    Args:
        pdf_directory: Directory containing PDF files to combine
        output_filename: Name of the combined output PDF file
        pdf_pattern: Glob pattern to match PDF files (default: "*.pdf")
        exclude_patterns: Optional list of filename patterns to exclude from combination

    Example:
        >>> # Combine all PDFs from prior predictive checks
        >>> combine_pdfs(
        ...     pdf_directory="reports/docs/prior_predictive",
        ...     output_filename="combined_prior_checks.pdf",
        ...     exclude_patterns=["combined_*.pdf"]  # Don't include previous combined files
        ... )
    """
    from pypdf import PdfReader, PdfWriter

    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory {pdf_directory} does not exist")

    # Find all PDF files matching the pattern
    pdf_files = list(pdf_dir.glob(pdf_pattern))

    # Apply exclusion patterns
    if exclude_patterns:
        filtered_files = []
        for pdf_file in pdf_files:
            exclude = False
            for pattern in exclude_patterns:
                if pdf_file.match(pattern):
                    exclude = True
                    break
            if not exclude:
                filtered_files.append(pdf_file)
        pdf_files = filtered_files

    if not pdf_files:
        print(f"No PDF files found in {pdf_directory} matching pattern '{pdf_pattern}'")
        return

    # Sort files for consistent ordering
    pdf_files.sort()

    # Create PDF writer object
    pdf_writer = PdfWriter()

    # Add each PDF to the writer
    for pdf_file in pdf_files:
        try:
            pdf_reader = PdfReader(str(pdf_file))
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)
            print(f"Added {pdf_file.name} ({len(pdf_reader.pages)} pages)")
        except Exception as e:
            print(f"Warning: Could not read {pdf_file.name}: {e}")
            continue

    # Write combined PDF
    output_path = pdf_dir / output_filename
    with open(output_path, 'wb') as output_file:
        pdf_writer.write(output_file)

    print(f"Combined {len(pdf_files)} PDFs into {output_path}")
    print(f"Total pages: {len(pdf_writer.pages)}")


def _get_available_parameters(
    parameters: Dict[str, torch.Tensor],
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

@beartype
def plot_parameter_marginals(
    parameters: Dict[str, torch.Tensor],
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
        parameters: Dictionary of parameter tensors
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
    available_params = _get_available_parameters(parameters, exclude_params, model)

    if not available_params:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'No parameters available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Parameter Marginals')
        return fig

    # Extract true parameters if provided for validation
    true_parameters = {}
    global_true_params = ['T_M_star', 't_loc', 't_scale']  # Global parameters that should appear on global marginals

    if true_parameters_adata is not None and 'true_parameters' in true_parameters_adata.uns:
        true_params_dict = true_parameters_adata.uns['true_parameters']
        for param_name in available_params:
            if param_name in true_params_dict and param_name in global_true_params:
                true_value = true_params_dict[param_name]
                # Convert to tensor if needed
                if not isinstance(true_value, torch.Tensor):
                    true_value = torch.tensor(true_value)
                # Only store scalar global parameters
                if true_value.numel() == 1:
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
        values = parameters[param_name].flatten().numpy()

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

        # Try to infer component name from all parameters if model not provided
        component_name = None
        if model is None:
            component_name = infer_component_name_from_parameters(parameters)

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
    parameters: Dict[str, torch.Tensor],
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
        parameters: Dictionary of parameter tensors
        check_type: Type of check ("prior" or "posterior")
        figsize: Figure size (width, height)
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        model: Optional PyroVelocity model instance for parameter metadata

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Hierarchical time structure
    _plot_hierarchical_time_structure(parameters, axes[0], check_type, model=model, default_fontsize=default_fontsize)

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

    Args:
        observed_adata: AnnData object with observed data
        predicted_adata: AnnData object with predicted data
        num_genes: Number of genes to select
        layer: Layer to use for MAE computation (default: "spliced")
        select_highest_error: If True, select genes with highest MAE instead of lowest.
                             Genes are always sorted from lowest to highest error (default: False)

    Returns:
        Tuple of (gene_indices, gene_names) for selected genes, sorted from lowest to highest error
    """
    from pyrovelocity.analysis.analyze import mae_per_gene

    # Get count data from appropriate layers
    if layer in observed_adata.layers and layer in predicted_adata.layers:
        observed_counts = observed_adata.layers[layer]
        predicted_counts = predicted_adata.layers[layer]
    else:
        # Fallback to X if layer not found
        observed_counts = observed_adata.X
        predicted_counts = predicted_adata.X

    # Convert sparse matrices to dense if needed
    if hasattr(observed_counts, 'toarray'):
        observed_counts = observed_counts.toarray()
    if hasattr(predicted_counts, 'toarray'):
        predicted_counts = predicted_counts.toarray()

    # Compute MAE per gene (returns negative values, higher is better)
    mae_scores = mae_per_gene(predicted_counts, observed_counts)

    # Store MAE scores in predicted_adata for transparency
    predicted_adata.var['mae_score'] = mae_scores

    # Sort all genes by MAE (lowest error to highest error)
    # Since mae_per_gene returns negative values, we sort in descending order
    # to get lowest error (highest negative value) to highest error (lowest negative value)
    sorted_indices = np.argsort(mae_scores)[::-1]

    if select_highest_error:
        # Select genes with highest error (from the end of the sorted list)
        # but maintain the lowest-to-highest error ordering
        selected_indices = sorted_indices[-num_genes:]
    else:
        # Select genes with lowest error (from the beginning of the sorted list)
        selected_indices = sorted_indices[:num_genes]

    # Ensure the selected genes are ordered from lowest to highest error
    # by sorting the selected indices by their MAE scores (descending order for negative values)
    selected_mae_scores = mae_scores[selected_indices]
    reorder_indices = np.argsort(selected_mae_scores)[::-1]
    final_gene_indices = selected_indices[reorder_indices]
    final_gene_names = [predicted_adata.var_names[i] for i in final_gene_indices]

    return final_gene_indices.tolist(), final_gene_names


@beartype
def plot_parameter_marginals_by_gene(
    posterior_parameters: Dict[str, torch.Tensor],
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
        ... )
    """
    from matplotlib.gridspec import GridSpec

    from pyrovelocity.plots.parameter_metadata import get_parameter_label

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

    # Filter parameters to show based on availability
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
                if not isinstance(true_value, torch.Tensor):
                    true_value = torch.tensor(true_value)
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
                param_reshaped = param_tensor.view(num_samples, num_genes_in_param)
            else:
                # Fallback: treat as single sample per gene
                param_reshaped = param_tensor.unsqueeze(0)  # [1, num_genes]
        elif param_tensor.ndim == 2:
            # Already shaped: [num_samples, num_genes]
            param_reshaped = param_tensor
        else:
            # Higher dimensions: flatten and reshape
            param_reshaped = param_tensor.view(-1, param_tensor.shape[-1])

        param_samples_by_gene[param_name] = param_reshaped

        # Compute global range for this parameter across all genes
        param_min = float(param_reshaped.min())
        param_max = float(param_reshaped.max())

        # Include true parameter values in range calculation if available
        if param_name in true_parameters:
            true_param = true_parameters[param_name]
            if true_param.numel() == 1:
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
        all_values = param_samples.numpy().flatten()
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
                gene_param_samples = param_samples[:, gene_idx].numpy()
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
                if true_param.numel() == 1:
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
                # Get parameter short label using metadata system
                param_short_label = get_parameter_label(
                    param_name=param_name,
                    label_type="short",
                    model=model,
                    fallback_to_legacy=True
                )
                ax.set_xlabel(param_short_label, fontsize=default_fontsize)
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
        _plot_gene_spliced_dynamics_rainbow(adata, axes_dict, n, gene_idx, gene_name, check_type, available_genes)

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
    parameters: Dict[str, torch.Tensor],
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
            u_star_log2 = torch.log2(u_star)
            s_star_log2 = torch.log2(s_star)

            # Plot unspliced (log2 scale)
            axes[pattern_idx, 0].plot(
                t_star.numpy(), u_star_log2.numpy(),
                color=color, alpha=0.7, linewidth=2,
                label=f'Example {example_idx+1}' if example_idx < 3 else None
            )

            # Plot spliced (log2 scale)
            axes[pattern_idx, 1].plot(
                t_star.numpy(), s_star_log2.numpy(),
                color=color, alpha=0.7, linewidth=2,
                label=f'Example {example_idx+1}' if example_idx < 3 else None
            )

            # Plot phase portrait (both axes log2)
            axes[pattern_idx, 2].plot(
                u_star_log2.numpy(), s_star_log2.numpy(),
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
        axes[pattern_idx, 0].set_ylabel(_latex_safe_text('log2(Unspliced) (u*)'), fontsize=default_fontsize * 0.9)
        axes[pattern_idx, 0].set_title(f'{formatted_pattern}: Unspliced', fontsize=default_fontsize)
        axes[pattern_idx, 0].grid(True, alpha=0.3)
        axes[pattern_idx, 0].legend(fontsize=default_fontsize * 0.8)
        axes[pattern_idx, 0].tick_params(labelsize=default_fontsize * 0.75)

        axes[pattern_idx, 1].set_xlabel(_latex_safe_text(f'{time_label}'), fontsize=default_fontsize * 0.9)
        axes[pattern_idx, 1].set_ylabel(_latex_safe_text('log2(Spliced) (s*)'), fontsize=default_fontsize * 0.9)
        axes[pattern_idx, 1].set_title(f'{formatted_pattern}: Spliced', fontsize=default_fontsize)
        axes[pattern_idx, 1].grid(True, alpha=0.3)
        axes[pattern_idx, 1].tick_params(labelsize=default_fontsize * 0.75)

        axes[pattern_idx, 2].set_xlabel(_latex_safe_text('log2(Unspliced) (u*)'), fontsize=default_fontsize * 0.9)
        axes[pattern_idx, 2].set_ylabel(_latex_safe_text('log2(Spliced) (s*)'), fontsize=default_fontsize * 0.9)
        axes[pattern_idx, 2].set_title(f'{formatted_pattern}: Phase Portrait', fontsize=default_fontsize)
        axes[pattern_idx, 2].grid(True, alpha=0.3)
        axes[pattern_idx, 2].tick_params(labelsize=default_fontsize * 0.75)

    plt.tight_layout()

    if save_path is not None:
        prefix = f"{file_prefix}_" if file_prefix else ""
        _save_figure(fig, save_path, f"{prefix}{check_type}_temporal_trajectories")

    return fig


@beartype
def plot_pattern_analysis(
    adata: AnnData,
    parameters: Dict[str, torch.Tensor],
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


@beartype
def plot_temporal_coordinate_validation(
    model: Any,
    adata: AnnData,
    parameters: Dict[str, torch.Tensor],
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
        ... )
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

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
    parameters: Dict[str, torch.Tensor],
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
        if isinstance(true_t_star, torch.Tensor):
            true_t_star = true_t_star.detach().cpu().numpy()
        if isinstance(estimated_t_star, torch.Tensor):
            estimated_t_star = estimated_t_star.detach().cpu().numpy()

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
            label_type="short",
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
    parameters: Dict[str, torch.Tensor],
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


def _compute_temporal_uncertainty(
    t_star_samples: torch.Tensor,
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
                samples_reshaped = t_star_samples.view(num_samples, num_cells)
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
                samples_reshaped = samples_flat.view(num_samples, num_cells)
            else:
                return None, None

        # Ensure we have the right number of cells
        if samples_reshaped.shape[1] != num_cells:
            return None, None

        # Compute statistics across samples (dim=0)
        samples_np = samples_reshaped.detach().cpu().numpy()
        mean_values = np.mean(samples_np, axis=0)
        std_values = np.std(samples_np, axis=0)

        # Compute coefficient of variation (CV = std/mean)
        # Avoid division by zero
        cv_values = np.where(mean_values > 1e-8, std_values / mean_values, 0.0)

        return std_values, cv_values

    except Exception as e:
        print(f"Warning: Could not compute temporal uncertainty: {e}")
        return None, None


def _classify_parameters_into_patterns(
    parameters: Dict[str, torch.Tensor],
    n_examples: int = 10
) -> Dict[str, List[Dict[str, torch.Tensor]]]:
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
    alpha_off = torch.ones_like(R_on)
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
            return torch.sigmoid(torch.tensor(steepness * (value - threshold))).item()
        else:  # direction == '<'
            return torch.sigmoid(torch.tensor(steepness * (threshold - value))).item()

    # Compute pattern scores for each sample using independent absolute parameters
    pattern_scores = {pattern: torch.zeros(min_length) for pattern in pattern_examples.keys()}

    for i in range(min_length):
        # Use independent absolute temporal parameters with mathematical specification thresholds
        
        # Pre-activation: negative onset time
        pattern_scores['pre_activation'][i] = torch.prod(torch.tensor([
            sigmoid_score(R_on[i].item(), 2.0, '>'),
            sigmoid_score(t_on_star[i].item(), 0.0, '<')
        ])) ** (1.0 / 2)

        # Transient: positive onset, early timing, short duration
        pattern_scores['transient'][i] = torch.prod(torch.tensor([
            sigmoid_score(R_on[i].item(), 2.0, '>'),
            sigmoid_score(t_on_star[i].item(), 0.0, '>'),
            sigmoid_score(t_on_star[i].item(), 1.5, '<'),  # Absolute early onset
            sigmoid_score(delta_star[i].item(), 2.0, '<')  # Absolute short duration
        ])) ** (1.0 / 4)

        # Sustained: positive onset, early timing, long duration
        pattern_scores['sustained'][i] = torch.prod(torch.tensor([
            sigmoid_score(R_on[i].item(), 2.0, '>'),
            sigmoid_score(t_on_star[i].item(), 0.0, '>'),
            sigmoid_score(t_on_star[i].item(), 1.5, '<'),  # Absolute early onset
            sigmoid_score(delta_star[i].item(), 2.5, '>')  # Absolute long duration
        ])) ** (1.0 / 4)

    # Assign each sample to the pattern with highest score and collect examples
    pattern_names = list(pattern_examples.keys())

    for i in range(min_length):
        scores = [pattern_scores[pattern][i] for pattern in pattern_names]
        best_pattern_idx = torch.argmax(torch.tensor(scores))
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

            # Add T_M_star if available
            if use_relative_params and len(T_M_star) > 0:
                if len(T_M_star) == 1:
                    params['T_M_star'] = T_M_star[0]
                elif len(T_M_star) == min_length:
                    params['T_M_star'] = T_M_star[i]
                else:
                    # Use first T_M_star value as it's a global parameter
                    params['T_M_star'] = T_M_star[0]

            pattern_examples[best_pattern].append(params)

    # Remove empty patterns
    pattern_examples = {k: v for k, v in pattern_examples.items() if v}

    return pattern_examples


def _compute_adaptive_time_range(
    pattern_examples: Dict[str, List[Dict[str, torch.Tensor]]],
    buffer_factor: float = 1.2,
    n_points: int = 300,
    adata: Optional[AnnData] = None
) -> torch.Tensor:
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
        Time tensor for trajectory evaluation
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
            return torch.linspace(0, time_range_max, n_points)

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

    return torch.linspace(0, time_range_max, n_points)


def _compute_time_course(
    t_star: torch.Tensor,
    params: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    u_star = torch.zeros_like(t_star)
    s_star = torch.zeros_like(t_star)

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
    tau_on: torch.Tensor,
    alpha_on: torch.Tensor,
    gamma_star: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute analytical solution for ON phase."""
    # Initial conditions: u*_0 = 1.0, s*_0 = 1.0/γ*
    u_0 = 1.0
    s_0 = 1.0 / gamma_star

    # Analytical solution for ON phase
    exp_tau = torch.exp(-tau_on)
    exp_gamma_tau = torch.exp(-gamma_star * tau_on)

    u_on = alpha_on + (u_0 - alpha_on) * exp_tau

    s_on = (alpha_on / gamma_star +
            (s_0 - alpha_on / gamma_star) * exp_gamma_tau +
            (alpha_on - u_0) * (exp_tau - exp_gamma_tau) / (gamma_star - 1))

    return u_on, s_on


def _compute_off_phase_solution(
    tau_off: torch.Tensor,
    alpha_off: torch.Tensor,
    alpha_on: torch.Tensor,
    gamma_star: torch.Tensor,
    delta_star: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute analytical solution for return to OFF phase."""
    # Initial conditions: endpoint values from ON phase
    u_end, s_end = _compute_on_phase_solution(
        delta_star, alpha_on, gamma_star
    )

    # Analytical solution for OFF phase
    exp_tau = torch.exp(-tau_off)
    exp_gamma_tau = torch.exp(-gamma_star * tau_off)

    u_off = alpha_off + (u_end - alpha_off) * exp_tau

    s_off = (alpha_off / gamma_star +
            (s_end - alpha_off / gamma_star) * exp_gamma_tau +
            (alpha_off - u_end) * (exp_tau - exp_gamma_tau) / (gamma_star - 1))

    return u_off, s_off


@beartype
def plot_prior_predictive_checks(
    model: Any,
    prior_adata: AnnData,
    prior_parameters: Dict[str, torch.Tensor],
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
) -> plt.Figure:
    """
    Generate comprehensive predictive check plots for PyroVelocity models.

    This function creates a multi-panel figure showing parameter distributions,
    expression data validation, temporal dynamics, and biological plausibility checks.
    Can be used for both prior and posterior predictive checks.

    Optionally creates individual modular plots in addition to the overview.

    Args:
        model: PyroVelocity model instance (unused but kept for compatibility)
        prior_adata: AnnData object with predictive samples
        prior_parameters: Dictionary of parameter samples
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
        ... )
    """
    # Create individual modular plots if requested
    if create_individual_plots and save_path is not None:
        # Clean up numbered files from previous executions before creating new plots
        print("🧹 Cleaning up numbered files from previous executions...")
        cleanup_numbered_files(save_path)

        # Process parameters for plotting compatibility (handle batch dimensions)
        processed_parameters = _process_parameters_for_plotting(prior_parameters)

        # Create plots in logical order with numbered prefixes for proper PDF combination ordering
        plot_parameter_marginals(processed_parameters, check_type, save_path=save_path, file_prefix="02", model=model, default_fontsize=default_fontsize, true_parameters_adata=true_parameters_adata)
        plot_parameter_relationships(processed_parameters, check_type, save_path=save_path, file_prefix="03", model=model, default_fontsize=default_fontsize)
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
                default_fontsize=default_fontsize
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

        # Plot 13: Training loss (ELBO) - only for posterior checks when model has been trained
        if check_type == "posterior":
            try:
                plot_training_loss(
                    model=model,
                    save_path=save_path,
                    file_prefix="13",
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
    _plot_activation_timing(processed_parameters, ax4, check_type, model=model, default_fontsize=default_fontsize)

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
        # Compute UMAP if not available and UMAP is installed
        if UMAP_AVAILABLE and 'X_pca' in adata.obsm:
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


def _process_parameters_for_plotting(
    parameters: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Process parameters to make them compatible with plotting functions.

    The plotting functions expect parameters to be flattened 1D arrays, but posterior
    samples from SVI have batch dimensions. This method handles the tensor reshaping
    to make the samples compatible with the plotting code.

    Also filters out guide-specific parameters that shouldn't be displayed.

    Args:
        parameters: Raw parameters with potential batch dimensions

    Returns:
        Processed parameters suitable for plotting functions
    """
    processed_parameters = {}

    # Define patterns for guide parameters to exclude
    # Be specific to avoid filtering legitimate model parameters like t_loc, t_scale
    guide_param_patterns = [
        'AutoLowRankMultivariateNormal',
        'AutoNormal',
        'AutoDelta',
        'AutoGuide',
        '_latent',
        'auto_',
        'guide_'
    ]

    for key, value in parameters.items():
        # Skip guide-specific parameters
        if any(pattern in key for pattern in guide_param_patterns):
            continue

        if isinstance(value, torch.Tensor):
            # Handle different tensor shapes
            if value.ndim == 1:
                # Already 1D, use as-is
                processed_parameters[key] = value
            elif value.ndim == 2:
                # 2D tensor: [num_samples, param_dim] or [batch_size, param_dim]
                # Flatten to 1D for plotting
                processed_parameters[key] = value.flatten()
            elif value.ndim == 3:
                # 3D tensor: [batch_size, num_samples, param_dim]
                # Remove batch dimension and flatten
                if value.shape[0] == 1:
                    # Remove batch dimension: [1, num_samples, param_dim] -> [num_samples, param_dim]
                    squeezed = value.squeeze(0)
                    processed_parameters[key] = squeezed.flatten()
                else:
                    # Multiple batches: flatten everything
                    processed_parameters[key] = value.flatten()
            else:
                # Higher dimensions: flatten everything
                processed_parameters[key] = value.flatten()
        else:
            # Non-tensor values: keep as-is
            processed_parameters[key] = value

    # Compute t_star if missing but required components are available
    # t_star = T_M_star * max(tilde_t, epsilon) where tilde_t ~ Normal(t_loc, t_scale)
    if ('t_star' not in processed_parameters and
        'T_M_star' in processed_parameters and
        'tilde_t' in processed_parameters):

        T_M_star = processed_parameters['T_M_star']
        tilde_t = processed_parameters['tilde_t']

        # Use same epsilon as in the prior model (default 1e-6)
        t_epsilon = 1e-6

        # Compute t_star: T_M_star * max(tilde_t, epsilon)
        # Handle broadcasting: T_M_star is [num_samples], tilde_t is [num_samples * num_cells]
        if T_M_star.ndim == 1 and tilde_t.ndim == 1:
            # Determine number of cells from tilde_t length
            num_samples = len(T_M_star)
            num_cells = len(tilde_t) // num_samples

            if len(tilde_t) % num_samples == 0:  # Valid division
                # Reshape tilde_t to [num_samples, num_cells] for broadcasting
                tilde_t_reshaped = tilde_t.view(num_samples, num_cells)
                t_star_computed = T_M_star.unsqueeze(-1) * torch.clamp(tilde_t_reshaped, min=t_epsilon)
                # Flatten back to 1D for consistency with other parameters
                processed_parameters['t_star'] = t_star_computed.flatten()

                print(f"ℹ️  Computed missing t_star from T_M_star and tilde_t (shape: {t_star_computed.shape} -> flattened)")
            else:
                print(f"⚠️  Cannot compute t_star: tilde_t length {len(tilde_t)} not divisible by T_M_star length {num_samples}")
        else:
            print(f"⚠️  Cannot compute t_star: unexpected tensor dimensions T_M_star: {T_M_star.shape}, tilde_t: {tilde_t.shape}")

    # Note: t_on_star and delta_star are now independent absolute parameters
    # No computation from tilde_* parameters needed with independent parameterization

    return processed_parameters


def _plot_parameter_marginals_summary(
    parameters: Dict[str, torch.Tensor],
    ax: plt.Axes,
    check_type: str,
    model: Optional[Any] = None
) -> None:
    """Plot marginal distributions of key parameters (summary version for overview)."""
    from pyrovelocity.plots.parameter_metadata import (
        get_parameter_label,
        infer_component_name_from_parameters,
    )

    # Try to infer component name from all parameters if model not provided
    component_name = None
    if model is None:
        component_name = infer_component_name_from_parameters(parameters)

    # Focus on piecewise activation parameters (use R_on instead of deprecated alpha_on)
    key_params = ['R_on', 't_on_star', 'delta_star', 'gamma_star']

    colors = sns.color_palette("husl", len(key_params))

    for i, param_name in enumerate(key_params):
        if param_name in parameters:
            values = parameters[param_name].flatten().numpy()

            # Get parameter label using new metadata system
            param_label = get_parameter_label(
                param_name=param_name,
                label_type="display",
                model=model,
                component_name=component_name,
                fallback_to_legacy=True
            )

            # Use relative frequency for consistency with main marginals plot
            ax.hist(values, bins=30, alpha=0.6, color=colors[i],
                   label=param_label, density=False,
                   weights=np.ones(len(values)) / len(values))

    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Relative Frequency')
    ax.set_title(f'{check_type.title()} Parameter Marginals')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)



def _plot_hierarchical_time_structure(
    parameters: Dict[str, torch.Tensor],
    ax: plt.Axes,
    check_type: str,
    model: Optional[Any] = None,
    default_fontsize: Union[int, float] = 8
) -> None:
    """Plot hierarchical time parameter relationships."""
    from pyrovelocity.plots.parameter_metadata import (
        get_parameter_label,
        infer_component_name_from_parameters,
    )

    # Try to infer component name from all parameters if model not provided
    component_name = None
    if model is None:
        component_name = infer_component_name_from_parameters(parameters)

    # Check for hierarchical time parameters
    time_params = ['T_M_star', 't_loc', 't_scale']
    available_time_params = [p for p in time_params if p in parameters]

    if len(available_time_params) >= 2:
        # Plot T_M_star vs population time spread (t_scale)
        if 'T_M_star' in parameters and 't_scale' in parameters:
            T_M = parameters['T_M_star'].flatten().numpy()
            t_scale = parameters['t_scale'].flatten().numpy()

            ax.scatter(T_M, t_scale, alpha=0.6, s=5, color='purple')

            # Get parameter labels using new metadata system
            T_M_label = get_parameter_label(
                param_name="T_M_star",
                label_type="display",
                model=model,
                component_name=component_name,
                fallback_to_legacy=True
            )
            t_scale_label = get_parameter_label(
                param_name="t_scale",
                label_type="display",
                model=model,
                component_name=component_name,
                fallback_to_legacy=True
            )

            ax.set_xlabel(f'Global Time Scale ({T_M_label})', fontsize=default_fontsize * 0.9)
            ax.set_ylabel(f'Population Time Spread ({t_scale_label})', fontsize=default_fontsize * 0.9)
            ax.set_title(f'{check_type.title()} Hierarchical Time Structure', fontsize=default_fontsize)
            ax.tick_params(labelsize=default_fontsize * 0.75)

            # Add adaptive interpretation guidelines
            # Compute adaptive thresholds based on actual data
            t_scale_median = np.median(t_scale)
            T_M_median = np.median(T_M)

            ax.axhline(t_scale_median, color='orange', linestyle='--', alpha=0.7,
                      label=f'Median temporal spread: {t_scale_median:.2f}')
            ax.axvline(T_M_median, color='green', linestyle='--', alpha=0.7,
                      label=f'Median process duration: {T_M_median:.1f}')
            ax.legend(fontsize=default_fontsize * 0.8)

        # Alternative: t_loc vs t_scale if T_M_star not available
        elif 't_loc' in parameters and 't_scale' in parameters:
            t_loc = parameters['t_loc'].flatten().numpy()
            t_scale = parameters['t_scale'].flatten().numpy()

            ax.scatter(t_loc, t_scale, alpha=0.6, s=5, color='purple')

            # Get parameter labels using new metadata system
            t_loc_label = get_parameter_label(
                param_name="t_loc",
                label_type="display",
                model=model,
                component_name=component_name,
                fallback_to_legacy=True
            )
            t_scale_label = get_parameter_label(
                param_name="t_scale",
                label_type="display",
                model=model,
                component_name=component_name,
                fallback_to_legacy=True
            )

            ax.set_xlabel(f'Population Time Location ({t_loc_label})', fontsize=default_fontsize)
            ax.set_ylabel(f'Population Time Spread ({t_scale_label})', fontsize=default_fontsize)
            ax.set_title(f'{check_type.title()} Population Time Parameters', fontsize=default_fontsize)
            ax.tick_params(labelsize=default_fontsize * 0.75)  # Reduce tick label size
            ax.legend(fontsize=default_fontsize * 0.75)
    else:
        ax.text(0.5, 0.5, 'Hierarchical time parameters\nnot available',
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{check_type.title()} Hierarchical Time Structure', fontsize=8)

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
    parameters: Dict[str, torch.Tensor],
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
    component_name = None
    if model is None:
        component_name = infer_component_name_from_parameters(parameters)

    # Use R_on directly (preferred) or fall back to alpha_on/alpha_off ratio
    if 'R_on' in parameters:
        fold_change = parameters['R_on'].flatten().numpy()
        param_source = "R_on"
    elif 'alpha_off' in parameters and 'alpha_on' in parameters:
        alpha_off = parameters['alpha_off'].flatten()
        alpha_on = parameters['alpha_on'].flatten()
        fold_change = (alpha_on / alpha_off).numpy()
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
    parameters: Dict[str, torch.Tensor]
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
    parameters: Dict[str, torch.Tensor],
    ax: plt.Axes,
    check_type: str,
    model: Optional[Any] = None,
    default_fontsize: Union[int, float] = 8
) -> None:
    """Plot activation timing and duration distributions."""
    from pyrovelocity.plots.parameter_metadata import (
        get_parameter_label,
        infer_component_name_from_parameters,
    )

    # Try to infer component name from all parameters if model not provided
    component_name = None
    if model is None:
        component_name = infer_component_name_from_parameters(parameters)

    # Use independent absolute parameters only
    if 't_on_star' in parameters and 'delta_star' in parameters:
        t_on = parameters['t_on_star'].flatten().numpy()
        delta = parameters['delta_star'].flatten().numpy()

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
    total_genes: int,
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
            'mae_score' in adata.var and
            gene_idx < len(adata.var['mae_score'])):

            # Get the stored MAE score (negative value from mae_per_gene)
            mae_score = adata.var['mae_score'].iloc[gene_idx]

            # Convert to positive value for display (lower = better)
            display_mae = -mae_score

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
    total_genes: int
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
                          alpha=0.6, s=3, color=cluster_color_map[cluster], edgecolors='none')
        else:
            axes_dict[f"dynamics_{n}"].scatter(time_sorted, s_sorted, alpha=0.6, s=3, color='steelblue', edgecolors='none')
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
    from pyrovelocity.plots._common import set_colorbar

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
    from pyrovelocity.plots._common import set_colorbar

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

        # Compute and display Wasserstein distance as a simple error metric
        try:
            from scipy.stats import wasserstein_distance
            wd = wasserstein_distance(predicted_log, observed_log)
            axes_dict[f"marginal_{n}"].text(
                0.02, 0.98, f'WD: {wd:.2f}',
                transform=axes_dict[f"marginal_{n}"].transAxes,
                fontsize=default_fontsize * 0.7, va='top', ha='left',
                color='black', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none')
            )
        except ImportError:
            # Fallback to simple MAE if scipy not available
            mae = np.mean(np.abs(predicted_log - np.interp(
                np.linspace(0, 1, len(predicted_log)),
                np.linspace(0, 1, len(observed_log)),
                np.sort(observed_log)
            )))
            axes_dict[f"marginal_{n}"].text(
                0.02, 0.98, f'MAE: {mae:.2f}',
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


def _plot_pattern_proportions(
    adata: AnnData,
    parameters: Dict[str, torch.Tensor],
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
        def extract_gene_suffix(gene_name: str) -> str:
            """Extract numeric suffix from gene names like 'gene_23' -> '23'."""
            digits = "".join(filter(str.isdigit, gene_name))
            return digits if digits else gene_name[:6]  # Fallback to truncated name

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


def _classify_patterns_from_parameters(parameters: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """
    Classify expression patterns from parameter samples using soft scoring approach.

    This function uses the same pattern classification logic as the prior hyperparameter
    calibration script to ensure consistency between validation scripts.

    Updated to use the simplified 3-pattern classification system:
    - pre_activation: All patterns where activation occurred before observation window
    - transient: Complete activation-decay cycle within observation window
    - sustained: Net increase over observation window (includes late activation)
    """
    # Check for required parameters - use independent absolute parameters only
    required_params = ['R_on', 't_on_star', 'delta_star']
    if not all(key in parameters for key in required_params):
        return {}

    # Extract parameter arrays
    R_on = parameters['R_on'].flatten()
    t_on_star = parameters['t_on_star'].flatten()
    delta_star = parameters['delta_star'].flatten()

    n_samples = len(R_on)

    # Updated pattern constraints using independent absolute parameters
    # These match the mathematical specification thresholds
    pattern_constraints = {
        'pre_activation': {
            # All patterns where activation occurred before observation window
            # Results in observable decay-only dynamics from activated steady-state
            't_on_star': ('<', 0.0),      # Absolute activation before observation starts
            'R_on': ('>', 2.0),           # Moderate to strong fold change
        },
        'transient': {
            # Complete activation-decay cycle within observation window
            # Activation early enough and pulse short enough to see full cycle
            't_on_star': ('>', 0.0),      # Absolute activation within observation window
            't_on_star_upper': ('<', 1.5), # Early enough to complete cycle (absolute early onset)
            'delta_star': ('<', 2.0),     # Short enough pulse to see decay (absolute short duration)
            'R_on': ('>', 2.0),           # Sufficient fold change to observe
        },
        'sustained': {
            # Net increase over observation window (includes late activation)
            # Either long pulse or late activation that doesn't complete decay
            't_on_star': ('>', 0.0),      # Absolute activation within observation window
            't_on_star_upper': ('<', 1.5), # Early activation onset (absolute early onset)
            'delta_star': ('>', 2.5),     # Long activation duration (absolute long duration)
            'R_on': ('>', 2.0),           # Strong fold change
        }
    }

    # Soft scoring function (same as in calibration script)
    def sigmoid_score(value: float, threshold: float, direction: str, steepness: float = 5.0) -> float:
        """Compute soft score using sigmoid function."""
        if direction == '>':
            return torch.sigmoid(torch.tensor(steepness * (value - threshold))).item()
        else:  # direction == '<'
            return torch.sigmoid(torch.tensor(steepness * (threshold - value))).item()

    # Compute pattern scores for each sample
    pattern_scores = {}
    for pattern in pattern_constraints.keys():
        pattern_scores[pattern] = torch.zeros(n_samples)

    for i in range(n_samples):
        for pattern, constraints in pattern_constraints.items():
            scores = []

            # Use independent absolute temporal parameters with mathematical specification thresholds
            if pattern == 'pre_activation':
                scores = [
                    sigmoid_score(R_on[i], 2.0, '>'),
                    sigmoid_score(t_on_star[i], 0.0, '<')
                ]
            elif pattern == 'transient':
                scores = [
                    sigmoid_score(R_on[i], 2.0, '>'),
                    sigmoid_score(t_on_star[i], 0.0, '>'),
                    sigmoid_score(t_on_star[i], 1.5, '<'),  # Absolute early onset
                    sigmoid_score(delta_star[i], 2.0, '<')  # Absolute short duration
                ]
            elif pattern == 'sustained':
                scores = [
                    sigmoid_score(R_on[i], 2.0, '>'),
                    sigmoid_score(t_on_star[i], 0.0, '>'),
                    sigmoid_score(t_on_star[i], 1.5, '<'),  # Absolute early onset
                    sigmoid_score(delta_star[i], 2.5, '>')  # Absolute long duration
                ]

            # Compute geometric mean of scores
            if scores:
                pattern_scores[pattern][i] = torch.prod(torch.tensor(scores)) ** (1.0 / len(scores))

    # Assign each sample to the pattern with highest score
    assignments = torch.zeros(n_samples, dtype=torch.long)
    pattern_names = list(pattern_constraints.keys())

    for i in range(n_samples):
        scores = [pattern_scores[pattern][i] for pattern in pattern_names]
        assignments[i] = torch.argmax(torch.tensor(scores))

    # Count pattern assignments
    pattern_counts = {}
    for i, pattern in enumerate(pattern_names):
        count = (assignments == i).sum().item()
        if count > 0:
            pattern_counts[pattern] = count

    # Add unknown category for any unassigned samples (shouldn't happen with soft scoring)
    total_assigned = sum(pattern_counts.values())
    if total_assigned < n_samples:
        pattern_counts['unknown'] = n_samples - total_assigned

    return pattern_counts


# Alias for posterior predictive checks (same function, different check_type)
@beartype
def plot_posterior_predictive_checks(
    model: Any,
    posterior_adata: AnnData,
    posterior_parameters: Dict[str, torch.Tensor],
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
        ... )
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
def plot_training_loss(
    model: Any,
    figsize: Tuple[Union[int, float], Union[int, float]] = (7.5, 5.0),
    save_path: Optional[str] = None,
    file_prefix: str = "",
    default_fontsize: int = 8,
    moving_average_window: int = 50,
) -> plt.Figure:
    """
    Plot training loss (ELBO) over epochs.

    This function plots the Evidence Lower BOund (ELBO) during training, showing both
    raw loss values and a moving average. The ELBO is plotted as positive values
    (negative of the minimization objective) to align with Bayesian model selection
    conventions used in WAIC and LOO-CV.

    Args:
        model: PyroVelocity model instance with training history
        figsize: Figure size (width, height)
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        default_fontsize: Default font size for all text elements
        moving_average_window: Window size for moving average calculation

    Returns:
        matplotlib Figure object

    Raises:
        ValueError: If model has not been trained or training history is not available

    Example:
        >>> fig = plot_training_loss(
        ...     model=trained_model,
        ...     save_path="reports/docs/posterior_predictive",
        ...     file_prefix="07",
        ...     moving_average_window=100
        ... )
    """
    # Extract training history from model state
    if not hasattr(model, 'state') or model.state is None:
        raise ValueError("Model has no state - has it been trained?")

    inference_state = model.state.metadata.get("inference_state")
    if inference_state is None:
        raise ValueError("Model has no inference state - has it been trained?")

    training_state = inference_state.training_state
    if training_state is None or not training_state.loss_history:
        raise ValueError("Model has no training history - has it been trained with SVI?")

    # Extract loss history (negative ELBO values from minimization)
    loss_history = training_state.loss_history
    epochs = list(range(1, len(loss_history) + 1))

    # Convert to positive ELBO (negate the minimization objective)
    elbo_values = [-loss for loss in loss_history]

    # Calculate moving average
    moving_avg_values = []
    moving_avg_epochs = []

    if len(elbo_values) >= moving_average_window:
        # Use simple moving average to avoid pandas dependency issues
        for i in range(len(elbo_values)):
            start_idx = max(0, i - moving_average_window // 2)
            end_idx = min(len(elbo_values), i + moving_average_window // 2 + 1)
            avg_val = sum(elbo_values[start_idx:end_idx]) / (end_idx - start_idx)
            moving_avg_values.append(avg_val)
            moving_avg_epochs.append(epochs[i])
    else:
        # If not enough data points for moving average, skip it
        print(f"Warning: Not enough data points ({len(elbo_values)}) for moving average window ({moving_average_window})")
        moving_avg_values = []
        moving_avg_epochs = []

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot raw ELBO values
    ax.scatter(epochs, elbo_values, alpha=0.6, s=8, color='steelblue',
               label='ELBO', zorder=2)

    # Plot connecting line for raw values
    ax.plot(epochs, elbo_values, alpha=0.3, linewidth=0.5, color='steelblue', zorder=1)

    # Plot moving average if available
    if moving_avg_values:
        ax.plot(moving_avg_epochs, moving_avg_values, color='red', linewidth=1.5,
                label=f'Moving avg. ({moving_average_window})', zorder=3)

    # Set labels and title
    ax.set_xlabel('Epoch', fontsize=default_fontsize)
    ax.set_ylabel('ELBO', fontsize=default_fontsize)
    ax.set_title('Training Loss (Evidence Lower Bound)', fontsize=default_fontsize)

    # Add legend
    ax.legend(fontsize=default_fontsize * 0.9, loc='lower right')

    # Add grid
    ax.grid(True, alpha=0.3)

    # Set tick label size
    ax.tick_params(labelsize=default_fontsize * 0.9)

    # Tight layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path is not None:
        if file_prefix:
            figure_name = f"{file_prefix}_training_loss"
        else:
            figure_name = "training_loss"
        _save_figure(fig, save_path, figure_name)

    return fig


@beartype
def plot_parameter_recovery_correlation(
    posterior_parameters: Dict[str, torch.Tensor],
    true_parameters_adata: AnnData,
    parameters_to_validate: List[str] = ["R_on", "gamma_star", "t_on_star", "delta_star"],
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    save_path: Optional[str] = None,
    file_prefix: str = "",
    model: Optional[Any] = None,
    default_fontsize: int = 8,
    summary_statistic: str = "median"
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
        ... )
        >>> print(f"Mean correlation: {metrics['summary']['mean_pearson_r']:.3f}")
    """
    from pyrovelocity.plots.parameter_metadata import get_parameter_label

    # Validate inputs
    if 'true_parameters' not in true_parameters_adata.uns:
        raise ValueError("true_parameters_adata must contain 'true_parameters' in adata.uns")

    true_params_dict = true_parameters_adata.uns['true_parameters']

    # Filter parameters to only those available in both posterior and true parameters
    available_params = []
    for param_name in parameters_to_validate:
        if param_name in posterior_parameters and param_name in true_params_dict:
            available_params.append(param_name)
        else:
            print(f"Warning: Parameter '{param_name}' not found in both posterior and true parameters")

    if not available_params:
        raise ValueError("No valid parameters found for correlation analysis")

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
        if not isinstance(true_param, torch.Tensor):
            true_param = torch.tensor(true_param)

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
                posterior_reshaped = posterior_param.view(num_samples, num_genes)
            else:
                # Fallback: treat as single sample per gene
                posterior_reshaped = posterior_param.unsqueeze(0)
        elif posterior_param.ndim == 2:
            # Already shaped: [num_samples, num_genes]
            posterior_reshaped = posterior_param
        else:
            # Higher dimensions: flatten and reshape
            posterior_reshaped = posterior_param.view(-1, posterior_param.shape[-1])

        # Compute summary statistic and standard deviation across samples
        if summary_statistic == "median":
            posterior_summary = torch.median(posterior_reshaped, dim=0)[0]
            # For median, use MAD (median absolute deviation) scaled to approximate std
            mad = torch.median(torch.abs(posterior_reshaped - posterior_summary.unsqueeze(0)), dim=0)[0]
            posterior_std = 1.4826 * mad  # Scale factor to approximate std from MAD
        elif summary_statistic == "mean":
            posterior_summary = torch.mean(posterior_reshaped, dim=0)
            posterior_std = torch.std(posterior_reshaped, dim=0)
        else:
            raise ValueError(f"Unknown summary_statistic: {summary_statistic}")

        # Ensure we have the same number of genes
        n_genes = min(len(true_param_flat), len(posterior_summary))
        true_values = true_param_flat[:n_genes].numpy()
        estimated_values = posterior_summary[:n_genes].detach().numpy()
        estimated_std = posterior_std[:n_genes].detach().numpy()

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

        # Create scatter plot with error bars
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
            label_type="short",
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
