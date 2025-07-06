"""
Utility functions for predictive checks.

This module contains shared utility functions for file management,
text formatting, and other common operations.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from beartype import beartype


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
        't_star_normalized': r't^*_N',
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
    # Import here to avoid circular imports
    from .core import compute_and_store_mae
    
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