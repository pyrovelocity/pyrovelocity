"""
Training analysis plotting functions for predictive checks.

This module contains functions for analyzing training progress
and performance.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from beartype import beartype
from scipy.stats import pearsonr

from .core import compute_and_store_mae
from .parameters import _save_figure


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

    # Use the new type-safe inference state field instead of metadata
    inference_state = model.state.inference_state
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
def plot_mae_vs_spliced_count(
    predicted_adata: AnnData,
    observed_adata: Optional[AnnData] = None,
    figsize: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    save_path: Optional[str] = None,
    file_prefix: str = "",
    default_fontsize: int = 8,
    check_type: str = "posterior"
) -> plt.Figure:
    """
    Plot relationship between MAE and maximum/median spliced count per gene.
    
    This function creates scatter plots showing the relationship between gene-specific
    Mean Absolute Error (MAE) and the maximum or median spliced count for each gene,
    helping identify whether expression level correlates with model fitting accuracy.
    
    Args:
        predicted_adata: AnnData object with predicted data containing MAE scores
        observed_adata: Optional AnnData object with observed data. If None, uses predicted_adata
        figsize: Optional figure size (auto-calculated if None)
        save_path: Optional directory path to save figures
        file_prefix: Prefix for saved file names
        default_fontsize: Default font size for all text elements
        check_type: Type of check ("prior" or "posterior")
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> fig = plot_mae_vs_spliced_count(
        ...     predicted_adata=posterior_adata,
        ...     observed_adata=original_adata,
        ...     save_path="reports/docs/posterior_predictive",
        ...     file_prefix="15"
        ... )
    """
    # Use observed_adata if provided, otherwise use predicted_adata
    data_adata = observed_adata if observed_adata is not None else predicted_adata
    
    # Get spliced counts
    spliced_counts = data_adata.layers['spliced']
    if hasattr(spliced_counts, 'toarray'):
        spliced_counts = spliced_counts.toarray()
    
    # Compute max and median spliced counts per gene
    max_spliced = np.max(spliced_counts, axis=0)
    median_spliced = np.median(spliced_counts, axis=0)
    
    # Get MAE scores - check if already computed in predicted_adata
    if 'mae_combined' in predicted_adata.var.columns:
        mae_scores_positive = predicted_adata.var['mae_combined'].values
    elif 'mae_score' in predicted_adata.var.columns:
        mae_scores_positive = predicted_adata.var['mae_score'].values
    else:
        # Compute MAE if not already available
        if observed_adata is None:
            raise ValueError("Cannot compute MAE without observed_adata when MAE scores not pre-computed")
        
        # Use our centralized MAE computation function
        mae_scores_positive = compute_and_store_mae(predicted_adata, observed_adata, store_in_var=True)
    
    # Create figure with two subplots
    if figsize is None:
        figsize = (10, 4)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: MAE vs Maximum Spliced Count
    ax1.scatter(max_spliced, mae_scores_positive, alpha=0.6, s=20, color='steelblue')
    ax1.set_xlabel('Maximum Spliced Count', fontsize=default_fontsize)
    ax1.set_ylabel('Mean Absolute Error', fontsize=default_fontsize)
    ax1.set_title(f'{check_type.title()}: MAE vs Maximum Spliced Count', fontsize=default_fontsize)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=default_fontsize * 0.8)
    
    # Add correlation coefficient
    corr_max, p_val_max = pearsonr(max_spliced[~np.isnan(mae_scores_positive)], 
                                    mae_scores_positive[~np.isnan(mae_scores_positive)])
    ax1.text(0.02, 0.98, f'$r = {corr_max:.3f}$\n$p = {p_val_max:.3e}$', 
             transform=ax1.transAxes, fontsize=default_fontsize * 0.9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: MAE vs Median Spliced Count
    ax2.scatter(median_spliced, mae_scores_positive, alpha=0.6, s=20, color='darkorange')
    ax2.set_xlabel('Median Spliced Count', fontsize=default_fontsize)
    ax2.set_ylabel('Mean Absolute Error', fontsize=default_fontsize)
    ax2.set_title(f'{check_type.title()}: MAE vs Median Spliced Count', fontsize=default_fontsize)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=default_fontsize * 0.8)
    
    # Add correlation coefficient
    corr_median, p_val_median = pearsonr(median_spliced[~np.isnan(mae_scores_positive)], 
                                         mae_scores_positive[~np.isnan(mae_scores_positive)])
    ax2.text(0.02, 0.98, f'$r = {corr_median:.3f}$\n$p = {p_val_median:.3e}$', 
             transform=ax2.transAxes, fontsize=default_fontsize * 0.9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Log scale for x-axis if values span multiple orders of magnitude
    if max_spliced.max() / (max_spliced[max_spliced > 0].min() + 1e-10) > 100:
        ax1.set_xscale('log')
        ax1.set_xlabel('Maximum Spliced Count (log scale)', fontsize=default_fontsize)
    
    if median_spliced.max() / (median_spliced[median_spliced > 0].min() + 1e-10) > 100:
        ax2.set_xscale('log')
        ax2.set_xlabel('Median Spliced Count (log scale)', fontsize=default_fontsize)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        if file_prefix:
            figure_name = f"{file_prefix}_{check_type}_mae_vs_spliced_count"
        else:
            figure_name = f"{check_type}_mae_vs_spliced_count"
        _save_figure(fig, save_path, figure_name)
    
    return fig