"""
Core computation functions for predictive checks.

This module contains the fundamental computation functions that are used
across multiple plotting functions, including MAE computation, parameter processing,
and temporal calculations.
"""

from typing import Dict

import numpy as np
from anndata import AnnData
from beartype import beartype
from numpy.typing import ArrayLike
from pyrovelocity.plots.tensor_utils import convert_to_numpy
from scipy.stats import linregress


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
def _process_parameters_for_plotting(
    parameters: Dict[str, ArrayLike]
) -> Dict[str, ArrayLike]:
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
        'guide_',
        '_unconstrained'  # Filter out unconstrained parameters from AutoGuides
    ]

    for key, value in parameters.items():
        # Skip guide-specific parameters
        if any(pattern in key for pattern in guide_param_patterns):
            continue

        if hasattr(value, 'shape'):  # Works for both torch.Tensor and np.ndarray
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

    # Note: With independent absolute parameterization, no hierarchical computation needed
    # t_on_star and delta_star are now independent absolute parameters sampled directly

    return processed_parameters