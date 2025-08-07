"""
Standard likelihood functions for PyroVelocity JAX/NumPyro implementation.

This module registers standard likelihood functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.jax.registry import register_likelihood


@jaxtyped(typechecker=beartype)
def piecewise_activation_likelihood_function(
    context: Dict[str, Any]
) -> None:
    """
    Piecewise activation Poisson likelihood function for NumPyro.
    
    This function implements the observation model for the piecewise activation
    RNA velocity model. It receives expected RNA concentrations and scaling
    parameters via the context dictionary.
    
    Mathematical Formulation:
    
    Observation Model:
        u_{ij} ∼ Poisson(λ_j · U_{0i} · u*_{ij})
        s_{ij} ∼ Poisson(λ_j · U_{0i} · s*_{ij})
        
    Where:
        λ_j = cell-specific effective capture efficiency
        U_{0i} = gene-specific characteristic concentration scale
        u*_{ij}, s*_{ij} = expected RNA concentrations

    Args:
        context: Dictionary containing:
            - u_obs: Observed unspliced RNA counts
            - s_obs: Observed spliced RNA counts  
            - u_expected: Expected unspliced concentrations
            - s_expected: Expected spliced concentrations
            - u_log_library: Log library size for unspliced counts (optional)
            - s_log_library: Log library size for spliced counts (optional)
            - Additional likelihood parameters
        
    Returns:
        None: Uses numpyro.sample to register observations in probabilistic model
    """
    # Extract required parameters from context
    # Use get() method to allow None values for prior predictive sampling
    u_obs = context.get("u_obs")
    s_obs = context.get("s_obs")
    u_star = context["u_star"]  # Dimensionless concentrations from dynamics
    s_star = context["s_star"]  # Dimensionless concentrations from dynamics
    
    # Extract scaling parameters from context
    lambda_j = context["lambda_j"]  # Cell-specific capture efficiency [n_cells]
    U_0i = context["U_0i"]  # Gene-specific concentration scale [n_genes]
    eps = context.get("eps", 1e-6)
    
    # Apply observation model scaling according to mathematical specification
    # u_{ij} ~ Poisson(λ_j · U_{0i} · u*_{ij})
    # s_{ij} ~ Poisson(λ_j · U_{0i} · s*_{ij})
    
    # Handle potential extra dimensions from explicit plate dimensions
    # Ensure lambda_j is 1D [n_cells]
    if lambda_j.ndim > 1:
        lambda_j = jnp.squeeze(lambda_j)
    # Ensure U_0i is 1D [n_genes]  
    if U_0i.ndim > 1:
        U_0i = jnp.squeeze(U_0i)
    
    # Proper 2D broadcasting to compute rates:
    # lambda_j: [N] -> [N, 1], U_0i: [G] -> [1, G] 
    # Result: [N, 1] * [1, G] * [N, G] = [N, G]
    lambda_j_expanded = lambda_j[:, jnp.newaxis]  # [N, 1]
    U_0i_expanded = U_0i[jnp.newaxis, :]  # [1, G]
    
    # Apply scaling to get Poisson rates
    u_rate = lambda_j_expanded * U_0i_expanded * u_star
    s_rate = lambda_j_expanded * U_0i_expanded * s_star
    
    # Ensure positive rates for numerical stability
    u_rate = jnp.maximum(u_rate, eps)
    s_rate = jnp.maximum(s_rate, eps)
    
    # Apply library size scaling if available
    # if u_log_library is not None:
    #     u_rate = u_rate * jnp.exp(u_log_library)[..., jnp.newaxis]
    # if s_log_library is not None:
    #     s_rate = s_rate * jnp.exp(s_log_library)[..., jnp.newaxis]
    
    # Ensure positive rates for numerical stability
    u_rate = jnp.maximum(u_rate, eps)
    s_rate = jnp.maximum(s_rate, eps)
    
    # Convert observations to integers for Poisson distribution compatibility
    # Handle None observations for prior predictive sampling
    u_obs_int = None if u_obs is None else jnp.round(u_obs).astype(jnp.int32)
    s_obs_int = None if s_obs is None else jnp.round(s_obs).astype(jnp.int32)
    
    # Sample from Poisson distributions with proper independence structure
    # When obs=None, NumPyro performs unconditional sampling (prior predictive)
    # Get dimensions for proper plate notation
    batch_shape = u_rate.shape
    if len(batch_shape) >= 2:
        n_cells = batch_shape[-2]
        n_genes = batch_shape[-1]
    else:
        raise ValueError(f"Expected at least 2D tensor for rates, got shape {batch_shape}")
    
    # Use proper plate notation with consistent names and explicit dimensions
    with numpyro.plate("cells", n_cells, dim=-2):
        with numpyro.plate("genes", n_genes, dim=-1):
            numpyro.sample("u_obs", dist.Poisson(rate=u_rate), obs=u_obs_int)
            numpyro.sample("s_obs", dist.Poisson(rate=s_rate), obs=s_obs_int)


def register_standard_likelihoods():
    """Register standard likelihood functions."""
    register_likelihood("piecewise_activation", piecewise_activation_likelihood_function)
    register_likelihood("piecewise_activation_poisson_likelihood", piecewise_activation_likelihood_function)  # Alias for tests
    
    # Register alternative observation models to demonstrate multi-model extensibility
    from pyrovelocity.models.jax.components.alternative_likelihoods import (
        negative_binomial_likelihood_function,
        gaussian_likelihood_function
    )
    register_likelihood("negative_binomial", negative_binomial_likelihood_function)
    register_likelihood("gaussian", gaussian_likelihood_function)
