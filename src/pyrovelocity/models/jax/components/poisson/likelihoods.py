"""
Poisson-only likelihood functions for PyroVelocity JAX/NumPyro implementation.

This module registers Poisson-only likelihood functions for the JAX implementation of PyroVelocity.
The Poisson likelihood model implements a pure Poisson observation model without
temporal dynamics, focusing solely on modeling RNA count data.
"""

from typing import Any, Dict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.jax.registry import register_likelihood


@jaxtyped(typechecker=beartype)
def poisson_likelihood_function(
    context: Dict[str, Any]
) -> None:
    """
    Poisson-only likelihood function for NumPyro.

    This function implements a pure Poisson observation model without temporal
    dynamics. It models RNA count data directly using Poisson distributions
    with the correct multiplicative structure from the mathematical specification.

    Mathematical Formulation:

    Rate Structure:
        u_rate_ij = λ_j × U_{0i} × r_u,i
        s_rate_ij = λ_j × U_{0i} × r_s,i

    Observation Model:
        u_{ij} ∼ Poisson(u_rate_ij)
        s_{ij} ∼ Poisson(s_rate_ij)

    Where:
        λ_j = cell-specific library size scaling (capture efficiency)
        U_{0i} = gene-specific expression capacity
        r_u,i = gene-specific unspliced rate multiplier
        r_s,i = gene-specific spliced rate multiplier

    Args:
        context: Dictionary containing:
            - u_obs: Observed unspliced RNA counts
            - s_obs: Observed spliced RNA counts  
            - u_star: Expected unspliced concentrations (constant from dynamics)
            - s_star: Expected spliced concentrations (constant from dynamics)
            - lambda_j: Cell-specific library scaling parameters
            - U_0i: Gene-specific expression capacity parameters
            - r_u_i: Gene-specific unspliced rate multipliers
            - r_s_i: Gene-specific spliced rate multipliers
            - Additional likelihood parameters

    Returns:
        None: Uses numpyro.sample to register observations in probabilistic model
    """
    # Extract required parameters from context
    # Use get() method to allow None values for prior predictive sampling
    u_obs = context.get("u_obs")
    s_obs = context.get("s_obs")
    u_star = context["u_star"]  # Constant concentrations from trivial dynamics
    s_star = context["s_star"]  # Constant concentrations from trivial dynamics

    # Extract scaling parameters from context (corrected parameter names)
    lambda_j = context["lambda_j"]  # Cell-specific library scaling [n_cells]
    U_0i = context["U_0i"]         # Gene expression capacity [n_genes] 
    r_u_i = context["r_u_i"]       # Unspliced rate multiplier [n_genes]
    r_s_i = context["r_s_i"]       # Spliced rate multiplier [n_genes]
    eps = context.get("eps", 1e-6)

    # Handle potential extra dimensions from explicit plate dimensions
    # Ensure lambda_j is 1D [n_cells] but maintain at least 1D shape
    if lambda_j.ndim > 1:
        lambda_j = jnp.squeeze(lambda_j)
    if lambda_j.ndim == 0:  # Handle scalar case
        lambda_j = jnp.array([lambda_j])
        
    # Ensure gene scales are 1D [n_genes] but maintain at least 1D shape 
    if U_0i.ndim > 1:
        U_0i = jnp.squeeze(U_0i)
    if U_0i.ndim == 0:  # Handle scalar case
        U_0i = jnp.array([U_0i])
        
    if r_u_i.ndim > 1:
        r_u_i = jnp.squeeze(r_u_i)
    if r_u_i.ndim == 0:  # Handle scalar case
        r_u_i = jnp.array([r_u_i])
        
    if r_s_i.ndim > 1:
        r_s_i = jnp.squeeze(r_s_i)
    if r_s_i.ndim == 0:  # Handle scalar case
        r_s_i = jnp.array([r_s_i])

    # Proper 2D broadcasting to compute Poisson rates:
    # lambda_j: [N] -> [N, 1], scales: [G] -> [1, G]
    # Result: [N, 1] * [1, G] * [1, G] = [N, G]
    lambda_j_expanded = lambda_j[:, jnp.newaxis]  # [N, 1]
    U_0i_expanded = U_0i[jnp.newaxis, :]          # [1, G]
    r_u_i_expanded = r_u_i[jnp.newaxis, :]       # [1, G]
    r_s_i_expanded = r_s_i[jnp.newaxis, :]       # [1, G]

    # Apply scaling to get Poisson rates using correct multiplicative structure
    # Note: u_star and s_star are constant from trivial dynamics (typically 1.0)
    u_rate = lambda_j_expanded * U_0i_expanded * r_u_i_expanded * u_star
    s_rate = lambda_j_expanded * U_0i_expanded * r_s_i_expanded * s_star

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


def register_poisson_likelihoods():
    """Register Poisson-only likelihood functions."""
    register_likelihood("poisson", poisson_likelihood_function)
    register_likelihood("poisson_likelihood", poisson_likelihood_function)  # Alias