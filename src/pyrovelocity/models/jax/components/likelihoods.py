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


def poisson_likelihood_function(
    u_obs: Float[Array, "batch_size n_cells n_genes"],
    s_obs: Float[Array, "batch_size n_cells n_genes"],
    u_logits: Float[Array, "batch_size n_cells n_genes"],
    s_logits: Float[Array, "batch_size n_cells n_genes"],
    likelihood_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Poisson likelihood function for RNA velocity.

    This function samples observed RNA counts from Poisson distributions.

    Args:
        u_obs: Observed unspliced counts
        s_obs: Observed spliced counts
        u_logits: Expected unspliced counts
        s_logits: Expected spliced counts
        likelihood_params: Dictionary of likelihood parameters
    """
    if likelihood_params is None:
        likelihood_params = {}

    # Get library size scaling
    u_log_library = likelihood_params.get("u_log_library")
    s_log_library = likelihood_params.get("s_log_library")

    # Apply library size scaling if provided
    if u_log_library is not None:
        u_logits = u_logits * jnp.exp(u_log_library[:, :, jnp.newaxis])
    if s_log_library is not None:
        s_logits = s_logits * jnp.exp(s_log_library[:, :, jnp.newaxis])

    # Ensure positive values for Poisson distribution
    u_logits = jnp.maximum(u_logits, 1e-6)
    s_logits = jnp.maximum(s_logits, 1e-6)

    # Sample from Poisson distribution
    numpyro.sample("u", dist.Poisson(u_logits).to_event(2), obs=u_obs)
    numpyro.sample("s", dist.Poisson(s_logits).to_event(2), obs=s_obs)


def negative_binomial_likelihood_function(
    u_obs: Float[Array, "batch_size n_cells n_genes"],
    s_obs: Float[Array, "batch_size n_cells n_genes"],
    u_logits: Float[Array, "batch_size n_cells n_genes"],
    s_logits: Float[Array, "batch_size n_cells n_genes"],
    likelihood_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Negative binomial likelihood function for RNA velocity.

    This function samples observed RNA counts from negative binomial distributions.

    Args:
        u_obs: Observed unspliced counts
        s_obs: Observed spliced counts
        u_logits: Expected unspliced counts
        s_logits: Expected spliced counts
        likelihood_params: Dictionary of likelihood parameters
    """
    if likelihood_params is None:
        likelihood_params = {}

    # Get library size scaling
    u_log_library = likelihood_params.get("u_log_library")
    s_log_library = likelihood_params.get("s_log_library")

    # Apply library size scaling if provided
    if u_log_library is not None:
        u_logits = u_logits * jnp.exp(u_log_library[:, :, jnp.newaxis])
    if s_log_library is not None:
        s_logits = s_logits * jnp.exp(s_log_library[:, :, jnp.newaxis])

    # Get dispersion parameters
    u_dispersion = likelihood_params.get("u_dispersion", 1.0)
    s_dispersion = likelihood_params.get("s_dispersion", 1.0)

    # Ensure positive values for NegativeBinomial distribution
    u_logits = jnp.maximum(u_logits, 1e-6)
    s_logits = jnp.maximum(s_logits, 1e-6)

    # Sample from negative binomial distribution
    numpyro.sample(
        "u",
        dist.NegativeBinomial2(u_logits, u_dispersion).to_event(2),
        obs=u_obs,
    )
    numpyro.sample(
        "s",
        dist.NegativeBinomial2(s_logits, s_dispersion).to_event(2),
        obs=s_obs,
    )


@jaxtyped(typechecker=beartype)
def piecewise_activation_likelihood_function(
    u_obs: Float[Array, "batch_size n_cells n_genes"],
    s_obs: Float[Array, "batch_size n_cells n_genes"], 
    u_logits: Float[Array, "batch_size n_cells n_genes"],
    s_logits: Float[Array, "batch_size n_cells n_genes"],
    likelihood_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Piecewise activation Poisson likelihood function for NumPyro.
    
    This function implements the observation model for the piecewise activation
    RNA velocity model. Following the modular pattern, it receives latent
    dimensionless concentrations (ut, st) and technical parameters (lambda_j, U_0i)
    via likelihood_params, then computes rates internally.
    
    Mathematical Formulation:
    
    Observation Model:
        u_{ij} ∼ Poisson(λ_j · U_{0i} · u*_{ij})
        s_{ij} ∼ Poisson(λ_j · U_{0i} · s*_{ij})
        
    Where:
        λ_j = cell-specific effective capture efficiency [n_cells]
        U_{0i} = gene-specific characteristic concentration scale [n_genes]
        u*_{ij}, s*_{ij} = latent dimensionless RNA concentrations [n_cells, n_genes]
        
    Note: This function receives u_logits/s_logits to match the JAX interface,
    but interprets them as latent concentrations ut/st following the modular pattern.

    Args:
        u_obs: Observed unspliced RNA counts [batch_size, n_cells, n_genes]
        s_obs: Observed spliced RNA counts [batch_size, n_cells, n_genes]
        u_logits: Latent dimensionless unspliced concentrations (ut) [batch_size, n_cells, n_genes]
        s_logits: Latent dimensionless spliced concentrations (st) [batch_size, n_cells, n_genes]
        likelihood_params: Dictionary containing:
            - lambda_j: Cell-specific capture efficiency [n_cells]
            - U_0i: Gene-specific concentration scale [n_genes]
            - eps: Numerical stability epsilon (optional)
        
    Returns:
        None: Uses numpyro.sample to register observations in probabilistic model
    """
    if likelihood_params is None:
        likelihood_params = {}
    
    # Extract technical parameters from likelihood_params (following modular pattern)
    lambda_j = likelihood_params.get("lambda_j")
    U_0i = likelihood_params.get("U_0i")
    eps = likelihood_params.get("eps", 1e-6)
    
    # For piecewise activation, u_logits/s_logits are actually ut/st (latent concentrations)
    ut = u_logits
    st = s_logits
    
    # Compute rates following the modular mathematical specification
    if lambda_j is not None and U_0i is not None:
        # Handle tensor broadcasting for rate computation
        # lambda_j: [n_cells] -> [batch_size, n_cells, 1] 
        # U_0i: [n_genes] -> [batch_size, 1, n_genes]
        # ut, st: [batch_size, n_cells, n_genes]
        lambda_j_expanded = lambda_j[jnp.newaxis, :, jnp.newaxis]  # [1, n_cells, 1]
        U_0i_expanded = U_0i[jnp.newaxis, jnp.newaxis, :]         # [1, 1, n_genes]
        
        # Compute rates: [1, n_cells, 1] * [1, 1, n_genes] * [batch_size, n_cells, n_genes]
        u_rate = lambda_j_expanded * U_0i_expanded * ut
        s_rate = lambda_j_expanded * U_0i_expanded * st
    else:
        # Fallback: use latent concentrations directly as rates
        u_rate = ut
        s_rate = st
    
    # Ensure positive rates for numerical stability
    u_rate = jnp.maximum(u_rate, eps)
    s_rate = jnp.maximum(s_rate, eps)
    
    # Convert observations to integers for Poisson distribution compatibility
    u_obs_int = jnp.round(u_obs).astype(jnp.int32)
    s_obs_int = jnp.round(s_obs).astype(jnp.int32)
    
    # Sample from Poisson distributions
    numpyro.sample("u_obs", dist.Poisson(rate=u_rate).to_event(2), obs=u_obs_int)
    numpyro.sample("s_obs", dist.Poisson(rate=s_rate).to_event(2), obs=s_obs_int)


def register_standard_likelihoods():
    """Register standard likelihood functions."""
    register_likelihood("poisson", poisson_likelihood_function)
    register_likelihood(
        "negative_binomial", negative_binomial_likelihood_function
    )
    register_likelihood("piecewise_activation", piecewise_activation_likelihood_function)
