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
    u_obs = context["u_obs"]
    s_obs = context["s_obs"]
    u_expected = context["u_expected"]
    s_expected = context["s_expected"]
    
    # Extract optional scaling parameters
    u_log_library = context.get("u_log_library")
    s_log_library = context.get("s_log_library")
    eps = context.get("eps", 1e-6)
    
    # Use expected concentrations directly as rates
    ut = u_expected
    st = s_expected
    
    # Apply library size scaling if available
    if u_log_library is not None:
        u_rate = ut * jnp.exp(u_log_library)[:, jnp.newaxis]
    else:
        u_rate = ut
        
    if s_log_library is not None:
        s_rate = st * jnp.exp(s_log_library)[:, jnp.newaxis]
    else:
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
    register_likelihood("piecewise_activation", piecewise_activation_likelihood_function)
    register_likelihood("piecewise_activation_poisson_likelihood", piecewise_activation_likelihood_function)  # Alias for tests
