"""
Standard observation functions for PyroVelocity JAX/NumPyro implementation.

This module registers standard observation functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Dict, Tuple

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.jax.registry import register_observation


@jaxtyped(typechecker=beartype)
def standard_observation_function(
    u_obs: Float[Array, "batch_size n_cells n_genes"],
    s_obs: Float[Array, "batch_size n_cells n_genes"],
    observation_params: Dict[str, Any]
) -> Tuple[Float[Array, "batch_size n_cells n_genes"], Float[Array, "batch_size n_cells n_genes"]]:
    """
    Standard observation function that passes through observations unchanged.
    
    This function implements the identity transformation for observations,
    returning the input arrays unchanged. This is the simplest observation
    function and is suitable for most use cases.
    
    Args:
        u_obs: Observed unspliced counts
        s_obs: Observed spliced counts
        observation_params: Additional parameters (unused in standard function)
        
    Returns:
        Tuple of (u_transformed, s_transformed) where transformations are identity
    """
    return u_obs, s_obs


def register_standard_observations():
    """Register standard observation functions."""
    register_observation("standard", standard_observation_function)