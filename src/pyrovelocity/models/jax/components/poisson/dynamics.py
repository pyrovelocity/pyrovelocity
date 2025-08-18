"""
Poisson-only dynamics functions for PyroVelocity JAX/NumPyro implementation.

This module registers Poisson-only dynamics functions for the JAX implementation of PyroVelocity.
The Poisson dynamics model implements a trivial dynamics function that returns constant
RNA concentrations, effectively removing temporal evolution from the model.
"""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.jax.registry import register_dynamics


@jax.jit
@jaxtyped(typechecker=beartype)
def poisson_dynamics_function(
    t_star: Float[Array, "batch_size n_cells"],
    u0_star: Float[Array, "batch_size n_cells n_genes"],
    params: Dict[str, Float[Array, "..."]],
) -> Tuple[
    Float[Array, "batch_size n_cells n_genes"],
    Float[Array, "batch_size n_cells n_genes"],
]:
    """
    Poisson-only dynamics function with no temporal evolution.

    This function implements a trivial dynamics model for the Poisson-only
    RNA velocity model. It returns constant RNA concentrations that do not
    evolve over time, effectively reducing the model to a pure observation
    model without temporal dynamics.
    
    Mathematical Formulation:
    
    The dynamics equations are trivially satisfied:
        du*/dt* = 0  =>  u*(t*) = u*_0 (constant)
        ds*/dt* = 0  =>  s*(t*) = s*_0 (constant)
    
    This removes all temporal complexity and focuses purely on modeling
    the Poisson observation process for RNA count data.

    Args:
        t_star: Dimensionless time parameter [batch_size, n_cells] (unused)
        u0_star: Initial dimensionless unspliced RNA [batch_size, n_cells, n_genes]
        params: Dictionary of model parameters (unused for this trivial dynamics)

    Returns:
        Tuple of (u_star, s_star):
            - u_star: Dimensionless unspliced RNA = u0_star (constant)
            - s_star: Dimensionless spliced RNA = u0_star (constant, same as unspliced)
    """
    # Trivial dynamics: RNA concentrations remain constant over time
    # This effectively removes temporal evolution from the model
    u_star = u0_star  # Unspliced RNA remains at initial concentration
    s_star = u0_star  # Spliced RNA equals unspliced (no splicing dynamics)
    
    # Ensure positive values for numerical stability
    eps = 1e-6
    u_star = jnp.maximum(u_star, eps)
    s_star = jnp.maximum(s_star, eps)
    
    return u_star, s_star


def register_poisson_dynamics():
    """Register Poisson-only dynamics functions."""
    register_dynamics("poisson", poisson_dynamics_function)
    register_dynamics("poisson_dynamics", poisson_dynamics_function)  # Alias