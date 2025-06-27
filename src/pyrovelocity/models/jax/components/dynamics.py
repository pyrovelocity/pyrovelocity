"""
Piecewise activation dynamics functions for PyroVelocity JAX/NumPyro implementation.

This module registers piecewise activation dynamics functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrovelocity.models.jax.registry import register_dynamics


@jax.jit
def piecewise_activation_dynamics_function(
    t_star: Float[Array, "batch_size n_cells n_genes"],
    u0_star: Float[Array, "batch_size n_cells n_genes"],
    s0_star: Float[Array, "batch_size n_cells n_genes"],
    params: Dict[str, Float[Array, "..."]],
) -> Tuple[
    Float[Array, "batch_size n_cells n_genes"],
    Float[Array, "batch_size n_cells n_genes"],
]:
    """
    Piecewise activation RNA velocity dynamics function.

    This function implements the piecewise activation RNA velocity model:

    du*/dt* = α*(t*) - u*
    ds*/dt* = u* - γ*s*

    Where α*(t*) is piecewise:
    - Phase 1 (Off): t* < t*_on, α*(t*) = 1.0
    - Phase 2 (On): t*_on ≤ t* < t*_on + δ*, α*(t*) = R_on  
    - Phase 3 (Return to Off): t* ≥ t*_on + δ*, α*(t*) = 1.0

    Args:
        t_star: Dimensionless time parameter 
        u0_star: Initial dimensionless unspliced RNA (= 1.0)
        s0_star: Initial dimensionless spliced RNA (= 1.0/γ*)
        params: Dictionary of parameters (R_on, gamma_star, t_on_star, delta_star)

    Returns:
        Tuple of (dimensionless unspliced, dimensionless spliced) RNA
    """
    R_on = params["R_on"]
    gamma_star = params["gamma_star"]
    t_on_star = params["t_on_star"]
    delta_star = params["delta_star"]
    
    # Numerical stability epsilon for gamma_star near 1
    eps = 1e-8
    gamma_stable = jnp.where(jnp.abs(gamma_star - 1.0) < eps, 1.0 + eps, gamma_star)
    
    # Phase boundaries
    t_start_on = t_on_star
    t_end_on = t_on_star + delta_star
    
    # Phase 1: t* < t*_on (OFF phase)
    # Solutions: u*(t*) = 1, s*(t*) = 1/γ* 
    phase1_mask = t_star < t_start_on
    u_phase1 = jnp.ones_like(t_star)
    s_phase1 = 1.0 / gamma_stable
    
    # Phase 2: t*_on ≤ t* < t*_on + δ* (ON phase)
    # Analytical solution with α* = R_on
    phase2_mask = (t_star >= t_start_on) & (t_star < t_end_on)
    dt2 = t_star - t_start_on
    
    # Phase 2 solutions starting from steady state (1, 1/γ*)
    exp_dt2 = jnp.exp(-dt2)
    u_phase2 = R_on + (1.0 - R_on) * exp_dt2
    
    # s*(t*) solution for phase 2
    gamma_diff = gamma_stable - 1.0
    s_phase2 = jnp.where(
        jnp.abs(gamma_diff) < eps,
        # Near γ* = 1 case: analytical limit
        (1.0 / gamma_stable) + dt2 * (R_on - 1.0) * exp_dt2,
        # General case
        (R_on / gamma_stable) + 
        ((1.0 / gamma_stable) - (R_on / gamma_stable)) * jnp.exp(-gamma_stable * dt2) +
        ((R_on - 1.0) / gamma_diff) * (exp_dt2 - jnp.exp(-gamma_stable * dt2))
    )
    
    # Phase 3: t* ≥ t*_on + δ* (Return to OFF)
    # Solutions starting from end of phase 2
    phase3_mask = t_star >= t_end_on
    dt3 = t_star - t_end_on
    
    # Initial conditions for phase 3 (end values of phase 2)
    exp_delta = jnp.exp(-delta_star)
    u2_end = R_on + (1.0 - R_on) * exp_delta
    
    gamma_diff = gamma_stable - 1.0
    s2_end = jnp.where(
        jnp.abs(gamma_diff) < eps,
        (1.0 / gamma_stable) + delta_star * (R_on - 1.0) * exp_delta,
        (R_on / gamma_stable) + 
        ((1.0 / gamma_stable) - (R_on / gamma_stable)) * jnp.exp(-gamma_stable * delta_star) +
        ((R_on - 1.0) / gamma_diff) * (exp_delta - jnp.exp(-gamma_stable * delta_star))
    )
    
    # Phase 3 solutions with α* = 1.0
    exp_dt3 = jnp.exp(-dt3)
    u_phase3 = 1.0 + (u2_end - 1.0) * exp_dt3
    
    s_phase3 = jnp.where(
        jnp.abs(gamma_diff) < eps,
        (1.0 / gamma_stable) + dt3 * (u2_end - 1.0) * exp_dt3,
        (1.0 / gamma_stable) + 
        (s2_end - (1.0 / gamma_stable)) * jnp.exp(-gamma_stable * dt3) +
        ((u2_end - 1.0) / gamma_diff) * (exp_dt3 - jnp.exp(-gamma_stable * dt3))
    )
    
    # Combine phases using masks
    u_star = jnp.where(phase1_mask, u_phase1,
                      jnp.where(phase2_mask, u_phase2, u_phase3))
    s_star = jnp.where(phase1_mask, s_phase1,
                      jnp.where(phase2_mask, s_phase2, s_phase3))
    
    return u_star, s_star


def register_piecewise_activation_dynamics():
    """Register piecewise activation dynamics functions."""
    register_dynamics("piecewise_activation", piecewise_activation_dynamics_function)
