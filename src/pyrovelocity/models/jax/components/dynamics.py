"""
Piecewise activation dynamics functions for PyroVelocity JAX/NumPyro implementation.

This module registers piecewise activation dynamics functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.jax.registry import register_dynamics


def _handle_gamma_boundary_case(gamma_star: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Handle numerical stability for γ* ≈ 1 boundary case.
    
    This implements the safe_gamma_minus_1 pattern from the modular implementation
    to prevent division by zero when computing analytical solutions.
    
    Args:
        gamma_star: Dimensionless degradation rate parameter
        
    Returns:
        Numerically stable gamma_minus_1 = γ* - 1 for use in denominators
    """
    gamma_minus_1 = gamma_star - 1.0
    return jnp.where(
        jnp.abs(gamma_minus_1) > 1e-8,
        gamma_minus_1,
        jnp.sign(gamma_minus_1) * 1e-8
    )


def _ensure_positive_values(
    u_star: Float[Array, "..."], 
    s_star: Float[Array, "..."]
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Ensure RNA concentrations are positive for numerical stability.
    
    Applies ReLU + small epsilon pattern from modular implementation.
    
    Args:
        u_star: Dimensionless unspliced RNA concentrations
        s_star: Dimensionless spliced RNA concentrations
        
    Returns:
        Tuple of (u_star, s_star) with guaranteed positive values
    """
    eps = 1e-6
    u_positive = jnp.maximum(u_star, eps)
    s_positive = jnp.maximum(s_star, eps)
    return u_positive, s_positive


@jaxtyped(typechecker=beartype)  
def _compute_piecewise_solution(
    t_star: Float[Array, "batch_size n_cells n_genes"],
    params: Dict[str, Float[Array, "..."]],
) -> Tuple[
    Float[Array, "batch_size n_cells n_genes"],
    Float[Array, "batch_size n_cells n_genes"],
]:
    """
    Compute piecewise activation analytical solutions.
    
    This is the core mathematical computation extracted as a separate function
    for testing and modularity. Implements the 3-phase piecewise activation
    dynamics with proper continuity conditions and numerical stability.
    
    Args:
        t_star: Dimensionless time parameter
        params: Dictionary of model parameters
        
    Returns:
        Tuple of (u_star, s_star) solutions
    """
    R_on = params["R_on"]
    gamma_star = params["gamma_star"]
    t_on_star = params["t_on_star"] 
    delta_star = params["delta_star"]
    
    # Numerical stability for γ* ≈ 1 case
    safe_gamma_minus_1 = _handle_gamma_boundary_case(gamma_star)
    
    # Phase boundaries
    t_switch_on = t_on_star
    t_switch_off = t_on_star + delta_star
    
    # Phase masks
    phase1_mask = t_star < t_switch_on
    phase2_mask = (t_star >= t_switch_on) & (t_star < t_switch_off)
    phase3_mask = t_star >= t_switch_off
    
    # Phase 1: t* < t*_on (OFF phase, α* = 1.0)
    # Fixed initial conditions: u0 = 1.0, s0 = 1.0/γ*
    # In steady state with α*=1.0, u*=1.0 and s*=1.0/γ*
    exp_minus_t = jnp.exp(-t_star)
    exp_minus_gamma_t = jnp.exp(-gamma_star * t_star)
    
    u_phase1 = jnp.ones_like(t_star)  # Always 1.0 in steady state
    
    # Special handling for γ* ≈ 1 case in s_phase1
    # When γ*=1, the system is at steady state and s* = 1.0
    s_phase1 = jnp.where(
        jnp.abs(gamma_star - 1.0) < 1e-8,
        jnp.ones_like(t_star),  # γ* ≈ 1: s* = 1.0 (steady state)
        1.0 / gamma_star  # General case: s* = 1.0/γ* (steady state)
    )
    
    # Phase 2: t*_on ≤ t* < t*_on + δ* (ON phase, α* = R_on)
    t_rel = t_star - t_switch_on
    exp_minus_t_rel = jnp.exp(-t_rel)
    exp_minus_gamma_t_rel = jnp.exp(-gamma_star * t_rel)
    
    # Continuity: use phase 1 end values as initial conditions
    u_at_ton = 1.0
    s_at_ton = jnp.where(
        jnp.abs(gamma_star - 1.0) < 1e-8,
        1.0,  # γ* ≈ 1: s = 1.0
        1.0 / gamma_star  # General case: s = 1.0/γ*
    )
    
    # Phase 2 analytical solutions with α* = R_on
    u_phase2 = R_on + (u_at_ton - R_on) * exp_minus_t_rel
    
    # Handle γ* ≈ 1 case for s_phase2 analytical solution
    s_phase2 = jnp.where(
        jnp.abs(gamma_star - 1.0) < 1e-8,
        # γ* ≈ 1 limit: s*(t) = s₀ + (u₀ - α*) * t + α* * t
        s_at_ton + (u_at_ton - R_on) * t_rel + R_on * t_rel,
        # General case: full analytical solution
        (R_on / gamma_star + 
         (u_at_ton - R_on) / safe_gamma_minus_1 * (exp_minus_t_rel - exp_minus_gamma_t_rel) +
         (s_at_ton - R_on / gamma_star) * exp_minus_gamma_t_rel)
    )
    
    # Phase 3: t* ≥ t*_on + δ* (Return to OFF, α* = 1.0)  
    t_rel_off = t_star - t_switch_off
    exp_minus_t_rel_off = jnp.exp(-t_rel_off)
    exp_minus_gamma_t_rel_off = jnp.exp(-gamma_star * t_rel_off)
    
    # Continuity: use phase 2 end values as initial conditions
    delta_rel = delta_star
    exp_minus_delta = jnp.exp(-delta_rel)
    exp_minus_gamma_delta = jnp.exp(-gamma_star * delta_rel)
    
    u_at_toff = R_on + (u_at_ton - R_on) * exp_minus_delta
    s_at_toff = (R_on / gamma_star + 
                (u_at_ton - R_on) / safe_gamma_minus_1 * (exp_minus_delta - exp_minus_gamma_delta) +
                (s_at_ton - R_on / gamma_star) * exp_minus_gamma_delta)
    
    # Phase 3 analytical solutions with α* = 1.0
    u_phase3 = 1.0 + (u_at_toff - 1.0) * exp_minus_t_rel_off
    s_phase3 = (1.0 / gamma_star + 
               (u_at_toff - 1.0) / safe_gamma_minus_1 * (exp_minus_t_rel_off - exp_minus_gamma_t_rel_off) +
               (s_at_toff - 1.0 / gamma_star) * exp_minus_gamma_t_rel_off)
    
    # Combine phases using masks
    u_star = jnp.where(phase1_mask, u_phase1,
                      jnp.where(phase2_mask, u_phase2, u_phase3))
    s_star = jnp.where(phase1_mask, s_phase1,
                      jnp.where(phase2_mask, s_phase2, s_phase3))
    
    return u_star, s_star


@jax.jit
@jaxtyped(typechecker=beartype)
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

    This function implements the piecewise activation RNA velocity model with
    dimensionless analytical dynamics and piecewise constant transcription rates.
    
    Mathematical Formulation:
    
    The dimensionless system equations are:
        du*/dt* = α*(t*) - u*
        ds*/dt* = u* - γ*s*
    
    Where α*(t*) is piecewise constant:
        - Phase 1 (Off): t* < t*_on, α*(t*) = 1.0 (reference)
        - Phase 2 (On): t*_on ≤ t* < t*_on + δ*, α*(t*) = R_on (fold-change)
        - Phase 3 (Return to Off): t* ≥ t*_on + δ*, α*(t*) = 1.0
    
    Fixed steady-state initial conditions:
        u*_0 = 1.0, s*_0 = 1.0/γ*
        
    Numerical Stability:
    Handles γ* ≈ 1 singularities using safe_gamma_minus_1 pattern
    to prevent division by zero in analytical solutions.

    Args:
        t_star: Dimensionless time parameter [batch_size, n_cells, n_genes]
        u0_star: Initial dimensionless unspliced RNA (fixed = 1.0, unused)
        s0_star: Initial dimensionless spliced RNA (fixed = 1.0/γ*, unused)
        params: Dictionary with keys:
            - R_on: Fold-change transcription rate during activation
            - gamma_star: Dimensionless degradation rate γ*
            - t_on_star: Dimensionless activation start time t*_on  
            - delta_star: Dimensionless activation duration δ*

    Returns:
        Tuple of (u_star, s_star):
            - u_star: Dimensionless unspliced RNA [batch_size, n_cells, n_genes]
            - s_star: Dimensionless spliced RNA [batch_size, n_cells, n_genes]
    """
    # Compute analytical solutions using modular approach
    u_star, s_star = _compute_piecewise_solution(t_star, params)
    
    # Ensure positive values for numerical stability
    u_positive, s_positive = _ensure_positive_values(u_star, s_star)
    
    return u_positive, s_positive


def register_piecewise_activation_dynamics():
    """Register piecewise activation dynamics functions."""
    register_dynamics("piecewise_activation", piecewise_activation_dynamics_function)
