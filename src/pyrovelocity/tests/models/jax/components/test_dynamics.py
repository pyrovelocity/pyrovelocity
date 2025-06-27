"""
Tests for PyroVelocity JAX/NumPyro piecewise activation dynamics components.

This module contains tests for the piecewise activation dynamics components, including:

- test_piecewise_activation_dynamics_function: Test piecewise activation dynamics function
- test_register_piecewise_activation_dynamics: Test registration of piecewise activation dynamics functions
"""

import jax.numpy as jnp

from pyrovelocity.models.jax.components.dynamics import (
    piecewise_activation_dynamics_function,
)
from pyrovelocity.models.jax.registry import get_dynamics


def test_piecewise_activation_dynamics_function():
    """Test piecewise activation dynamics function."""
    # Create test data
    batch_size = 1
    n_cells = 2
    n_genes = 3

    # Test time points: before activation, during activation, after activation
    t_star = jnp.array([[[0.1, 0.2, 0.3], [0.6, 0.8, 1.2]]])  # Shape: (batch_size, n_cells, n_genes)

    # Initial conditions (dimensionless steady state)
    u0_star = jnp.ones((batch_size, n_cells, n_genes))  # u*_0 = 1.0
    s0_star = jnp.ones((batch_size, n_cells, n_genes))  # s*_0 = 1.0/γ* (for γ*=1)

    # Create parameters for piecewise activation
    params = {
        "R_on": jnp.array([[2.0, 3.0, 1.5]]),  # Activation fold-change
        "gamma_star": jnp.array([[1.0, 0.8, 1.2]]),  # Relative degradation rate
        "t_on_star": jnp.array([[0.5, 0.5, 0.5]]),  # Activation onset time
        "delta_star": jnp.array([[0.3, 0.4, 0.2]]),  # Activation duration
    }

    # Call function
    u_star, s_star = piecewise_activation_dynamics_function(t_star, u0_star, s0_star, params)

    # Check shapes
    assert u_star.shape == (batch_size, n_cells, n_genes)
    assert s_star.shape == (batch_size, n_cells, n_genes)

    # Check that values are finite and positive
    assert jnp.all(jnp.isfinite(u_star))
    assert jnp.all(jnp.isfinite(s_star))
    assert jnp.all(u_star > 0)
    assert jnp.all(s_star > 0)

    # Test phase behavior
    # Phase 1 (t* < t*_on): should be close to steady state (1, 1/γ*)
    phase1_mask = t_star < params["t_on_star"]
    u_phase1 = u_star[phase1_mask]
    s_phase1 = s_star[phase1_mask]
    
    # For phase 1, u* should be close to 1.0
    assert jnp.allclose(u_phase1, 1.0, atol=1e-6)


def test_piecewise_activation_dynamics_edge_cases():
    """Test edge cases for piecewise activation dynamics."""
    batch_size = 1
    n_cells = 1  
    n_genes = 1

    # Test case: gamma_star near 1.0 (numerical stability)
    t_star = jnp.array([[[0.6]]])  # During activation phase
    u0_star = jnp.ones((batch_size, n_cells, n_genes))
    s0_star = jnp.ones((batch_size, n_cells, n_genes))

    params = {
        "R_on": jnp.array([[2.0]]),
        "gamma_star": jnp.array([[1.0 + 1e-10]]),  # Very close to 1.0
        "t_on_star": jnp.array([[0.5]]),
        "delta_star": jnp.array([[0.2]]),
    }

    # Should not raise numerical errors
    u_star, s_star = piecewise_activation_dynamics_function(t_star, u0_star, s0_star, params)
    
    assert jnp.all(jnp.isfinite(u_star))
    assert jnp.all(jnp.isfinite(s_star))


def test_register_piecewise_activation_dynamics():
    """Test registration of piecewise activation dynamics functions."""
    # Check that piecewise activation dynamics function is registered
    piecewise_fn = get_dynamics("piecewise_activation")
    assert piecewise_fn is not None
