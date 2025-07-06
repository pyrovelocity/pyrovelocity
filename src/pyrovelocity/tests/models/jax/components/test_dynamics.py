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
    t_star = jnp.array([[0.3, 0.8]])  # Shape: (batch_size, n_cells)

    # Initial conditions (dimensionless steady state)
    u0_star = jnp.ones((batch_size, n_cells, n_genes))  # u*_0 = 1.0

    # Create parameters for piecewise activation
    # Avoid gamma_star = 1.0 exactly to prevent numerical issues
    params = {
        "R_on": jnp.array([2.0, 3.0, 1.5]),  # Activation fold-change [n_genes]
        "gamma_star": jnp.array([1.1, 0.8, 1.2]),  # Relative degradation rate [n_genes]
        "t_on_star": jnp.array([0.5, 0.5, 0.5]),  # Activation onset time [n_genes]
        "delta_star": jnp.array([0.3, 0.4, 0.2]),  # Activation duration [n_genes]
    }

    # Call function (no s0_star parameter)
    u_star, s_star = piecewise_activation_dynamics_function(t_star, u0_star, params)

    # Check shapes
    assert u_star.shape == (batch_size, n_cells, n_genes)
    assert s_star.shape == (batch_size, n_cells, n_genes)

    # Check that values are finite and positive
    assert jnp.all(jnp.isfinite(u_star))
    assert jnp.all(jnp.isfinite(s_star))
    assert jnp.all(u_star > 0)
    assert jnp.all(s_star > 0)

    # Test phase behavior
    # For cell 0 (t*=0.3 < t_on=0.5): should be in phase 1 (steady state)
    # For cell 1 (t*=0.8 > t_on=0.5): should be in phase 2 or 3
    
    # Check cell 0 is at steady state (phase 1)
    # u* should be close to 1.0
    assert jnp.allclose(u_star[0, 0, :], 1.0, atol=1e-6)
    # s* should be close to 1.0/gamma_star
    expected_s_phase1 = 1.0 / params["gamma_star"]
    assert jnp.allclose(s_star[0, 0, :], expected_s_phase1, atol=1e-6)


def test_piecewise_activation_dynamics_edge_cases():
    """Test edge cases for piecewise activation dynamics."""
    batch_size = 1
    n_cells = 1  
    n_genes = 1

    # Test case: gamma_star near 1.0 (numerical stability)
    t_star = jnp.array([[0.6]])  # During activation phase, shape: (batch_size, n_cells)
    u0_star = jnp.ones((batch_size, n_cells, n_genes))

    params = {
        "R_on": jnp.array([2.0]),  # Shape: (n_genes,)
        "gamma_star": jnp.array([1.0 + 1e-10]),  # Very close to 1.0, shape: (n_genes,)
        "t_on_star": jnp.array([0.5]),  # Shape: (n_genes,)
        "delta_star": jnp.array([0.2]),  # Shape: (n_genes,)
    }

    # Should not raise numerical errors
    u_star, s_star = piecewise_activation_dynamics_function(t_star, u0_star, params)
    
    assert jnp.all(jnp.isfinite(u_star))
    assert jnp.all(jnp.isfinite(s_star))


def test_register_piecewise_activation_dynamics():
    """Test registration of piecewise activation dynamics functions."""
    # Check that piecewise activation dynamics function is registered
    piecewise_fn = get_dynamics("piecewise_activation")
    assert piecewise_fn is not None
