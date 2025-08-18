"""
Tests for PyroVelocity JAX/NumPyro Poisson-only dynamics components.

This module contains tests for the Poisson-only dynamics components, including:

- test_poisson_dynamics_function: Test Poisson-only dynamics function
- test_register_poisson_dynamics: Test registration of Poisson-only dynamics functions
"""

import jax.numpy as jnp

from pyrovelocity.models.jax.components import (
    poisson_dynamics_function,
)
from pyrovelocity.models.jax.registry import get_dynamics


def test_poisson_dynamics_function():
    """Test Poisson-only dynamics function."""
    # Create test data
    batch_size = 1
    n_cells = 2
    n_genes = 3

    # Time points (should be ignored in trivial dynamics)
    t_star = jnp.array([[0.3, 0.8]])  # Shape: (batch_size, n_cells)

    # Initial conditions (constant RNA concentrations)
    u0_star = jnp.ones((batch_size, n_cells, n_genes))  # Fixed to 1.0

    # Create trivial parameters (should be unused)
    params = {
        "dummy_param": jnp.array([1.0]),  # Unused in trivial dynamics
    }

    # Call function
    u_star, s_star = poisson_dynamics_function(t_star, u0_star, params)

    # Check shapes
    assert u_star.shape == (batch_size, n_cells, n_genes)
    assert s_star.shape == (batch_size, n_cells, n_genes)

    # Check that values are finite and positive
    assert jnp.all(jnp.isfinite(u_star))
    assert jnp.all(jnp.isfinite(s_star))
    assert jnp.all(u_star > 0)
    assert jnp.all(s_star > 0)

    # Check trivial dynamics behavior:
    # u_star should equal u0_star (no temporal evolution)
    assert jnp.allclose(u_star, u0_star, atol=1e-6)
    
    # s_star should equal u0_star (no splicing dynamics)
    assert jnp.allclose(s_star, u0_star, atol=1e-6)

    # Test that time does not affect the result (trivial dynamics)
    t_star_different = jnp.array([[1.5, 2.8]])  # Different time points
    u_star_different, s_star_different = poisson_dynamics_function(
        t_star_different, u0_star, params
    )
    
    # Results should be identical regardless of time
    assert jnp.allclose(u_star, u_star_different, atol=1e-6)
    assert jnp.allclose(s_star, s_star_different, atol=1e-6)


def test_poisson_dynamics_edge_cases():
    """Test edge cases for Poisson-only dynamics."""
    batch_size = 1
    n_cells = 1  
    n_genes = 1

    # Test case: various initial conditions
    t_star = jnp.array([[0.5]])  # Shape: (batch_size, n_cells)
    
    # Test with different initial conditions
    test_u0_values = [0.1, 1.0, 10.0]
    
    for u0_val in test_u0_values:
        u0_star = jnp.full((batch_size, n_cells, n_genes), u0_val)
        params = {"dummy": jnp.array([1.0])}
        
        u_star, s_star = poisson_dynamics_function(t_star, u0_star, params)
        
        # Should return the initial conditions unchanged
        assert jnp.allclose(u_star, u0_star, atol=1e-6)
        assert jnp.allclose(s_star, u0_star, atol=1e-6)
        
        # Should be finite and positive
        assert jnp.all(jnp.isfinite(u_star))
        assert jnp.all(jnp.isfinite(s_star))
        assert jnp.all(u_star > 0)
        assert jnp.all(s_star > 0)


def test_register_poisson_dynamics():
    """Test registration of Poisson-only dynamics functions."""
    # Check that Poisson dynamics function is registered
    poisson_fn = get_dynamics("poisson")
    assert poisson_fn is not None
    
    # Check alias registration
    poisson_dynamics_fn = get_dynamics("poisson_dynamics")
    assert poisson_dynamics_fn is not None