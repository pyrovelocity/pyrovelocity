"""
Tests for PyroVelocity JAX/NumPyro Poisson-only likelihood components.

This module contains tests for the Poisson-only likelihood components, including:

- test_poisson_likelihood_function: Test Poisson-only likelihood function
- test_register_poisson_likelihoods: Test registration of Poisson-only likelihood functions
"""

import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace

from pyrovelocity.models.jax.components import (
    poisson_likelihood_function,
)
from pyrovelocity.models.jax.registry import get_likelihood


def test_poisson_likelihood_function():
    """Test Poisson-only likelihood function."""
    # Create test data
    batch_size = 1
    n_cells = 2
    n_genes = 3

    u_obs = jnp.ones((batch_size, n_cells, n_genes))
    s_obs = jnp.ones((batch_size, n_cells, n_genes))
    u_star = jnp.ones((batch_size, n_cells, n_genes))  # Constant concentrations
    s_star = jnp.ones((batch_size, n_cells, n_genes))  # Constant concentrations
    
    # Add required scaling parameters
    lambda_j = jnp.ones(n_cells)  # Cell-specific library scaling
    U_0i = jnp.ones(n_genes)      # Gene-specific expression capacity
    r_u_i = jnp.ones(n_genes)     # Unspliced rate multiplier
    r_s_i = jnp.ones(n_genes)     # Spliced rate multiplier

    # Create context for likelihood function
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_star": u_star,
        "s_star": s_star,
        "lambda_j": lambda_j,
        "U_0i": U_0i,
        "r_u_i": r_u_i,
        "r_s_i": r_s_i,
    }

    # Create a model that uses the likelihood function
    def model():
        poisson_likelihood_function(context)

    # Trace the model to check that it samples from the correct distributions
    tr = trace(seed(model, 0)).get_trace()

    # Check that the model samples from distributions
    assert "u_obs" in tr
    assert "s_obs" in tr

    # Check that the model uses the correct observations
    np.testing.assert_allclose(tr["u_obs"]["value"], u_obs.astype(jnp.int32))
    np.testing.assert_allclose(tr["s_obs"]["value"], s_obs.astype(jnp.int32))


def test_poisson_likelihood_function_prior_predictive():
    """Test Poisson likelihood function for prior predictive sampling."""
    batch_size = 1
    n_cells = 2
    n_genes = 3

    # No observations (prior predictive)
    u_obs = None
    s_obs = None
    u_star = jnp.ones((batch_size, n_cells, n_genes))
    s_star = jnp.ones((batch_size, n_cells, n_genes))
    
    # Scaling parameters
    lambda_j = jnp.ones(n_cells)
    U_0i = jnp.ones(n_genes)
    r_u_i = jnp.ones(n_genes)
    r_s_i = jnp.ones(n_genes)

    # Create context for likelihood function
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_star": u_star,
        "s_star": s_star,
        "lambda_j": lambda_j,
        "U_0i": U_0i,
        "r_u_i": r_u_i,
        "r_s_i": r_s_i,
    }

    # Create a model that uses the likelihood function
    def model():
        poisson_likelihood_function(context)

    # Trace the model to check prior predictive behavior
    tr = trace(seed(model, 0)).get_trace()

    # Check that the model samples from distributions
    assert "u_obs" in tr
    assert "s_obs" in tr
    
    # Check that samples have correct shapes
    assert tr["u_obs"]["value"].shape == (batch_size, n_cells, n_genes)
    assert tr["s_obs"]["value"].shape == (batch_size, n_cells, n_genes)
    
    # Check that samples are non-negative integers (Poisson)
    assert jnp.all(tr["u_obs"]["value"] >= 0)
    assert jnp.all(tr["s_obs"]["value"] >= 0)


def test_poisson_likelihood_function_scaling():
    """Test Poisson likelihood function with different scaling parameters."""
    batch_size = 1
    n_cells = 2
    n_genes = 2

    u_obs = jnp.array([[[1, 2], [3, 4]]])  # Shape: (1, 2, 2)
    s_obs = jnp.array([[[2, 3], [4, 5]]])  # Shape: (1, 2, 2)
    u_star = jnp.ones((batch_size, n_cells, n_genes))
    s_star = jnp.ones((batch_size, n_cells, n_genes))
    
    # Different scaling parameters
    lambda_j = jnp.array([1.5, 2.0])      # Cell-specific scaling
    U_0i = jnp.array([0.8, 1.2])          # Gene-specific expression capacity
    r_u_i = jnp.array([0.9, 1.1])         # Unspliced rate multiplier
    r_s_i = jnp.array([1.0, 1.2])         # Spliced rate multiplier

    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_star": u_star,
        "s_star": s_star,
        "lambda_j": lambda_j,
        "U_0i": U_0i,
        "r_u_i": r_u_i,
        "r_s_i": r_s_i,
    }

    # Create a model that uses the likelihood function
    def model():
        poisson_likelihood_function(context)

    # Should not raise errors
    tr = trace(seed(model, 0)).get_trace()
    
    # Check that observations are properly handled
    assert "u_obs" in tr
    assert "s_obs" in tr


def test_poisson_likelihood_function_numerical_stability():
    """Test numerical stability of Poisson likelihood function."""
    batch_size = 1
    n_cells = 1
    n_genes = 1

    u_obs = jnp.zeros((batch_size, n_cells, n_genes))
    s_obs = jnp.zeros((batch_size, n_cells, n_genes))
    u_star = jnp.ones((batch_size, n_cells, n_genes)) * 1e-10  # Very small
    s_star = jnp.ones((batch_size, n_cells, n_genes)) * 1e-10  # Very small
    
    # Small scaling parameters
    lambda_j = jnp.array([1e-8])
    U_0i = jnp.array([1e-8])
    r_u_i = jnp.array([1e-8])
    r_s_i = jnp.array([1e-8])

    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_star": u_star,
        "s_star": s_star,
        "lambda_j": lambda_j,
        "U_0i": U_0i,
        "r_u_i": r_u_i,
        "r_s_i": r_s_i,
    }

    # Should handle small values without numerical issues
    def model():
        poisson_likelihood_function(context)

    # Should not raise numerical errors
    tr = trace(seed(model, 42)).get_trace()
    assert "u_obs" in tr
    assert "s_obs" in tr


def test_register_poisson_likelihoods():
    """Test registration of Poisson-only likelihood functions."""
    # Check that Poisson likelihood function is registered
    poisson_fn = get_likelihood("poisson")
    assert poisson_fn is not None
    
    # Check alias registration
    poisson_likelihood_fn = get_likelihood("poisson_likelihood")
    assert poisson_likelihood_fn is not None