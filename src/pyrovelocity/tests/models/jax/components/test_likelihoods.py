"""
Tests for PyroVelocity JAX/NumPyro likelihood components.

This module contains tests for the likelihood components, including:

- test_piecewise_activation_likelihood_function: Test piecewise activation likelihood function
- test_register_standard_likelihoods: Test registration of standard likelihood functions
"""

import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace

from pyrovelocity.models.jax.components import (
    piecewise_activation_likelihood_function,
)
from pyrovelocity.models.jax.registry import get_likelihood


def test_piecewise_activation_likelihood_function():
    """Test piecewise activation likelihood function."""
    # Create test data
    batch_size = 1
    n_cells = 2
    n_genes = 3

    u_obs = jnp.ones((batch_size, n_cells, n_genes))
    s_obs = jnp.ones((batch_size, n_cells, n_genes))
    u_star = jnp.ones((batch_size, n_cells, n_genes))  # Dimensionless concentrations
    s_star = jnp.ones((batch_size, n_cells, n_genes))  # Dimensionless concentrations
    
    # Add required scaling parameters
    lambda_j = jnp.ones(n_cells)  # Cell-specific capture efficiency
    U_0i = jnp.ones(n_genes)  # Gene-specific concentration scale

    # Create context for likelihood function
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_star": u_star,
        "s_star": s_star,
        "lambda_j": lambda_j,
        "U_0i": U_0i,
    }

    # Create a model that uses the likelihood function
    def model():
        piecewise_activation_likelihood_function(context)

    # Trace the model to check that it samples from the correct distributions
    tr = trace(seed(model, 0)).get_trace()

    # Check that the model samples from distributions
    assert "u_obs" in tr
    assert "s_obs" in tr

    # Check that the model uses the correct observations
    np.testing.assert_allclose(tr["u_obs"]["value"], u_obs.astype(jnp.int32))
    np.testing.assert_allclose(tr["s_obs"]["value"], s_obs.astype(jnp.int32))

    # Test with library size scaling
    u_log_library = jnp.zeros((n_cells,))
    s_log_library = jnp.zeros((n_cells,))
    context_with_scaling = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_star": u_star,
        "s_star": s_star,
        "lambda_j": lambda_j,
        "U_0i": U_0i,
        "u_log_library": u_log_library,
        "s_log_library": s_log_library,
    }

    # Create a model that uses the likelihood function with library size scaling
    def model_with_scaling():
        piecewise_activation_likelihood_function(context_with_scaling)

    # Trace the model to check that it samples from the correct distributions
    tr = trace(seed(model_with_scaling, 0)).get_trace()

    # Check that the model samples from distributions
    assert "u_obs" in tr
    assert "s_obs" in tr




def test_register_standard_likelihoods():
    """Test registration of standard likelihood functions."""
    # Check that piecewise activation likelihood function is registered
    piecewise_fn = get_likelihood("piecewise_activation")
    assert piecewise_fn is not None
