"""
Tests for PyroVelocity JAX/NumPyro likelihood components.

This module contains tests for the likelihood components, including:

- test_piecewise_activation_likelihood_function: Test piecewise activation likelihood function
- test_register_standard_likelihoods: Test registration of standard likelihood functions
"""

import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace

from pyrovelocity.models.jax.components.likelihoods import (
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
    u_expected = jnp.ones((batch_size, n_cells, n_genes))
    s_expected = jnp.ones((batch_size, n_cells, n_genes))

    # Create context for likelihood function
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_expected": u_expected,
        "s_expected": s_expected,
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
        "u_expected": u_expected,
        "s_expected": s_expected,
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
