"""
Tests for PyroVelocity JAX/NumPyro Poisson-only prior components.

This module contains tests for the Poisson-only prior components, including:

- test_poisson_prior_function: Test Poisson-only prior function
- test_register_poisson_priors: Test registration of Poisson-only prior functions
"""

import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import seed, trace

from pyrovelocity.models.jax.components import (
    poisson_prior_function,
)
from pyrovelocity.models.jax.registry import get_prior


def test_poisson_prior_function():
    """Test Poisson-only prior function."""
    # Test parameters
    num_genes = 5
    n_cells = 10
    
    # Create test model using the prior
    def test_model():
        prior_params = {"n_cells": n_cells}
        sampled_params = poisson_prior_function(
            num_genes=num_genes,
            prior_params=prior_params
        )
        return sampled_params

    # Sample from the prior
    with seed(rng_seed=42):
        samples = test_model()

    # Check that required parameters are present
    expected_params = ["lambda_j", "U_0i", "S_0i"]
    for param in expected_params:
        assert param in samples
        
    # Check parameter shapes
    assert samples["lambda_j"].shape == (n_cells, 1)  # Cell parameters have plate dimension
    assert samples["U_0i"].shape == (num_genes,)
    assert samples["S_0i"].shape == (num_genes,)
    
    # Check that all parameters are positive (from LogNormal priors)
    assert jnp.all(samples["lambda_j"] > 0)
    assert jnp.all(samples["U_0i"] > 0)
    assert jnp.all(samples["S_0i"] > 0)
    
    # Check that all parameters are finite
    assert jnp.all(jnp.isfinite(samples["lambda_j"]))
    assert jnp.all(jnp.isfinite(samples["U_0i"]))
    assert jnp.all(jnp.isfinite(samples["S_0i"]))


def test_poisson_prior_function_with_custom_hyperparameters():
    """Test Poisson prior function with custom hyperparameters."""
    num_genes = 3
    n_cells = 5
    
    # Custom hyperparameters
    custom_prior_params = {
        "n_cells": n_cells,
        "lambda_loc": 1.0,
        "lambda_scale": 0.3,
        "U_0i_loc": 0.5,
        "U_0i_scale": 0.2,
        "S_0i_loc": -0.5,
        "S_0i_scale": 0.4,
    }
    
    # Create test model
    def test_model():
        return poisson_prior_function(
            num_genes=num_genes,
            prior_params=custom_prior_params
        )

    # Sample from the prior
    with seed(rng_seed=123):
        samples = test_model()

    # Check parameter shapes
    assert samples["lambda_j"].shape == (n_cells, 1)  # Cell parameters have plate dimension
    assert samples["U_0i"].shape == (num_genes,)
    assert samples["S_0i"].shape == (num_genes,)
    
    # Check positivity (LogNormal distributions)
    assert jnp.all(samples["lambda_j"] > 0)
    assert jnp.all(samples["U_0i"] > 0)
    assert jnp.all(samples["S_0i"] > 0)


def test_poisson_prior_missing_n_cells():
    """Test that missing n_cells parameter raises ValueError."""
    num_genes = 2
    
    def test_model():
        # Missing n_cells in prior_params
        return poisson_prior_function(num_genes=num_genes)

    # Should raise ValueError for missing n_cells
    try:
        with seed(rng_seed=42):
            test_model()
        assert False, "Should have raised ValueError for missing n_cells"
    except ValueError as e:
        assert "n_cells must be provided" in str(e)


def test_poisson_prior_trace_structure():
    """Test that the prior function creates proper NumPyro trace structure."""
    num_genes = 2
    n_cells = 3
    
    def test_model():
        prior_params = {"n_cells": n_cells}
        return poisson_prior_function(
            num_genes=num_genes,
            prior_params=prior_params
        )

    # Trace the model
    tr = trace(seed(test_model, 42)).get_trace()

    # Check that all expected parameters are in the trace
    expected_sites = ["lambda_j", "U_0i", "S_0i"]
    for site in expected_sites:
        assert site in tr, f"Site {site} not found in trace"
        
    # Check that sites have correct shapes
    assert tr["lambda_j"]["value"].shape == (n_cells, 1)  # Cell parameters have plate dimension
    assert tr["U_0i"]["value"].shape == (num_genes,)
    assert tr["S_0i"]["value"].shape == (num_genes,)


def test_register_poisson_priors():
    """Test registration of Poisson-only prior functions."""
    # Check that Poisson prior function is registered
    poisson_fn = get_prior("poisson")
    assert poisson_fn is not None
    
    # Check alias registration
    poisson_prior_fn = get_prior("poisson_prior")
    assert poisson_prior_fn is not None