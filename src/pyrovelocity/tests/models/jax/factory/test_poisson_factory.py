"""
Tests for PyroVelocity JAX/NumPyro Poisson-only model factory functions.

This module contains integration tests for the Poisson-only model factory functions, including:

- test_create_poisson_model: Test creation of Poisson-only model
- test_poisson_model_config: Test Poisson-only model configuration
- test_poisson_model_functionality: Test full model functionality
"""

import jax
import jax.numpy as jnp
import numpyro
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

from pyrovelocity.models.jax.factory import (
    create_poisson_model,
    create_poisson_model_jax,
    poisson_model_config,
)


def test_poisson_model_config():
    """Test Poisson-only model configuration."""
    config = poisson_model_config()
    
    # Check that configuration has the correct components
    assert config.dynamics_function.name == "poisson"
    assert config.prior_function.name == "poisson"
    assert config.likelihood_function.name == "poisson"
    assert config.guide_function.name == "auto"


def test_create_poisson_model():
    """Test creation of Poisson-only model."""
    # Create the model
    model = create_poisson_model()
    
    # Check that model is callable
    assert callable(model)
    
    # Test model parameters
    num_cells = 5
    num_genes = 3
    
    # Test prior predictive sampling (no observations)
    rng_key = jax.random.PRNGKey(42)
    predictive = Predictive(model, num_samples=1)
    samples = predictive(
        rng_key,
        u_obs=None,
        s_obs=None,
        num_cells=num_cells,
        num_genes=num_genes,
    )
    
    # Should return a dictionary with sampled parameters
    assert isinstance(samples, dict)
    
    # Should contain required parameters from prior
    expected_params = ["lambda_j", "U_0i", "S_0i", "u_star", "s_star"]
    for param in expected_params:
        assert param in samples


def test_create_poisson_model_jax():
    """Test creation of JAX-native Poisson-only model."""
    # Create the model using registry system
    model = create_poisson_model_jax()
    
    # Check that model is callable
    assert callable(model)
    
    # Test model parameters
    num_cells = 4
    num_genes = 2
    
    # Test prior predictive sampling
    rng_key = jax.random.PRNGKey(42)
    predictive = Predictive(model, num_samples=1)
    samples = predictive(
        rng_key,
        u_obs=None,
        s_obs=None,
        num_cells=num_cells,
        num_genes=num_genes,
    )
    
    # Should return a dictionary with sampled parameters
    assert isinstance(samples, dict)
    
    # Should contain required parameters
    expected_params = ["lambda_j", "U_0i", "S_0i", "u_star", "s_star"]
    for param in expected_params:
        assert param in samples


def test_poisson_model_with_observations():
    """Test Poisson model with observed data."""
    # Create the model
    model = create_poisson_model()
    
    # Test data
    batch_size = 1
    num_cells = 3
    num_genes = 2
    
    u_obs = jnp.ones((batch_size, num_cells, num_genes), dtype=jnp.int32)
    s_obs = jnp.ones((batch_size, num_cells, num_genes), dtype=jnp.int32)
    
    # Test model with observations
    rng_key = jax.random.PRNGKey(42)
    predictive = Predictive(model, num_samples=1)
    samples = predictive(
        rng_key,
        u_obs=u_obs,
        s_obs=s_obs,
        num_cells=num_cells,
        num_genes=num_genes,
    )
    
    # Should return parameters even with observations
    assert isinstance(samples, dict)
    
    # Check parameter shapes (Predictive adds a sample dimension)
    assert samples["lambda_j"].shape == (1, num_cells, 1)  # [num_samples, num_cells, plate_dim]
    assert samples["U_0i"].shape == (1, num_genes)
    assert samples["S_0i"].shape == (1, num_genes)
    assert samples["u_star"].shape == (1, batch_size, num_cells, num_genes)
    assert samples["s_star"].shape == (1, batch_size, num_cells, num_genes)


def test_poisson_model_predictive_sampling():
    """Test Poisson model with NumPyro Predictive."""
    # Create the model
    model = create_poisson_model()
    
    # Test parameters
    num_cells = 2
    num_genes = 2
    
    # Use Predictive for prior sampling
    predictive = Predictive(model, num_samples=10)
    
    with seed(rng_seed=42):
        samples = predictive(
            jax.random.PRNGKey(0),
            u_obs=None,
            s_obs=None,
            num_cells=num_cells,
            num_genes=num_genes,
        )
    
    # Check sample shapes (with plate dimensions)
    assert samples["lambda_j"].shape == (10, num_cells, 1)  # Cell parameters have plate dimension
    assert samples["U_0i"].shape == (10, num_genes)
    assert samples["S_0i"].shape == (10, num_genes)
    assert samples["u_obs"].shape == (10, 1, num_cells, num_genes)
    assert samples["s_obs"].shape == (10, 1, num_cells, num_genes)
    
    # Check that generated observations are non-negative
    assert jnp.all(samples["u_obs"] >= 0)
    assert jnp.all(samples["s_obs"] >= 0)


def test_poisson_model_trace_structure():
    """Test that Poisson model creates proper trace structure."""
    # Create the model
    model = create_poisson_model()
    
    # Test parameters
    num_cells = 2
    num_genes = 2
    
    u_obs = jnp.ones((1, num_cells, num_genes), dtype=jnp.int32)
    s_obs = jnp.ones((1, num_cells, num_genes), dtype=jnp.int32)
    
    # Create a wrapped model for tracing
    def traced_model():
        rng_key = jax.random.PRNGKey(42)
        predictive = Predictive(model, num_samples=1)
        return predictive(
            rng_key,
            u_obs=u_obs,
            s_obs=s_obs,
            num_cells=num_cells,
            num_genes=num_genes,
        )
    
    # Get the traced model samples
    samples = traced_model()
    
    # Check that all expected parameters are present
    expected_params = ["lambda_j", "U_0i", "S_0i", "u_obs", "s_obs"]
    for param in expected_params:
        assert param in samples, f"Parameter {param} not found in samples"
    
    # Check deterministic parameters
    expected_deterministic = ["u_star", "s_star"]
    for param in expected_deterministic:
        assert param in samples, f"Deterministic parameter {param} not found in samples"


def test_poisson_model_numerical_stability():
    """Test numerical stability of Poisson model."""
    # Create the model
    model = create_poisson_model()
    
    # Test with edge case data
    num_cells = 1
    num_genes = 1
    
    # Zero observations
    u_obs = jnp.zeros((1, num_cells, num_genes), dtype=jnp.int32)
    s_obs = jnp.zeros((1, num_cells, num_genes), dtype=jnp.int32)
    
    # Should handle zeros without numerical issues
    rng_key = jax.random.PRNGKey(42)
    predictive = Predictive(model, num_samples=1)
    samples = predictive(
        rng_key,
        u_obs=u_obs,
        s_obs=s_obs,
        num_cells=num_cells,
        num_genes=num_genes,
    )
    
    # Check that all parameters are finite
    for param_name, param_value in samples.items():
        assert jnp.all(jnp.isfinite(param_value)), f"Parameter {param_name} has non-finite values"
    
    # Check that scaling parameters are positive (access first sample)
    assert jnp.all(samples["lambda_j"][0] > 0)
    assert jnp.all(samples["U_0i"][0] > 0)
    assert jnp.all(samples["S_0i"][0] > 0)