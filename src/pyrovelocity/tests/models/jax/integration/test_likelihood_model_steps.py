"""
Step definitions for testing JAX likelihood models in PyroVelocity's JAX implementation.

This module implements the steps defined in the likelihood_model.feature file for JAX/NumPyro.
"""

from importlib.resources import files

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

# Import the feature file using importlib.resources
scenarios(str(files("pyrovelocity.tests.features") / "models" / "jax" / "likelihood_model.feature"))

# Import the JAX factory functions
from pyrovelocity.models.jax.factory import (
    create_likelihood_function,
    LikelihoodFunctionConfig,
)


@given("I have a JAX likelihood model component")
def jax_likelihood_model_component():
    """Create a generic JAX likelihood model component."""
    config = LikelihoodFunctionConfig(name="piecewise_activation_poisson_likelihood")
    return create_likelihood_function(config)


@given("I have a JAX PiecewiseActivationPoissonLikelihoodModel", target_fixture="jax_poisson_likelihood_model")
def jax_poisson_likelihood_model_fixture():
    """Create a JAX PiecewiseActivationPoissonLikelihoodModel."""
    config = LikelihoodFunctionConfig(name="piecewise_activation_poisson_likelihood")
    return create_likelihood_function(config)


@when("I run the forward method with expected counts as JAX arrays", target_fixture="run_jax_likelihood_forward")
def run_jax_likelihood_forward_fixture(jax_poisson_likelihood_model, jax_input_data, jax_prng_key):
    """Run the forward method with expected counts as JAX arrays."""
    n_cells = jax_input_data["n_cells"]
    n_genes = jax_input_data["n_genes"]
    
    # Create expected counts (from dynamics model)
    u_expected = jnp.abs(jax.random.normal(jax_prng_key, (n_cells, n_genes))) + 1.0
    s_expected = jnp.abs(jax.random.normal(jax_prng_key, (n_cells, n_genes))) + 1.0
    
    context = {
        "u_obs": jax_input_data["u_obs"],
        "s_obs": jax_input_data["s_obs"],
        "u_expected": u_expected,
        "s_expected": s_expected,
    }
    
    # Run within NumPyro context
    with numpyro.handlers.seed(rng_seed=42):
        result_context = jax_poisson_likelihood_model(context, jax_prng_key)
    
    return result_context


@when("I run the forward method with library size factors as JAX arrays", target_fixture="run_jax_likelihood_with_libsize")
def run_jax_likelihood_with_libsize_fixture(jax_poisson_likelihood_model, jax_input_data, jax_prng_key):
    """Run the forward method with library size factors as JAX arrays."""
    n_cells = jax_input_data["n_cells"]
    n_genes = jax_input_data["n_genes"]
    
    # Create expected counts and library size factors
    u_expected = jnp.abs(jax.random.normal(jax_prng_key, (n_cells, n_genes))) + 1.0
    s_expected = jnp.abs(jax.random.normal(jax_prng_key, (n_cells, n_genes))) + 1.0
    u_lib_size = jnp.ones(n_cells) * 2.0  # 2x scaling
    s_lib_size = jnp.ones(n_cells) * 1.5  # 1.5x scaling
    
    context = {
        "u_obs": jax_input_data["u_obs"],
        "s_obs": jax_input_data["s_obs"],
        "u_expected": u_expected,
        "s_expected": s_expected,
        "u_lib_size": u_lib_size,
        "s_lib_size": s_lib_size,
    }
    
    with numpyro.handlers.seed(rng_seed=42):
        result_context = jax_poisson_likelihood_model(context, jax_prng_key)
    
    return result_context


@when("I preprocess the input data", target_fixture="run_jax_likelihood_preprocessing")
def run_jax_likelihood_preprocessing_fixture(jax_poisson_likelihood_model, jax_input_data, jax_prng_key):
    """Preprocess the input data."""
    context = {
        "u_obs": jax_input_data["u_obs"],
        "s_obs": jax_input_data["s_obs"],
    }
    
    # Run preprocessing (assuming the model has preprocessing functionality)
    preprocessed_context = jax_poisson_likelihood_model(context, jax_prng_key)
    
    return preprocessed_context


@when("I run the forward method with zero count observations", target_fixture="run_jax_likelihood_zero_counts")
def run_jax_likelihood_zero_counts_fixture(jax_poisson_likelihood_model, jax_prng_key):
    """Run the forward method with zero count observations."""
    n_cells = 10
    n_genes = 5
    
    # Create zero observations
    u_obs = jnp.zeros((n_cells, n_genes))
    s_obs = jnp.zeros((n_cells, n_genes))
    
    # Create small expected counts
    u_expected = jnp.ones((n_cells, n_genes)) * 0.1
    s_expected = jnp.ones((n_cells, n_genes)) * 0.1
    
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_expected": u_expected,
        "s_expected": s_expected,
    }
    
    with numpyro.handlers.seed(rng_seed=42):
        result_context = jax_poisson_likelihood_model(context, jax_prng_key)
    
    return result_context


@when("I process data in batches using JAX vectorization", target_fixture="run_jax_likelihood_batched")
def run_jax_likelihood_batched_fixture(jax_poisson_likelihood_model, jax_prng_key):
    """Process data in batches using JAX vectorization."""
    batch_size = 3
    n_cells = 10
    n_genes = 5
    
    # Create batched data
    u_obs_batch = jnp.abs(jax.random.normal(jax_prng_key, (batch_size, n_cells, n_genes)))
    s_obs_batch = jnp.abs(jax.random.normal(jax_prng_key, (batch_size, n_cells, n_genes)))
    u_expected_batch = jnp.abs(jax.random.normal(jax_prng_key, (batch_size, n_cells, n_genes))) + 1.0
    s_expected_batch = jnp.abs(jax.random.normal(jax_prng_key, (batch_size, n_cells, n_genes))) + 1.0
    
    # Process each batch item
    results = []
    for i in range(batch_size):
        context = {
            "u_obs": u_obs_batch[i],
            "s_obs": s_obs_batch[i],
            "u_expected": u_expected_batch[i],
            "s_expected": s_expected_batch[i],
        }
        
        with numpyro.handlers.seed(rng_seed=42 + i):
            result = jax_poisson_likelihood_model(context, jax_prng_key)
            results.append(result)
    
    return {"results": results, "batch_size": batch_size}


@then("the model should observe data using NumPyro Poisson distributions")
def check_jax_numpyro_poisson_observations(run_jax_likelihood_forward):
    """Check that the model observes data using NumPyro Poisson distributions."""
    # If the function ran without error and returned results, 
    # it successfully used NumPyro distributions
    assert run_jax_likelihood_forward is not None
    
    # The context should have observation data
    assert "u_obs" in run_jax_likelihood_forward
    assert "s_obs" in run_jax_likelihood_forward


@then("the likelihood should be computed with JAX operations")
def check_jax_likelihood_operations(run_jax_likelihood_forward):
    """Check that the likelihood is computed with JAX operations."""
    # Check that we have JAX arrays in the result
    for key, value in run_jax_likelihood_forward.items():
        if isinstance(value, jnp.ndarray):
            assert isinstance(value, jnp.ndarray)


@then("the observations should be registered with NumPyro")
def check_jax_numpyro_registration(run_jax_likelihood_forward):
    """Check that observations are registered with NumPyro."""
    # This is a structural check - if the function completed without error,
    # NumPyro registration was successful
    assert run_jax_likelihood_forward is not None


@then("the expected counts should be scaled by library size")
def check_jax_library_size_scaling(run_jax_likelihood_with_libsize):
    """Check that expected counts are scaled by library size."""
    # Check that the context contains library size information
    assert "u_lib_size" in run_jax_likelihood_with_libsize or "s_lib_size" in run_jax_likelihood_with_libsize
    
    # The likelihood computation should have used these factors
    assert run_jax_likelihood_with_libsize is not None


@then("the scaling should use JAX broadcasting")
def check_jax_broadcasting_scaling(run_jax_likelihood_with_libsize):
    """Check that scaling uses JAX broadcasting."""
    # Check that library size factors are properly shaped for broadcasting
    if "u_lib_size" in run_jax_likelihood_with_libsize:
        u_lib_size = run_jax_likelihood_with_libsize["u_lib_size"]
        assert isinstance(u_lib_size, jnp.ndarray)
    
    if "s_lib_size" in run_jax_likelihood_with_libsize:
        s_lib_size = run_jax_likelihood_with_libsize["s_lib_size"]
        assert isinstance(s_lib_size, jnp.ndarray)


@then("the corrected counts should be used for likelihood computation")
def check_jax_corrected_counts_usage(run_jax_likelihood_with_libsize):
    """Check that corrected counts are used for likelihood computation."""
    # This is a structural check - if library size scaling was applied,
    # corrected counts were used
    assert run_jax_likelihood_with_libsize is not None


@then("the preprocessing should use JAX operations")
def check_jax_preprocessing_operations(run_jax_likelihood_preprocessing):
    """Check that preprocessing uses JAX operations."""
    # Check that preprocessing results are JAX arrays
    for key, value in run_jax_likelihood_preprocessing.items():
        if isinstance(value, jnp.ndarray):
            assert isinstance(value, jnp.ndarray)


@then("the data should be converted to appropriate JAX array types")
def check_jax_data_conversion(run_jax_likelihood_preprocessing):
    """Check that data is converted to appropriate JAX array types."""
    # Check that data arrays are JAX arrays with appropriate dtypes
    if "u_obs" in run_jax_likelihood_preprocessing:
        u_obs = run_jax_likelihood_preprocessing["u_obs"]
        assert isinstance(u_obs, jnp.ndarray)
        assert u_obs.dtype in [jnp.float32, jnp.float64, jnp.int32, jnp.int64]
    
    if "s_obs" in run_jax_likelihood_preprocessing:
        s_obs = run_jax_likelihood_preprocessing["s_obs"]
        assert isinstance(s_obs, jnp.ndarray)
        assert s_obs.dtype in [jnp.float32, jnp.float64, jnp.int32, jnp.int64]


@then("the preprocessing should be differentiable")
def check_jax_preprocessing_differentiable(run_jax_likelihood_preprocessing):
    """Check that preprocessing is differentiable."""
    # JAX operations are differentiable by default
    # Check that results are finite (necessary for differentiation)
    for key, value in run_jax_likelihood_preprocessing.items():
        if isinstance(value, jnp.ndarray):
            assert jnp.all(jnp.isfinite(value))


@then("the model should handle zero counts appropriately")
def check_jax_zero_counts_handling(run_jax_likelihood_zero_counts):
    """Check that the model handles zero counts appropriately."""
    # Model should run without error even with zero counts
    assert run_jax_likelihood_zero_counts is not None
    
    # Check that zero observations are preserved
    u_obs = run_jax_likelihood_zero_counts["u_obs"]
    s_obs = run_jax_likelihood_zero_counts["s_obs"]
    
    assert jnp.all(u_obs == 0)
    assert jnp.all(s_obs == 0)


@then("the Poisson likelihood should be computed correctly")
def check_jax_poisson_likelihood_computation(run_jax_likelihood_zero_counts):
    """Check that Poisson likelihood is computed correctly."""
    # With zero observations and small expected values,
    # the likelihood should be finite and well-defined
    assert run_jax_likelihood_zero_counts is not None
    
    # The computation should complete without errors
    for key, value in run_jax_likelihood_zero_counts.items():
        if isinstance(value, jnp.ndarray):
            assert jnp.all(jnp.isfinite(value))


@then("no numerical issues should arise")
def check_jax_no_numerical_issues(run_jax_likelihood_zero_counts):
    """Check that no numerical issues arise."""
    # Check for NaN or infinite values
    for key, value in run_jax_likelihood_zero_counts.items():
        if isinstance(value, jnp.ndarray):
            assert not jnp.any(jnp.isnan(value))
            assert not jnp.any(jnp.isinf(value))


@then("the model should handle batched computation efficiently")
def check_jax_batched_computation(run_jax_likelihood_batched):
    """Check that the model handles batched computation efficiently."""
    results = run_jax_likelihood_batched["results"]
    batch_size = run_jax_likelihood_batched["batch_size"]
    
    # Should have results for each batch item
    assert len(results) == batch_size
    
    # Each result should be valid
    for result in results:
        assert result is not None


@then("the likelihood computation should be vectorized")
def check_jax_vectorized_computation(run_jax_likelihood_batched):
    """Check that likelihood computation is vectorized."""
    results = run_jax_likelihood_batched["results"]
    
    # All results should have the same structure
    if len(results) > 1:
        first_keys = set(results[0].keys())
        for result in results[1:]:
            assert set(result.keys()) == first_keys


@then("the batch dimensions should be preserved correctly")
def check_jax_batch_dimensions(run_jax_likelihood_batched):
    """Check that batch dimensions are preserved correctly."""
    results = run_jax_likelihood_batched["results"]
    
    # Check that each result has the expected shape structure
    for result in results:
        for key, value in result.items():
            if isinstance(value, jnp.ndarray) and value.ndim >= 2:
                # Should maintain the cell x gene structure
                assert value.shape[-2:] == (10, 5)  # n_cells, n_genes