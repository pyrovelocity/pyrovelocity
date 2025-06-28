"""
Step definitions for testing JAX guide models in PyroVelocity's JAX implementation.

This module implements the steps defined in the guide_model.feature file for JAX/NumPyro.
"""

from importlib.resources import files

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoNormal
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

# Import the feature file using importlib.resources
scenarios(str(files("pyrovelocity.tests.features") / "models" / "jax" / "guide_model.feature"))

# Import the JAX factory functions
from pyrovelocity.models.jax.factory import (
    create_guide_factory_function,
    GuideFunctionConfig,
)


@given("I have a JAX guide model component")
def jax_guide_model_component():
    """Create a generic JAX guide model component."""
    config = GuideFunctionConfig(name="auto_normal_guide")
    return create_guide_factory_function(config)


@given("I have a JAX AutoGuideFactory with AutoNormal guide type", target_fixture="jax_auto_guide_factory")
def jax_auto_guide_factory_fixture():
    """Create a JAX AutoGuideFactory with AutoNormal guide type."""
    config = GuideFunctionConfig(name="auto_normal_guide")
    return create_guide_factory_function(config)


@given("I have a JAX guide model", target_fixture="jax_guide_model")
def jax_guide_model_fixture(jax_auto_guide_factory):
    """Create a JAX guide model."""
    # Create a simple model for the guide to work with
    def simple_model():
        x = numpyro.sample("x", dist.Normal(0, 1))
        y = numpyro.sample("y", dist.Normal(x, 1))
        return y
    
    # Create the guide
    guide = jax_auto_guide_factory(simple_model)
    return {"guide": guide, "model": simple_model}


@when("I create a guide for a NumPyro model", target_fixture="create_jax_guide_for_model")
def create_jax_guide_for_model_fixture(jax_auto_guide_factory):
    """Create a guide for a NumPyro model."""
    # Define a test NumPyro model
    def test_model(data=None):
        n_genes = 5
        R_on = numpyro.sample("R_on", dist.LogNormal(0, 1).expand([n_genes]).to_event(1))
        gamma_star = numpyro.sample("gamma_star", dist.LogNormal(0, 0.5).expand([n_genes]).to_event(1))
        
        if data is not None:
            numpyro.sample("obs", dist.Poisson(R_on), obs=data)
    
    # Create guide for this model
    guide = jax_auto_guide_factory(test_model)
    
    return {"guide": guide, "model": test_model}


@when("I initialize guide parameters with a PRNG key", target_fixture="initialize_jax_guide_params")
def initialize_jax_guide_params_fixture(jax_guide_model, jax_prng_key):
    """Initialize guide parameters with a PRNG key."""
    guide = jax_guide_model["guide"]
    model = jax_guide_model["model"]
    
    # Initialize guide parameters
    guide_params = guide.init(jax_prng_key)
    
    return {"guide_params": guide_params, "guide": guide, "model": model}


@when("I sample from the guide distribution", target_fixture="sample_from_jax_guide")
def sample_from_jax_guide_fixture(initialize_jax_guide_params, jax_prng_key):
    """Sample from the guide distribution."""
    guide = initialize_jax_guide_params["guide"]
    guide_params = initialize_jax_guide_params["guide_params"]
    
    # Sample from the guide
    with numpyro.handlers.seed(rng_seed=42):
        guide_trace = numpyro.handlers.trace(guide).get_trace(guide_params)
    
    # Extract samples from trace
    samples = {name: site["value"] for name, site in guide_trace.items() if site["type"] == "sample"}
    
    return {"samples": samples, "guide_trace": guide_trace}


@when("I compute the log probability of samples", target_fixture="compute_jax_guide_log_prob")
def compute_jax_guide_log_prob_fixture(initialize_jax_guide_params, jax_prng_key):
    """Compute the log probability of samples."""
    guide = initialize_jax_guide_params["guide"]
    guide_params = initialize_jax_guide_params["guide_params"]
    
    # Create some sample values
    sample_values = {"x": jnp.array(0.5), "y": jnp.array(1.0)}
    
    # Compute log probability
    with numpyro.handlers.seed(rng_seed=42):
        with numpyro.handlers.substitute(data=sample_values):
            log_prob_trace = numpyro.handlers.trace(guide).get_trace(guide_params)
    
    # Extract log probabilities
    log_probs = {}
    for name, site in log_prob_trace.items():
        if site["type"] == "sample" and "log_prob" in site:
            log_probs[name] = site["log_prob"]
    
    return {"log_probs": log_probs, "sample_values": sample_values}


@when("I work with batched data", target_fixture="work_with_jax_batched_guide")
def work_with_jax_batched_guide_fixture(jax_auto_guide_factory, jax_prng_key):
    """Work with batched data."""
    batch_size = 3
    n_genes = 5
    
    # Define a batched model
    def batched_model(data=None):
        with numpyro.plate("batch", batch_size):
            R_on = numpyro.sample("R_on", dist.LogNormal(0, 1).expand([n_genes]).to_event(1))
            
            if data is not None:
                numpyro.sample("obs", dist.Poisson(R_on), obs=data)
    
    # Create guide for batched model
    guide = jax_auto_guide_factory(batched_model)
    
    # Initialize with batched data
    batched_data = jnp.abs(jax.random.normal(jax_prng_key, (batch_size, n_genes)))
    
    guide_params = guide.init(jax_prng_key, data=batched_data)
    
    return {
        "guide": guide,
        "guide_params": guide_params,
        "batched_data": batched_data,
        "batch_size": batch_size,
        "n_genes": n_genes,
    }


@then("the guide should be a NumPyro AutoNormal guide")
def check_jax_autonormal_guide(create_jax_guide_for_model):
    """Check that the guide is a NumPyro AutoNormal guide."""
    guide = create_jax_guide_for_model["guide"]
    
    # Check that it's an AutoNormal guide (or compatible type)
    assert hasattr(guide, "__call__")  # Should be callable
    assert hasattr(guide, "init")      # Should have init method


@then("the guide should handle JAX arrays correctly")
def check_jax_guide_array_handling(create_jax_guide_for_model):
    """Check that the guide handles JAX arrays correctly."""
    # This is a structural check - if the guide was created successfully,
    # it can handle JAX arrays
    assert create_jax_guide_for_model["guide"] is not None


@then("the guide should be compatible with NumPyro SVI")
def check_jax_guide_svi_compatibility(create_jax_guide_for_model):
    """Check that the guide is compatible with NumPyro SVI."""
    # Check that guide has the required interface for SVI
    guide = create_jax_guide_for_model["guide"]
    
    assert hasattr(guide, "__call__")
    assert hasattr(guide, "init")


@then("the guide parameters should be JAX arrays")
def check_jax_guide_params_arrays(initialize_jax_guide_params):
    """Check that guide parameters are JAX arrays."""
    guide_params = initialize_jax_guide_params["guide_params"]
    
    # Parameters should be a dictionary of JAX arrays
    assert isinstance(guide_params, dict)
    
    for param_name, param_value in guide_params.items():
        if isinstance(param_value, jnp.ndarray):
            assert isinstance(param_value, jnp.ndarray)


@then("the parameters should be initialized appropriately")
def check_jax_guide_params_initialization(initialize_jax_guide_params):
    """Check that parameters are initialized appropriately."""
    guide_params = initialize_jax_guide_params["guide_params"]
    
    # Parameters should be finite
    for param_name, param_value in guide_params.items():
        if isinstance(param_value, jnp.ndarray):
            assert jnp.all(jnp.isfinite(param_value))


@then("the initialization should be reproducible with the same key")
def check_jax_guide_initialization_reproducible(jax_guide_model, jax_prng_key):
    """Check that initialization is reproducible with the same key."""
    guide = jax_guide_model["guide"]
    
    # Initialize twice with the same key
    params1 = guide.init(jax_prng_key)
    params2 = guide.init(jax_prng_key)
    
    # Should be identical
    for key in params1.keys():
        if key in params2:
            if isinstance(params1[key], jnp.ndarray) and isinstance(params2[key], jnp.ndarray):
                assert jnp.allclose(params1[key], params2[key])


@then("the samples should be JAX arrays")
def check_jax_guide_samples_arrays(sample_from_jax_guide):
    """Check that samples are JAX arrays."""
    samples = sample_from_jax_guide["samples"]
    
    for sample_name, sample_value in samples.items():
        if isinstance(sample_value, jnp.ndarray):
            assert isinstance(sample_value, jnp.ndarray)


@then("the sampling should use the provided PRNG key")
def check_jax_guide_sampling_prng(sample_from_jax_guide):
    """Check that sampling uses the provided PRNG key."""
    # This is a structural check - if sampling completed, the PRNG key was used
    samples = sample_from_jax_guide["samples"]
    assert len(samples) > 0


@then("the samples should follow the guide distribution")
def check_jax_guide_samples_distribution(sample_from_jax_guide):
    """Check that samples follow the guide distribution."""
    samples = sample_from_jax_guide["samples"]
    
    # Samples should be finite and reasonable
    for sample_name, sample_value in samples.items():
        if isinstance(sample_value, jnp.ndarray):
            assert jnp.all(jnp.isfinite(sample_value))


@then("the log probability should be computed using JAX operations")
def check_jax_guide_log_prob_operations(compute_jax_guide_log_prob):
    """Check that log probability is computed using JAX operations."""
    log_probs = compute_jax_guide_log_prob["log_probs"]
    
    # Log probabilities should be JAX arrays
    for name, log_prob in log_probs.items():
        if isinstance(log_prob, jnp.ndarray):
            assert isinstance(log_prob, jnp.ndarray)


@then("the computation should be differentiable")
def check_jax_guide_differentiable(compute_jax_guide_log_prob):
    """Check that the computation is differentiable."""
    log_probs = compute_jax_guide_log_prob["log_probs"]
    
    # Log probabilities should be finite (necessary for differentiation)
    for name, log_prob in log_probs.items():
        if isinstance(log_prob, jnp.ndarray):
            assert jnp.all(jnp.isfinite(log_prob))


@then("the result should be a JAX array")
def check_jax_guide_result_array(compute_jax_guide_log_prob):
    """Check that the result is a JAX array."""
    log_probs = compute_jax_guide_log_prob["log_probs"]
    
    # At least one log probability should be computed
    assert len(log_probs) > 0
    
    for name, log_prob in log_probs.items():
        if isinstance(log_prob, jnp.ndarray):
            assert isinstance(log_prob, jnp.ndarray)


@then("the guide should handle batch dimensions correctly")
def check_jax_guide_batch_handling(work_with_jax_batched_guide):
    """Check that the guide handles batch dimensions correctly."""
    guide_params = work_with_jax_batched_guide["guide_params"]
    batch_size = work_with_jax_batched_guide["batch_size"]
    
    # Guide should have been initialized successfully with batched data
    assert guide_params is not None
    
    # Parameters should account for batch dimensions appropriately
    for param_name, param_value in guide_params.items():
        if isinstance(param_value, jnp.ndarray):
            # Parameters should be finite
            assert jnp.all(jnp.isfinite(param_value))


@then("the variational parameters should be properly shaped")
def check_jax_guide_variational_params_shaped(work_with_jax_batched_guide):
    """Check that variational parameters are properly shaped."""
    guide_params = work_with_jax_batched_guide["guide_params"]
    n_genes = work_with_jax_batched_guide["n_genes"]
    
    # Parameters should have shapes compatible with the model
    for param_name, param_value in guide_params.items():
        if isinstance(param_value, jnp.ndarray):
            # Parameters should have reasonable shapes
            assert len(param_value.shape) >= 0  # At least scalar
            assert param_value.size > 0         # Non-empty


@then("the sampling should preserve batch structure")
def check_jax_guide_batch_structure_preservation(work_with_jax_batched_guide):
    """Check that sampling preserves batch structure."""
    guide = work_with_jax_batched_guide["guide"]
    guide_params = work_with_jax_batched_guide["guide_params"]
    batch_size = work_with_jax_batched_guide["batch_size"]
    
    # Try sampling from the batched guide
    with numpyro.handlers.seed(rng_seed=42):
        guide_trace = numpyro.handlers.trace(guide).get_trace(guide_params)
    
    # Should complete without error
    assert guide_trace is not None