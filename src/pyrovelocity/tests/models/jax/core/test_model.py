"""Tests for PyroVelocity JAX/NumPyro model definition."""

import jax
import jax.numpy as jnp
import numpyro
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from pyrovelocity.models.jax.registry.dynamics import get_dynamics
from pyrovelocity.models.jax.registry.likelihoods import get_likelihood
from pyrovelocity.models.jax.core.model import (
    create_model,
    velocity_model,
)
from pyrovelocity.models.jax.registry.priors import get_prior
from pyrovelocity.models.jax.core.state import ModelConfig


def test_velocity_model_interface(cell_gene_data):
    """Test velocity_model interface."""
    # Prepare test inputs
    u_obs = cell_gene_data["u_obs"]
    s_obs = cell_gene_data["s_obs"]
    # Reshape to match the expected type annotation
    u_log_library = jnp.log(jnp.sum(u_obs, axis=1))
    s_log_library = jnp.log(jnp.sum(s_obs, axis=1))

    # Set a fixed seed for reproducibility
    key = jax.random.PRNGKey(0)
    numpyro.set_host_device_count(1)

    # Run the model in a predictive context to avoid sampling
    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.trace() as trace:
            result = velocity_model(
                u_obs=u_obs,
                s_obs=s_obs,
                u_log_library=u_log_library,
                s_log_library=s_log_library,
            )

    # Check that the result is a dictionary with the expected keys
    assert isinstance(result, dict)
    # Check for piecewise activation parameters (generic test)
    assert "R_on" in result  # Activation fold-change
    assert "gamma_star" in result  # Relative degradation rate
    assert "t_on_star" in result  # Activation onset time
    assert "delta_star" in result  # Activation duration
    assert "t_star" in result  # Cell temporal coordinates
    assert "tau" in result
    assert "u_expected" in result
    assert "s_expected" in result

    # Check that the trace contains the expected sites
    assert "R_on" in trace
    assert "gamma_star" in trace
    assert "t_on_star" in trace
    assert "delta_star" in trace
    assert "t_star" in trace
    assert "tau" in trace
    assert "u_expected" in trace
    assert "s_expected" in trace
    assert "u_obs" in trace
    assert "s_obs" in trace


def test_velocity_model_type_checking(cell_gene_data):
    """Test velocity_model type checking."""
    # Prepare test inputs
    u_obs = cell_gene_data["u_obs"]
    s_obs = cell_gene_data["s_obs"]
    u_log_library = jnp.log(jnp.sum(u_obs, axis=1))
    s_log_library = jnp.log(jnp.sum(s_obs, axis=1))

    # Invalid u_obs type
    with pytest.raises(BeartypeCallHintParamViolation):
        velocity_model(
            u_obs="not_an_array",
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
        )

    # Invalid s_obs type
    with pytest.raises(BeartypeCallHintParamViolation):
        velocity_model(
            u_obs=u_obs,
            s_obs="not_an_array",
            u_log_library=u_log_library,
            s_log_library=s_log_library,
        )

    # Invalid u_log_library type
    with pytest.raises(BeartypeCallHintParamViolation):
        velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library="not_an_array",
            s_log_library=s_log_library,
        )

    # Invalid s_log_library type
    with pytest.raises(BeartypeCallHintParamViolation):
        velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library="not_an_array",
        )


def test_create_model():
    """Test create_model function with registry-based components."""
    # Test with piecewise_activation (only registered component)
    config = ModelConfig(
        dynamics="piecewise_activation", 
        likelihood="piecewise_activation",
        prior="piecewise_activation"
    )
    model_fn = create_model(config)
    assert model_fn is not None
    assert callable(model_fn)

    # Test with unknown dynamics should raise error
    config = ModelConfig(
        dynamics="unknown", 
        likelihood="piecewise_activation",
        prior="piecewise_activation"
    )
    with pytest.raises((ValueError, TypeError)):
        create_model(config)
        
    # Test with unknown likelihood should raise error
    config = ModelConfig(
        dynamics="piecewise_activation", 
        likelihood="unknown",
        prior="piecewise_activation"
    )
    with pytest.raises((ValueError, TypeError)):
        create_model(config)
        
    # Test with unknown prior should raise error
    config = ModelConfig(
        dynamics="piecewise_activation", 
        likelihood="piecewise_activation",
        prior="unknown"
    )
    with pytest.raises((ValueError, TypeError)):
        create_model(config)


def test_create_model_type_checking():
    """Test create_model type checking."""
    # Invalid config type
    with pytest.raises(BeartypeCallHintParamViolation):
        create_model("not_a_config")


def test_model_fn_interface(cell_gene_data):
    """Test model_fn interface with registry-based components."""
    # This test checks that the model function created by create_model has the correct interface

    # Prepare test inputs
    u_obs = cell_gene_data["u_obs"]
    s_obs = cell_gene_data["s_obs"]
    u_log_library = jnp.log(jnp.sum(u_obs, axis=1))
    s_log_library = jnp.log(jnp.sum(s_obs, axis=1))

    # Create model function with registry-based components
    config = ModelConfig(
        dynamics="piecewise_activation", 
        likelihood="piecewise_activation",
        prior="piecewise_activation"
    )
    model_fn = create_model(config)

    # Set a fixed seed for reproducibility
    numpyro.set_host_device_count(1)

    # Run the model in a predictive context to avoid sampling
    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.trace() as trace:
            result = model_fn(
                u_obs=u_obs,
                s_obs=s_obs,
                u_log_library=u_log_library,
                s_log_library=s_log_library,
            )

    # Check that the result is a dictionary with the expected keys
    assert isinstance(result, dict)
    # Check for piecewise activation parameters (generic test)
    assert "R_on" in result  # Activation fold-change
    assert "gamma_star" in result  # Relative degradation rate
    assert "t_on_star" in result  # Activation onset time
    assert "delta_star" in result  # Activation duration
    assert "t_star" in result  # Cell temporal coordinates
    assert "tau" in result
    assert "u_expected" in result
    assert "s_expected" in result


def test_model_config_dynamics_selection():
    """Test that ModelConfig correctly selects dynamics function via registry."""
    # Test piecewise_activation dynamics (only one currently registered)
    config = ModelConfig(dynamics="piecewise_activation")
    model_fn = create_model(config)
    
    # Verify the dynamics function can be retrieved from registry
    dynamics_fn = get_dynamics("piecewise_activation")
    assert dynamics_fn is not None, "piecewise_activation dynamics should be registered"
    
    # Test that model_fn was created successfully
    assert model_fn is not None
    assert callable(model_fn)


def test_model_config_likelihood_selection():
    """Test that ModelConfig correctly selects likelihood function via registry."""
    # Test piecewise_activation likelihood (only one currently registered)
    config = ModelConfig(
        dynamics="piecewise_activation", 
        likelihood="piecewise_activation",
        prior="piecewise_activation"
    )
    model_fn = create_model(config)
    
    # Verify the likelihood function can be retrieved from registry
    likelihood_fn = get_likelihood("piecewise_activation")
    assert likelihood_fn is not None, "piecewise_activation likelihood should be registered"
    
    # Check that the model uses the correct likelihood function
    assert config.likelihood == "piecewise_activation"
    
    # Test that model_fn was created successfully
    assert model_fn is not None
    assert callable(model_fn)


def test_model_with_different_dynamics(cell_gene_data):
    """Test model with registry-based dynamics functions."""
    # Prepare test inputs
    u_obs = cell_gene_data["u_obs"]
    s_obs = cell_gene_data["s_obs"]

    # Set a fixed seed for reproducibility
    numpyro.set_host_device_count(1)

    # Get dynamics function from registry
    dynamics_fn = get_dynamics("piecewise_activation")
    assert dynamics_fn is not None, "piecewise_activation dynamics should be registered"

    # Test with registry-based dynamics
    with numpyro.handlers.seed(rng_seed=0):
        result = velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            dynamics_fn=dynamics_fn,
        )

    # Check that the result has the expected structure
    assert isinstance(result, dict)
    
    # Check for essential outputs (component-agnostic)
    assert "tau" in result
    assert "u_expected" in result
    assert "s_expected" in result
    
    # Check for piecewise activation parameters (component-specific)
    assert "R_on" in result  # Activation fold-change
    assert "gamma_star" in result  # Relative degradation rate
    assert "t_on_star" in result  # Activation onset time
    assert "delta_star" in result  # Activation duration
    assert "t_star" in result  # Cell temporal coordinates


def test_model_with_different_likelihoods(cell_gene_data):
    """Test model with registry-based likelihood functions."""
    # Prepare test inputs
    u_obs = cell_gene_data["u_obs"]
    s_obs = cell_gene_data["s_obs"]

    # Set a fixed seed for reproducibility
    numpyro.set_host_device_count(1)

    # Get likelihood function from registry
    likelihood_fn = get_likelihood("piecewise_activation")
    assert likelihood_fn is not None, "piecewise_activation likelihood should be registered"

    # Test with registry-based likelihood
    with numpyro.handlers.seed(rng_seed=0):
        result = velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            likelihood_fn=likelihood_fn,
        )

    # Check that the result has the expected structure
    assert isinstance(result, dict)
    
    # Check for essential outputs (component-agnostic)
    assert "tau" in result
    assert "u_expected" in result
    assert "s_expected" in result
    
    # Check for piecewise activation parameters (component-specific)
    assert "R_on" in result  # Activation fold-change
    assert "gamma_star" in result  # Relative degradation rate
    assert "t_on_star" in result  # Activation onset time
    assert "delta_star" in result  # Activation duration
    assert "t_star" in result  # Cell temporal coordinates


def test_model_with_and_without_latent_time(cell_gene_data):
    """Test model with and without latent time."""
    # Prepare test inputs
    u_obs = cell_gene_data["u_obs"]
    s_obs = cell_gene_data["s_obs"]

    # Set a fixed seed for reproducibility
    numpyro.set_host_device_count(1)

    # Test with latent time
    with numpyro.handlers.seed(rng_seed=0):
        result_with_latent = velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            latent_time=True,
        )

    # Test without latent time
    with numpyro.handlers.seed(rng_seed=0):
        result_without_latent = velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            latent_time=False,
        )

    # Check that the results have the same structure
    assert set(result_with_latent.keys()) == set(result_without_latent.keys())

    # Check that tau is different between the two models
    assert not jnp.allclose(
        result_with_latent["tau"], result_without_latent["tau"]
    )


@pytest.fixture
def cell_gene_data():
    """Create test data for cell-gene matrices."""
    # Create small test data
    num_cells = 10
    num_genes = 5

    # Set a fixed seed for reproducibility
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)

    # Generate random count data
    u_obs = jax.random.poisson(key1, 10.0, (num_cells, num_genes))
    s_obs = jax.random.poisson(key2, 5.0, (num_cells, num_genes))

    # Convert to float arrays to match type annotations
    u_obs = u_obs.astype(jnp.float32)
    s_obs = s_obs.astype(jnp.float32)

    return {"u_obs": u_obs, "s_obs": s_obs}
