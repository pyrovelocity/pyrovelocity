"""
Tests for the MCMC inference components.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS

from pyrovelocity.models.jax.state import InferenceConfig, InferenceState
from pyrovelocity.models.jax.factory.factory import create_model
from pyrovelocity.models.jax.inference.mcmc import (
    create_inference_state,
    create_mcmc,
    extract_posterior_samples,
    mcmc_diagnostics,
    run_mcmc_inference,
)


# Simple model for testing
def simple_model(x=None):
    # Sample parameters
    alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
    beta = numpyro.sample("beta", dist.LogNormal(0.0, 1.0))

    # Sample observations
    with numpyro.plate("data", 10):
        if x is not None:
            numpyro.sample("x_obs", dist.Poisson(alpha * beta), obs=x)

    # Return expected values
    return {
        "expected": alpha * beta,
    }


@pytest.fixture
def test_data():
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(0)

    # Generate synthetic data
    alpha = jnp.exp(jnp.array(0.0))
    beta = jnp.exp(jnp.array(0.0))
    expected = alpha * beta

    # Sample observations
    key, subkey = jax.random.split(key)
    x = jax.random.poisson(subkey, expected, shape=(10,))

    return x


def test_create_mcmc():
    """Test creating an MCMC object."""
    # Create MCMC object with default parameters
    mcmc = create_mcmc(simple_model)

    # Check that MCMC object is an MCMC
    assert isinstance(mcmc, MCMC)

    # Create MCMC object with custom parameters
    mcmc = create_mcmc(
        model=simple_model,
        kernel=NUTS(simple_model),
        num_warmup=100,
        num_samples=200,
        num_chains=2,
        chain_method="sequential",
        progress_bar=False,
    )

    # Check that MCMC object is an MCMC
    assert isinstance(mcmc, MCMC)

    # Check that MCMC object has the correct parameters
    assert mcmc.num_warmup == 100
    assert mcmc.num_samples == 200
    assert mcmc.num_chains == 2


def test_run_mcmc_inference(test_data):
    """Test running MCMC inference with direct model function."""
    # Get test data
    x = test_data

    # Create inference config
    config = InferenceConfig(
        method="mcmc",
        num_samples=5,
        num_warmup=5,
        num_chains=1,
    )

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(0)

    # Run MCMC inference
    mcmc, posterior_samples = run_mcmc_inference(
        model=simple_model,
        args=(x,),
        kwargs={},
        config=config,
        key=key,
    )

    # Check that MCMC object is an MCMC
    assert isinstance(mcmc, MCMC)

    # Check that posterior samples are present
    assert "alpha" in posterior_samples
    assert "beta" in posterior_samples

    # Check that posterior samples have the correct shape
    assert posterior_samples["alpha"].shape == (5,)
    assert posterior_samples["beta"].shape == (5,)


def test_run_mcmc_inference_with_model_config(test_data):
    """Test running MCMC inference with model configuration."""
    # Get test data
    x = test_data

    # Create model configuration
    model_config = {
        "type": "simple",  # This would be a registered model type in a real scenario
        "params": {
            "num_data": 10,
        },
    }

    # Create a simple model factory for testing
    def create_model_mock(config):
        """Simple model factory for testing."""

        def model(x=None):
            # Sample parameters
            alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
            beta = numpyro.sample("beta", dist.LogNormal(0.0, 1.0))

            # Sample observations
            with numpyro.plate("data", config["params"]["num_data"]):
                if x is not None:
                    numpyro.sample("x_obs", dist.Poisson(alpha * beta), obs=x)

            # Return expected values
            return {"expected": alpha * beta}

        return model

    # Create inference config
    config = InferenceConfig(
        method="mcmc",
        num_samples=5,
        num_warmup=5,
        num_chains=1,
    )

    # Patch the create_model function in the mcmc module
    import pyrovelocity.models.jax.inference.mcmc as mcmc_module

    original_create_model = mcmc_module.create_model
    mcmc_module.create_model = create_model_mock

    try:
        # Run MCMC inference with model configuration
        mcmc, posterior_samples = run_mcmc_inference(
            model=model_config,
            kwargs={"x": x},
            config=config,
            key=jax.random.PRNGKey(0),
        )

        # Check that MCMC object is an MCMC
        assert isinstance(mcmc, MCMC)

        # Check that posterior samples are present
        assert "alpha" in posterior_samples
        assert "beta" in posterior_samples

        # Check that posterior samples have the correct shape
        assert posterior_samples["alpha"].shape == (5,)
        assert posterior_samples["beta"].shape == (5,)
    finally:
        # Restore the original create_model function
        mcmc_module.create_model = original_create_model


def test_mcmc_diagnostics(test_data):
    """Test computing MCMC diagnostics."""
    # Get test data
    x = test_data

    # Create inference config
    config = InferenceConfig(
        method="mcmc",
        num_samples=5,
        num_warmup=5,
        num_chains=1,
    )

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(0)

    # Run MCMC inference
    mcmc, _ = run_mcmc_inference(
        model=simple_model,
        args=(x,),
        kwargs={},
        config=config,
        key=key,
    )

    # Compute MCMC diagnostics
    diagnostics = mcmc_diagnostics(mcmc)

    # Check that diagnostics are present
    assert "samples" in diagnostics

    # Check that we have mean and std for parameters
    assert "alpha_mean" in diagnostics
    assert "alpha_std" in diagnostics


def test_extract_posterior_samples(test_data):
    """Test extracting posterior samples from MCMC."""
    # Get test data
    x = test_data

    # Create inference config
    config = InferenceConfig(
        method="mcmc",
        num_samples=5,
        num_warmup=5,
        num_chains=1,
    )

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(0)

    # Run MCMC inference
    mcmc, _ = run_mcmc_inference(
        model=simple_model,
        args=(x,),
        kwargs={},
        config=config,
        key=key,
    )

    # Extract posterior samples
    posterior_samples = extract_posterior_samples(mcmc)

    # Check that posterior samples are present
    assert "alpha" in posterior_samples
    assert "beta" in posterior_samples

    # Check that posterior samples have the correct shape
    assert posterior_samples["alpha"].shape == (5,)
    assert posterior_samples["beta"].shape == (5,)


def test_create_inference_state():
    """Test creating an inference state."""
    # Create posterior samples
    posterior_samples = {
        "alpha": jnp.ones((10,)),
        "beta": jnp.ones((10,)),
    }

    # Create diagnostics
    diagnostics = {
        "summary": {"r_hat": 1.0},
    }

    # Create inference state
    inference_state = create_inference_state(
        posterior_samples=posterior_samples,
        diagnostics=diagnostics,
    )

    # Check that inference state is an InferenceState
    assert isinstance(inference_state, InferenceState)

    # Check that posterior samples are present
    assert inference_state.posterior_samples == posterior_samples

    # Check that diagnostics are present
    assert inference_state.diagnostics == diagnostics
