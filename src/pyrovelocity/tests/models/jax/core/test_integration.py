"""Integration tests for PyroVelocity JAX/NumPyro core components."""

import jax
import jax.numpy as jnp
import pytest

from pyrovelocity.models.jax.core.utils import create_key, split_key
from pyrovelocity.models.jax.core.state import (
    VelocityModelState,
    TrainingState,
    InferenceState,
    ModelConfig,
    InferenceConfig,
)


def test_state_with_utils_integration(jax_key, model_parameters):
    """Test integration between state containers and utility functions."""
    # Create a random key
    key = create_key(42)

    # Use the piecewise activation model parameters from fixture
    parameters = model_parameters
    model_state = VelocityModelState(parameters=parameters)

    # Create a training state
    training_state = TrainingState(
        step=0,
        params=parameters,
        opt_state={},
        key=key,
    )

    # Verify integration
    assert model_state.parameters == parameters
    assert training_state.params == parameters
    assert jnp.array_equal(training_state.key, key)


def test_model_config_inference_config_integration():
    """Test integration between ModelConfig and InferenceConfig."""
    # Create configs
    model_config = ModelConfig()
    inference_config = InferenceConfig()

    # Verify that inference method in ModelConfig matches method in InferenceConfig
    assert model_config.inference == inference_config.method

    # Create new configs with updated inference method
    new_model_config = model_config.replace(inference="mcmc")
    new_inference_config = inference_config.replace(method="mcmc")

    # Verify that the updated configs match
    assert new_model_config.inference == new_inference_config.method


def test_training_state_update_workflow(jax_key, model_parameters):
    """Test a typical workflow for updating TrainingState."""
    # Create initial training state
    initial_state = TrainingState(
        step=0,
        params=model_parameters,
        opt_state={"momentum": jnp.zeros_like(model_parameters["R_on"])},
        key=jax_key,
    )

    # Simulate a training step
    new_key, _ = split_key(initial_state.key)
    new_params = {k: v + 0.01 for k, v in initial_state.params.items()}
    new_opt_state = {"momentum": jnp.ones_like(model_parameters["R_on"]) * 0.1}
    loss = 1.5

    # Update training state
    updated_state = initial_state.replace(
        step=initial_state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        loss_history=initial_state.loss_history + [loss],
        key=new_key,
    )

    # Verify update
    assert updated_state.step == 1
    assert updated_state.params == new_params
    assert updated_state.opt_state == new_opt_state
    assert updated_state.loss_history == [loss]
    assert jnp.array_equal(updated_state.key, new_key)

    # Verify original state is unchanged
    assert initial_state.step == 0
    assert initial_state.params == model_parameters
    assert initial_state.loss_history == []


def test_inference_state_update_workflow():
    """Test a typical workflow for updating InferenceState."""
    # Create initial posterior samples
    posterior_samples = {
        "R_on": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "gamma_star": jnp.array([[0.5, 1.0], [1.5, 2.0]]),
    }

    # Create initial inference state
    initial_state = InferenceState(posterior_samples=posterior_samples)

    # Simulate posterior predictive sampling
    posterior_predictive = {
        "u_obs": jnp.array([[10.0, 20.0], [30.0, 40.0]]),
        "s_obs": jnp.array([[5.0, 10.0], [15.0, 20.0]]),
    }

    # Simulate diagnostics
    diagnostics = {
        "r_hat": {"R_on": 1.01, "gamma_star": 1.02},
        "n_eff": {"R_on": 950, "gamma_star": 980},
    }

    # Update inference state
    updated_state = initial_state.replace(
        posterior_predictive=posterior_predictive,
        diagnostics=diagnostics,
    )

    # Verify update
    assert updated_state.posterior_samples == posterior_samples
    assert updated_state.posterior_predictive == posterior_predictive
    assert updated_state.diagnostics == diagnostics

    # Verify original state is unchanged
    assert initial_state.posterior_samples == posterior_samples
    assert initial_state.posterior_predictive is None
    assert initial_state.diagnostics is None


def test_velocity_model_state_update_workflow(model_parameters):
    """Test a typical workflow for updating VelocityModelState."""
    # Create initial model state
    initial_state = VelocityModelState(parameters=model_parameters)

    # Simulate dynamics computation
    dynamics_output = (
        jnp.array([1.0, 2.0, 3.0]),  # ut
        jnp.array([0.5, 1.0, 1.5]),  # st
    )

    # Update model state with dynamics output
    dynamics_state = initial_state.replace(dynamics_output=dynamics_output)

    # Verify update
    assert dynamics_state.parameters == model_parameters
    assert dynamics_state.dynamics_output == dynamics_output
    assert dynamics_state.distributions is None
    assert dynamics_state.observations is None

    # Simulate distributions computation
    class MockDistribution:
        def __init__(self, rate):
            self.rate = rate

    distributions = (
        MockDistribution(dynamics_output[0]),  # u_dist
        MockDistribution(dynamics_output[1]),  # s_dist
    )

    # Update model state with distributions
    distributions_state = dynamics_state.replace(distributions=distributions)

    # Verify update
    assert distributions_state.parameters == model_parameters
    assert distributions_state.dynamics_output == dynamics_output
    assert distributions_state.distributions == distributions
    assert distributions_state.observations is None

    # Simulate observations
    observations = {
        "u_obs": jnp.array([1.1, 2.1, 3.1]),
        "s_obs": jnp.array([0.6, 1.1, 1.6]),
    }

    # Update model state with observations
    observations_state = distributions_state.replace(observations=observations)

    # Verify update
    assert observations_state.parameters == model_parameters
    assert observations_state.dynamics_output == dynamics_output
    assert observations_state.distributions == distributions
    assert observations_state.observations == observations

    # Verify original state is unchanged
    assert initial_state.parameters == model_parameters
    assert initial_state.dynamics_output is None
    assert initial_state.distributions is None
    assert initial_state.observations is None
