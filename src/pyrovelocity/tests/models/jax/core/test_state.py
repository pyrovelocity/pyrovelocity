"""Tests for PyroVelocity JAX/NumPyro state containers."""

import jax
import jax.numpy as jnp
import pytest
from dataclasses import FrozenInstanceError

from pyrovelocity.models.jax.core.state import (
    VelocityModelState,
    TrainingState,
    InferenceState,
    ModelConfig,
    InferenceConfig,
)


def test_velocity_model_state_creation(model_parameters):
    """Test VelocityModelState creation."""
    state = VelocityModelState(parameters=model_parameters)

    assert state.parameters == model_parameters
    assert state.dynamics_output is None
    assert state.distributions is None
    assert state.observations is None


def test_velocity_model_state_immutability(model_parameters):
    """Test VelocityModelState immutability."""
    state = VelocityModelState(parameters=model_parameters)

    # Test immutability
    with pytest.raises(FrozenInstanceError):
        state.parameters = {}


def test_velocity_model_state_replace(model_parameters):
    """Test VelocityModelState replace method."""
    state = VelocityModelState(parameters=model_parameters)

    # Create new state with updated parameters
    new_parameters = {"R_on": jnp.array([2.0, 3.0, 4.0])}
    new_state = state.replace(parameters=new_parameters)

    # Check that the original state is unchanged
    assert state.parameters == model_parameters

    # Check that the new state has the updated parameters
    assert new_state.parameters == new_parameters
    assert new_state.dynamics_output is None
    assert new_state.distributions is None
    assert new_state.observations is None


def test_training_state_creation(jax_key):
    """Test TrainingState creation."""
    params = {"weights": jnp.array([1.0, 2.0, 3.0])}
    opt_state = {"momentum": jnp.array([0.1, 0.2, 0.3])}

    state = TrainingState(
        step=0,
        params=params,
        opt_state=opt_state,
        key=jax_key,
    )

    assert state.step == 0
    assert state.params == params
    assert state.opt_state == opt_state
    assert state.loss_history == []
    assert state.best_params is None
    assert state.best_loss is None
    assert jnp.array_equal(state.key, jax_key)


def test_training_state_immutability(jax_key):
    """Test TrainingState immutability."""
    params = {"weights": jnp.array([1.0, 2.0, 3.0])}
    opt_state = {"momentum": jnp.array([0.1, 0.2, 0.3])}

    state = TrainingState(
        step=0,
        params=params,
        opt_state=opt_state,
        key=jax_key,
    )

    # Test immutability
    with pytest.raises(FrozenInstanceError):
        state.step = 1


def test_training_state_replace(jax_key):
    """Test TrainingState replace method."""
    params = {"weights": jnp.array([1.0, 2.0, 3.0])}
    opt_state = {"momentum": jnp.array([0.1, 0.2, 0.3])}

    state = TrainingState(
        step=0,
        params=params,
        opt_state=opt_state,
        key=jax_key,
    )

    # Create new state with updated step
    new_state = state.replace(step=1)

    # Check that the original state is unchanged
    assert state.step == 0

    # Check that the new state has the updated step
    assert new_state.step == 1
    assert new_state.params == params
    assert new_state.opt_state == opt_state
    assert jnp.array_equal(new_state.key, jax_key)


def test_inference_state_creation():
    """Test InferenceState creation."""
    posterior_samples = {
        "R_on": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "gamma_star": jnp.array([[0.5, 1.0], [1.5, 2.0]]),
    }

    state = InferenceState(posterior_samples=posterior_samples)

    assert state.posterior_samples == posterior_samples
    assert state.posterior_predictive is None
    assert state.diagnostics is None


def test_inference_state_immutability():
    """Test InferenceState immutability."""
    posterior_samples = {
        "R_on": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "gamma_star": jnp.array([[0.5, 1.0], [1.5, 2.0]]),
    }

    state = InferenceState(posterior_samples=posterior_samples)

    # Test immutability
    with pytest.raises(FrozenInstanceError):
        state.posterior_samples = {}


def test_inference_state_replace():
    """Test InferenceState replace method."""
    posterior_samples = {
        "R_on": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "gamma_star": jnp.array([[0.5, 1.0], [1.5, 2.0]]),
    }

    state = InferenceState(posterior_samples=posterior_samples)

    # Create new state with updated posterior_samples
    new_posterior_samples = {
        "t_on_star": jnp.array([[0.3, 0.6], [0.9, 1.2]]),
    }
    new_state = state.replace(posterior_samples=new_posterior_samples)

    # Check that the original state is unchanged
    assert state.posterior_samples == posterior_samples

    # Check that the new state has the updated posterior_samples
    assert new_state.posterior_samples == new_posterior_samples
    assert new_state.posterior_predictive is None
    assert new_state.diagnostics is None


def test_model_config_creation():
    """Test ModelConfig creation."""
    config = ModelConfig()

    assert config.dynamics == "piecewise_activation"
    assert config.likelihood == "piecewise_activation"
    assert config.prior == "piecewise_activation"
    assert config.inference == "svi"
    assert config.use_observed_lib_size is True
    assert config.latent_time is True
    assert config.latent_time_prior_mean == 0.0
    assert config.latent_time_prior_scale == 1.0
    assert config.include_prior is True


def test_model_config_immutability():
    """Test ModelConfig immutability."""
    config = ModelConfig()

    # Test immutability
    with pytest.raises(FrozenInstanceError):
        config.dynamics = "nonlinear"


def test_model_config_replace():
    """Test ModelConfig replace method."""
    config = ModelConfig()

    # Create new config with updated dynamics
    new_config = config.replace(dynamics="nonlinear")

    # Check that the original config is unchanged
    assert config.dynamics == "piecewise_activation"

    # Check that the new config has the updated dynamics
    assert new_config.dynamics == "nonlinear"
    assert new_config.likelihood == "piecewise_activation"
    assert new_config.prior == "piecewise_activation"
    assert new_config.inference == "svi"
    assert new_config.use_observed_lib_size is True
    assert new_config.latent_time is True
    assert new_config.latent_time_prior_mean == 0.0
    assert new_config.latent_time_prior_scale == 1.0
    assert new_config.include_prior is True


def test_inference_config_creation():
    """Test InferenceConfig creation."""
    config = InferenceConfig()

    assert config.method == "svi"
    assert config.num_samples == 1000
    assert config.num_warmup == 500
    assert config.num_chains == 1
    assert config.guide_type == "auto_normal"
    assert config.optimizer == "adam"
    assert config.learning_rate == 0.01
    assert config.num_epochs == 1000
    assert config.batch_size is None
    assert config.clip_norm is None
    assert config.early_stopping is True
    assert config.early_stopping_patience == 10


def test_inference_config_immutability():
    """Test InferenceConfig immutability."""
    config = InferenceConfig()

    # Test immutability
    with pytest.raises(FrozenInstanceError):
        config.method = "mcmc"


def test_inference_config_replace():
    """Test InferenceConfig replace method."""
    config = InferenceConfig()

    # Create new config with updated method
    new_config = config.replace(method="mcmc")

    # Check that the original config is unchanged
    assert config.method == "svi"

    # Check that the new config has the updated method
    assert new_config.method == "mcmc"
    assert new_config.num_samples == 1000
    assert new_config.num_warmup == 500
    assert new_config.num_chains == 1
    assert new_config.guide_type == "auto_normal"
    assert new_config.optimizer == "adam"
    assert new_config.learning_rate == 0.01
    assert new_config.num_epochs == 1000
    assert new_config.batch_size is None
    assert new_config.clip_norm is None
    assert new_config.early_stopping is True
    assert new_config.early_stopping_patience == 10
