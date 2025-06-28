"""
Tests for PyroVelocity JAX/NumPyro factory system.

This module contains tests for the factory system, including:

- test_config_classes: Test configuration classes
- test_create_dynamics_function: Test dynamics function factory
- test_create_prior_function: Test prior function factory
- test_create_likelihood_function: Test likelihood function factory
- test_create_observation_function: Test observation function factory
- test_create_guide_factory_function: Test guide factory function factory
- test_create_model: Test model factory
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import pytest
from jaxtyping import Array, Float

from pyrovelocity.models.jax.factory import (
    DynamicsFunctionConfig,
    GuideFunctionConfig,
    LikelihoodFunctionConfig,
    ModelConfig,
    ObservationFunctionConfig,
    PriorFunctionConfig,
    create_dynamics_function,
    create_guide_factory_function,
    create_likelihood_function,
    create_model,
    create_observation_function,
    create_prior_function,
    create_piecewise_activation_model,
    piecewise_activation_model_config,
)
from pyrovelocity.models.jax.registry import (
    register_dynamics,
    register_guide,
    register_likelihood,
    register_observation,
    register_prior,
)


# Mock implementations for testing
def mock_dynamics_function(
    tau: Float[Array, "batch_size n_cells n_genes"],
    u0: Float[Array, "batch_size n_cells n_genes"],
    s0: Float[Array, "batch_size n_cells n_genes"],
    params: Dict[str, Float[Array, "..."]],
) -> Tuple[
    Float[Array, "batch_size n_cells n_genes"],
    Float[Array, "batch_size n_cells n_genes"],
]:
    """Mock dynamics function for testing."""
    return u0, s0


def mock_prior_function(
    key: jnp.ndarray,
    num_genes: int,
    prior_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Float[Array, "n_genes"]]:
    """Mock prior function for testing."""
    return {
        "alpha": jnp.ones(num_genes),
        "beta": jnp.ones(num_genes),
        "gamma": jnp.ones(num_genes),
    }


def mock_likelihood_function(
    u_obs: Float[Array, "batch_size n_cells n_genes"],
    s_obs: Float[Array, "batch_size n_cells n_genes"],
    u_logits: Float[Array, "batch_size n_cells n_genes"],
    s_logits: Float[Array, "batch_size n_cells n_genes"],
    likelihood_params: Optional[Dict[str, Any]] = None,
) -> None:
    """Mock likelihood function for testing."""
    pass


def mock_observation_function(
    u_obs: Float[Array, "batch_size n_cells n_genes"],
    s_obs: Float[Array, "batch_size n_cells n_genes"],
    observation_params: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Float[Array, "batch_size n_cells n_genes"],
    Float[Array, "batch_size n_cells n_genes"],
]:
    """Mock observation function for testing."""
    return u_obs, s_obs


def mock_guide_factory_function(
    model: Callable,
    guide_params: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Mock guide factory function for testing."""
    return lambda *args, **kwargs: None


@pytest.fixture
def setup_registries():
    """Register mock functions for testing."""
    # Register mock functions
    try:
        register_dynamics("mock", mock_dynamics_function)
    except ValueError:
        pass

    try:
        register_prior("mock", mock_prior_function)
    except ValueError:
        pass

    try:
        register_likelihood("mock", mock_likelihood_function)
    except ValueError:
        pass

    try:
        register_observation("mock", mock_observation_function)
    except ValueError:
        pass

    try:
        register_guide("mock", mock_guide_factory_function)
    except ValueError:
        pass


def test_config_classes():
    """Test configuration classes."""
    # Test DynamicsFunctionConfig
    dynamics_config = DynamicsFunctionConfig(name="standard")
    assert dynamics_config.name == "standard"
    assert dynamics_config.params == {}

    dynamics_config = DynamicsFunctionConfig(
        name="standard", params={"param1": 1}
    )
    assert dynamics_config.name == "standard"
    assert dynamics_config.params == {"param1": 1}

    # Test PriorFunctionConfig
    prior_config = PriorFunctionConfig(name="lognormal")
    assert prior_config.name == "lognormal"
    assert prior_config.params == {}

    prior_config = PriorFunctionConfig(name="lognormal", params={"param1": 1})
    assert prior_config.name == "lognormal"
    assert prior_config.params == {"param1": 1}

    # Test LikelihoodFunctionConfig
    likelihood_config = LikelihoodFunctionConfig(name="poisson")
    assert likelihood_config.name == "poisson"
    assert likelihood_config.params == {}

    likelihood_config = LikelihoodFunctionConfig(
        name="poisson", params={"param1": 1}
    )
    assert likelihood_config.name == "poisson"
    assert likelihood_config.params == {"param1": 1}

    # Test ObservationFunctionConfig
    observation_config = ObservationFunctionConfig(name="standard")
    assert observation_config.name == "standard"
    assert observation_config.params == {}

    observation_config = ObservationFunctionConfig(
        name="standard", params={"param1": 1}
    )
    assert observation_config.name == "standard"
    assert observation_config.params == {"param1": 1}

    # Test GuideFunctionConfig
    guide_config = GuideFunctionConfig(name="auto")
    assert guide_config.name == "auto"
    assert guide_config.params == {}

    guide_config = GuideFunctionConfig(name="auto", params={"param1": 1})
    assert guide_config.name == "auto"
    assert guide_config.params == {"param1": 1}

    # Test ModelConfig
    model_config = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="standard"),
        prior_function=PriorFunctionConfig(name="lognormal"),
        likelihood_function=LikelihoodFunctionConfig(name="poisson"),
        observation_function=ObservationFunctionConfig(name="standard"),
        guide_function=GuideFunctionConfig(name="auto"),
    )
    assert model_config.dynamics_function.name == "standard"
    assert model_config.prior_function.name == "lognormal"
    assert model_config.likelihood_function.name == "poisson"
    assert model_config.observation_function.name == "standard"
    assert model_config.guide_function.name == "auto"
    assert model_config.metadata == {}

    model_config = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="standard"),
        prior_function=PriorFunctionConfig(name="lognormal"),
        likelihood_function=LikelihoodFunctionConfig(name="poisson"),
        observation_function=ObservationFunctionConfig(name="standard"),
        guide_function=GuideFunctionConfig(name="auto"),
        metadata={"param1": 1},
    )
    assert model_config.metadata == {"param1": 1}


def test_create_dynamics_function(setup_registries):
    """Test dynamics function factory."""
    # Test with string
    fn = create_dynamics_function("standard")
    assert callable(fn)

    # Test with config
    fn = create_dynamics_function(DynamicsFunctionConfig(name="standard"))
    assert callable(fn)

    # Test with dict
    fn = create_dynamics_function({"name": "standard"})
    assert callable(fn)

    # Test with mock
    fn = create_dynamics_function("mock")
    assert fn is mock_dynamics_function

    # Test with params
    fn = create_dynamics_function(
        DynamicsFunctionConfig(name="mock", params={"param1": 1})
    )
    assert fn is mock_dynamics_function

    # Test with invalid name
    with pytest.raises(ValueError):
        create_dynamics_function("invalid")


def test_create_prior_function(setup_registries):
    """Test prior function factory."""
    # Test with string
    fn = create_prior_function("lognormal")
    assert callable(fn)

    # Test with config
    fn = create_prior_function(PriorFunctionConfig(name="lognormal"))
    assert callable(fn)

    # Test with dict
    fn = create_prior_function({"name": "lognormal"})
    assert callable(fn)

    # Test with mock
    fn = create_prior_function("mock")
    assert fn is mock_prior_function

    # Test with params
    fn = create_prior_function(
        PriorFunctionConfig(name="mock", params={"param1": 1})
    )
    assert fn is mock_prior_function

    # Test with invalid name
    with pytest.raises(ValueError):
        create_prior_function("invalid")


def test_create_likelihood_function(setup_registries):
    """Test likelihood function factory."""
    # Test with string
    fn = create_likelihood_function("poisson")
    assert callable(fn)

    # Test with config
    fn = create_likelihood_function(LikelihoodFunctionConfig(name="poisson"))
    assert callable(fn)

    # Test with dict
    fn = create_likelihood_function({"name": "poisson"})
    assert callable(fn)

    # Test with mock
    fn = create_likelihood_function("mock")
    assert fn is mock_likelihood_function

    # Test with params
    fn = create_likelihood_function(
        LikelihoodFunctionConfig(name="mock", params={"param1": 1})
    )
    assert fn is mock_likelihood_function

    # Test with invalid name
    with pytest.raises(ValueError):
        create_likelihood_function("invalid")


def test_create_observation_function(setup_registries):
    """Test observation function factory."""
    # Test with string
    fn = create_observation_function("standard")
    assert callable(fn)

    # Test with config
    fn = create_observation_function(ObservationFunctionConfig(name="standard"))
    assert callable(fn)

    # Test with dict
    fn = create_observation_function({"name": "standard"})
    assert callable(fn)

    # Test with mock
    fn = create_observation_function("mock")
    assert fn is mock_observation_function

    # Test with params
    fn = create_observation_function(
        ObservationFunctionConfig(name="mock", params={"param1": 1})
    )
    assert fn is mock_observation_function

    # Test with invalid name
    with pytest.raises(ValueError):
        create_observation_function("invalid")


def test_create_guide_factory_function(setup_registries):
    """Test guide factory function factory."""
    # Test with string
    fn = create_guide_factory_function("auto")
    assert callable(fn)

    # Test with config
    fn = create_guide_factory_function(GuideFunctionConfig(name="auto"))
    assert callable(fn)

    # Test with dict
    fn = create_guide_factory_function({"name": "auto"})
    assert callable(fn)

    # Test with mock
    fn = create_guide_factory_function("mock")
    assert fn is mock_guide_factory_function

    # Test with params
    fn = create_guide_factory_function(
        GuideFunctionConfig(name="mock", params={"param1": 1})
    )
    assert fn is mock_guide_factory_function

    # Test with invalid name
    with pytest.raises(ValueError):
        create_guide_factory_function("invalid")


def test_create_model(setup_registries):
    """Test model factory."""
    # Test with config
    model_config = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="mock"),
        prior_function=PriorFunctionConfig(name="mock"),
        likelihood_function=LikelihoodFunctionConfig(name="mock"),
        observation_function=ObservationFunctionConfig(name="mock"),
        guide_function=GuideFunctionConfig(name="mock"),
    )
    model = create_model(model_config)
    assert callable(model)

    # Test with dict
    model = create_model(
        {
            "dynamics_function": {"name": "mock"},
            "prior_function": {"name": "mock"},
            "likelihood_function": {"name": "mock"},
            "observation_function": {"name": "mock"},
            "guide_function": {"name": "mock"},
        }
    )
    assert callable(model)

    # Test piecewise activation model config
    config = piecewise_activation_model_config()
    assert config.dynamics_function.name == "piecewise_activation"
    assert config.prior_function.name == "lognormal"
    assert config.likelihood_function.name == "poisson"
    assert config.observation_function.name == "standard"
    assert config.guide_function.name == "auto"

    # Test create piecewise activation model
    model = create_piecewise_activation_model()
    assert callable(model)


def test_model_execution(setup_registries):
    """Test model execution."""
    # Create a model
    model_config = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="mock"),
        prior_function=PriorFunctionConfig(name="mock"),
        likelihood_function=LikelihoodFunctionConfig(name="mock"),
        observation_function=ObservationFunctionConfig(name="mock"),
        guide_function=GuideFunctionConfig(name="mock"),
    )
    model = create_model(model_config)

    # Create test data with the correct shape (batch_size, n_cells, n_genes)
    key = jax.random.PRNGKey(0)
    u_obs = jnp.ones((1, 2, 3))  # batch_size=1, n_cells=2, n_genes=3
    s_obs = jnp.ones((1, 2, 3))  # batch_size=1, n_cells=2, n_genes=3

    # Test model execution
    with numpyro.handlers.seed(rng_seed=0):
        trace = numpyro.handlers.trace(model).get_trace(u_obs, s_obs)

    # Check that the model executed without errors
    assert "alpha" in trace
    assert "beta" in trace
    assert "gamma" in trace
