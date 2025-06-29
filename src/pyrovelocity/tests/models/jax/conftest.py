"""Common test fixtures for PyroVelocity JAX/NumPyro implementation."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
# Use standard JAX utilities instead of deleted core utilities
def create_key(seed: int):
    return jax.random.PRNGKey(seed)

def split_key(key):
    return jax.random.split(key)


@pytest.fixture
def jax_key():
    """Fixture for JAX random key."""
    return create_key(42)


@pytest.fixture
def jax_array_1d():
    """Fixture for 1D JAX array."""
    return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def jax_array_2d():
    """Fixture for 2D JAX array."""
    return jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def numpy_array_1d():
    """Fixture for 1D NumPy array."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def numpy_array_2d():
    """Fixture for 2D NumPy array."""
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def model_parameters():
    """Fixture for model parameters."""
    return {
        "R_on": jnp.array([2.0, 2.5, 1.8]),
        "gamma_star": jnp.array([0.7, 0.8, 0.6]),
        "t_on_star": jnp.array([1.2, 1.5, 1.0]),
        "delta_star": jnp.array([1.8, 2.0, 1.6]),
        "U_0i": jnp.array([8.0, 10.0, 12.0]),
    }


@pytest.fixture
def cell_gene_data():
    """Fixture for cell-gene data."""
    num_cells = 10
    num_genes = 5

    # Create random data
    key = create_key(42)
    key1, key2 = split_key(key)

    u_data = jnp.abs(jax.random.normal(key1, (num_cells, num_genes)))
    s_data = jnp.abs(jax.random.normal(key2, (num_cells, num_genes)))

    return {
        "u_obs": u_data,
        "s_obs": s_data,
        "num_cells": num_cells,
        "num_genes": num_genes,
    }


@pytest.fixture
def training_state(jax_key, model_parameters):
    """Fixture for training state."""
    # Simple training state dict without core dependencies
    return {
        "step": 0,
        "params": model_parameters,
        "opt_state": {},
        "key": jax_key,
    }


@pytest.fixture
def inference_state(model_parameters):
    """Fixture for inference state."""
    # Simple inference state dict with piecewise activation parameters
    posterior_samples = {
        "R_on": jnp.stack([model_parameters["R_on"] for _ in range(10)]),
        "gamma_star": jnp.stack([model_parameters["gamma_star"] for _ in range(10)]),
        "t_on_star": jnp.stack([model_parameters["t_on_star"] for _ in range(10)]),
        "delta_star": jnp.stack([model_parameters["delta_star"] for _ in range(10)]),
        "U_0i": jnp.stack([model_parameters["U_0i"] for _ in range(10)]),
    }

    return {"posterior_samples": posterior_samples}


@pytest.fixture
def model_config():
    """Fixture for model configuration."""
    from pyrovelocity.models.jax.factory import ModelConfig, DynamicsFunctionConfig, PriorFunctionConfig, LikelihoodFunctionConfig, GuideFunctionConfig

    return ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="piecewise_activation"),
        prior_function=PriorFunctionConfig(name="piecewise_activation"),
        likelihood_function=LikelihoodFunctionConfig(name="piecewise_activation"),
        guide_function=GuideFunctionConfig(name="auto"),
    )


@pytest.fixture
def inference_config():
    """Fixture for inference configuration."""
    # Simple inference config dict without core dependencies
    return {
        "method": "svi",
        "num_steps": 1000,
        "learning_rate": 0.01,
        "guide_type": "auto_normal",
    }
