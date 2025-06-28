"""
Fixtures for BDD testing of PyroVelocity's JAX implementation.

This module provides fixtures that are specific to BDD testing of JAX/NumPyro components,
complementing the fixtures defined in the main conftest.py file.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import pytest

# Import JAX factory functions and configurations
from pyrovelocity.models.jax.factory import (
    create_dynamics_function,
    create_prior_function,
    create_likelihood_function,
    create_guide_factory_function,
    DynamicsFunctionConfig,
    PriorFunctionConfig,
    LikelihoodFunctionConfig,
    GuideFunctionConfig,
)


@pytest.fixture
def bdd_jax_simple_data():
    """Create simple JAX data for BDD testing."""
    # Set JAX random seed for reproducibility
    key = jax.random.PRNGKey(42)
    n_cells = 10
    n_genes = 5

    # Generate random data as JAX arrays
    u_obs_key, s_obs_key = jax.random.split(key, 2)
    u_obs = jnp.abs(jax.random.normal(u_obs_key, (n_cells, n_genes)))
    s_obs = jnp.abs(jax.random.normal(s_obs_key, (n_cells, n_genes)))

    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture
def bdd_jax_model_parameters():
    """Create JAX model parameters for BDD testing."""
    key = jax.random.PRNGKey(42)
    n_genes = 5

    # Generate random parameters as JAX arrays
    keys = jax.random.split(key, 6)
    
    # Legacy-style parameters (for compatibility testing)
    alpha = jnp.abs(jax.random.normal(keys[0], (n_genes,)))
    beta = jnp.abs(jax.random.normal(keys[1], (n_genes,)))
    gamma = jnp.abs(jax.random.normal(keys[2], (n_genes,)))

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


@pytest.fixture
def bdd_jax_piecewise_model_parameters():
    """Create JAX piecewise activation model parameters for BDD testing."""
    key = jax.random.PRNGKey(42)
    n_genes = 5
    n_cells = 10

    # Generate random parameters for piecewise activation models as JAX arrays
    keys = jax.random.split(key, 7)
    
    R_on = jnp.abs(jax.random.normal(keys[0], (n_genes,))) * 2.0 + 1.0       # [1.0, 3.0]
    gamma_star = jnp.abs(jax.random.normal(keys[1], (n_genes,))) * 1.0 + 0.5 # [0.5, 1.5]
    t_on_star = jnp.abs(jax.random.normal(keys[2], (n_genes,))) * 0.3 + 0.2  # [0.2, 0.5]
    delta_star = jnp.abs(jax.random.normal(keys[3], (n_genes,))) * 0.4 + 0.3 # [0.3, 0.7]
    t_star = jnp.abs(jax.random.normal(keys[4], (n_cells,))) * 0.5 + 0.1     # [0.1, 0.6]

    return {
        "R_on": R_on,
        "gamma_star": gamma_star,
        "t_on_star": t_on_star,
        "delta_star": delta_star,
        "t_star": t_star,
    }


@pytest.fixture
def bdd_jax_piecewise_dynamics_model():
    """Create a JAX PiecewiseActivationDynamicsModel for BDD testing."""
    config = DynamicsFunctionConfig(name="piecewise_activation_dynamics")
    return create_dynamics_function(config)


@pytest.fixture
def bdd_jax_piecewise_prior_model():
    """Create a JAX PiecewiseActivationPriorModel for BDD testing."""
    config = PriorFunctionConfig(name="piecewise_activation_prior")
    return create_prior_function(config)


@pytest.fixture
def bdd_jax_poisson_likelihood_model():
    """Create a JAX PiecewiseActivationPoissonLikelihoodModel for BDD testing."""
    config = LikelihoodFunctionConfig(name="piecewise_activation_poisson_likelihood")
    return create_likelihood_function(config)


@pytest.fixture
def bdd_jax_auto_guide_factory():
    """Create a JAX AutoGuideFactory for BDD testing."""
    config = GuideFunctionConfig(name="auto_normal_guide")
    return create_guide_factory_function(config)


@pytest.fixture
def bdd_jax_pyrovelocity_model(
    bdd_jax_piecewise_dynamics_model,
    bdd_jax_piecewise_prior_model,
    bdd_jax_poisson_likelihood_model,
    bdd_jax_auto_guide_factory,
):
    """Create a complete JAX PyroVelocity model for BDD testing."""
    from pyrovelocity.models.jax.factory import create_piecewise_activation_model
    
    # Use the factory to create a complete model
    model, guide = create_piecewise_activation_model()
    
    return {
        "model": model,
        "guide": guide,
        "components": {
            "dynamics": bdd_jax_piecewise_dynamics_model,
            "prior": bdd_jax_piecewise_prior_model,
            "likelihood": bdd_jax_poisson_likelihood_model,
            "guide_factory": bdd_jax_auto_guide_factory,
        }
    }


@pytest.fixture
def bdd_jax_prng_key():
    """Create a JAX PRNG key for BDD testing."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def bdd_jax_prng_keys():
    """Create multiple JAX PRNG keys for BDD testing."""
    root_key = jax.random.PRNGKey(42)
    return jax.random.split(root_key, 10)


@pytest.fixture
def bdd_jax_boundary_gamma_data():
    """Create JAX data with gamma_star values near 1.0 for boundary testing."""
    key = jax.random.PRNGKey(42)
    n_cells = 10
    n_genes = 5

    # Generate data for boundary case testing
    u_obs_key, s_obs_key = jax.random.split(key, 2)
    u_obs = jnp.abs(jax.random.normal(u_obs_key, (n_cells, n_genes)))
    s_obs = jnp.abs(jax.random.normal(s_obs_key, (n_cells, n_genes)))

    # Create gamma_star values very close to 1.0
    gamma_star = jnp.ones(n_genes) + jax.random.normal(key, (n_genes,)) * 1e-6

    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "gamma_star": gamma_star,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture
def bdd_jax_batched_data():
    """Create batched JAX data for BDD testing."""
    key = jax.random.PRNGKey(42)
    batch_size = 3
    n_cells = 10
    n_genes = 5

    # Generate batched data
    keys = jax.random.split(key, 2)
    u_obs_batch = jnp.abs(jax.random.normal(keys[0], (batch_size, n_cells, n_genes)))
    s_obs_batch = jnp.abs(jax.random.normal(keys[1], (batch_size, n_cells, n_genes)))

    return {
        "u_obs_batch": u_obs_batch,
        "s_obs_batch": s_obs_batch,
        "batch_size": batch_size,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture
def bdd_jax_zero_count_data():
    """Create JAX data with zero counts for BDD testing."""
    n_cells = 10
    n_genes = 5

    # Create zero observations
    u_obs = jnp.zeros((n_cells, n_genes))
    s_obs = jnp.zeros((n_cells, n_genes))

    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture
def bdd_jax_library_size_data():
    """Create JAX data with library size factors for BDD testing."""
    key = jax.random.PRNGKey(42)
    n_cells = 10
    n_genes = 5

    # Generate data with library size factors
    keys = jax.random.split(key, 4)
    u_obs = jnp.abs(jax.random.normal(keys[0], (n_cells, n_genes)))
    s_obs = jnp.abs(jax.random.normal(keys[1], (n_cells, n_genes)))
    
    # Create library size factors
    u_lib_size = jnp.ones(n_cells) * 2.0  # 2x scaling
    s_lib_size = jnp.ones(n_cells) * 1.5  # 1.5x scaling

    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_lib_size": u_lib_size,
        "s_lib_size": s_lib_size,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture
def bdd_jax_anndata():
    """Create a simple AnnData object with JAX-compatible data for BDD testing."""
    from anndata import AnnData
    
    # Create random data
    np.random.seed(42)
    n_cells = 10
    n_genes = 5
    
    # Create data as numpy arrays (AnnData expects numpy)
    X = np.random.rand(n_cells, n_genes)
    layers = {
        "spliced": np.random.rand(n_cells, n_genes),
        "unspliced": np.random.rand(n_cells, n_genes),
    }
    
    # Create AnnData object
    adata = AnnData(X=X, layers=layers)
    
    return adata


@pytest.fixture(autouse=True)
def clear_numpyro_param_store():
    """Clear NumPyro's parameter store before and after each test."""
    # NumPyro doesn't have a global parameter store like Pyro
    # Instead, we ensure fresh state by setting random seeds
    yield


@pytest.fixture(autouse=True)
def set_jax_platform():
    """Set JAX platform configuration for testing."""
    # Ensure reproducible behavior
    import os
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    
    # Set JAX to use 64-bit precision for numerical stability
    jax.config.update("jax_enable_x64", True)
    
    yield
    
    # Reset to default after tests
    jax.config.update("jax_enable_x64", False)


@pytest.fixture
def bdd_jax_numerical_stability_config():
    """Configuration for numerical stability testing."""
    return {
        "gamma_star_epsilon": 1e-7,  # Very close to 1.0
        "small_count_threshold": 1e-6,
        "max_iterations": 100,
        "tolerance": 1e-4,
        "learning_rate": 0.001,  # Smaller learning rate for stability
    }


@pytest.fixture
def bdd_jax_model_config():
    """Create model configuration for BDD testing."""
    from pyrovelocity.models.jax.factory import ModelConfig, piecewise_activation_model_config
    
    # Use the predefined piecewise activation configuration
    return piecewise_activation_model_config()


@pytest.fixture
def bdd_jax_inference_config():
    """Create inference configuration for BDD testing."""
    return {
        "num_warmup": 50,
        "num_samples": 100,
        "num_chains": 1,
        "svi_num_steps": 100,
        "learning_rate": 0.01,
        "batch_size": None,  # Use full batch
    }


# JAX-specific helper fixtures
@pytest.fixture
def bdd_jax_test_model():
    """Create a simple test model for JAX BDD testing."""
    def simple_jax_model(u_obs, s_obs):
        n_genes = u_obs.shape[-1]
        
        # Simple prior
        rate = numpyro.sample("rate", numpyro.distributions.Exponential(1.0).expand([n_genes]).to_event(1))
        
        # Likelihood
        numpyro.sample("u_obs", numpyro.distributions.Poisson(rate), obs=u_obs)
        numpyro.sample("s_obs", numpyro.distributions.Poisson(rate), obs=s_obs)
    
    return simple_jax_model


@pytest.fixture
def bdd_jax_test_guide(bdd_jax_test_model):
    """Create a simple test guide for JAX BDD testing."""
    from numpyro.infer.autoguide import AutoNormal
    return AutoNormal(bdd_jax_test_model)


# Performance and debugging fixtures
@pytest.fixture
def bdd_jax_performance_data():
    """Create larger datasets for performance testing."""
    key = jax.random.PRNGKey(42)
    n_cells = 100
    n_genes = 50

    # Generate larger random data
    keys = jax.random.split(key, 2)
    u_obs = jnp.abs(jax.random.normal(keys[0], (n_cells, n_genes)))
    s_obs = jnp.abs(jax.random.normal(keys[1], (n_cells, n_genes)))

    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture
def bdd_jax_debug_mode():
    """Enable JAX debugging mode for BDD testing."""
    # Enable debugging
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
    
    yield
    
    # Disable debugging after test
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_debug_infs", False)