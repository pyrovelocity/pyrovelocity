"""
Tests for PyroVelocity JAX/NumPyro interface definitions.

This module contains tests for the interface definitions, including:

- test_dynamics_function_interface: Test dynamics function interface
- test_prior_function_interface: Test prior function interface
- test_likelihood_function_interface: Test likelihood function interface
- test_guide_factory_function_interface: Test guide factory function interface
- test_interface_validation: Test interface validation utilities
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.jax.interfaces import (
    DynamicsFunction,
    GuideFactoryFunction,
    LikelihoodFunction,
    PriorFunction,
    validate_dynamics_function,
    validate_guide_factory_function,
    validate_likelihood_function,
    validate_prior_function,
)


# Example implementations for testing
@jaxtyped(typechecker=beartype)
def example_dynamics_function(
    t_star: Float[Array, "batch_size n_cells"],
    u0_star: Float[Array, "batch_size n_cells n_genes"],
    params: Dict[str, Float[Array, "..."]],
) -> Tuple[
    Float[Array, "batch_size n_cells n_genes"],
    Float[Array, "batch_size n_cells n_genes"],
]:
    """Example dynamics function implementation for testing."""
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]

    # Expand dimensions for broadcasting
    # t_star is (batch_size, n_cells), need to add gene dimension
    t_star_expanded = t_star[..., jnp.newaxis]  # (batch_size, n_cells, 1)
    
    # Parameters are (n_genes,), need to broadcast
    alpha_expanded = alpha.reshape((1, 1, -1))
    beta_expanded = beta.reshape((1, 1, -1))
    gamma_expanded = gamma.reshape((1, 1, -1))

    # Compute dynamics
    ut = u0_star * jnp.exp(-beta_expanded * t_star_expanded) + (
        alpha_expanded / beta_expanded
    ) * (1 - jnp.exp(-beta_expanded * t_star_expanded))
    
    # Compute s0 from steady state: s0 = u0 / gamma (assuming steady state)
    s0 = u0_star / gamma_expanded
    
    st = s0 * jnp.exp(-gamma_expanded * t_star_expanded) + (
        beta_expanded * u0_star / (gamma_expanded - beta_expanded)
    ) * (jnp.exp(-beta_expanded * t_star_expanded) - jnp.exp(-gamma_expanded * t_star_expanded))

    return ut, st


@jaxtyped(typechecker=beartype)
def example_prior_function(
    num_genes: int,
    prior_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Float[Array, "..."]]:
    """Example prior function implementation for testing."""
    if prior_params is None:
        prior_params = {}

    # NumPyro handles randomness automatically
    alpha = numpyro.sample(
        "alpha", 
        dist.LogNormal(
            prior_params.get("alpha_loc", -0.5),
            prior_params.get("alpha_scale", 1.0)
        ).expand([num_genes]).to_event(1)
    )
    beta = numpyro.sample(
        "beta",
        dist.LogNormal(
            prior_params.get("beta_loc", -0.5),
            prior_params.get("beta_scale", 1.0)
        ).expand([num_genes]).to_event(1)
    )
    gamma = numpyro.sample(
        "gamma",
        dist.LogNormal(
            prior_params.get("gamma_loc", -0.5),
            prior_params.get("gamma_scale", 1.0)
        ).expand([num_genes]).to_event(1)
    )

    return {"alpha": alpha, "beta": beta, "gamma": gamma}


@jaxtyped(typechecker=beartype)
def example_likelihood_function(
    context: Dict[str, Any],
) -> None:
    """Example likelihood function implementation for testing."""
    # Extract required data from context
    u_obs = context["u_obs"]
    s_obs = context["s_obs"]
    u_expected = context["u_expected"]
    s_expected = context["s_expected"]
    
    # Sample from Poisson distribution with expected counts as rates
    numpyro.sample("u", dist.Poisson(u_expected).to_event(2), obs=u_obs)
    numpyro.sample("s", dist.Poisson(s_expected).to_event(2), obs=s_obs)




@jaxtyped(typechecker=beartype)
def example_guide_factory_function(
    model: Callable,
    guide_params: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Example guide factory function implementation for testing."""
    from numpyro.infer.autoguide import AutoNormal

    return AutoNormal(model)


def test_dynamics_function_interface():
    """Test dynamics function interface."""
    # Test that the example function conforms to the interface
    assert validate_dynamics_function(example_dynamics_function)

    # Create test data
    batch_size, n_cells, n_genes = 2, 3, 4
    t_star = jnp.ones((batch_size, n_cells))  # 2D time array
    u0_star = jnp.ones((batch_size, n_cells, n_genes))
    params = {
        "alpha": jnp.ones((n_genes,)),
        "beta": jnp.ones((n_genes,)),
        "gamma": jnp.ones((n_genes,)),
    }

    # Test function execution
    ut, st = example_dynamics_function(t_star, u0_star, params)

    # Check output shapes
    assert ut.shape == (batch_size, n_cells, n_genes)
    assert st.shape == (batch_size, n_cells, n_genes)

    # Test validation utility
    assert validate_dynamics_function(example_dynamics_function)


def test_prior_function_interface():
    """Test prior function interface."""
    # Test that the example function conforms to the interface
    assert validate_prior_function(example_prior_function)

    # Create test data
    num_genes = 10

    # Test function execution within a NumPyro context
    # Since prior functions use numpyro.sample, they need to be called within a model context
    def test_model():
        params = example_prior_function(num_genes)
        return params
    
    # Execute with trace handler to test
    with numpyro.handlers.seed(rng_seed=0):
        trace = numpyro.handlers.trace(test_model).get_trace()
    
    # Extract sampled parameters from trace
    alpha = trace["alpha"]["value"]
    beta = trace["beta"]["value"]
    gamma = trace["gamma"]["value"]
    
    # Check output shapes
    assert alpha.shape == (num_genes,)
    assert beta.shape == (num_genes,)
    assert gamma.shape == (num_genes,)

    # Test validation utility
    assert validate_prior_function(example_prior_function)


def test_likelihood_function_interface():
    """Test likelihood function interface."""
    # Test that the example function conforms to the interface
    assert validate_likelihood_function(example_likelihood_function)

    # Create test data
    batch_size, n_cells, n_genes = 2, 3, 4
    u_obs = jnp.ones((batch_size, n_cells, n_genes))
    s_obs = jnp.ones((batch_size, n_cells, n_genes))
    u_expected = jnp.ones((batch_size, n_cells, n_genes))
    s_expected = jnp.ones((batch_size, n_cells, n_genes))

    # Create context dictionary
    context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_expected": u_expected,
        "s_expected": s_expected,
    }

    # Test function execution in a numpyro model
    def model():
        example_likelihood_function(context)

    # This should not raise an error
    numpyro.handlers.trace(model).get_trace()

    # Test validation utility
    assert validate_likelihood_function(example_likelihood_function)




def test_guide_factory_function_interface():
    """Test guide factory function interface."""
    # Test that the example function conforms to the interface
    assert validate_guide_factory_function(example_guide_factory_function)

    # Create a simple model for testing
    def model():
        numpyro.sample("x", dist.Normal(0, 1))

    # Test function execution
    guide = example_guide_factory_function(model)

    # Check that the guide is callable
    assert callable(guide)

    # Test validation utility
    assert validate_guide_factory_function(example_guide_factory_function)


def test_interface_validation_with_invalid_functions():
    """Test interface validation with invalid functions."""

    # Test with a function that doesn't match the dynamics function interface
    def invalid_dynamics_function(x, y):
        return x, y

    # This should fail validation
    with pytest.raises(TypeError):
        validate_dynamics_function(invalid_dynamics_function)

    # Test with a function that doesn't match the prior function interface
    def invalid_prior_function(x):
        return x

    # This should fail validation
    with pytest.raises(TypeError):
        validate_prior_function(invalid_prior_function)

    # Test with a function that doesn't match the likelihood function interface
    def invalid_likelihood_function(x, y):
        pass

    # This should fail validation
    with pytest.raises(TypeError):
        validate_likelihood_function(invalid_likelihood_function)


    # Test with a function that doesn't match the guide factory function interface
    def invalid_guide_factory_function(x):
        return x

    # This should fail validation
    with pytest.raises(TypeError):
        validate_guide_factory_function(invalid_guide_factory_function)
