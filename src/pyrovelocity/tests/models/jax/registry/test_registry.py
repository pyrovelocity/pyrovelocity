"""
Tests for PyroVelocity JAX/NumPyro registry system.

This module contains tests for the registry system, including:

- test_base_registry: Test base registry functionality
- test_dynamics_registry: Test dynamics registry
- test_prior_registry: Test prior registry
- test_likelihood_registry: Test likelihood registry
- test_guide_registry: Test guide registry
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.jax.interfaces import (
    validate_dynamics_function,
    validate_guide_factory_function,
    validate_likelihood_function,
    validate_prior_function,
)
from pyrovelocity.models.jax.registry import (
    Registry,
    get_registry,
    register,
)
from pyrovelocity.models.jax.registry.dynamics import (
    DynamicsRegistry,
    get_dynamics,
    list_dynamics,
    register_dynamics,
)
from pyrovelocity.models.jax.registry.guides import (
    GuideRegistry,
    get_guide,
    list_guides,
    register_guide,
)
from pyrovelocity.models.jax.registry.likelihoods import (
    LikelihoodRegistry,
    get_likelihood,
    list_likelihoods,
    register_likelihood,
)
from pyrovelocity.models.jax.registry.priors import (
    PriorRegistry,
    get_prior,
    list_priors,
    register_prior,
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
    """Example dynamics function implementation for registry testing."""
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
    """Example prior function implementation for registry testing."""
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
    context: Dict[str, Any]
) -> None:
    """Example likelihood function implementation for registry testing."""
    # Extract observations and expected values from context
    u_obs = context["u_obs"]
    s_obs = context["s_obs"]
    u_expected = context["u_expected"]
    s_expected = context["s_expected"]
    
    # Sample from Poisson distribution
    numpyro.sample("u", dist.Poisson(u_expected).to_event(2), obs=u_obs)
    numpyro.sample("s", dist.Poisson(s_expected).to_event(2), obs=s_obs)




@jaxtyped(typechecker=beartype)
def example_guide_factory_function(
    model: Callable,
    guide_params: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Example guide factory function implementation for registry testing."""
    from numpyro.infer.autoguide import AutoNormal

    return AutoNormal(model)


def test_base_registry():
    """Test base registry functionality."""
    # Create a registry
    registry = Registry("test_registry")

    # Register a function
    registry.register("test_function", lambda x: x)

    # Get the function
    fn = registry.get("test_function")
    assert fn is not None
    assert fn(5) == 5

    # List registered functions
    functions = registry.list()
    assert "test_function" in functions

    # Test get_registry function
    registry2 = get_registry("test_registry")
    assert registry2 is registry

    # Test register decorator
    @register("test_registry", "decorated_function")
    def decorated_function(x):
        return x * 2

    fn = registry.get("decorated_function")
    assert fn is not None
    assert fn(5) == 10


def test_dynamics_registry():
    """Test dynamics registry."""
    # Validate the test function
    assert validate_dynamics_function(example_dynamics_function)

    # Register the function
    register_dynamics("test_dynamics", example_dynamics_function)

    # Get the function
    fn = get_dynamics("test_dynamics")
    assert fn is not None
    assert fn is example_dynamics_function

    # List registered functions
    functions = list_dynamics()
    assert "test_dynamics" in functions

    # Test registry instance
    registry = DynamicsRegistry()
    assert registry.name == "dynamics"

    # Test registration with invalid function
    def invalid_function(x):
        return x

    with pytest.raises(TypeError):
        register_dynamics("invalid", invalid_function)


def test_prior_registry():
    """Test prior registry."""
    # Validate the test function
    assert validate_prior_function(example_prior_function)

    # Register the function
    register_prior("test_prior", example_prior_function)

    # Get the function
    fn = get_prior("test_prior")
    assert fn is not None
    assert fn is example_prior_function

    # List registered functions
    functions = list_priors()
    assert "test_prior" in functions

    # Test registry instance
    registry = PriorRegistry()
    assert registry.name == "priors"

    # Test registration with invalid function
    def invalid_function(x):
        return x

    with pytest.raises(TypeError):
        register_prior("invalid", invalid_function)


def test_likelihood_registry():
    """Test likelihood registry."""
    # Validate the test function
    assert validate_likelihood_function(example_likelihood_function)

    # Register the function
    register_likelihood("test_likelihood", example_likelihood_function)

    # Get the function
    fn = get_likelihood("test_likelihood")
    assert fn is not None
    assert fn is example_likelihood_function

    # List registered functions
    functions = list_likelihoods()
    assert "test_likelihood" in functions

    # Test registry instance
    registry = LikelihoodRegistry()
    assert registry.name == "likelihoods"

    # Test registration with invalid function
    def invalid_function(x):
        return x

    with pytest.raises(TypeError):
        register_likelihood("invalid", invalid_function)




def test_guide_registry():
    """Test guide registry."""
    # Validate the test function
    assert validate_guide_factory_function(example_guide_factory_function)

    # Register the function
    register_guide("test_guide", example_guide_factory_function)

    # Get the function
    fn = get_guide("test_guide")
    assert fn is not None
    assert fn is example_guide_factory_function

    # List registered functions
    functions = list_guides()
    assert "test_guide" in functions

    # Test registry instance
    registry = GuideRegistry()
    assert registry.name == "guides"

    # Test registration with invalid function
    def invalid_function(x):
        return x

    with pytest.raises(TypeError):
        register_guide("invalid", invalid_function)
