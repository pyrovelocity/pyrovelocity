"""
NumPyro model definition for RNA velocity.

This module contains the NumPyro model definition for RNA velocity, including:

- velocity_model: Main NumPyro model for RNA velocity
- create_model: Factory function for creating models with different components

The JAX/NumPyro implementation of PyroVelocity provides several advantages:
1. JIT compilation for faster execution
2. Automatic vectorization for better hardware utilization
3. Functional programming approach for composability
4. Immutable state containers for thread safety
5. Automatic differentiation for gradient-based inference

Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpyro
    >>> from pyrovelocity.models.jax.core.model import create_model
    >>> from pyrovelocity.models.jax.core.state import ModelConfig
    >>>
    >>> # Create a model configuration
    >>> config = ModelConfig(
    ...     dynamics="standard",
    ...     likelihood="poisson",
    ...     prior="lognormal",
    ...     latent_time=True,
    ...     include_prior=True
    ... )
    >>>
    >>> # Create the model function
    >>> model_fn = create_model(config)
    >>>
    >>> # Generate synthetic data with proper type conversion
    >>> key = jax.random.PRNGKey(0)
    >>> key1, key2 = jax.random.split(key)
    >>> # Generate integer counts
    >>> u_counts = jax.random.poisson(key1, jnp.ones((10, 5)) * 5.0)
    >>> s_counts = jax.random.poisson(key2, jnp.ones((10, 5)) * 5.0)
    >>> # Convert to float arrays to satisfy type annotations
    >>> u_obs = jnp.asarray(u_counts, dtype=jnp.float32)
    >>> s_obs = jnp.asarray(s_counts, dtype=jnp.float32)
    >>>
    >>> # Run the model with numpyro seed handler
    >>> with numpyro.handlers.seed(rng_seed=0):
    ...     results = model_fn(u_obs=u_obs, s_obs=s_obs)
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, Float

from pyrovelocity.models.jax.registry.dynamics import get_dynamics
from pyrovelocity.models.jax.registry.likelihoods import get_likelihood
from pyrovelocity.models.jax.registry.priors import get_prior
from pyrovelocity.models.jax.core.likelihoods import (
    poisson_likelihood,
)
from pyrovelocity.models.jax.core.state import (
    ModelConfig,
)

# Default registry component accessors
def _get_default_dynamics():
    """Get default dynamics function from registry."""
    dynamics_fn = get_dynamics("piecewise_activation")
    if dynamics_fn is None:
        raise ValueError("piecewise_activation dynamics not found in registry")
    return dynamics_fn

def _get_default_likelihood():
    """Get default likelihood function from registry.""" 
    likelihood_fn = get_likelihood("piecewise_activation")
    if likelihood_fn is None:
        raise ValueError("piecewise_activation likelihood not found in registry")
    return likelihood_fn

def _get_default_prior():
    """Get default prior function from registry."""
    prior_fn = get_prior("piecewise_activation")
    if prior_fn is None:
        raise ValueError("piecewise_activation prior not found in registry")
    return prior_fn


@beartype
def velocity_model(
    u_obs: Float[Array, "cell gene"],
    s_obs: Float[Array, "cell gene"],
    u_log_library: Optional[Float[Array, "cell"]] = None,
    s_log_library: Optional[Float[Array, "cell"]] = None,
    dynamics_fn: Optional[Callable] = None,
    likelihood_fn: Optional[Callable] = None,
    prior_fn: Optional[Callable] = None,
    latent_time: bool = True,
    include_prior: bool = True,
) -> Dict[str, Float[Array, "..."]]:
    """Main NumPyro model for RNA velocity.

    This function defines a generic probabilistic model for RNA velocity analysis using
    NumPyro. The model consists of configurable components:

    1. Prior distributions for model parameters (determined by prior_fn)
    2. Latent time for each cell (optional)
    3. RNA dynamics equations (determined by dynamics_fn)
    4. Likelihood distributions for observed RNA counts (determined by likelihood_fn)

    The model is component-agnostic and works with any combination of registered
    dynamics, prior, and likelihood functions. Parameter structures are determined
    by the chosen prior function.

    Args:
        u_obs: Observed unspliced RNA counts of shape [cell, gene]
        s_obs: Observed spliced RNA counts of shape [cell, gene]
        u_log_library: Log library size for unspliced RNA of shape [cell]
        s_log_library: Log library size for spliced RNA of shape [cell]
        dynamics_fn: Function that implements RNA dynamics equations
        likelihood_fn: Function that defines the likelihood distribution
        prior_fn: Function that defines prior distributions for parameters
        latent_time: Whether to use latent time for each cell
        include_prior: Whether to include prior distributions in the model

    Returns:
        Dictionary of model outputs including:
        - All parameters sampled from the prior function
        - tau: Latent time of shape [cell] (if latent_time=True)
        - u_expected: Expected unspliced RNA counts of shape [cell, gene]
        - s_expected: Expected spliced RNA counts of shape [cell, gene]

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> tmp = getfixture("tmp_path")
        >>> # Generate synthetic data with proper type conversion
        >>> key = jax.random.PRNGKey(0)
        >>> key1, key2 = jax.random.split(key)
        >>> # Generate integer counts
        >>> u_counts = jax.random.poisson(key1, jnp.ones((10, 5)) * 5.0)
        >>> s_counts = jax.random.poisson(key2, jnp.ones((10, 5)) * 5.0)
        >>> # Convert to float arrays to satisfy type annotations
        >>> u_obs = jnp.asarray(u_counts, dtype=jnp.float32)
        >>> s_obs = jnp.asarray(s_counts, dtype=jnp.float32)
        >>> # Run the model
        >>> with numpyro.handlers.seed(rng_seed=0):
        ...     results = velocity_model(u_obs=u_obs, s_obs=s_obs)
    """
    # Set defaults from registry if not provided
    if dynamics_fn is None:
        dynamics_fn = _get_default_dynamics()
    if likelihood_fn is None:
        likelihood_fn = _get_default_likelihood()  
    if prior_fn is None:
        prior_fn = _get_default_prior()

    # Get dimensions
    num_cells, num_genes = u_obs.shape

    # Create default log library sizes if not provided
    if u_log_library is None:
        u_log_library = jnp.log(jnp.sum(u_obs, axis=1))
    if s_log_library is None:
        s_log_library = jnp.log(jnp.sum(s_obs, axis=1))

    # Sample model parameters using prior function
    # Create prior_params with required cell count
    prior_params = {"n_cells": num_cells}

    # Call the prior function within NumPyro context (it handles sampling internally)
    rng_key = jax.random.PRNGKey(0)
    sampled_params = prior_fn(rng_key, num_genes, prior_params)

    # Sample latent time for each cell
    if latent_time:
        with numpyro.plate("cell", num_cells):
            tau = numpyro.sample("tau", dist.Normal(0.0, 1.0))
    else:
        # Use fixed time points if latent_time is False
        tau = jnp.linspace(0.0, 1.0, num_cells)
        numpyro.deterministic("tau", tau)

    # Compute RNA dynamics with generic parameters
    # Use t_star from sampled parameters
    t_star = sampled_params["t_star"]  # Shape: (num_cells,)
    
    # Expand time coordinates for dynamics function: Shape (1, num_cells, num_genes)
    time_expanded = t_star[jnp.newaxis, :, jnp.newaxis] * jnp.ones((1, 1, num_genes))
    
    # Create generic initial conditions (dynamics function should handle specifics)  
    u0_expanded = jnp.ones((1, num_cells, num_genes))

    # Create parameters dictionary for dynamics function (exclude time coordinate)
    dynamics_params = {}
    for param_name, param_value in sampled_params.items():
        if param_name != "t_star":  # Don't pass time coordinate as parameter
            dynamics_params[param_name] = param_value

    # Apply dynamics model to get expected counts
    u_expected, s_expected = dynamics_fn(
        time_expanded, u0_expanded, dynamics_params
    )

    # Register expected counts with the model
    numpyro.deterministic("u_expected", u_expected)
    numpyro.deterministic("s_expected", s_expected)

    # Create context for likelihood function
    likelihood_context = {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "u_expected": u_expected,
        "s_expected": s_expected,
        "u_log_library": u_log_library,
        "s_log_library": s_log_library,
    }

    # Call likelihood function with context (handles sampling internally)
    likelihood_fn(likelihood_context)

    # Return model outputs (generic - includes all sampled parameters)
    return {
        **sampled_params,  # All parameters from prior function
        "tau": tau,
        "u_expected": u_expected,
        "s_expected": s_expected,
    }


@beartype
def create_model(
    config: ModelConfig,
) -> Callable:
    """Factory function for creating RNA velocity models with different components.

    This function creates a model function based on the provided configuration.
    It selects the appropriate dynamics function, likelihood function, and prior
    function based on the configuration, and returns a model function that can
    be used for inference.

    The factory pattern allows for flexible composition of different model components
    without modifying the core model definition. This enables users to experiment
    with different combinations of dynamics, likelihoods, and priors.

    Args:
        config: Model configuration object with the following attributes:
            - dynamics: Type of dynamics function ("standard", "nonlinear", or "ode")
            - likelihood: Type of likelihood function ("poisson" or "negative_binomial")
            - prior: Type of prior function ("lognormal" or "informative")
            - latent_time: Whether to use latent time
            - include_prior: Whether to include prior in the model

    Returns:
        A model function that takes observed data and returns model outputs

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from pyrovelocity.models.jax.core.state import ModelConfig
        >>> tmp = getfixture("tmp_path")
        >>> # Create a model configuration
        >>> config = ModelConfig(
        ...     dynamics="standard",
        ...     likelihood="poisson",
        ...     prior="lognormal",
        ...     latent_time=True,
        ...     include_prior=True
        ... )
        >>>
        >>> # Create a model function
        >>> model_fn = create_model(config)
        >>>
        >>> # Generate synthetic data with proper type conversion
        >>> key = jax.random.PRNGKey(0)
        >>> key1, key2 = jax.random.split(key)
        >>> # Generate integer counts
        >>> u_counts = jax.random.poisson(key1, jnp.ones((10, 5)) * 5.0)
        >>> s_counts = jax.random.poisson(key2, jnp.ones((10, 5)) * 5.0)
        >>> # Convert to float arrays to satisfy type annotations
        >>> u_obs = jnp.asarray(u_counts, dtype=jnp.float32)
        >>> s_obs = jnp.asarray(s_counts, dtype=jnp.float32)
        >>>
        >>> # Use the model function for inference with numpyro seed handler
        >>> import numpyro
        >>> with numpyro.handlers.seed(rng_seed=0):
        ...     results = model_fn(u_obs=u_obs, s_obs=s_obs)
    """
    # Select dynamics function using registry
    dynamics_fn = get_dynamics(config.dynamics)
    if dynamics_fn is None:
        raise ValueError(f"Unknown dynamics type: {config.dynamics}")

    # Select likelihood function using registry
    likelihood_fn = get_likelihood(config.likelihood)
    if likelihood_fn is None:
        raise ValueError(f"Unknown likelihood type: {config.likelihood}")

    # Select prior function using registry
    prior_fn = get_prior(config.prior)
    if prior_fn is None:
        raise ValueError(f"Unknown prior type: {config.prior}")

    # Create model function
    def model_fn(
        u_obs: Float[Array, "cell gene"],
        s_obs: Float[Array, "cell gene"],
        u_log_library: Optional[Float[Array, "cell"]] = None,
        s_log_library: Optional[Float[Array, "cell"]] = None,
    ) -> Dict[str, Float[Array, "..."]]:
        """NumPyro model function for RNA velocity with configured components.

        This function is the actual model function that will be used for inference.
        It's created by the factory function with the specified configuration and
        component functions.

        The function takes observed RNA counts and library sizes, and returns
        a dictionary of model outputs. It delegates to the velocity_model function
        with the configured dynamics, likelihood, and prior functions.

        Args:
            u_obs: Observed unspliced RNA counts of shape [cell, gene]
            s_obs: Observed spliced RNA counts of shape [cell, gene]
            u_log_library: Log library size for unspliced RNA of shape [cell]
            s_log_library: Log library size for spliced RNA of shape [cell]

        Returns:
            Dictionary of model outputs including:
            - All parameters sampled from the configured prior function
            - tau: Latent time of shape [cell] (if latent_time=True)
            - u_expected: Expected unspliced RNA counts of shape [cell, gene]
            - s_expected: Expected spliced RNA counts of shape [cell, gene]
        """
        return velocity_model(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
            dynamics_fn=dynamics_fn,
            likelihood_fn=likelihood_fn,
            prior_fn=prior_fn,
            latent_time=config.latent_time,
            include_prior=config.include_prior,
        )

    return model_fn
