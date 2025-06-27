"""
Factory functions for PyroVelocity JAX/NumPyro implementation.

This module provides factory functions for creating models and components
for the JAX implementation of PyroVelocity.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

from beartype import beartype
import jax
import jax.numpy as jnp
import numpyro
from jaxtyping import Array, Float

from pyrovelocity.models.jax.factory.config import (
    DynamicsFunctionConfig,
    PriorFunctionConfig,
    LikelihoodFunctionConfig,
    ObservationFunctionConfig,
    GuideFunctionConfig,
    ModelConfig,
)

from pyrovelocity.models.jax.registry import (
    get_dynamics,
    get_prior,
    get_likelihood,
    get_observation,
    get_guide,
)


@beartype
def create_dynamics_function(
    config: Union[str, Dict, DynamicsFunctionConfig]
) -> Callable:
    """
    Create a dynamics function from a configuration.

    Args:
        config: Configuration for the dynamics function, either as a string,
               a DynamicsFunctionConfig object, or a dictionary.

    Returns:
        The dynamics function.

    Raises:
        ValueError: If the specified function is not registered.
    """
    # Convert config to a DynamicsFunctionConfig object
    if isinstance(config, str):
        config = DynamicsFunctionConfig(name=config)
    elif isinstance(config, dict):
        config = DynamicsFunctionConfig(**config)

    # Get the function from the registry
    fn = get_dynamics(config.name)
    if fn is None:
        raise ValueError(f"Dynamics function '{config.name}' is not registered")

    return fn


@beartype
def create_prior_function(
    config: Union[str, Dict, PriorFunctionConfig]
) -> Callable:
    """
    Create a prior function from a configuration.

    Args:
        config: Configuration for the prior function, either as a string,
               a PriorFunctionConfig object, or a dictionary.

    Returns:
        The prior function.

    Raises:
        ValueError: If the specified function is not registered.
    """
    # Convert config to a PriorFunctionConfig object
    if isinstance(config, str):
        config = PriorFunctionConfig(name=config)
    elif isinstance(config, dict):
        config = PriorFunctionConfig(**config)

    # Get the function from the registry
    fn = get_prior(config.name)
    if fn is None:
        raise ValueError(f"Prior function '{config.name}' is not registered")

    return fn


@beartype
def create_likelihood_function(
    config: Union[str, Dict, LikelihoodFunctionConfig]
) -> Callable:
    """
    Create a likelihood function from a configuration.

    Args:
        config: Configuration for the likelihood function, either as a string,
               a LikelihoodFunctionConfig object, or a dictionary.

    Returns:
        The likelihood function.

    Raises:
        ValueError: If the specified function is not registered.
    """
    # Convert config to a LikelihoodFunctionConfig object
    if isinstance(config, str):
        config = LikelihoodFunctionConfig(name=config)
    elif isinstance(config, dict):
        config = LikelihoodFunctionConfig(**config)

    # Get the function from the registry
    fn = get_likelihood(config.name)
    if fn is None:
        raise ValueError(
            f"Likelihood function '{config.name}' is not registered"
        )

    return fn


@beartype
def create_observation_function(
    config: Union[str, Dict, ObservationFunctionConfig]
) -> Callable:
    """
    Create an observation function from a configuration.

    Args:
        config: Configuration for the observation function, either as a string,
               an ObservationFunctionConfig object, or a dictionary.

    Returns:
        The observation function.

    Raises:
        ValueError: If the specified function is not registered.
    """
    # Convert config to an ObservationFunctionConfig object
    if isinstance(config, str):
        config = ObservationFunctionConfig(name=config)
    elif isinstance(config, dict):
        config = ObservationFunctionConfig(**config)

    # Get the function from the registry
    fn = get_observation(config.name)
    if fn is None:
        raise ValueError(
            f"Observation function '{config.name}' is not registered"
        )

    return fn


@beartype
def create_guide_factory_function(
    config: Union[str, Dict, GuideFunctionConfig]
) -> Callable:
    """
    Create a guide factory function from a configuration.

    Args:
        config: Configuration for the guide factory function, either as a string,
               a GuideFunctionConfig object, or a dictionary.

    Returns:
        The guide factory function.

    Raises:
        ValueError: If the specified function is not registered.
    """
    # Convert config to a GuideFunctionConfig object
    if isinstance(config, str):
        config = GuideFunctionConfig(name=config)
    elif isinstance(config, dict):
        config = GuideFunctionConfig(**config)

    # Get the function from the registry
    fn = get_guide(config.name)
    if fn is None:
        raise ValueError(
            f"Guide factory function '{config.name}' is not registered"
        )

    return fn


@beartype
def create_model(config: Union[Dict, ModelConfig]) -> Callable:
    """
    Create a model from a configuration.

    Args:
        config: Configuration for the model, either as a ModelConfig object
               or a dictionary.

    Returns:
        The model function.

    Raises:
        ValueError: If any of the specified components are not registered.
    """
    # Convert config to a ModelConfig object
    if isinstance(config, dict):
        config = ModelConfig(**config)

    # Create each component
    dynamics_fn = create_dynamics_function(config.dynamics_function)
    prior_fn = create_prior_function(config.prior_function)
    likelihood_fn = create_likelihood_function(config.likelihood_function)
    observation_fn = create_observation_function(config.observation_function)
    guide_factory_fn = create_guide_factory_function(config.guide_function)

    # Create the model function
    def model(
        u_obs: Float[Array, "batch_size n_cells n_genes"],
        s_obs: Float[Array, "batch_size n_cells n_genes"],
        u_log_library: Optional[Float[Array, "batch_size n_cells"]] = None,
        s_log_library: Optional[Float[Array, "batch_size n_cells"]] = None,
    ) -> Dict[str, Float[Array, "..."]]:
        """
        PyroVelocity model function.

        Args:
            u_obs: Observed unspliced counts
            s_obs: Observed spliced counts
            u_log_library: Log library size for unspliced counts
            s_log_library: Log library size for spliced counts

        Returns:
            Dictionary of model outputs
        """
        # Get dimensions
        batch_size, n_cells, n_genes = u_obs.shape

        # Create default log library sizes if not provided
        if u_log_library is None:
            u_log_library = jnp.log(jnp.sum(u_obs, axis=-1))
        if s_log_library is None:
            s_log_library = jnp.log(jnp.sum(s_obs, axis=-1))

        # Apply observation function
        u_transformed, s_transformed = observation_fn(u_obs, s_obs)

        # Sample model parameters
        with numpyro.plate("gene", n_genes):
            # Sample RNA velocity parameters from prior
            # Create a deterministic key for reproducibility
            rng_key = jax.random.PRNGKey(0)

            # Use the prior function to sample parameters with a valid key
            params = prior_fn(rng_key, n_genes)

            # Register parameters with the model
            alpha = numpyro.sample(
                "alpha", numpyro.distributions.Delta(params["alpha"])
            )
            beta = numpyro.sample(
                "beta", numpyro.distributions.Delta(params["beta"])
            )
            gamma = numpyro.sample(
                "gamma", numpyro.distributions.Delta(params["gamma"])
            )

        # Sample latent time for each cell
        with numpyro.plate("cell", n_cells):
            tau = numpyro.sample("tau", numpyro.distributions.Normal(0.0, 1.0))

        # Compute RNA dynamics
        dynamics_params = {"alpha": alpha, "beta": beta, "gamma": gamma}

        # Initial conditions (steady state)
        u0 = alpha / beta
        s0 = alpha / gamma

        # Reshape parameters for broadcasting
        tau_expanded = tau[:, jnp.newaxis]  # Shape: (n_cells, 1)
        u0_expanded = u0[jnp.newaxis, :]  # Shape: (1, n_genes)
        s0_expanded = s0[jnp.newaxis, :]  # Shape: (1, n_genes)

        # Create expanded parameters dictionary
        expanded_params = {
            "alpha": alpha[jnp.newaxis, :],  # Shape: (1, n_genes)
            "beta": beta[jnp.newaxis, :],  # Shape: (1, n_genes)
            "gamma": gamma[jnp.newaxis, :],  # Shape: (1, n_genes)
        }

        # Apply dynamics model to get expected counts
        u_expected, s_expected = dynamics_fn(
            tau_expanded, u0_expanded, s0_expanded, expanded_params
        )

        # Register expected counts with the model
        numpyro.deterministic("u_expected", u_expected)
        numpyro.deterministic("s_expected", s_expected)

        # Create scaling parameters for likelihood
        scaling_params = {
            "u_log_library": u_log_library,
            "s_log_library": s_log_library,
        }

        # Apply likelihood function
        likelihood_fn(
            u_transformed, s_transformed, u_expected, s_expected, scaling_params
        )

        # Return model outputs
        return {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "tau": tau,
            "u_expected": u_expected,
            "s_expected": s_expected,
        }

    return model


def piecewise_activation_model_config() -> ModelConfig:
    """
    Create a configuration for a piecewise activation PyroVelocity model.

    This function returns a configuration for a PyroVelocity model with piecewise
    activation components: piecewise activation dynamics function, lognormal prior 
    function, poisson likelihood function, standard observation function, and auto 
    guide factory function.

    Returns:
        A ModelConfig object with piecewise activation component configurations.
    """
    return ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="piecewise_activation"),
        prior_function=PriorFunctionConfig(name="lognormal"),
        likelihood_function=LikelihoodFunctionConfig(name="poisson"),
        observation_function=ObservationFunctionConfig(name="standard"),
        guide_function=GuideFunctionConfig(name="auto"),
    )


def create_piecewise_activation_model() -> Callable:
    """
    Create a piecewise activation PyroVelocity model.

    This function creates a PyroVelocity model with piecewise activation components:
    piecewise activation dynamics function, lognormal prior function, poisson 
    likelihood function, standard observation function, and auto guide factory 
    function.

    Returns:
        A model function with piecewise activation components.
    """
    return create_model(piecewise_activation_model_config())
