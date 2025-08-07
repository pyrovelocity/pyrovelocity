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
    GuideFunctionConfig,
    ModelConfig,
)

from pyrovelocity.models.jax.registry import (
    get_dynamics,
    get_prior,
    get_likelihood,
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






def create_piecewise_activation_model_jax() -> Callable:
    """
    Create JAX piecewise activation model using registry system.
    
    This function creates a PyroVelocity model with native JAX piecewise activation
    components retrieved directly from the registry system:
    - Piecewise activation dynamics function
    - Piecewise activation prior function  
    - Piecewise activation likelihood function
    - Auto guide factory function
    
    Returns:
        A model function with JAX-native piecewise activation components.
        
    Raises:
        ValueError: If any required component is not registered.
    """
    # Retrieve JAX-native components from registry
    dynamics_fn = get_dynamics("piecewise_activation")
    prior_fn = get_prior("piecewise_activation")
    likelihood_fn = get_likelihood("piecewise_activation")
    guide_factory_fn = get_guide("auto")
    
    # Validate all components are available
    if dynamics_fn is None:
        raise ValueError("Piecewise activation dynamics function not registered")
    if prior_fn is None:
        raise ValueError("Piecewise activation prior function not registered")
    if likelihood_fn is None:
        raise ValueError("Piecewise activation likelihood function not registered")
    if guide_factory_fn is None:
        raise ValueError("Auto guide factory function not registered")
    
    # Create the JAX model configuration using registry components
    config = ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="piecewise_activation"),
        prior_function=PriorFunctionConfig(name="piecewise_activation"),
        likelihood_function=LikelihoodFunctionConfig(name="piecewise_activation"),
        guide_function=GuideFunctionConfig(name="auto"),
    )
    
    return create_model(config)


def piecewise_activation_model_config() -> ModelConfig:
    """
    Create a configuration for a piecewise activation PyroVelocity model.

    This function returns a configuration for a PyroVelocity model with piecewise
    activation components: piecewise activation dynamics function, piecewise 
    activation prior function, piecewise activation likelihood function, and auto 
    guide factory function.

    Returns:
        A ModelConfig object for piecewise activation model.
    """
    return ModelConfig(
        dynamics_function=DynamicsFunctionConfig(name="piecewise_activation"),
        prior_function=PriorFunctionConfig(name="piecewise_activation"),
        likelihood_function=LikelihoodFunctionConfig(name="piecewise_activation"),
        guide_function=GuideFunctionConfig(name="auto"),
    )


def create_piecewise_activation_model() -> Callable:
    """
    Create a piecewise activation PyroVelocity model.

    This function creates a PyroVelocity model with piecewise activation components:
    piecewise activation dynamics function, piecewise activation prior function, 
    piecewise activation likelihood function, and auto guide factory function.

    Returns:
        A model function with piecewise activation components.
    """
    return create_model(piecewise_activation_model_config())


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
    guide_factory_fn = create_guide_factory_function(config.guide_function)

    # Create the model function that uses registry components
    def model(
        u_obs: Optional[Float[Array, "batch_size n_cells n_genes"]] = None,
        s_obs: Optional[Float[Array, "batch_size n_cells n_genes"]] = None,
        u_log_library: Optional[Float[Array, "batch_size n_cells"]] = None,
        s_log_library: Optional[Float[Array, "batch_size n_cells"]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        num_cells: Optional[int] = None,
        num_genes: Optional[int] = None,
    ) -> Dict[str, Float[Array, "..."]]:
        """
        Flexible PyroVelocity model function using registry components.

        Args:
            u_obs: Observed unspliced counts (None for prior predictive sampling)
            s_obs: Observed spliced counts (None for prior predictive sampling)
            u_log_library: Log library size for unspliced counts
            s_log_library: Log library size for spliced counts
            model_params: Additional parameters for the model
            num_cells: Number of cells (required if u_obs/s_obs are None)
            num_genes: Number of genes (required if u_obs/s_obs are None)

        Returns:
            Dictionary of model outputs
        """
        if model_params is None:
            model_params = {}
            
        # Get dimensions from observations or parameters
        if u_obs is not None and s_obs is not None:
            batch_size, n_cells, n_genes = u_obs.shape
        else:
            # For prior predictive sampling, get dimensions from parameters
            if num_cells is None or num_genes is None:
                raise ValueError("num_cells and num_genes must be provided when u_obs/s_obs are None")
            batch_size = 1  # Default batch size for prior predictive
            n_cells = num_cells
            n_genes = num_genes

        # Create default log library sizes if not provided
        if u_log_library is None:
            if u_obs is not None:
                u_log_library = jnp.log(jnp.sum(u_obs, axis=-1) + 1e-6)
            else:
                # Default log library for prior predictive sampling
                u_log_library = jnp.zeros((batch_size, n_cells))
        if s_log_library is None:
            if s_obs is not None:
                s_log_library = jnp.log(jnp.sum(s_obs, axis=-1) + 1e-6)
            else:
                # Default log library for prior predictive sampling
                s_log_library = jnp.zeros((batch_size, n_cells))

        # Use observations directly (no transformation needed in focused architecture)
        # For prior predictive sampling, preserve None values for the likelihood
        u_transformed, s_transformed = u_obs, s_obs

        # Sample model parameters using the prior function
        # Pass dimensions via model_params to the prior function
        prior_params = model_params.get("prior_params", {})
        prior_params["n_cells"] = n_cells
        
        # Call the prior function (handles its own numpyro sampling)
        # NumPyro manages randomness automatically through numpyro.sample() calls
        sampled_params = prior_fn(
            num_genes=n_genes,
            prior_params=prior_params
        )

        # Call the dynamics function to get expected RNA counts
        dynamics_params = model_params.get("dynamics_params", {})
        
        # Extract time parameter from sampled parameters
        t_star_raw = sampled_params["t_star"]  # May have various shapes from NumPyro predictive
        
        # Ensure correct shape: handle different NumPyro predictive behaviors
        if t_star_raw.shape == (n_cells, batch_size):
            # Transpose if dimensions are swapped
            t_star = t_star_raw.T  # Shape: (batch_size, n_cells)
        elif t_star_raw.shape == (batch_size, n_cells):
            # Already correct shape
            t_star = t_star_raw
        else:
            # Try to squeeze and reshape
            t_star_squeezed = jnp.squeeze(t_star_raw)
            if t_star_squeezed.ndim == 1 and t_star_squeezed.shape[0] == n_cells:
                t_star = t_star_squeezed[jnp.newaxis, :]  # Shape: (1, n_cells)
            else:
                raise ValueError(f"Unexpected t_star shape: {t_star_raw.shape}, expected compatible with ({batch_size}, {n_cells})")
        
        # Create initial condition using dimensions instead of observations
        u0_star = jnp.ones((batch_size, n_cells, n_genes))  # Fixed initial condition
        
        # Call the dynamics function to get dimensionless concentrations
        u_star, s_star = dynamics_fn(
            t_star, u0_star, {**sampled_params, **dynamics_params}
        )

        # Register dimensionless concentrations as deterministic sites
        # These are the raw dynamics outputs without any observation model scaling
        numpyro.deterministic("u_star", u_star)
        numpyro.deterministic("s_star", s_star)

        # Call the likelihood function
        likelihood_params = model_params.get("likelihood_params", {})
        likelihood_fn(
            context={
                **sampled_params,
                "u_obs": u_transformed,
                "s_obs": s_transformed,
                "u_star": u_star,
                "s_star": s_star,
                "u_log_library": u_log_library,
                "s_log_library": s_log_library,
                "n_cells": n_cells,
                "n_genes": n_genes,
                **likelihood_params
            }
        )

        # Return all sampled parameters and computed values
        return {
            **sampled_params,
            "u_star": u_star,
            "s_star": s_star,
        }

    return model
