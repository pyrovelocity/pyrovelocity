"""
Factory system for PyroVelocity JAX/NumPyro implementation.

This module provides a factory system for creating models and components
for the JAX implementation of PyroVelocity.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

from beartype import beartype
import jax.numpy as jnp
from jaxtyping import Array, Float

from pyrovelocity.models.jax.factory.config import (
    DynamicsFunctionConfig,
    PriorFunctionConfig,
    LikelihoodFunctionConfig,
    ObservationFunctionConfig,
    GuideFunctionConfig,
    ModelConfig,
)

from pyrovelocity.models.jax.factory.factory import (
    create_dynamics_function,
    create_prior_function,
    create_likelihood_function,
    create_observation_function,
    create_guide_factory_function,
    create_model,
    piecewise_activation_model_config,
    create_piecewise_activation_model,
)

__all__ = [
    # Configuration classes
    "DynamicsFunctionConfig",
    "PriorFunctionConfig",
    "LikelihoodFunctionConfig",
    "ObservationFunctionConfig",
    "GuideFunctionConfig",
    "ModelConfig",
    # Factory functions
    "create_dynamics_function",
    "create_prior_function",
    "create_likelihood_function",
    "create_observation_function",
    "create_guide_factory_function",
    "create_model",
    # Predefined configurations
    "piecewise_activation_model_config",
    "create_piecewise_activation_model",
]
