"""
Type definitions for PyroVelocity JAX/NumPyro component interfaces.

This module defines function type definitions that establish the interfaces for
different component types in the JAX implementation of PyroVelocity.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float, Int
from beartype import beartype
import inspect

# Import interface definitions
from pyrovelocity.models.jax.interfaces.dynamics import (
    DynamicsFunction,
    validate_dynamics_function,
)
from pyrovelocity.models.jax.interfaces.priors import (
    PriorFunction,
    validate_prior_function,
)
from pyrovelocity.models.jax.interfaces.likelihoods import (
    LikelihoodFunction,
    validate_likelihood_function,
)
from pyrovelocity.models.jax.interfaces.guides import (
    GuideFactoryFunction,
    validate_guide_factory_function,
)

__all__ = [
    # Dynamics
    "DynamicsFunction",
    "validate_dynamics_function",
    # Priors
    "PriorFunction",
    "validate_prior_function",
    # Likelihoods
    "LikelihoodFunction",
    "validate_likelihood_function",
    # Guides
    "GuideFactoryFunction",
    "validate_guide_factory_function",
]
