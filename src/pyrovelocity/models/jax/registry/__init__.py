"""
Registry system for PyroVelocity JAX/NumPyro implementation.

This module provides a registry system for registering and retrieving functions
for different component types in the JAX implementation of PyroVelocity.
"""

from typing import Any, Callable, Dict, List, Optional, Type
import functools

from pyrovelocity.models.jax.registry.base import (
    Registry,
    register,
    get_registry,
)

from pyrovelocity.models.jax.registry.dynamics import (
    DynamicsRegistry,
    register_dynamics,
    get_dynamics,
    list_dynamics,
)

from pyrovelocity.models.jax.registry.priors import (
    PriorRegistry,
    register_prior,
    get_prior,
    list_priors,
)

from pyrovelocity.models.jax.registry.likelihoods import (
    LikelihoodRegistry,
    register_likelihood,
    get_likelihood,
    list_likelihoods,
)


from pyrovelocity.models.jax.registry.guides import (
    GuideRegistry,
    register_guide,
    get_guide,
    list_guides,
)

__all__ = [
    # Base registry
    "Registry",
    "register",
    "get_registry",
    # Dynamics registry
    "DynamicsRegistry",
    "register_dynamics",
    "get_dynamics",
    "list_dynamics",
    # Prior registry
    "PriorRegistry",
    "register_prior",
    "get_prior",
    "list_priors",
    # Likelihood registry
    "LikelihoodRegistry",
    "register_likelihood",
    "get_likelihood",
    "list_likelihoods",
    # Guide registry
    "GuideRegistry",
    "register_guide",
    "get_guide",
    "list_guides",
]
