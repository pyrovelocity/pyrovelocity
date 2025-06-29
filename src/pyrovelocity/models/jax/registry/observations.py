"""
Registry for observation functions in PyroVelocity JAX/NumPyro implementation.

This module provides a registry for observation functions, allowing registration
and retrieval of observation functions by name.
"""

from typing import Callable, List, Optional

from pyrovelocity.models.jax.interfaces.observations import validate_observation_function
from pyrovelocity.models.jax.registry.base import Registry

# Singleton registry instance
_OBSERVATIONS_REGISTRY = None


class ObservationRegistry(Registry):
    """
    Registry for observation functions.

    This class provides a registry for observation functions, allowing registration
    and retrieval of observation functions by name.
    """

    def __init__(self):
        """Initialize the observation registry."""
        super().__init__("observations")

    def register(self, name: str, fn: Callable) -> Callable:
        """
        Register an observation function in the registry.

        Args:
            name: Name to register the function under
            fn: Observation function to register

        Returns:
            The registered function

        Raises:
            ValueError: If a function with the same name is already registered
            TypeError: If the function does not conform to the observation function interface
        """
        # Validate the function
        validate_observation_function(fn)

        return super().register(name, fn)


def get_observation_registry() -> ObservationRegistry:
    """
    Get the observation registry.

    Returns:
        The observation registry
    """
    global _OBSERVATIONS_REGISTRY
    if _OBSERVATIONS_REGISTRY is None:
        _OBSERVATIONS_REGISTRY = ObservationRegistry()

    return _OBSERVATIONS_REGISTRY


def register_observation(name: str, fn: Callable) -> Callable:
    """
    Register an observation function.

    Args:
        name: Name to register the function under
        fn: Observation function to register

    Returns:
        The registered function

    Raises:
        ValueError: If a function with the same name is already registered
        TypeError: If the function does not conform to the observation function interface
    """
    registry = get_observation_registry()
    return registry.register(name, fn)


def get_observation(name: str) -> Optional[Callable]:
    """
    Get an observation function by name.

    Args:
        name: Name of the observation function to get

    Returns:
        The registered observation function, or None if not found
    """
    registry = get_observation_registry()
    return registry.get(name)


def list_observations() -> List[str]:
    """
    List all registered observation function names.

    Returns:
        List of registered observation function names
    """
    registry = get_observation_registry()
    return registry.list()