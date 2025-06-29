"""
Type definitions for likelihood functions in PyroVelocity JAX/NumPyro implementation.

This module defines the LikelihoodFunction type and validation utilities.
"""

from typing import Any, Callable, Dict, Optional, Tuple, get_type_hints
import jax.numpy as jnp
import inspect
from jaxtyping import Array, Float
from beartype import beartype
from beartype.door import is_bearable


# Type definition for likelihood functions
LikelihoodFunction = Callable[
    [
        Dict[str, Any],  # context dictionary containing all required parameters
    ],
    None,  # PyroEffect (implicit)
]


def validate_likelihood_function(fn: Callable) -> bool:
    """
    Validate that a function conforms to the LikelihoodFunction interface.

    This function checks if the provided function has the correct signature
    to be used as a likelihood function in PyroVelocity. Likelihood functions
    are expected to use numpyro.sample inside their implementation.

    Args:
        fn: Function to validate

    Returns:
        True if the function conforms to the LikelihoodFunction interface

    Raises:
        TypeError: If the function does not conform to the interface
    """
    # Check if the function is callable
    if not callable(fn):
        raise TypeError("Likelihood function must be callable")

    # Get function signature
    sig = inspect.signature(fn)
    params = sig.parameters

    # Check parameter count (expecting 1 parameter: context dict)
    if len(params) != 1:
        raise TypeError(
            f"Likelihood function must have 1 parameter (context dict), got {len(params)}"
        )

    # Check parameter names (expecting single 'context' parameter)
    param_names = list(params.keys())
    expected_name = "context"
    if param_names[0] != expected_name:
        raise TypeError(
            f"Parameter should be named '{expected_name}', got '{param_names[0]}'"
        )

    # Check return type annotation
    return_annotation = sig.return_annotation
    if return_annotation != None and return_annotation != type(None):
        raise TypeError(
            "Likelihood function must have None as return type annotation"
        )

    # Check if the function is bearable as a LikelihoodFunction
    if not is_bearable(fn, LikelihoodFunction):
        raise TypeError(
            "Function does not conform to LikelihoodFunction interface"
        )

    return True
