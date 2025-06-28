"""
Type definitions for prior functions in PyroVelocity JAX/NumPyro implementation.

This module defines the PriorFunction type and validation utilities.
"""

from typing import Any, Callable, Dict, Optional, Tuple, get_type_hints
import jax.numpy as jnp
import inspect
from jaxtyping import Array, Float
from beartype import beartype
from beartype.door import is_bearable


# Type definition for prior functions
PriorFunction = Callable[
    [
        jnp.ndarray,  # key (random key)
        int,  # num_genes
        Optional[Dict[str, Any]],  # prior_params (can include n_cells)
    ],
    Dict[str, Float[Array, "..."]],  # sampled parameters (mixed shapes allowed)
]


def validate_prior_function(fn: Callable) -> bool:
    """
    Validate that a function conforms to the PriorFunction interface.

    This function checks if the provided function has the correct signature
    and return type to be used as a prior function in PyroVelocity.

    Args:
        fn: Function to validate

    Returns:
        True if the function conforms to the PriorFunction interface

    Raises:
        TypeError: If the function does not conform to the interface
    """
    # Check if the function is callable
    if not callable(fn):
        raise TypeError("Prior function must be callable")

    # Get function signature
    sig = inspect.signature(fn)
    params = sig.parameters

    # Check parameter count
    if len(params) != 3:
        raise TypeError(
            f"Prior function must have 3 parameters, got {len(params)}"
        )

    # Check parameter names
    param_names = list(params.keys())
    expected_names = ["key", "num_genes", "prior_params"]
    for i, name in enumerate(param_names):
        if name != expected_names[i]:
            raise TypeError(
                f"Parameter {i+1} should be named '{expected_names[i]}', got '{name}'"
            )

    # Check if the third parameter is optional
    if params["prior_params"].default == inspect.Parameter.empty:
        raise TypeError("Parameter 'prior_params' should be optional")

    # Check return type annotation
    return_annotation = sig.return_annotation
    if return_annotation == inspect.Signature.empty:
        raise TypeError("Prior function must have a return type annotation")

    # Check if the function is bearable as a PriorFunction
    if not is_bearable(fn, PriorFunction):
        raise TypeError("Function does not conform to PriorFunction interface")

    return True
