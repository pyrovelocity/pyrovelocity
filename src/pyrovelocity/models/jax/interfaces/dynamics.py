"""
Type definitions for dynamics functions in PyroVelocity JAX/NumPyro implementation.

This module defines the DynamicsFunction type and validation utilities.
"""

from typing import Any, Callable, Dict, Optional, Tuple, get_type_hints
import jax.numpy as jnp
import inspect
from jaxtyping import Array, Float
from beartype import beartype
from beartype.door import is_bearable


# Type definition for dynamics functions
DynamicsFunction = Callable[
    [
        Float[Array, "batch_size n_cells n_genes"],  # t_star/tau (time points)
        Float[Array, "batch_size n_cells n_genes"],  # u0 (initial unspliced)
        Float[Array, "batch_size n_cells n_genes"],  # s0 (initial spliced)
        Dict[str, Float[Array, "..."]],  # params (model parameters)
    ],
    Tuple[
        Float[Array, "batch_size n_cells n_genes"],  # ut (unspliced at time t)
        Float[Array, "batch_size n_cells n_genes"],  # st (spliced at time t)
    ],
]


def validate_dynamics_function(fn: Callable) -> bool:
    """
    Validate that a function conforms to the DynamicsFunction interface.

    This function checks if the provided function has the correct signature
    and return type to be used as a dynamics function in PyroVelocity.

    Args:
        fn: Function to validate

    Returns:
        True if the function conforms to the DynamicsFunction interface

    Raises:
        TypeError: If the function does not conform to the interface
    """
    # Check if the function is callable
    if not callable(fn):
        raise TypeError("Dynamics function must be callable")

    # Get function signature
    sig = inspect.signature(fn)
    params = sig.parameters

    # Check parameter count
    if len(params) != 4:
        raise TypeError(
            f"Dynamics function must have 4 parameters, got {len(params)}"
        )

    # Check parameter names - use modular implementation names
    param_names = list(params.keys())
    expected_names = ["t_star", "u0_star", "s0_star", "params"]  # Match modular metadata
    for i, name in enumerate(param_names):
        if i == 0 and name in ["tau", "t_star"]:  # Accept both legacy tau and modular t_star
            continue
        elif i == 1 and name in ["u0", "u0_star"]:  # Accept both legacy u0 and modular u0_star  
            continue
        elif i == 2 and name in ["s0", "s0_star"]:  # Accept both legacy s0 and modular s0_star
            continue
        elif i == 3 and name == "params":
            continue
        else:
            raise TypeError(
                f"Parameter {i+1} should be named '{expected_names[i]}' (or legacy equivalent), got '{name}'"
            )

    # Check return type annotation
    return_annotation = sig.return_annotation
    if return_annotation == inspect.Signature.empty:
        raise TypeError("Dynamics function must have a return type annotation")

    # Check if the function is bearable as a DynamicsFunction
    if not is_bearable(fn, DynamicsFunction):
        raise TypeError(
            "Function does not conform to DynamicsFunction interface"
        )

    return True
