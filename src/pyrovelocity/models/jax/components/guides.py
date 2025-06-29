"""
Standard guide factory functions for PyroVelocity JAX/NumPyro implementation.

This module registers standard guide factory functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Callable, Dict, Optional

import numpyro
from numpyro.infer.autoguide import AutoDelta, AutoNormal

from pyrovelocity.models.jax.registry import register_guide


def auto_normal_guide_factory(
    model: Callable,
    guide_params: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Auto normal guide factory function.

    This function creates an AutoNormal guide for the given model.

    Args:
        model: Model function
        guide_params: Dictionary of guide parameters

    Returns:
        Guide function
    """
    if guide_params is None:
        guide_params = {}

    # Get guide parameters
    init_loc_fn = guide_params.get("init_loc_fn", None)
    init_scale = guide_params.get("init_scale", 0.1)

    # Create guide
    return AutoNormal(
        model,
        init_loc_fn=init_loc_fn,
        init_scale=init_scale,
    )


def auto_delta_guide_factory(
    model: Callable,
    guide_params: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Auto delta guide factory function.

    This function creates an AutoDelta guide for the given model.

    Args:
        model: Model function
        guide_params: Dictionary of guide parameters

    Returns:
        Guide function
    """
    if guide_params is None:
        guide_params = {}

    # Get guide parameters
    init_loc_fn = guide_params.get("init_loc_fn", None)

    # Create guide
    return AutoDelta(
        model,
        init_loc_fn=init_loc_fn,
    )




def register_standard_guides():
    """Register standard guide factory functions."""
    register_guide("auto", auto_normal_guide_factory)
    register_guide("auto_normal", auto_normal_guide_factory)
    register_guide("auto_normal_guide", auto_normal_guide_factory)  # Alias for tests
    register_guide("auto_delta", auto_delta_guide_factory)
