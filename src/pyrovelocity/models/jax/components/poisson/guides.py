"""
Guide factory functions for Poisson-only PyroVelocity JAX/NumPyro implementation.

This module registers guide factory functions for the Poisson-only JAX implementation of PyroVelocity.
These functions create variational distributions (guides) for SVI inference.
"""

from typing import Any, Callable, Dict, Optional

import numpyro
from numpyro import distributions as dist
from numpyro.infer.autoguide import AutoDelta, AutoNormal

from pyrovelocity.models.jax.registry import register_guide


def auto_normal_guide_factory(
    model: Callable,
    guide_params: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Create an AutoNormal guide for Poisson-only model.

    This function creates an AutoNormal variational guide that approximates
    all latent variables with independent normal distributions. This is
    appropriate for the Poisson model's log-normal priors.

    Args:
        model: The model function to create a guide for
        guide_params: Optional parameters for guide configuration

    Returns:
        AutoNormal guide function
    """
    if guide_params is None:
        guide_params = {}
    
    # Create AutoNormal guide with optional initialization
    init_loc_fn = guide_params.get("init_loc_fn", None)
    
    return AutoNormal(
        model,
        init_loc_fn=init_loc_fn,
        create_plates=True,
    )


def auto_delta_guide_factory(
    model: Callable,
    guide_params: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Create an AutoDelta guide for Poisson-only model.

    This function creates an AutoDelta variational guide that approximates
    all latent variables with delta (point mass) distributions. This is
    useful for MAP estimation.

    Args:
        model: The model function to create a guide for
        guide_params: Optional parameters for guide configuration

    Returns:
        AutoDelta guide function
    """
    if guide_params is None:
        guide_params = {}
    
    # Create AutoDelta guide with optional initialization
    init_loc_fn = guide_params.get("init_loc_fn", None)
    
    return AutoDelta(
        model,
        init_loc_fn=init_loc_fn,
        create_plates=True,
    )


def register_poisson_guides():
    """Register Poisson-only guide factory functions."""
    # Only register poisson-specific guides to avoid collisions with piecewise components
    # The "auto", "auto_normal", "auto_delta" guides are already registered by piecewise components
    # and can be reused by poisson models since they're generic guide factories
    pass  # No additional guides needed - reuse existing ones