"""
Piecewise activation components for PyroVelocity JAX/NumPyro implementation.

This module registers piecewise activation components for the JAX implementation of PyroVelocity.
"""

from pyrovelocity.models.jax.components.dynamics import (
    register_piecewise_activation_dynamics,
)
from pyrovelocity.models.jax.components.priors import (
    register_standard_priors,
)
from pyrovelocity.models.jax.components.likelihoods import (
    register_standard_likelihoods,
)


def register_piecewise_activation_components():
    """Register all piecewise activation components."""
    register_piecewise_activation_dynamics()
    register_standard_priors()
    register_standard_likelihoods()


# Register piecewise activation components when the module is imported
register_piecewise_activation_components()
