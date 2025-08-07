"""
Piecewise activation components for PyroVelocity JAX/NumPyro implementation.

This module registers piecewise activation components for the JAX implementation of PyroVelocity.
"""

from pyrovelocity.models.jax.components.piecewise.dynamics import (
    piecewise_activation_dynamics_function,
    register_piecewise_activation_dynamics,
)
from pyrovelocity.models.jax.components.piecewise.priors import (
    register_standard_priors,
)
from pyrovelocity.models.jax.components.piecewise.likelihoods import (
    piecewise_activation_likelihood_function,
    register_standard_likelihoods,
)
from pyrovelocity.models.jax.components.piecewise.guides import (
    auto_delta_guide_factory,
    auto_normal_guide_factory,
    register_standard_guides,
)


def register_piecewise_activation_components():
    """Register all piecewise activation components."""
    register_piecewise_activation_dynamics()
    register_standard_priors()
    register_standard_likelihoods()
    register_standard_guides()


# Register piecewise activation components when the module is imported
register_piecewise_activation_components()