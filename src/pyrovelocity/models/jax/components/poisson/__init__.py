"""
Poisson-only components for PyroVelocity JAX/NumPyro implementation.

This module registers Poisson-only components for the JAX implementation of PyroVelocity.
The Poisson model is a simplified RNA velocity model without temporal dynamics,
focusing purely on the Poisson observation model for single-cell RNA count data.
"""

from pyrovelocity.models.jax.components.poisson.dynamics import (
    poisson_dynamics_function,
    register_poisson_dynamics,
)
from pyrovelocity.models.jax.components.poisson.priors import (
    poisson_prior_function,
    register_poisson_priors,
)
from pyrovelocity.models.jax.components.poisson.likelihoods import (
    poisson_likelihood_function,
    register_poisson_likelihoods,
)
from pyrovelocity.models.jax.components.poisson.guides import (
    register_poisson_guides,
)


def register_poisson_components():
    """Register all Poisson-only components."""
    register_poisson_dynamics()
    register_poisson_priors()
    register_poisson_likelihoods()
    register_poisson_guides()


# Register Poisson components when the module is imported
register_poisson_components()