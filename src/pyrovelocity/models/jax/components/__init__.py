"""
JAX/NumPyro components for PyroVelocity models.

This module provides backward compatibility re-exports for components that have been
reorganized into model-specific subdirectories.
"""

# Re-export all functions from piecewise components for backward compatibility
from pyrovelocity.models.jax.components.piecewise import (
    register_piecewise_activation_components,
)

# Re-export specific functions that were previously available directly from components
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

# Note: Registration happens automatically when piecewise module is imported
# due to the import statement above, so no explicit call needed here
