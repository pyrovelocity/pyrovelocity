"""
JAX/NumPyro components for PyroVelocity models.

This module provides backward compatibility re-exports for components that have been
reorganized into model-specific subdirectories.
"""

# Re-export all functions from piecewise components for backward compatibility
from pyrovelocity.models.jax.components.piecewise import (
    register_piecewise_activation_components,
)

# Re-export all functions from Poisson components
from pyrovelocity.models.jax.components.poisson import (
    register_poisson_components,
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

# Re-export specific functions from Poisson components
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

# Note: Registration happens automatically when piecewise and poisson modules are imported
# due to the import statements above, so no explicit call needed here
