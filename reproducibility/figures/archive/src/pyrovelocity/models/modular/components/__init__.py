"""
PyroVelocity modular component implementations.

This package contains the implementations of the various components used in the
PyroVelocity modular architecture, including dynamics models, prior models,
likelihood models, observation models, and inference guides.

All components directly implement Protocol interfaces defined in interfaces.py,
following a Protocol-First approach that embraces composition over inheritance.
This approach reduces code complexity, enhances flexibility, and creates
architectural consistency with the JAX implementation's pure functional approach.

This package has been simplified to include only the essential components needed for
validation against the legacy implementation.
"""

# Import component implementations
from pyrovelocity.models.modular.components.dynamics import (
    PiecewiseActivationDynamicsModel,
)
from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
)
from pyrovelocity.models.modular.components.likelihoods import (
    PiecewiseActivationPoissonLikelihoodModel,
)
from pyrovelocity.models.modular.components.priors import (
    PiecewiseActivationPriorModel,
)

__all__ = [
    # Component implementations
    "PiecewiseActivationDynamicsModel",
    "PiecewiseActivationPriorModel",
    "PiecewiseActivationPoissonLikelihoodModel",
    "AutoGuideFactory",
]
