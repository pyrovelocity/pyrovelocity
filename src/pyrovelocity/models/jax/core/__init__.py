"""
Core JAX/NumPyro implementation for PyroVelocity.

This module contains the core components for the JAX/NumPyro implementation
of PyroVelocity, including dynamics models, prior distributions, likelihood
models, and utility functions.
"""

from pyrovelocity.models.jax.core.utils import (
    create_key,
    split_key,
    set_platform_device,
    enable_x64,
    disable_x64,
    get_device_count,
    get_devices,
    check_array_shape,
    check_array_dtype,
    ensure_array,
)

from pyrovelocity.models.jax.core.state import (
    VelocityModelState,
    TrainingState,
    InferenceState,
    ModelConfig,
    InferenceConfig,
)



from pyrovelocity.models.jax.core.likelihoods import (
    poisson_likelihood,
    negative_binomial_likelihood,
    create_likelihood,
)

from pyrovelocity.models.jax.core.model import (
    velocity_model,
    create_model,
)

__all__ = [
    # Utils
    "create_key",
    "split_key",
    "set_platform_device",
    "enable_x64",
    "disable_x64",
    "get_device_count",
    "get_devices",
    "check_array_shape",
    "check_array_dtype",
    "ensure_array",
    # State
    "VelocityModelState",
    "TrainingState",
    "InferenceState",
    "ModelConfig",
    "InferenceConfig",
    # Likelihoods
    "poisson_likelihood",
    "negative_binomial_likelihood",
    "create_likelihood",
    # Model
    "velocity_model",
    "create_model",
]
