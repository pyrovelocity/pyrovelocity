"""
PyroVelocity JAX/NumPyro Implementation.

This module contains the JAX/NumPyro implementation of PyroVelocity, a probabilistic
model for RNA velocity analysis. This is a prototype implementation focused on
core functionality required for model validation.
"""

# Register piecewise activation components on import
from pyrovelocity.models.jax.components import register_piecewise_activation_components

# Factory functions for model creation
from pyrovelocity.models.jax.factory import (
    ModelConfig,
    create_piecewise_activation_model,
)

# Inference system
from pyrovelocity.models.jax.inference import (
    create_inference_config,
    run_inference,
)

# State containers
from pyrovelocity.models.jax.state import (
    InferenceConfig,
    InferenceState,
    TrainingState,
)

__all__ = [
    # Component registration (happens on import)
    "register_piecewise_activation_components",
    # Factory
    "ModelConfig", 
    "create_piecewise_activation_model",
    # Inference
    "create_inference_config",
    "run_inference",
    # State
    "InferenceConfig",
    "InferenceState", 
    "TrainingState",
]