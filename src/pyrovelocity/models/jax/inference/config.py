"""
Inference configuration utilities for PyroVelocity JAX/NumPyro implementation.

This module contains inference configuration utilities, including:

- InferenceConfig: Configuration dataclass for inference methods
- create_inference_config: Factory function for creating configurations
- validate_config: Validate inference configuration
"""

from typing import Dict, Tuple, Optional, Any, List, Union
import jax
import jax.numpy as jnp
import numpyro
from jaxtyping import Array, Float
from beartype import beartype

from pyrovelocity.models.jax.state import InferenceConfig


@beartype
def create_inference_config(method: str = "svi", **kwargs) -> InferenceConfig:
    """Factory function for creating inference configurations.

    Args:
        method: Inference method ("svi" or "mcmc")
        **kwargs: Additional configuration parameters

    Returns:
        InferenceConfig object
    """
    # Create default config based on method
    if method == "svi":
        config = InferenceConfig(
            method="svi",
            num_samples=1000,
            guide_type="auto_normal",
            optimizer="adam",
            learning_rate=0.01,
            num_epochs=1000,
            batch_size=None,
            clip_norm=None,
            early_stopping=True,
            early_stopping_patience=10,
        )
    elif method == "mcmc":
        config = InferenceConfig(
            method="mcmc",
            num_samples=1000,
            num_warmup=500,
            num_chains=1,
        )
    else:
        raise ValueError(f"Unknown inference method: {method}")

    # Update config with kwargs
    return config.replace(**kwargs)


@beartype
def validate_config(config: InferenceConfig) -> bool:
    """Validate inference configuration.

    Args:
        config: InferenceConfig object

    Returns:
        True if the configuration is valid, False otherwise
    """
    # Validate method
    if config.method not in ["svi", "mcmc"]:
        return False
    
    # Common validations
    if config.num_samples <= 0:
        return False
    
    # SVI-specific validations
    if config.method == "svi":
        if config.learning_rate <= 0:
            return False
        if config.num_epochs <= 0:
            return False
        if config.early_stopping_patience <= 0:
            return False
        if config.guide_type not in [
            "auto_normal", 
            "auto_diagonal_normal", 
            "auto_multivariate_normal", 
            "auto_lowrank_multivariate_normal"
        ]:
            return False
        if config.optimizer not in ["adam", "adamax", "rmsprop"]:
            return False
        if config.batch_size is not None and config.batch_size <= 0:
            return False
        if config.clip_norm is not None and config.clip_norm <= 0:
            return False
    
    # MCMC-specific validations
    elif config.method == "mcmc":
        if config.num_warmup < 0:
            return False
        if config.num_chains <= 0:
            return False
    
    return True


@beartype
def get_default_config(method: str = "svi") -> InferenceConfig:
    """Get default inference configuration.

    Args:
        method: Inference method ("svi" or "mcmc")

    Returns:
        Default InferenceConfig object
    """
    return create_inference_config(method=method)
