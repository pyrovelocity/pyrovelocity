"""
Tests for the inference configuration components.
"""

import pytest
import jax
import jax.numpy as jnp

from pyrovelocity.models.jax.inference.config import (
    create_inference_config,
    get_default_config,
)
from pyrovelocity.models.jax.state import InferenceConfig


def test_create_inference_config():
    """Test creating an inference configuration."""
    # Create config with default parameters
    config = create_inference_config()

    # Check that config has expected attributes
    assert hasattr(config, "method")
    assert hasattr(config, "num_samples")
    assert hasattr(config, "num_warmup")
    assert hasattr(config, "num_chains")
    assert hasattr(config, "guide_type")
    assert hasattr(config, "optimizer")
    assert hasattr(config, "learning_rate")
    assert hasattr(config, "num_epochs")
    assert config.num_samples == 1000
    assert config.num_warmup == 500
    assert config.num_chains == 1

    # Create SVI config with custom parameters
    config = create_inference_config(
        method="svi",
        guide_type="auto_delta",
        optimizer="sgd",
        learning_rate=0.1,
        num_epochs=500,
        batch_size=32,
        clip_norm=1.0,
        early_stopping=False,
    )

    # Check that config is an InferenceConfig
    assert isinstance(config, InferenceConfig)

    # Check that config has the correct method
    assert config.method == "svi"

    # Check that config has custom SVI parameters
    assert config.guide_type == "auto_delta"
    assert config.optimizer == "sgd"
    assert config.learning_rate == 0.1
    assert config.num_epochs == 500
    assert config.batch_size == 32
    assert config.clip_norm == 1.0
    assert config.early_stopping == False

    # Create MCMC config with custom parameters
    config = create_inference_config(
        method="mcmc",
        num_samples=2000,
        num_warmup=1000,
        num_chains=4,
    )

    # Check that config is an InferenceConfig
    assert isinstance(config, InferenceConfig)

    # Check that config has the correct method
    assert config.method == "mcmc"

    # Check that config has custom MCMC parameters
    assert config.num_samples == 2000
    assert config.num_warmup == 1000
    assert config.num_chains == 4

    # Check that an error is raised for unknown method
    with pytest.raises(ValueError):
        create_inference_config(method="unknown")


def test_get_default_config():
    """Test getting a default inference configuration."""
    # Get default SVI config
    config = get_default_config(method="svi")

    # Check that config is an InferenceConfig
    assert isinstance(config, InferenceConfig)

    # Check that config has the correct method
    assert config.method == "svi"

    # Check that config has default SVI parameters
    assert config.guide_type == "auto_normal"
    assert config.optimizer == "adam"
    assert config.learning_rate == 0.01
    assert config.num_epochs == 1000
    assert config.early_stopping == True
    assert config.early_stopping_patience == 10

    # Get default MCMC config
    config = get_default_config(method="mcmc")

    # Check that config is an InferenceConfig
    assert isinstance(config, InferenceConfig)

    # Check that config has the correct method
    assert config.method == "mcmc"

    # Check that config has default MCMC parameters
    assert config.num_samples == 1000
    assert config.num_warmup == 500
    assert config.num_chains == 1
