"""
Inference configuration utilities for PyroVelocity PyTorch/Pyro modular implementation.

This module contains utilities for creating and validating inference configurations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from beartype import beartype


@dataclass
class InferenceConfig:
    """
    Configuration for inference.

    This class contains configuration parameters for inference, including
    SVI and MCMC parameters.
    """

    # General inference parameters
    method: str = "svi"  # "svi" or "mcmc"
    num_samples: int = 1000  # Number of posterior samples
    seed: Optional[int] = None  # Random seed

    # SVI parameters
    num_epochs: int = 1000  # Number of epochs for SVI
    learning_rate: float = 0.01  # Learning rate for SVI
    optimizer: str = "adam"  # Optimizer for SVI
    guide: str = "auto_normal"  # Guide for SVI
    batch_size: Optional[int] = None  # Batch size for SVI
    early_stopping: bool = True  # Whether to use early stopping
    early_stopping_patience: int = 10  # Patience for early stopping
    clip_norm: Optional[float] = None  # Gradient clipping norm
    train_size: float = 0.8  # Fraction of data to use for training
    valid_size: Optional[float] = 0.2  # Fraction of data to use for validation

    # MCMC parameters
    num_warmup: int = 500  # Number of warmup steps for MCMC
    num_chains: int = 1  # Number of chains for MCMC
    chain_method: str = "parallel"  # Method for running chains
    kernel: str = "nuts"  # MCMC kernel
    target_accept_prob: float = 0.8  # Target acceptance probability for NUTS

    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@beartype
def create_inference_config(
    method: str = "svi",
    num_samples: int = 1000,
    seed: Optional[int] = None,
    num_epochs: int = 1000,
    learning_rate: float = 0.01,
    optimizer: str = "adam",
    guide: str = "auto_normal",
    batch_size: Optional[int] = None,
    early_stopping: bool = True,
    early_stopping_patience: int = 10,
    clip_norm: Optional[float] = None,
    train_size: float = 0.8,
    valid_size: Optional[float] = 0.2,
    num_warmup: int = 500,
    num_chains: int = 1,
    chain_method: str = "parallel",
    kernel: str = "nuts",
    target_accept_prob: float = 0.8,
    **kwargs: Any,
) -> InferenceConfig:
    """
    Create an inference configuration.

    Args:
        method: Inference method ("svi" or "mcmc")
        num_samples: Number of posterior samples
        seed: Random seed
        num_epochs: Number of epochs for SVI
        learning_rate: Learning rate for SVI
        optimizer: Optimizer for SVI
        guide: Guide for SVI
        batch_size: Batch size for SVI
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        clip_norm: Gradient clipping norm
        train_size: Fraction of data to use for training
        valid_size: Fraction of data to use for validation
        num_warmup: Number of warmup steps for MCMC
        num_chains: Number of chains for MCMC
        chain_method: Method for running chains
        kernel: MCMC kernel
        target_accept_prob: Target acceptance probability for NUTS
        **kwargs: Additional parameters

    Returns:
        Inference configuration
    """
    return InferenceConfig(
        method=method,
        num_samples=num_samples,
        seed=seed,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        optimizer=optimizer,
        guide=guide,
        batch_size=batch_size,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        clip_norm=clip_norm,
        train_size=train_size,
        valid_size=valid_size,
        num_warmup=num_warmup,
        num_chains=num_chains,
        chain_method=chain_method,
        kernel=kernel,
        target_accept_prob=target_accept_prob,
        extra_params=kwargs,
    )


@beartype
def validate_config(
    config: Union[InferenceConfig, Dict[str, Any]]
) -> InferenceConfig:
    """
    Validate an inference configuration.

    Args:
        config: Inference configuration or dictionary

    Returns:
        Validated inference configuration
    """
    # Convert dictionary to InferenceConfig if necessary
    if isinstance(config, dict):
        config = create_inference_config(**config)

    # Validate method
    if config.method not in ["svi", "mcmc"]:
        raise ValueError(f"Invalid inference method: {config.method}")

    # Validate SVI parameters if using SVI
    if config.method == "svi":
        if config.num_epochs <= 0:
            raise ValueError(f"Invalid number of epochs: {config.num_epochs}")
        if config.learning_rate <= 0:
            raise ValueError(f"Invalid learning rate: {config.learning_rate}")
        if config.optimizer not in ["adam", "sgd", "rmsprop"]:
            raise ValueError(f"Invalid optimizer: {config.optimizer}")
        # Allow all AutoGuide types and common variations
        valid_guides = [
            "auto_normal", "AutoNormal",
            "auto_diagonal_normal", "AutoDiagonalNormal",
            "auto_multivariate_normal", "AutoMultivariateNormal",
            "auto_low_rank_multivariate_normal", "AutoLowRankMultivariateNormal",
            "auto_delta", "AutoDelta",
            "auto_guide_list", "AutoGuideList",
            "auto_iaf_normal", "AutoIAFNormal",
            "auto_normalizing_flow", "AutoNormalizingFlow",
            "custom"
        ]
        if config.guide not in valid_guides:
            raise ValueError(f"Invalid guide: {config.guide}. Valid options: {valid_guides}")
        if config.batch_size is not None and config.batch_size <= 0:
            raise ValueError(f"Invalid batch size: {config.batch_size}")
        if config.early_stopping_patience <= 0:
            raise ValueError(
                f"Invalid early stopping patience: {config.early_stopping_patience}"
            )
        if config.clip_norm is not None and config.clip_norm <= 0:
            raise ValueError(f"Invalid clip norm: {config.clip_norm}")
        if config.train_size <= 0 or config.train_size > 1:
            raise ValueError(f"Invalid train size: {config.train_size}")
        if config.valid_size is not None and (config.valid_size <= 0 or config.valid_size > 1):
            raise ValueError(f"Invalid validation size: {config.valid_size}")

    # Validate MCMC parameters if using MCMC
    if config.method == "mcmc":
        if config.num_warmup <= 0:
            raise ValueError(
                f"Invalid number of warmup steps: {config.num_warmup}"
            )
        if config.num_chains <= 0:
            raise ValueError(f"Invalid number of chains: {config.num_chains}")
        if config.chain_method not in ["parallel", "sequential"]:
            raise ValueError(f"Invalid chain method: {config.chain_method}")
        if config.kernel not in ["nuts", "hmc"]:
            raise ValueError(f"Invalid MCMC kernel: {config.kernel}")
        if config.target_accept_prob <= 0 or config.target_accept_prob >= 1:
            raise ValueError(
                f"Invalid target acceptance probability: {config.target_accept_prob}"
            )

    # Validate general parameters
    if config.num_samples <= 0:
        raise ValueError(f"Invalid number of samples: {config.num_samples}")

    return config


@beartype
def get_default_config(method: str = "svi") -> InferenceConfig:
    """
    Get default inference configuration.

    Args:
        method: Inference method ("svi" or "mcmc")

    Returns:
        Default inference configuration
    """
    if method == "svi":
        return create_inference_config(
            method="svi",
            num_samples=1000,
            num_epochs=1000,
            learning_rate=0.01,
            optimizer="adam",
            guide="auto_normal",
            batch_size=None,
            early_stopping=True,
            early_stopping_patience=10,
            clip_norm=None,
            train_size=0.8,
            valid_size=0.2,
        )
    elif method == "mcmc":
        return create_inference_config(
            method="mcmc",
            num_samples=1000,
            num_warmup=500,
            num_chains=1,
            chain_method="parallel",
            kernel="nuts",
            target_accept_prob=0.8,
        )
    else:
        raise ValueError(f"Invalid inference method: {method}")
