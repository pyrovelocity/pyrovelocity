"""
Immutable state containers for PyroVelocity JAX/NumPyro implementation.

This module contains immutable state containers for model state, training state,
and inference state.
"""

from typing import Dict, Tuple, Optional, Any, List, Union
from dataclasses import dataclass, field
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float, PyTree
from beartype import beartype


@dataclass(frozen=True)
class VelocityModelState:
    """Immutable container for model state.

    Attributes:
        parameters: Dictionary of model parameters
        dynamics_output: Optional tuple of (unspliced, spliced) RNA counts
        distributions: Optional tuple of (unspliced_dist, spliced_dist) distributions
        observations: Optional dictionary of observation results
    """

    parameters: Dict[str, jnp.ndarray]
    dynamics_output: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    distributions: Optional[Tuple[dist.Distribution, dist.Distribution]] = None
    observations: Optional[Dict[str, jnp.ndarray]] = None

    def replace(self, **kwargs) -> "VelocityModelState":
        """Create a new VelocityModelState with updated values.

        Args:
            **kwargs: Keyword arguments with new values

        Returns:
            New VelocityModelState with updated values
        """
        return dataclass(type(self))(**{**self.__dict__, **kwargs})


@dataclass(frozen=True)
class TrainingState:
    """Immutable container for training state.

    Attributes:
        step: Current training step
        params: Current parameter values
        opt_state: Optimizer state
        loss_history: History of loss values
        best_params: Best parameters found so far
        best_loss: Best loss value found so far
        key: JAX random key
    """

    step: int
    params: PyTree
    opt_state: PyTree
    loss_history: List[float] = field(default_factory=list)
    best_params: Optional[PyTree] = None
    best_loss: Optional[float] = None
    key: Optional[jnp.ndarray] = None

    def replace(self, **kwargs) -> "TrainingState":
        """Create a new TrainingState with updated values.

        Args:
            **kwargs: Keyword arguments with new values

        Returns:
            New TrainingState with updated values
        """
        # Handle loss_history specially to avoid reference issues
        if "loss_history" in kwargs and self.loss_history is not None:
            kwargs["loss_history"] = list(kwargs["loss_history"])

        return dataclass(type(self))(**{**self.__dict__, **kwargs})


@dataclass(frozen=True)
class InferenceState:
    """Immutable container for inference state.

    Attributes:
        posterior_samples: Dictionary of posterior samples
        posterior_predictive: Optional dictionary of posterior predictive samples
        diagnostics: Optional dictionary of inference diagnostics
        training_state: Optional training state containing loss history
    """

    posterior_samples: Dict[str, jnp.ndarray]
    posterior_predictive: Optional[Dict[str, jnp.ndarray]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    training_state: Optional["TrainingState"] = None

    def replace(self, **kwargs) -> "InferenceState":
        """Create a new InferenceState with updated values.

        Args:
            **kwargs: Keyword arguments with new values

        Returns:
            New InferenceState with updated values
        """
        return dataclass(type(self))(**{**self.__dict__, **kwargs})


@dataclass(frozen=True)
class ModelConfig:
    """Immutable container for model configuration.

    Attributes:
        dynamics: Dynamics model type
        likelihood: Likelihood model type
        prior: Prior distribution type
        inference: Inference method type
        use_observed_lib_size: Whether to use observed library size
        latent_time: Whether to use latent time
        latent_time_prior_mean: Prior mean for latent time
        latent_time_prior_scale: Prior scale for latent time
        include_prior: Whether to include prior in the model
    """

    dynamics: str = "piecewise_activation"
    likelihood: str = "piecewise_activation"
    prior: str = "piecewise_activation"
    inference: str = "svi"
    use_observed_lib_size: bool = True
    latent_time: bool = True
    latent_time_prior_mean: float = 0.0
    latent_time_prior_scale: float = 1.0
    include_prior: bool = True

    def replace(self, **kwargs) -> "ModelConfig":
        """Create a new ModelConfig with updated values.

        Args:
            **kwargs: Keyword arguments with new values

        Returns:
            New ModelConfig with updated values
        """
        return dataclass(type(self))(**{**self.__dict__, **kwargs})


@dataclass(frozen=True)
class InferenceConfig:
    """Immutable container for inference configuration.

    Attributes:
        method: Inference method ("svi" or "mcmc")
        num_samples: Number of posterior samples
        num_warmup: Number of warmup steps for MCMC
        num_chains: Number of MCMC chains
        guide_type: Guide type for SVI
        optimizer: Optimizer for SVI
        learning_rate: Learning rate for SVI
        num_epochs: Number of epochs for SVI
        batch_size: Batch size for SVI
        clip_norm: Gradient clipping norm for SVI
        early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
    """

    method: str = "svi"
    num_samples: int = 1000
    num_warmup: int = 500
    num_chains: int = 1
    guide_type: str = "auto_normal"
    optimizer: str = "adam"
    learning_rate: float = 0.01
    num_epochs: int = 1000
    batch_size: Optional[int] = None
    clip_norm: Optional[float] = None
    early_stopping: bool = True
    early_stopping_patience: int = 10

    def replace(self, **kwargs) -> "InferenceConfig":
        """Create a new InferenceConfig with updated values.

        Args:
            **kwargs: Keyword arguments with new values

        Returns:
            New InferenceConfig with updated values
        """
        return dataclass(type(self))(**{**self.__dict__, **kwargs})
