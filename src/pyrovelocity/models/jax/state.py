"""State containers for PyroVelocity JAX/NumPyro implementation."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import jax.numpy as jnp
from jaxtyping import PyTree
from beartype import beartype


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for inference.
    
    Attributes:
        method: Inference method ("svi" or "mcmc")
        num_samples: Number of posterior samples
        num_epochs: Number of training epochs (SVI only)
        learning_rate: Learning rate (SVI only)
        num_warmup: Number of warmup steps (MCMC only)
        num_chains: Number of MCMC chains (MCMC only)
        guide_type: Type of guide for SVI
        early_stopping: Whether to use early stopping (SVI only)
        early_stopping_patience: Patience for early stopping
        optimizer: Optimizer type for SVI
        batch_size: Batch size for training
        clip_norm: Gradient clipping norm
    """
    method: str = "svi"
    num_samples: int = 1000
    num_epochs: int = 1000
    learning_rate: float = 0.01
    num_warmup: int = 500
    num_chains: int = 1
    guide_type: str = "auto_normal"
    early_stopping: bool = False
    early_stopping_patience: int = 10
    optimizer: str = "adam"
    batch_size: Optional[int] = None
    clip_norm: Optional[float] = None

    def replace(self, **kwargs) -> "InferenceConfig":
        """Create a new InferenceConfig with updated values."""
        return dataclass(type(self))(**{**self.__dict__, **kwargs})


@dataclass(frozen=True)
class InferenceState:
    """State container for inference results.
    
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
        """Create a new InferenceState with updated values."""
        return dataclass(type(self))(**{**self.__dict__, **kwargs})


@dataclass(frozen=True)
class TrainingState:
    """State container for SVI training.
    
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
        """Create a new TrainingState with updated values."""
        if "loss_history" in kwargs and self.loss_history is not None:
            kwargs["loss_history"] = list(kwargs["loss_history"])
        return dataclass(type(self))(**{**self.__dict__, **kwargs})