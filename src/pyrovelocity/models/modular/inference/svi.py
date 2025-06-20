"""
SVI utilities for PyroVelocity PyTorch/Pyro modular implementation.

This module contains SVI utilities, including:

- create_optimizer: Create an optimizer
- create_svi: Create an SVI object
- svi_step: Single SVI step
- run_svi_inference: Run SVI inference
- extract_posterior_samples: Extract posterior samples from SVI results
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pyro
import pyro.infer
import pyro.optim
import torch
from beartype import beartype
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoGuide

from pyrovelocity.models.modular.components.guides import InferenceGuide
from pyrovelocity.models.modular.inference.config import InferenceConfig


@dataclass
class TrainingState:
    """
    State of the training process.

    This class contains the state of the training process, including
    parameters, optimizer state, loss history, and best parameters.
    """

    step: int = 0
    params: Dict[str, torch.Tensor] = field(default_factory=dict)
    opt_state: Any = None
    loss_history: List[float] = field(default_factory=list)
    best_params: Dict[str, torch.Tensor] = field(default_factory=dict)
    best_loss: Optional[float] = None
    key: Optional[torch.Tensor] = None


@beartype
def create_optimizer(
    optimizer_name: str,
    learning_rate: float = 0.01,
    clip_norm: Optional[float] = None,
    max_epochs: Optional[int] = 1000,
    **kwargs: Any,
) -> Any:
    """
    Create a Pyro optimizer.

    Args:
        optimizer_name: Name of the optimizer ("adam", "sgd", "rmsprop", "velocity_clipped_adam")
        learning_rate: Learning rate
        clip_norm: Gradient clipping norm
        max_epochs: Maximum number of epochs (used for learning rate decay in VelocityClippedAdam)
        **kwargs: Additional optimizer parameters

    Returns:
        Pyro optimizer
    """
    # Import VelocityClippedAdam from the legacy implementation
    from pyrovelocity.models._trainer import VelocityClippedAdam

    # Set up optimizer parameters
    optim_args = {"lr": learning_rate, **kwargs}

    # Create optimizer based on name
    if optimizer_name.lower() == "velocity_clipped_adam":
        # Use VelocityClippedAdam with learning rate decay like in the legacy implementation
        return VelocityClippedAdam({"lr": learning_rate, "lrd": 0.1 ** (1 / max_epochs)})
    elif optimizer_name.lower() == "adam":
        if clip_norm is not None:
            return pyro.optim.ClippedAdam(optim_args)
        else:
            return pyro.optim.Adam(optim_args)
    elif optimizer_name.lower() == "sgd":
        return pyro.optim.SGD(optim_args)
    elif optimizer_name.lower() == "rmsprop":
        return pyro.optim.RMSprop(optim_args)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


@beartype
def create_svi(
    model: Callable,
    guide: Union[AutoGuide, Callable, InferenceGuide],
    optimizer: Union[str, Any],
    loss: Optional[Any] = None,
    learning_rate: float = 0.01,
    clip_norm: Optional[float] = None,
    max_epochs: Optional[int] = 1000,
    **kwargs: Any,
) -> SVI:
    """
    Create an SVI object for variational inference.

    This function creates a Stochastic Variational Inference (SVI) object with the
    specified model, guide, optimizer, and loss function.

    Args:
        model: Pyro model function that defines the probabilistic model
        guide: Pyro guide function or AutoGuide object that defines the variational distribution
        optimizer: Pyro optimizer or optimizer name ("adam", "sgd", "velocity_clipped_adam", etc.)
        loss: ELBO loss function (defaults to Trace_ELBO if None)
        learning_rate: Learning rate for the optimizer (used if optimizer is a string)
        clip_norm: Gradient clipping norm (used if optimizer is a string)
        max_epochs: Maximum number of epochs (used for learning rate decay in VelocityClippedAdam)
        **kwargs: Additional optimizer parameters

    Returns:
        An SVI object for performing variational inference
    """
    # Use default loss if not provided
    if loss is None:
        # Use Trace_ELBO with strict_enumeration_warning=True like in the legacy implementation
        loss = Trace_ELBO(strict_enumeration_warning=True)

    # Handle string optimizer
    if isinstance(optimizer, str):
        # Use VelocityClippedAdam by default to match the legacy implementation
        if optimizer.lower() == "adam":
            optimizer = "velocity_clipped_adam"

        optimizer = create_optimizer(
            optimizer, learning_rate, clip_norm=clip_norm, max_epochs=max_epochs, **kwargs
        )

    # Create SVI object
    return SVI(model, guide, optimizer, loss)


@beartype
def svi_step(svi: SVI, *args: Any, **kwargs: Any) -> Any:
    """
    Perform a single SVI step.

    Args:
        svi: SVI object
        *args: Positional arguments to pass to the model and guide
        **kwargs: Keyword arguments to pass to the model and guide

    Returns:
        Loss value for this step
    """
    return svi.step(*args, **kwargs)


@beartype
def extract_posterior_samples(
    guide: Union[AutoGuide, Callable, InferenceGuide],
    params: Optional[
        Dict[str, torch.Tensor]
    ] = None,  # Unused but kept for API compatibility
    num_samples: int = 1000,
    seed: Optional[int] = None,
    model_args: Optional[Tuple] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract posterior samples from SVI results.

    This function properly handles parameter constraints to ensure that scale parameters
    (like t_scale) remain positive. It uses the guide's sample_posterior method when
    available to get properly constrained samples.

    Args:
        guide: Guide function or AutoGuide object
        params: Parameters from SVI
        num_samples: Number of posterior samples to extract
        seed: Random seed
        model_args: Arguments to pass to the model for deterministic site computation
        model_kwargs: Keyword arguments to pass to the model for deterministic site computation

    Returns:
        Dictionary of posterior samples with properly constrained parameters
    """
    # Set seed if provided
    if seed is not None:
        pyro.set_rng_seed(seed)

    # For all guide types, create a predictive object
    # First, get samples from the guide (latent variables)
    guide_predictive = pyro.infer.Predictive(guide, num_samples=num_samples)
    guide_samples = guide_predictive()

    # Post-process samples to ensure constraints are satisfied
    # This is a safety measure in case the guide doesn't properly handle constraints
    guide_samples = _ensure_parameter_constraints(guide_samples)

    # Now, we need to get deterministic sites from the model
    # We'll use the guide samples as input to the model
    # This is critical for getting ut and st which are deterministic sites

    # Get the model function from the guide if it's an AutoGuide
    if hasattr(guide, "model"):
        model_fn = guide.model
    else:
        # If guide is not an AutoGuide, we can't get the model
        # In this case, we'll just return the guide samples
        return guide_samples

    # Create a predictive object for the model, using the guide samples
    # Use None to return all sites, including deterministic ones
    # Note: When using posterior_samples, do NOT specify num_samples as it's already determined
    model_predictive = pyro.infer.Predictive(
        model_fn,
        posterior_samples=guide_samples,
        return_sites=None,  # Return all sites, including deterministic
    )

    # Run the model predictive to get all sites, including deterministic ones
    # We need to run the model in unconditioned mode to generate deterministic sites
    # from the posterior samples, not conditioning on observations
    if model_args is not None or model_kwargs is not None:
        args = model_args or ()
        kwargs = model_kwargs or {}

        # Create unconditioned version of the model arguments
        # Remove observation conditioning by creating dummy observations with the right shape
        unconditioned_kwargs = {}
        for key, value in kwargs.items():
            if key in ['u_obs', 's_obs'] and isinstance(value, torch.Tensor):
                # Create dummy observations with the same shape but don't condition on them
                unconditioned_kwargs[key] = torch.zeros_like(value)
            else:
                unconditioned_kwargs[key] = value

        # Use pyro.poutine.uncondition to remove observation conditioning
        unconditioned_model = pyro.poutine.uncondition(model_fn)
        unconditioned_predictive = pyro.infer.Predictive(
            unconditioned_model,
            posterior_samples=guide_samples,
            return_sites=None,  # Return all sites, including deterministic
        )
        model_samples = unconditioned_predictive(*args, **unconditioned_kwargs)
    else:
        # Fallback to no arguments (this will fail for models that require arguments)
        model_samples = model_predictive()

    # Combine guide and model samples
    # Guide samples take precedence if there's a conflict
    combined_samples = {**model_samples, **guide_samples}

    return combined_samples


@beartype
def _ensure_parameter_constraints(samples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Ensure that parameter samples satisfy their constraints.

    This function applies constraints to parameter samples to ensure they remain
    in valid ranges. This is particularly important for scale parameters that
    must be positive.

    Args:
        samples: Dictionary of parameter samples

    Returns:
        Dictionary of constrained parameter samples
    """
    constrained_samples = {}

    for name, value in samples.items():
        if isinstance(value, torch.Tensor):
            # Apply constraints based on parameter name patterns
            if any(scale_param in name.lower() for scale_param in ['scale', 't_scale', 'sigma']):
                # Scale parameters must be positive
                # Use softplus to ensure positivity: softplus(x) = log(1 + exp(x))
                # But first check if values are already positive
                if torch.any(value <= 0):
                    # Apply softplus transformation to ensure positivity
                    constrained_samples[name] = torch.nn.functional.softplus(value) + 1e-6
                else:
                    constrained_samples[name] = value
            elif any(rate_param in name.lower() for rate_param in ['rate', 'beta', 'gamma']):
                # Rate parameters must be positive
                if torch.any(value <= 0):
                    constrained_samples[name] = torch.nn.functional.softplus(value) + 1e-6
                else:
                    constrained_samples[name] = value
            elif any(conc_param in name.lower() for conc_param in ['alpha', 'concentration']):
                # Concentration parameters must be positive
                if torch.any(value <= 0):
                    constrained_samples[name] = torch.nn.functional.softplus(value) + 1e-6
                else:
                    constrained_samples[name] = value
            else:
                # No constraint needed for other parameters
                constrained_samples[name] = value
        else:
            # Non-tensor values: keep as-is
            constrained_samples[name] = value

    return constrained_samples


@beartype
def run_svi_inference(
    model: Callable,
    guide: Union[AutoGuide, Callable, InferenceGuide],
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[InferenceConfig] = None,
    seed: Optional[int] = None,
) -> Tuple[TrainingState, Dict[str, torch.Tensor]]:
    """
    Run SVI inference with a model and guide.

    This function performs Stochastic Variational Inference (SVI) with the specified model
    and guide.

    Args:
        model: Pyro model function
        guide: Pyro guide function, AutoGuide object, or guide name
        args: Positional arguments to pass to the model and guide
        kwargs: Keyword arguments to pass to the model and guide
        config: Inference configuration
        seed: Random seed

    Returns:
        Tuple of (training state, posterior samples)
    """
    # Set default kwargs if None
    if kwargs is None:
        kwargs = {}

    # Set default config if None
    if config is None:
        config = InferenceConfig()

    # Set seed if provided
    if seed is not None:
        pyro.set_rng_seed(seed)
    elif config.seed is not None:
        pyro.set_rng_seed(config.seed)

    # Create SVI object
    svi = create_svi(
        model=model,
        guide=guide,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        clip_norm=config.clip_norm,
        max_epochs=config.num_epochs,
    )

    # Initialize training state
    state = TrainingState(step=0)

    # Run SVI for the specified number of epochs
    for _ in range(config.num_epochs):


        # Perform a single SVI step
        loss = svi.step(*args, **kwargs)
        loss_value = float(loss)  # Convert to float

        # Update training state
        state.step += 1
        state.loss_history.append(loss_value)

        # Update best parameters if this is the best loss so far
        if state.best_loss is None or loss_value < state.best_loss:
            state.best_loss = loss_value
            # In Pyro, we need to get the parameters from the param store
            state.best_params = {
                name: param.detach().clone()
                for name, param in pyro.get_param_store().items()
            }

        # Early stopping
        if (
            config.early_stopping
            and len(state.loss_history) > config.early_stopping_patience
            and all(
                state.loss_history[-i - 1] >= state.loss_history[-i - 2]
                for i in range(config.early_stopping_patience)
            )
        ):
            break

    # Use the best parameters
    # In Pyro, we need to set the parameters in the param store
    for name, param in state.best_params.items():
        pyro.param(name, param)
    state.params = state.best_params

    # Extract posterior samples
    # We need to pass both the guide and the model to extract_posterior_samples
    # This is critical for getting deterministic sites like ut and st
    posterior_samples = extract_posterior_samples(
        guide, state.params, config.num_samples, seed, args, kwargs
    )

    return state, posterior_samples
