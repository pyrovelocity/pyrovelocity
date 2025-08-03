"""
SVI utilities for PyroVelocity JAX/NumPyro implementation.

This module contains SVI utilities, including:

- create_svi: Create an SVI object
- svi_step: Single SVI step
- run_svi_inference: Run SVI inference
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
from beartype import beartype
from jaxtyping import Array, Float, PyTree
from numpyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from numpyro.infer.autoguide import AutoGuide

from pyrovelocity.models.jax.state import InferenceConfig, TrainingState
from pyrovelocity.models.jax.factory.factory import create_model
from pyrovelocity.models.jax.inference.guide import create_guide


@beartype
def create_optimizer(
    optimizer_name: str = "adam", learning_rate: float = 0.01, **kwargs
) -> optax.GradientTransformation:
    """Create an optimizer for SVI.

    This function creates an Optax optimizer based on the specified name and parameters.
    Supported optimizers include Adam, SGD, RMSProp, and Adagrad.

    Args:
        optimizer_name: Name of the optimizer ("adam", "sgd", "rmsprop", or "adagrad")
        learning_rate: Learning rate for the optimizer
        **kwargs: Additional optimizer-specific parameters

    Returns:
        An Optax optimizer (GradientTransformation)

    Raises:
        ValueError: If an unknown optimizer name is provided
    """
    # Create optimizer based on name
    if optimizer_name.lower() == "adam":
        return optax.adam(learning_rate=learning_rate, **kwargs)
    elif optimizer_name.lower() == "sgd":
        return optax.sgd(learning_rate=learning_rate, **kwargs)
    elif optimizer_name.lower() == "rmsprop":
        return optax.rmsprop(learning_rate=learning_rate, **kwargs)
    elif optimizer_name.lower() == "adagrad":
        return optax.adagrad(learning_rate=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


@beartype
def create_svi(
    model: Callable,
    guide: Union[AutoGuide, Callable],
    optimizer: Union[str, optax.GradientTransformation],
    loss: Optional[numpyro.infer.ELBO] = None,
    learning_rate: float = 0.01,
    **kwargs,
) -> SVI:
    """Create an SVI object for variational inference.

    This function creates a Stochastic Variational Inference (SVI) object with the
    specified model, guide, optimizer, and loss function.

    Args:
        model: NumPyro model function that defines the probabilistic model
        guide: NumPyro guide function or AutoGuide object that defines the variational distribution
        optimizer: Optax optimizer or optimizer name ("adam", "sgd", etc.)
        loss: ELBO loss function (defaults to Trace_ELBO if None)
        learning_rate: Learning rate for the optimizer (used if optimizer is a string)
        **kwargs: Additional optimizer parameters

    Returns:
        An SVI object for performing variational inference
    """
    # Use default loss if not provided
    if loss is None:
        loss = Trace_ELBO()

    # Handle string optimizer
    if isinstance(optimizer, str):
        optimizer = create_optimizer(optimizer, learning_rate, **kwargs)

    # Create SVI object
    return SVI(model, guide, optimizer, loss)


@beartype
def svi_step(svi: SVI, state: TrainingState, *args, **kwargs) -> TrainingState:
    """Single SVI step.

    Args:
        svi: SVI object
        state: Training state
        *args: Additional positional arguments for the model
        **kwargs: Additional keyword arguments for the model

    Returns:
        Updated training state
    """
    # Extract state components
    params = state.params
    opt_state = state.opt_state
    key = state.key
    step = state.step
    loss_history = state.loss_history
    best_params = state.best_params
    best_loss = state.best_loss

    # Generate a new key for this step
    key, subkey = jax.random.split(key)

    # Perform SVI step
    # Only use keyword arguments to avoid passing 'x' twice
    try:
        result = svi.update(params, opt_state, **kwargs)

        # Unpack the result based on the NumPyro SVI.update return value
        # In NumPyro, SVI.update returns (loss, optim_state) or ((loss, mutable_state), optim_state)
        if isinstance(result, tuple) and len(result) == 2:
            if isinstance(result[0], tuple) and len(result[0]) == 2:
                # Case: ((loss, mutable_state), optim_state)
                (loss_val, _), new_opt_state = result
            else:
                # Case: (loss, optim_state)
                loss_val, new_opt_state = result

            # Handle scalar loss (0-d array)
            if isinstance(loss_val, (jnp.ndarray, float, int)):
                # Convert to Python float to avoid iteration issues with 0-d arrays
                loss_val = float(loss_val)

            # Get the new parameters from the optimizer
            # Handle the case where new_opt_state might be a scalar
            try:
                new_params = svi.optim.get_params(new_opt_state)
            except TypeError:
                # If we get a TypeError (iteration over 0-d array), convert to a tuple
                if (
                    isinstance(new_opt_state, jnp.ndarray)
                    and new_opt_state.ndim == 0
                ):
                    # In case of error, keep the old parameters
                    new_params = params
                else:
                    raise  # Re-raise if it's a different TypeError
        else:
            raise ValueError(
                f"Unexpected return value from SVI.update: {result}"
            )

        # Update loss history
        new_loss_history = loss_history + [float(loss_val)]
    except Exception as e:
        # If there's an error, log it and return the original state
        print(f"Error in SVI step: {e}")
        # Keep the original parameters and state
        new_params = params
        new_opt_state = opt_state
        new_loss_history = loss_history
        # Add a placeholder loss value
        new_loss_history = loss_history + [float("nan")]

    # Update best parameters if this is the best loss so far
    try:
        if (
            "loss_val" in locals()
            and best_loss is None
            or ("loss_val" in locals() and loss_val < best_loss)
        ):
            new_best_params = new_params
            new_best_loss = (
                float(loss_val) if "loss_val" in locals() else float("nan")
            )
        else:
            new_best_params = best_params
            new_best_loss = best_loss
    except Exception as e:
        print(f"Error updating best parameters: {e}")
        new_best_params = best_params
        new_best_loss = best_loss

    # Create updated state
    new_state = TrainingState(
        step=step + 1,
        params=new_params,
        opt_state=new_opt_state,
        loss_history=new_loss_history,
        best_params=new_best_params,
        best_loss=new_best_loss,
        key=key,
    )

    return new_state


@beartype
def run_svi_inference(
    model: Union[Callable, Dict[str, Any]],
    guide: Optional[Union[AutoGuide, Callable, str]] = None,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    config: InferenceConfig = None,
    key: Optional[jnp.ndarray] = None,
) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
    """Run SVI inference with a model and guide.

    This function performs Stochastic Variational Inference (SVI) with the specified model
    and guide. It supports both direct model/guide functions and model configurations that
    can be used with the factory system.

    Args:
        model: Either a NumPyro model function or a model configuration dictionary
        guide: NumPyro guide function, AutoGuide object, guide type string, or None (auto-created from model)
        args: Positional arguments for the model
        kwargs: Keyword arguments for the model
        config: Inference configuration
        key: JAX random key (created automatically if None)

    Returns:
        A tuple containing:
            - training_state: The final training state after inference
            - posterior_samples: Dictionary of posterior samples
    """
    # Initialize kwargs if None
    if kwargs is None:
        kwargs = {}

    # Initialize config if None
    if config is None:
        config = InferenceConfig()

    # Initialize key if None
    if key is None:
        key = jax.random.PRNGKey(0)

    # Handle model configuration dictionary
    if isinstance(model, dict):
        # Create model from configuration
        model_fn = create_model(model)
    else:
        # Use model function directly
        model_fn = model

    # Handle guide creation if needed
    if guide is None or isinstance(guide, str):
        # Create guide from model
        guide_type = guide if isinstance(guide, str) else config.guide_type
        key, subkey = jax.random.split(key)
        # Don't pass the key to create_guide since it doesn't accept it
        guide = create_guide(model_fn, guide_type=guide_type)

    # Create SVI object
    svi = create_svi(
        model=model_fn,
        guide=guide,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
    )

    # Run SVI using the built-in run method
    key, subkey = jax.random.split(key)
    svi_result = svi.run(subkey, config.num_epochs, *args, **kwargs)

    # Extract the final parameters and losses
    params = svi_result.params
    losses = svi_result.losses

    # Create a training state from the results
    state = TrainingState(
        step=config.num_epochs,
        params=params,
        opt_state=None,  # We don't need to store the optimizer state
        loss_history=list(losses),
        best_params=params,  # Use the final parameters as the best
        best_loss=losses[-1] if len(losses) > 0 else None,
        key=key,
    )

    # Extract posterior samples
    key, subkey = jax.random.split(key)
    posterior_samples = extract_posterior_samples(
        guide, params, config.num_samples, subkey
    )

    return state, posterior_samples


@beartype
def extract_posterior_samples(
    guide: Union[AutoGuide, Callable],
    params: PyTree,
    num_samples: int,
    key: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Extract posterior samples from a guide.

    This function extracts posterior samples from a guide using the provided parameters.
    It handles both AutoGuide objects and custom guide functions.

    Args:
        guide: NumPyro guide (AutoGuide object or callable function)
        params: Guide parameters (PyTree)
        num_samples: Number of posterior samples to generate
        key: JAX random key for sampling

    Returns:
        Dictionary mapping parameter names to arrays of samples

    Raises:
        TypeError: If the guide type is not supported
    """
    # Handle different guide types
    if isinstance(guide, AutoGuide):
        # Use AutoGuide's sample_posterior method
        return guide.sample_posterior(
            rng_key=key, params=params, sample_shape=(num_samples,)
        )
    elif callable(guide):
        # For custom guides, we need to create a predictive object
        predictive = numpyro.infer.Predictive(
            guide,
            params=params,
            num_samples=num_samples,
        )
        # Sample from the guide
        samples = predictive(key)
        # Remove the "_auto_latent" prefix if present and other guide-specific prefixes
        cleaned_samples = {}
        for k, v in samples.items():
            # Remove common prefixes used in guides
            cleaned_key = k
            for prefix in ["_auto_latent", "_auto", "guide$"]:
                cleaned_key = cleaned_key.replace(prefix, "")
            cleaned_samples[cleaned_key] = v
        return cleaned_samples
    else:
        raise TypeError(f"Unsupported guide type: {type(guide)}")
