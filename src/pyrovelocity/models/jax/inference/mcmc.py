"""
MCMC utilities for PyroVelocity JAX/NumPyro implementation.

This module contains MCMC utilities, including:

- create_mcmc: Create an MCMC object
- run_mcmc_inference: Run MCMC inference
- mcmc_diagnostics: MCMC diagnostics
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, Float, PyTree
from numpyro.infer import HMC, MCMC, NUTS, DiscreteHMCGibbs

from pyrovelocity.models.jax.state import InferenceConfig, InferenceState
from pyrovelocity.models.jax.factory.factory import create_model


@beartype
def create_mcmc(
    model: Callable,
    kernel: Optional[numpyro.infer.mcmc.MCMCKernel] = None,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    chain_method: str = "parallel",
    progress_bar: bool = True,
    **kwargs,
) -> MCMC:
    """Create an MCMC object.

    Args:
        model: NumPyro model function
        kernel: MCMC kernel
        num_warmup: Number of warmup steps
        num_samples: Number of samples
        num_chains: Number of chains
        chain_method: Method for running chains ("parallel" or "sequential")
        progress_bar: Whether to show progress bar
        **kwargs: Additional MCMC parameters

    Returns:
        MCMC object
    """
    # Create default kernel if not provided
    if kernel is None:
        # Use NUTS as the default kernel
        kernel = NUTS(model, **kwargs)

    # Check if the model has discrete variables
    if "discrete" in kwargs and kwargs["discrete"]:
        # Use DiscreteHMCGibbs for models with discrete variables
        kernel = DiscreteHMCGibbs(kernel)

    # Create MCMC object
    return MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )


@beartype
def run_mcmc_inference(
    model: Union[Callable, Dict[str, Any]],
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[InferenceConfig] = None,
    key: Optional[jnp.ndarray] = None,
) -> Tuple[MCMC, Dict[str, jnp.ndarray]]:
    """Run MCMC inference with a model.

    This function performs Markov Chain Monte Carlo (MCMC) inference with the specified model.
    It supports both direct model functions and model configurations that can be used with
    the factory system.

    Args:
        model: Either a NumPyro model function or a model configuration dictionary
        args: Positional arguments for the model (default: empty tuple)
        kwargs: Keyword arguments for the model (default: empty dict)
        config: Inference configuration (default: default InferenceConfig)
        key: JAX random key (created automatically if None)

    Returns:
        A tuple containing:
            - mcmc: The MCMC object after inference
            - posterior_samples: Dictionary of posterior samples
    """
    # Initialize kwargs if None
    if kwargs is None:
        kwargs = {}

    # Initialize config if None
    if config is None:
        config = InferenceConfig(method="mcmc")

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

    # Create MCMC object
    mcmc = create_mcmc(
        model=model_fn,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
    )

    # Run MCMC
    mcmc.run(key, *args, **kwargs)

    # Extract posterior samples
    posterior_samples = mcmc.get_samples()

    return mcmc, posterior_samples


@beartype
def mcmc_diagnostics(
    mcmc: MCMC,
) -> Dict[str, Any]:
    """MCMC diagnostics.

    Args:
        mcmc: MCMC object

    Returns:
        Dictionary of diagnostic results
    """
    # Get diagnostic summaries
    diagnostics = {}

    # Get summary statistics
    # NumPyro MCMC doesn't have get_diagnostics(), use get_samples() instead
    samples = mcmc.get_samples()
    diagnostics["samples"] = samples

    # Add basic summary statistics
    for key, value in samples.items():
        diagnostics[f"{key}_mean"] = jnp.mean(value, axis=0)
        diagnostics[f"{key}_std"] = jnp.std(value, axis=0)

    # Get effective sample size
    if mcmc.num_chains > 1:
        try:
            ess = mcmc.get_ess()
            diagnostics["ess"] = ess
        except:
            # ESS calculation may fail for some models
            pass

    # Get R-hat statistics for convergence diagnostics
    if mcmc.num_chains > 1:
        try:
            r_hat = mcmc.get_hmc_diagnostics()["r_hat"]
            diagnostics["r_hat"] = r_hat
        except:
            # R-hat calculation may fail for some models
            pass

    # Get acceptance rate
    try:
        accept_rate = mcmc.get_extra_fields()["accept_prob"].mean()
        diagnostics["accept_rate"] = accept_rate
    except:
        # Accept rate may not be available for all kernels
        pass

    return diagnostics


@beartype
def extract_posterior_samples(
    mcmc: MCMC,
) -> Dict[str, jnp.ndarray]:
    """Extract posterior samples from MCMC.

    Args:
        mcmc: MCMC object

    Returns:
        Dictionary of posterior samples
    """
    # Get samples from MCMC
    samples = mcmc.get_samples()

    # Reshape samples if needed
    if mcmc.num_chains > 1:
        # Combine samples from multiple chains
        samples = {k: v.reshape(-1, *v.shape[2:]) for k, v in samples.items()}

    return samples


@beartype
def create_inference_state(
    posterior_samples: Dict[str, jnp.ndarray],
    diagnostics: Optional[Dict[str, Any]] = None,
) -> InferenceState:
    """Create an inference state from MCMC results.

    Args:
        posterior_samples: Dictionary of posterior samples
        diagnostics: Optional dictionary of diagnostic results

    Returns:
        InferenceState object
    """
    return InferenceState(
        posterior_samples=posterior_samples,
        diagnostics=diagnostics,
    )
