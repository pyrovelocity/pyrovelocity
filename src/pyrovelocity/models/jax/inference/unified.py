"""
Unified inference interface for PyroVelocity JAX/NumPyro implementation.

This module contains the unified inference interface, including:

- run_inference: Run inference using either SVI or MCMC
- create_guide: Create appropriate guide based on configuration
- extract_posterior_samples: Extract posterior samples from inference results
"""

from typing import Dict, Tuple, Optional, Any, List, Union, Callable
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, MCMC
from numpyro.infer.autoguide import AutoGuide
from jaxtyping import Array, Float, PyTree
from beartype import beartype

from pyrovelocity.models.jax.core.state import InferenceConfig, InferenceState
from pyrovelocity.models.jax.inference.config import create_inference_config
from pyrovelocity.models.jax.inference.guide import create_guide
from pyrovelocity.models.jax.inference.svi import run_svi_inference
from pyrovelocity.models.jax.inference.mcmc import (
    run_mcmc_inference,
    mcmc_diagnostics,
)


@beartype
def run_inference(
    model: Callable,
    args: Tuple,
    kwargs: Dict[str, Any],
    config: Optional[Union[InferenceConfig, Dict[str, Any]]] = None,
    key: Optional[jnp.ndarray] = None,
) -> Tuple[Union[SVI, MCMC, AutoGuide], InferenceState]:
    """Run inference using either SVI or MCMC.

    Args:
        model: NumPyro model function
        args: Positional arguments for the model
        kwargs: Keyword arguments for the model
        config: Inference configuration
        key: JAX random key

    Returns:
        Tuple of (inference_object, inference_state), where inference_object
        can be an SVI, MCMC, or AutoGuide object depending on the inference method.
    """
    # Generate a random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)

    # Create inference config if not provided or convert dict to InferenceConfig
    if config is None:
        config = create_inference_config()
    elif isinstance(config, dict):
        config = create_inference_config(**config)

    # Run inference based on method
    if config.method.lower() == "svi":
        # Create guide
        guide = create_guide(model, guide_type=config.guide_type)

        # Run SVI inference
        training_state, posterior_samples = run_svi_inference(
            model=model,
            guide=guide,
            args=args,
            kwargs=kwargs,
            config=config,
            key=key,
        )

        # Create inference state
        inference_state = create_inference_state(
            posterior_samples=posterior_samples
        )

        # Return SVI object and inference state
        return guide, inference_state

    elif config.method.lower() == "mcmc":
        # Run MCMC inference
        mcmc, posterior_samples = run_mcmc_inference(
            model=model,
            args=args,
            kwargs=kwargs,
            config=config,
            key=key,
        )

        # Get diagnostics
        diagnostics = mcmc_diagnostics(mcmc)

        # Create inference state
        inference_state = create_inference_state(
            posterior_samples=posterior_samples,
            diagnostics=diagnostics,
        )

        # Return MCMC object and inference state
        return mcmc, inference_state

    else:
        raise ValueError(f"Unknown inference method: {config.method}")


@beartype
def extract_posterior_samples(
    inference_object: Union[SVI, MCMC, Dict[str, jnp.ndarray], AutoGuide],
    params: Optional[PyTree] = None,
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """Extract posterior samples from inference results.

    Args:
        inference_object: SVI, MCMC, AutoGuide, or dictionary of samples
        params: Optional parameters (for SVI/AutoGuide)
        num_samples: Number of samples (for SVI/AutoGuide)
        key: JAX random key (for SVI/AutoGuide)

    Returns:
        Dictionary of posterior samples
    """
    # If inference_object is already a dictionary of samples, return it
    if isinstance(inference_object, dict):
        return inference_object

    # If inference_object is an MCMC object, extract samples
    elif isinstance(inference_object, MCMC):
        return inference_object.get_samples()

    # If inference_object is an AutoGuide, sample from the guide
    elif isinstance(inference_object, AutoGuide):
        if params is None:
            raise ValueError("Parameters must be provided for AutoGuide")

        if key is None:
            key = jax.random.PRNGKey(0)

        try:
            # Try to sample from the guide directly
            return inference_object.sample_posterior(
                rng_key=key, params=params, sample_shape=(num_samples,)
            )
        except KeyError as e:
            # If we get a KeyError, it might be due to parameter naming issues
            # Create a predictive object instead
            predictive = numpyro.infer.Predictive(
                inference_object,
                params=params,
                num_samples=num_samples,
            )
            # Sample from the guide
            return predictive(key)

    # If inference_object is an SVI object, raise an error (should use AutoGuide)
    elif isinstance(inference_object, SVI):
        raise ValueError(
            "SVI object cannot be used directly to extract samples. "
            "Use the guide object with parameters instead."
        )

    else:
        raise TypeError(
            f"Unsupported inference object type: {type(inference_object)}"
        )


@beartype
def posterior_predictive(
    model: Callable,
    posterior_samples: Dict[str, jnp.ndarray],
    args: Tuple,
    kwargs: Dict[str, Any],
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
    parallel: bool = True,
    return_sites: Optional[List[str]] = None,
) -> Dict[str, jnp.ndarray]:
    """Generate posterior predictive samples.

    Args:
        model: NumPyro model function
        posterior_samples: Dictionary of posterior samples
        args: Positional arguments for the model
        kwargs: Keyword arguments for the model
        num_samples: Number of samples
        key: JAX random key
        parallel: Whether to run in parallel
        return_sites: Optional list of sites to return

    Returns:
        Dictionary of posterior predictive samples
    """
    # Generate a random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)

    # If return_sites is None, include all sites including latent variables
    if return_sites is None:
        # Include both observed and latent sites
        return_sites = list(posterior_samples.keys())

        # Add observed sites if they're not already included
        if "x_obs" not in return_sites:
            return_sites.append("x_obs")
        if "y_obs" not in return_sites:
            return_sites.append("y_obs")

    # Create a predictive object
    predictive = numpyro.infer.Predictive(
        model,
        posterior_samples=posterior_samples,
        num_samples=num_samples,
        return_sites=return_sites,
        parallel=parallel,
    )

    # Generate posterior predictive samples
    result = predictive(key, *args, **kwargs)

    # Merge with posterior_samples to include all parameters
    for k, v in posterior_samples.items():
        if k not in result:
            # Take only the first num_samples if there are more
            if v.shape[0] > num_samples:
                result[k] = v[:num_samples]
            else:
                result[k] = v

    return result


@beartype
def create_inference_state(
    posterior_samples: Dict[str, jnp.ndarray],
    posterior_predictive_samples: Optional[Dict[str, jnp.ndarray]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> InferenceState:
    """Create an inference state from inference results.

    Args:
        posterior_samples: Dictionary of posterior samples
        posterior_predictive_samples: Optional dictionary of posterior predictive samples
        diagnostics: Optional dictionary of diagnostic results

    Returns:
        InferenceState object
    """
    return InferenceState(
        posterior_samples=posterior_samples,
        posterior_predictive=posterior_predictive_samples,
        diagnostics=diagnostics,
    )
