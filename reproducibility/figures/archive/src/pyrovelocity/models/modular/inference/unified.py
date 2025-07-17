"""
Unified inference interface for PyroVelocity PyTorch/Pyro modular implementation.

This module contains the unified inference interface, including:

- run_inference: Run inference using either SVI or MCMC
- extract_posterior_samples: Extract posterior samples from inference results
- posterior_predictive: Generate posterior predictive samples
- create_inference_state: Create an inference state
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pyro
import torch
from beartype import beartype
from pyro.infer import MCMC, SVI, Predictive
from pyro.infer.autoguide import AutoGuide

from pyrovelocity.models.modular.components.guides import InferenceGuide
from pyrovelocity.models.modular.inference.config import InferenceConfig
from pyrovelocity.models.modular.inference.mcmc import (
    extract_posterior_samples as extract_mcmc_posterior_samples,
)
from pyrovelocity.models.modular.inference.mcmc import (
    run_mcmc_inference,
)
from pyrovelocity.models.modular.inference.svi import (
    TrainingState,
    run_svi_inference,
)
from pyrovelocity.models.modular.inference.svi import (
    extract_posterior_samples as extract_svi_posterior_samples,
)


@dataclass
class InferenceState:
    """
    State of the inference process.

    This class contains the state of the inference process, including
    the method, parameters, posterior samples, and diagnostics.
    """

    method: str
    params: Dict[str, torch.Tensor] = field(default_factory=dict)
    posterior_samples: Dict[str, torch.Tensor] = field(default_factory=dict)
    training_state: Optional[TrainingState] = None
    mcmc: Optional[MCMC] = None
    svi: Optional[SVI] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@beartype
def create_inference_state(
    method: str,
    params: Optional[Dict[str, torch.Tensor]] = None,
    posterior_samples: Optional[Dict[str, torch.Tensor]] = None,
    training_state: Optional[TrainingState] = None,
    mcmc: Optional[MCMC] = None,
    svi: Optional[SVI] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> InferenceState:
    """
    Create an inference state.

    Args:
        method: Inference method ("svi" or "mcmc")
        params: Parameters from inference
        posterior_samples: Posterior samples from inference
        training_state: Training state from SVI
        mcmc: MCMC object from MCMC
        svi: SVI object from SVI
        diagnostics: Diagnostics from inference

    Returns:
        Inference state
    """
    return InferenceState(
        method=method,
        params=params or {},
        posterior_samples=posterior_samples or {},
        training_state=training_state,
        mcmc=mcmc,
        svi=svi,
        diagnostics=diagnostics or {},
    )


@beartype
def run_inference(
    model: Callable,
    guide: Optional[Union[AutoGuide, Callable, InferenceGuide]] = None,
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[InferenceConfig] = None,
    seed: Optional[int] = None,
) -> InferenceState:
    """
    Run inference with a model and guide.

    This function performs inference with the specified model and guide,
    using either SVI or MCMC based on the configuration.

    Args:
        model: Pyro model function
        guide: Pyro guide function, AutoGuide object, or guide name (required for SVI)
        args: Positional arguments to pass to the model and guide
        kwargs: Keyword arguments to pass to the model and guide
        config: Inference configuration
        seed: Random seed

    Returns:
        Inference state
    """
    # Set default kwargs if None
    if kwargs is None:
        kwargs = {}

    # Set default config if None
    if config is None:
        config = InferenceConfig()

    # Let the global RNG state continue from top-level script setting
    # Don't reset seed here as it would reset RNG state and cause identical samples

    # Run inference based on method
    if config.method == "svi":
        # Check that guide is provided
        if guide is None:
            raise ValueError("Guide is required for SVI inference")

        # Run SVI inference
        training_state, posterior_samples = run_svi_inference(
            model=model,
            guide=guide,
            args=args,
            kwargs=kwargs,
            config=config,
            seed=seed,
        )

        # Create inference state
        return create_inference_state(
            method="svi",
            params=training_state.params,
            posterior_samples=posterior_samples,
            training_state=training_state,
        )
    elif config.method == "mcmc":
        # Run MCMC inference
        mcmc, posterior_samples = run_mcmc_inference(
            model=model,
            args=args,
            kwargs=kwargs,
            config=config,
            seed=seed,
        )

        # Create inference state
        return create_inference_state(
            method="mcmc",
            posterior_samples=posterior_samples,
            mcmc=mcmc,
        )
    else:
        raise ValueError(f"Unsupported inference method: {config.method}")


@beartype
def extract_posterior_samples(
    state: InferenceState,
    num_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract posterior samples from inference results.

    Args:
        state: Inference state
        num_samples: Number of posterior samples to extract (if None, use all samples)

    Returns:
        Dictionary of posterior samples
    """
    # First get the full posterior samples
    if state.method == "svi":
        if state.svi is not None and state.training_state is not None:
            samples = extract_svi_posterior_samples(
                state.svi.guide,
                state.training_state.params,
                num_samples or 1000,
            )
        else:
            samples = state.posterior_samples
    elif state.method == "mcmc":
        if state.mcmc is not None:
            samples = extract_mcmc_posterior_samples(state.mcmc, num_samples)
        else:
            samples = state.posterior_samples
    else:
        raise ValueError(f"Unsupported inference method: {state.method}")

    # If num_samples is specified and different from the current number of samples,
    # subsample the posterior samples
    if (
        num_samples is not None
        and len(next(iter(samples.values()))) != num_samples
    ):
        subsampled = {}
        for k, v in samples.items():
            indices = torch.randperm(len(v))[:num_samples]
            subsampled[k] = v[indices]
        return subsampled

    return samples


@beartype
def posterior_predictive(
    model: Callable,
    posterior_samples: Dict[str, torch.Tensor],
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate posterior predictive samples.

    Args:
        model: Pyro model function
        posterior_samples: Posterior samples from inference
        args: Positional arguments to pass to the model
        kwargs: Keyword arguments to pass to the model
        num_samples: Number of posterior predictive samples to generate
        seed: Random seed

    Returns:
        Dictionary of posterior predictive samples
    """
    # Set default kwargs if None
    if kwargs is None:
        kwargs = {}

    # Let the global RNG state continue from top-level script setting
    # Don't reset seed here as it would reset RNG state and cause identical samples

    # Determine number of samples
    if num_samples is None:
        num_samples = len(next(iter(posterior_samples.values())))
    else:
        # If num_samples is specified, we need to subsample the posterior samples
        subsampled_posterior = {}
        for k, v in posterior_samples.items():
            indices = torch.randperm(len(v))[:num_samples]
            subsampled_posterior[k] = v[indices]
        posterior_samples = subsampled_posterior

    # Create predictive object
    predictive = Predictive(
        model,
        posterior_samples=posterior_samples,
        num_samples=num_samples,
        return_sites=None,  # Return all sites
    )

    # Generate posterior predictive samples
    return predictive(*args, **kwargs)
