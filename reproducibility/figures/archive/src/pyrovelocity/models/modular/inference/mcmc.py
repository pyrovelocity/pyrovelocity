"""
MCMC utilities for PyroVelocity PyTorch/Pyro modular implementation.

This module contains MCMC utilities, including:

- create_mcmc: Create an MCMC object
- run_mcmc_inference: Run MCMC inference
- mcmc_diagnostics: Compute MCMC diagnostics
- extract_posterior_samples: Extract posterior samples from MCMC results
"""

from typing import Any, Callable, Dict, Optional, Tuple

import pyro
import torch
from beartype import beartype
from pyro.infer import HMC, MCMC, NUTS

from pyrovelocity.models.modular.inference.config import InferenceConfig


@beartype
def create_mcmc(
    model: Callable,
    kernel: Optional[Any] = None,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    chain_method: str = "parallel",
    progress_bar: bool = True,
    **kwargs: Any,
) -> MCMC:
    """
    Create an MCMC object.

    Args:
        model: Pyro model function
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

    # Create MCMC object with appropriate parameters
    # According to Pyro documentation, the parameters are:
    # kernel, num_samples, warmup_steps, initial_params, num_chains, hook_fn, mp_context, disable_progbar, disable_validation, transforms
    return MCMC(
        kernel,
        num_samples=num_samples,
        warmup_steps=num_warmup,
        num_chains=num_chains,
        disable_progbar=not progress_bar,
    )


@beartype
def run_mcmc_inference(
    model: Callable,
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    config: Optional[InferenceConfig] = None,
    seed: Optional[int] = None,
) -> Tuple[MCMC, Dict[str, torch.Tensor]]:
    """
    Run MCMC inference with a model.

    This function performs Markov Chain Monte Carlo (MCMC) inference with the specified model.

    Args:
        model: Pyro model function
        args: Positional arguments to pass to the model
        kwargs: Keyword arguments to pass to the model
        config: Inference configuration
        seed: Random seed

    Returns:
        Tuple of (MCMC object, posterior samples)
    """
    # Set default kwargs if None
    if kwargs is None:
        kwargs = {}

    # Set default config if None
    if config is None:
        config = InferenceConfig(method="mcmc")

    # Let the global RNG state continue from top-level script setting
    # Don't reset seed here as it would reset RNG state and cause identical samples

    # Create kernel based on config
    if config.kernel.lower() == "nuts":
        kernel = NUTS(
            model,
            target_accept_prob=config.target_accept_prob,
        )
    elif config.kernel.lower() == "hmc":
        kernel = HMC(model)
    else:
        raise ValueError(f"Unsupported MCMC kernel: {config.kernel}")

    # Create MCMC object
    mcmc = create_mcmc(
        model=model,
        kernel=kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        chain_method=config.chain_method,
    )

    # Run MCMC
    mcmc.run(*args, **kwargs)

    # Extract posterior samples
    posterior_samples = mcmc.get_samples()

    return mcmc, posterior_samples


@beartype
def mcmc_diagnostics(mcmc: MCMC) -> Dict[str, Any]:
    """
    Compute MCMC diagnostics.

    Args:
        mcmc: MCMC object

    Returns:
        Dictionary of diagnostics
    """
    # Get diagnostics from MCMC object
    diagnostics = {}

    # Add summary statistics
    summary = mcmc.summary()
    diagnostics["summary"] = summary

    # Add additional diagnostics if available
    try:
        # Get effective sample size
        ess = mcmc.diagnostics()["effective_sample_size"]
        diagnostics["effective_sample_size"] = ess

        # Get r_hat
        r_hat = mcmc.diagnostics()["r_hat"]
        diagnostics["r_hat"] = r_hat
    except Exception:
        pass

    return diagnostics


@beartype
def extract_posterior_samples(
    mcmc: MCMC,
    num_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract posterior samples from MCMC results.

    Args:
        mcmc: MCMC object
        num_samples: Number of posterior samples to extract (if None, use all samples)

    Returns:
        Dictionary of posterior samples
    """
    # Get all samples
    samples = mcmc.get_samples()

    # Subsample if num_samples is specified
    if num_samples is not None and num_samples < len(
        next(iter(samples.values()))
    ):
        # Get indices for subsampling
        indices = torch.randperm(len(next(iter(samples.values()))))[
            :num_samples
        ]

        # Subsample
        samples = {k: v[indices] for k, v in samples.items()}

    return samples
