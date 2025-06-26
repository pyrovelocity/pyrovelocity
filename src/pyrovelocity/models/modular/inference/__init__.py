"""
Inference utilities for PyroVelocity PyTorch/Pyro modular implementation.

This module contains utilities for inference, including variational guides,
SVI, MCMC, and posterior analysis.
"""

from pyrovelocity.models.modular.inference.config import (
    create_inference_config,
    validate_config,
    get_default_config,
)

from pyrovelocity.models.modular.inference.svi import (
    create_optimizer,
    create_svi,
    svi_step,
    run_svi_inference,
    extract_posterior_samples as extract_svi_posterior_samples,
)

from pyrovelocity.models.modular.inference.mcmc import (
    create_mcmc,
    run_mcmc_inference,
    mcmc_diagnostics,
    extract_posterior_samples as extract_mcmc_posterior_samples,
)

from pyrovelocity.models.modular.inference.unified import (
    run_inference,
    extract_posterior_samples,
    posterior_predictive,
    create_inference_state,
)


__all__ = [
    # Config
    "create_inference_config",
    "validate_config",
    "get_default_config",
    # SVI
    "create_optimizer",
    "create_svi",
    "svi_step",
    "run_svi_inference",
    "extract_svi_posterior_samples",
    # MCMC
    "create_mcmc",
    "run_mcmc_inference",
    "mcmc_diagnostics",
    "extract_mcmc_posterior_samples",
    # Unified
    "run_inference",
    "extract_posterior_samples",
    "posterior_predictive",
    "create_inference_state",
]
