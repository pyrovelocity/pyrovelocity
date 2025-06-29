"""
Inference utilities for PyroVelocity JAX/NumPyro implementation.

This module contains utilities for inference, including variational guides,
SVI, MCMC, and posterior analysis.
"""

from pyrovelocity.models.jax.inference.config import (
    create_inference_config,
    validate_config,
    get_default_config,
)

from pyrovelocity.models.jax.inference.guide import (
    auto_normal_guide,
    auto_delta_guide,
    custom_guide,
    create_guide,
)

from pyrovelocity.models.jax.inference.posterior import (
    sample_posterior,
    compute_velocity,
    compute_uncertainty,
    analyze_posterior,
    create_inference_data,
    format_anndata_output,
)


from pyrovelocity.models.jax.inference.svi import (
    create_optimizer,
    create_svi,
    svi_step,
    run_svi_inference,
    extract_posterior_samples as extract_svi_posterior_samples,
)

from pyrovelocity.models.jax.inference.mcmc import (
    create_mcmc,
    run_mcmc_inference,
    mcmc_diagnostics,
    extract_posterior_samples as extract_mcmc_posterior_samples,
    create_inference_state as create_mcmc_inference_state,
)

from pyrovelocity.models.jax.inference.unified import (
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
    # Guide
    "auto_normal_guide",
    "auto_delta_guide",
    "custom_guide",
    "create_guide",
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
    "create_mcmc_inference_state",
    # Unified
    "run_inference",
    "extract_posterior_samples",
    "posterior_predictive",
    "create_inference_state",
    # Posterior
    "sample_posterior",
    "compute_velocity",
    "compute_uncertainty",
    "analyze_posterior",
    "create_inference_data",
    "format_anndata_output",
]
