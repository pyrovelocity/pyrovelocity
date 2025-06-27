"""
PyroVelocity JAX/NumPyro Implementation.

This module contains the JAX/NumPyro implementation of PyroVelocity, a probabilistic
model for RNA velocity analysis.
"""

# Register piecewise activation components
from pyrovelocity.models.jax.components import register_piecewise_activation_components
from pyrovelocity.models.jax.core import (
    InferenceConfig,
    InferenceState,
    TrainingState,
    # State
    VelocityModelState,
    check_array_dtype,
    check_array_shape,
    # Utils
    create_key,
    create_likelihood,
    disable_x64,
    dynamics_ode_model,
    enable_x64,
    ensure_array,
    get_device_count,
    get_devices,
    informative_prior,
    # Priors
    lognormal_prior,
    negative_binomial_likelihood,
    nonlinear_dynamics_model,
    # Likelihoods
    poisson_likelihood,
    sample_prior_parameters,
    set_platform_device,
    split_key,
    # Dynamics
    standard_dynamics_model,
    # Model
    velocity_model,
)
from pyrovelocity.models.jax.core import (
    ModelConfig as CoreModelConfig,
)
from pyrovelocity.models.jax.core import (
    create_model as core_create_model,
)
from pyrovelocity.models.jax.data import (
    _internal_transform,
    batch_data,
    compute_size_factors,
    create_batch_iterator,
    extract_layers,
    filter_genes,
    get_library_size,
    # Preprocessing
    normalize_counts,
    # AnnData integration
    prepare_anndata,
    # Batch processing
    random_batch_indices,
    store_results,
    vmap_batch_function,
)
from pyrovelocity.models.jax.factory import (
    # Configuration classes
    DynamicsFunctionConfig,
    GuideFunctionConfig,
    LikelihoodFunctionConfig,
    ModelConfig,
    ObservationFunctionConfig,
    PriorFunctionConfig,
    # Factory functions
    create_dynamics_function,
    create_guide_factory_function,
    create_likelihood_function,
    create_model,
    create_observation_function,
    create_prior_function,
    create_piecewise_activation_model,
    # Predefined configurations
    piecewise_activation_model_config,
)
from pyrovelocity.models.jax.inference import (
    analyze_posterior,
    auto_delta_guide,
    # Guide
    auto_normal_guide,
    compute_uncertainty,
    compute_velocity,
    create_guide,
    # Config
    create_inference_config,
    create_inference_data,
    create_inference_state,
    # MCMC
    create_mcmc,
    # SVI
    create_optimizer,
    create_svi,
    custom_guide,
    extract_posterior_samples,
    format_anndata_output,
    get_default_config,
    mcmc_diagnostics,
    posterior_predictive,
    # Unified
    run_inference,
    run_mcmc_inference,
    run_svi_inference,
    # Posterior
    sample_posterior,
    svi_step,
    validate_config,
)
from pyrovelocity.models.jax.train import (
    clip_gradients,
    compute_elbo,
    # Metrics
    compute_loss,
    compute_metrics,
    compute_predictive_log_likelihood,
    compute_validation_metrics,
    # Optimizer
    create_optimizer,
    create_optimizer_with_schedule,
    evaluate_model,
    learning_rate_schedule,
    train_epoch,
    # Loop
    train_model,
    train_with_early_stopping,
)

__all__ = [
    # Utils
    "create_key",
    "split_key",
    "set_platform_device",
    "enable_x64",
    "disable_x64",
    "get_device_count",
    "get_devices",
    "check_array_shape",
    "check_array_dtype",
    "ensure_array",
    # State
    "VelocityModelState",
    "TrainingState",
    "InferenceState",
    "ModelConfig",
    "CoreModelConfig",
    "InferenceConfig",
    # Factory
    "DynamicsFunctionConfig",
    "PriorFunctionConfig",
    "LikelihoodFunctionConfig",
    "ObservationFunctionConfig",
    "GuideFunctionConfig",
    "create_dynamics_function",
    "create_prior_function",
    "create_likelihood_function",
    "create_observation_function",
    "create_guide_factory_function",
    "create_model",
    "piecewise_activation_model_config",
    "create_piecewise_activation_model",
    "register_piecewise_activation_components",
    # Dynamics
    "standard_dynamics_model",
    "nonlinear_dynamics_model",
    "dynamics_ode_model",
    # Priors
    "lognormal_prior",
    "informative_prior",
    "sample_prior_parameters",
    # Likelihoods
    "poisson_likelihood",
    "negative_binomial_likelihood",
    "create_likelihood",
    # Model
    "velocity_model",
    "core_create_model",
    # AnnData integration
    "prepare_anndata",
    "extract_layers",
    "store_results",
    "get_library_size",
    # Batch processing
    "random_batch_indices",
    "create_batch_iterator",
    "batch_data",
    "vmap_batch_function",
    # Preprocessing
    "normalize_counts",
    "compute_size_factors",
    "filter_genes",
    "_internal_transform",
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
    # MCMC
    "create_mcmc",
    "run_mcmc_inference",
    "mcmc_diagnostics",
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
    # Loop
    "train_model",
    "evaluate_model",
    "train_with_early_stopping",
    "train_epoch",
    # Optimizer
    "learning_rate_schedule",
    "clip_gradients",
    "create_optimizer_with_schedule",
    # Metrics
    "compute_loss",
    "compute_elbo",
    "compute_predictive_log_likelihood",
    "compute_metrics",
    "compute_validation_metrics",
]
