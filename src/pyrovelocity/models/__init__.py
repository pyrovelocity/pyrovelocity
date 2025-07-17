"""PyroVelocity models package.

This package contains the modular components of the PyroVelocity model architecture,
including factory methods for model creation and configuration management,
direct AnnData integration, and Bayesian model comparison tools.

This package has been simplified to include only the essential components needed for
validation against the legacy implementation.
"""

# Import from modular components
# Import the legacy PyroVelocity model for backward compatibility
from pyrovelocity.models._velocity import PyroVelocity

# Import experimental implementations
from pyrovelocity.models.experimental import (
    deterministic_transcription_splicing_probabilistic_model,
    generate_posterior_inference_data,
    generate_prior_inference_data,
    generate_test_data_for_deterministic_model_inference,
    lognormal_tail_probability,
    plot_sample_phase_portraits,
    plot_sample_trajectories,
    plot_sample_trajectories_with_percentiles,
    save_inference_plots,
    solve_for_lognormal_mu_given_threshold_and_tail_mass,
    solve_for_lognormal_sigma_given_threshold_and_tail_mass,
    solve_transcription_splicing_model,
    solve_transcription_splicing_model_analytical,
)

# Modular PyTorch/Pyro implementation has been archived
# Only JAX and legacy models remain available

__all__ = [
    # Legacy model
    "PyroVelocity",
    # Experimental implementations
    "deterministic_transcription_splicing_probabilistic_model",
    "generate_test_data_for_deterministic_model_inference",
    "generate_prior_inference_data",
    "generate_posterior_inference_data",
    "plot_sample_phase_portraits",
    "plot_sample_trajectories",
    "plot_sample_trajectories_with_percentiles",
    "save_inference_plots",
    "solve_transcription_splicing_model",
    "solve_transcription_splicing_model_analytical",
    "lognormal_tail_probability",
    "solve_for_lognormal_sigma_given_threshold_and_tail_mass",
    "solve_for_lognormal_mu_given_threshold_and_tail_mass",
]
