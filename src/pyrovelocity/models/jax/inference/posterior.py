"""
Posterior analysis utilities for PyroVelocity JAX/NumPyro implementation.

This module contains posterior analysis utilities, including:

- sample_posterior: Sample from the posterior
- posterior_predictive: Generate posterior predictive samples
- compute_velocity: Compute RNA velocity from posterior samples
- compute_uncertainty: Compute uncertainty in RNA velocity
- analyze_posterior: Analyze posterior samples from either SVI or MCMC
- create_inference_data: Create ArviZ InferenceData object from posterior samples
- format_anndata_output: Format results into AnnData object
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anndata
import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import torch
from beartype import beartype
from jaxtyping import Array, Float, PyTree

from pyrovelocity.models.jax.registry.dynamics import get_dynamics

# Get the default/standard dynamics function
def get_standard_dynamics_model():
    """Get the standard dynamics model function."""
    # Use piecewise_activation as the standard dynamics
    dynamics_fn = get_dynamics("piecewise_activation")
    if dynamics_fn is None:
        raise ValueError("Standard dynamics function 'piecewise_activation' not found in registry")
    return dynamics_fn

standard_dynamics_model = get_standard_dynamics_model()
from pyrovelocity.models.jax.core.state import InferenceState
from pyrovelocity.models.jax.factory.config import ModelConfig
from pyrovelocity.models.jax.factory.factory import create_model


@beartype
def sample_posterior(
    inference_state: InferenceState,
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """Sample from the posterior.

    Args:
        inference_state: Inference state
        num_samples: Number of samples
        key: JAX random key

    Returns:
        Dictionary of posterior samples
    """
    # Get posterior samples from inference state
    posterior_samples = inference_state.posterior_samples

    # If we already have the requested number of samples, return them
    if all(v.shape[0] >= num_samples for v in posterior_samples.values()):
        # Truncate to the requested number of samples if needed
        return {k: v[:num_samples] for k, v in posterior_samples.items()}

    # If we need more samples, we need to resample
    if key is None:
        key = jax.random.PRNGKey(0)

    # Resample from the posterior
    resampled_samples = {}
    for param_name, param_samples in posterior_samples.items():
        # Get the number of existing samples
        n_existing = param_samples.shape[0]

        # Generate indices for resampling with replacement
        indices = jax.random.randint(
            key, shape=(num_samples,), minval=0, maxval=n_existing
        )

        # Resample
        resampled_samples[param_name] = param_samples[indices]

        # Get a new key for the next parameter
        key, _ = jax.random.split(key)

    return resampled_samples


@beartype
def posterior_predictive(
    model: Callable,
    posterior_samples: Dict[str, jnp.ndarray],
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
    return_sites: Optional[List[str]] = None,
    parallel: bool = True,
) -> Dict[str, jnp.ndarray]:
    """Generate posterior predictive samples.

    Args:
        model: NumPyro model function
        posterior_samples: Dictionary of posterior samples
        args: Positional arguments for the model
        kwargs: Keyword arguments for the model
        num_samples: Number of samples
        key: JAX random key
        return_sites: Names of sites to return
        parallel: Whether to run in parallel

    Returns:
        Dictionary of posterior predictive samples
    """
    # Generate random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)

    # Subsample if we have more samples than requested
    if all(v.shape[0] > num_samples for v in posterior_samples.values()):
        subkey, key = jax.random.split(key)
        idx = jax.random.choice(
            subkey,
            posterior_samples[list(posterior_samples.keys())[0]].shape[0],
            shape=(num_samples,),
            replace=False,
        )
        posterior_samples = {k: v[idx] for k, v in posterior_samples.items()}

    # Create predictive object
    predictive = numpyro.infer.Predictive(
        model,
        posterior_samples=posterior_samples,
        num_samples=num_samples,
        return_sites=return_sites,
        parallel=parallel,
    )

    # Generate posterior predictive samples
    if kwargs is None:
        kwargs = {}
    samples = predictive(key, *args, **kwargs)

    # If the samples don't include the original parameters, add them
    for param_name, param_samples in posterior_samples.items():
        if param_name not in samples:
            # Take only the first num_samples samples if we have more
            if param_samples.shape[0] > num_samples:
                samples[param_name] = param_samples[:num_samples]
            else:
                samples[param_name] = param_samples

    return samples


@beartype
def compute_velocity(
    posterior_samples: Dict[str, jnp.ndarray],
    dynamics_fn: Callable = standard_dynamics_model,
) -> Dict[str, jnp.ndarray]:
    """Compute RNA velocity from posterior samples.

    Args:
        posterior_samples: Dictionary of posterior samples
        dynamics_fn: Dynamics function

    Returns:
        Dictionary of velocity results
    """
    # Check if we have piecewise activation parameters or legacy alpha/beta/gamma
    has_piecewise_params = all(
        param in posterior_samples 
        for param in ["R_on", "gamma_star", "t_on_star", "delta_star"]
    )
    has_legacy_params = all(
        param in posterior_samples 
        for param in ["alpha", "beta", "gamma"]
    )
    
    if has_piecewise_params:
        # Use piecewise activation parameters
        return _compute_velocity_piecewise(posterior_samples, dynamics_fn)
    elif has_legacy_params:
        # Use legacy alpha/beta/gamma parameters
        return _compute_velocity_legacy(posterior_samples, dynamics_fn)
    else:
        raise ValueError(
            f"Posterior samples must contain either piecewise activation parameters "
            f"(R_on, gamma_star, t_on_star, delta_star) or legacy parameters (alpha, beta, gamma). "
            f"Found parameters: {list(posterior_samples.keys())}"
        )


@beartype
def _compute_velocity_piecewise(
    posterior_samples: Dict[str, jnp.ndarray],
    dynamics_fn: Callable,
) -> Dict[str, jnp.ndarray]:
    """Compute RNA velocity from piecewise activation posterior samples."""
    # Extract piecewise activation parameters
    R_on = posterior_samples["R_on"]  # Shape: (num_samples,) or (num_samples, num_genes)
    gamma_star = posterior_samples["gamma_star"]  # Shape: (num_samples,) or (num_samples, num_genes)
    t_on_star = posterior_samples["t_on_star"]  # Shape: (num_samples,) or (num_samples, num_genes)
    delta_star = posterior_samples["delta_star"]  # Shape: (num_samples,) or (num_samples, num_genes)
    
    # Get time coordinate - could be tau, t_star, or similar
    time_param = None
    for time_key in ["t_star", "tau"]:
        if time_key in posterior_samples:
            time_param = posterior_samples[time_key]
            break
    
    if time_param is None:
        raise ValueError("No time parameter found in posterior samples (expected 't_star' or 'tau')")

    # Get dimensions
    num_samples = R_on.shape[0]

    # Handle both 1D and 2D parameter cases
    if len(R_on.shape) == 1:
        # Single gene case - reshape to 2D
        R_on = R_on.reshape(-1, 1)  # Shape: (num_samples, 1)
        gamma_star = gamma_star.reshape(-1, 1)  # Shape: (num_samples, 1)
        t_on_star = t_on_star.reshape(-1, 1)  # Shape: (num_samples, 1)
        delta_star = delta_star.reshape(-1, 1)  # Shape: (num_samples, 1)
        num_genes = 1
    else:
        # Multiple genes case
        num_genes = R_on.shape[1]

    # Handle both 1D and 2D time cases
    if len(time_param.shape) == 1:
        # Single cell case - reshape to 2D
        time_param = time_param.reshape(-1, 1)  # Shape: (num_samples, 1)
        num_cells = 1
    else:
        # Multiple cells case
        num_cells = time_param.shape[1]

    # Reshape parameters for dynamics function
    # The dynamics function expects t_star with shape (batch_size, n_cells)
    # and parameters with shape (n_genes,) that will be broadcast internally
    
    # Keep time as (num_samples, num_cells) for dynamics function
    t_star = time_param  # Shape: (num_samples, num_cells)
    
    # For parameters, the dynamics function expects them as 1D arrays (n_genes,)
    # but we have (num_samples, n_genes). We'll need to process each sample separately
    # or vectorize over the sample dimension
    
    # Vectorized approach: use vmap to apply dynamics function to each sample
    def single_sample_dynamics(R_on_i, gamma_star_i, t_on_star_i, delta_star_i, t_star_i):
        params_i = {
            "R_on": R_on_i,  # Shape: (n_genes,)
            "gamma_star": gamma_star_i,  # Shape: (n_genes,)
            "t_on_star": t_on_star_i,  # Shape: (n_genes,)
            "delta_star": delta_star_i,  # Shape: (n_genes,)
        }
        # Initial conditions for single sample
        u0_i = jnp.ones((1, num_cells, num_genes))  # Shape: (1, n_cells, n_genes)
        t_star_i_batch = t_star_i[jnp.newaxis, :]  # Shape: (1, n_cells)
        
        return dynamics_fn(t_star_i_batch, u0_i, params_i)
    
    # Apply to all samples using vmap
    u_expected, s_expected = jax.vmap(single_sample_dynamics, in_axes=(0, 0, 0, 0, 0))(
        R_on, gamma_star, t_on_star, delta_star, t_star
    )
    
    # u_expected and s_expected now have shape (num_samples, 1, num_cells, num_genes)
    # Remove the extra batch dimension
    u_expected = u_expected.squeeze(axis=1)  # Shape: (num_samples, num_cells, num_genes)
    s_expected = s_expected.squeeze(axis=1)  # Shape: (num_samples, num_cells, num_genes)

    # Compute velocity as time derivative of spliced counts
    # For piecewise activation: ds*/dt* = u* - γ*s*
    # Need to broadcast gamma_star correctly: (num_samples, num_genes) -> (num_samples, num_cells, num_genes)
    gamma_star_bc = gamma_star[:, jnp.newaxis, :]  # Shape: (num_samples, 1, num_genes)
    velocity = u_expected - gamma_star_bc * s_expected

    # Compute acceleration as second time derivative
    # For piecewise activation: d²s*/dt*² = du*/dt* - γ*ds*/dt*
    # Use finite differences to approximate du*/dt*
    dt = 1e-4
    t_star_plus_dt = t_star + dt  # Shape: (num_samples, num_cells)
    
    # Apply dynamics at t + dt
    u_plus_dt, s_plus_dt = jax.vmap(single_sample_dynamics, in_axes=(0, 0, 0, 0, 0))(
        R_on, gamma_star, t_on_star, delta_star, t_star_plus_dt
    )
    u_plus_dt = u_plus_dt.squeeze(axis=1)  # Shape: (num_samples, num_cells, num_genes)
    s_plus_dt = s_plus_dt.squeeze(axis=1)
    
    du_dt = (u_plus_dt - u_expected) / dt
    acceleration = du_dt - gamma_star_bc * velocity

    # Return results
    return {
        "u_expected": u_expected,
        "s_expected": s_expected,
        "velocity": velocity,
        "acceleration": acceleration,
    }


@beartype
def _compute_velocity_legacy(
    posterior_samples: Dict[str, jnp.ndarray],
    dynamics_fn: Callable,
) -> Dict[str, jnp.ndarray]:
    """Compute RNA velocity from legacy alpha/beta/gamma posterior samples."""
    # Extract parameters
    alpha = posterior_samples["alpha"]  # Shape: (num_samples, num_genes) or (num_samples,)
    beta = posterior_samples["beta"]  # Shape: (num_samples, num_genes) or (num_samples,)
    gamma = posterior_samples["gamma"]  # Shape: (num_samples, num_genes) or (num_samples,)
    tau = posterior_samples["tau"]  # Shape: (num_samples, num_cells) or (num_samples,)

    # Get dimensions
    num_samples = alpha.shape[0]

    # Handle both 1D and 2D parameter cases
    if len(alpha.shape) == 1:
        # Single gene case - reshape to 2D
        alpha = alpha.reshape(-1, 1)  # Shape: (num_samples, 1)
        beta = beta.reshape(-1, 1)  # Shape: (num_samples, 1)
        gamma = gamma.reshape(-1, 1)  # Shape: (num_samples, 1)
        num_genes = 1
    else:
        # Multiple genes case
        num_genes = alpha.shape[1]

    # Handle both 1D and 2D tau cases
    if len(tau.shape) == 1:
        # Single cell case - reshape to 2D
        tau = tau.reshape(-1, 1)  # Shape: (num_samples, 1)
        num_cells = 1
    else:
        # Multiple cells case
        num_cells = tau.shape[1]

    # Reshape parameters for broadcasting
    alpha_expanded = alpha[:, :, jnp.newaxis]  # Shape: (num_samples, num_genes, 1)
    beta_expanded = beta[:, :, jnp.newaxis]  # Shape: (num_samples, num_genes, 1)
    gamma_expanded = gamma[:, :, jnp.newaxis]  # Shape: (num_samples, num_genes, 1)
    tau_expanded = tau[:, jnp.newaxis, :]  # Shape: (num_samples, 1, num_cells)

    # Create expanded parameters dictionary
    expanded_params = {
        "alpha": alpha_expanded,
        "beta": beta_expanded,
        "gamma": gamma_expanded,
    }

    # Add scaling parameter if it exists in the parameters
    if "scaling" in posterior_samples:
        scaling = posterior_samples["scaling"]
        expanded_params["scaling"] = scaling[:, :, jnp.newaxis]

    # Initial conditions (steady state)
    u0 = alpha / beta
    s0 = alpha / gamma

    # Reshape for broadcasting
    u0_expanded = u0[:, :, jnp.newaxis]  # Shape: (num_samples, num_genes, 1)
    s0_expanded = s0[:, :, jnp.newaxis]  # Shape: (num_samples, num_genes, 1)

    # Apply dynamics model to get expected counts
    dynamics_result = dynamics_fn(
        tau_expanded, u0_expanded, expanded_params
    )

    # Handle both tuple and single return value cases
    if isinstance(dynamics_result, tuple) and len(dynamics_result) == 2:
        u_expected, s_expected = dynamics_result
    else:
        # If only one value is returned, assume it's the spliced counts
        # and compute unspliced counts from the parameters
        s_expected = dynamics_result
        # For standard dynamics, u = (ds/dt + gamma*s) / beta
        u_expected = (
            beta_expanded * u0_expanded * jnp.exp(-beta_expanded * tau_expanded)
        )

    # Compute velocity as time derivative of spliced counts
    # For standard dynamics: ds/dt = beta*u - gamma*s
    velocity = beta_expanded * u_expected - gamma_expanded * s_expected

    # Compute acceleration as second time derivative
    # For standard dynamics: d²s/dt² = beta*du/dt - gamma*ds/dt
    # du/dt = alpha - beta*u
    du_dt = alpha_expanded - beta_expanded * u_expected
    acceleration = beta_expanded * du_dt - gamma_expanded * velocity

    # Return results
    return {
        "u_expected": u_expected,
        "s_expected": s_expected,
        "velocity": velocity,
        "acceleration": acceleration,
    }


@beartype
def compute_uncertainty(
    velocity_samples: Union[Dict[str, jnp.ndarray], jnp.ndarray, np.ndarray, torch.Tensor],
) -> Dict[str, jnp.ndarray]:
    """Compute uncertainty in RNA velocity.

    This function computes uncertainty measures for RNA velocity. It can handle
    different input formats:
    - Dictionary of velocity samples (e.g., {"velocity": samples, "u_expected": samples})
    - Array of velocity samples (e.g., samples with shape [num_samples, num_genes] or [num_samples, num_cells, num_genes])

    Args:
        velocity_samples: Velocity samples, either as a dictionary or an array

    Returns:
        Dictionary of uncertainty measures
    """
    # Initialize uncertainty measures
    uncertainty = {}

    # Handle different input types
    if isinstance(velocity_samples, dict):
        # Dictionary input - process each key
        for key, samples in velocity_samples.items():
            # Convert to JAX array if needed
            if isinstance(samples, np.ndarray):
                samples = jnp.array(samples)
            elif isinstance(samples, torch.Tensor):
                samples = jnp.array(samples.detach().cpu().numpy())

            # Ensure samples has at least 2 dimensions
            if samples.ndim == 1:
                samples = samples.reshape(-1, 1)

            # Compute mean across samples (first dimension)
            mean = jnp.mean(samples, axis=0)
            uncertainty[f"{key}_mean"] = mean

            # Compute standard deviation across samples
            std = jnp.std(samples, axis=0)
            uncertainty[f"{key}_std"] = std

            # Compute coefficient of variation (CV = std/mean)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            cv = std / (jnp.abs(mean) + epsilon)
            uncertainty[f"{key}_cv"] = cv

            # Compute quantiles
            q10 = jnp.quantile(samples, 0.1, axis=0)
            q90 = jnp.quantile(samples, 0.9, axis=0)
            uncertainty[f"{key}_q10"] = q10
            uncertainty[f"{key}_q90"] = q90

            # Compute credible interval width
            ci_width = q90 - q10
            uncertainty[f"{key}_ci_width"] = ci_width

            # Compute normalized credible interval width
            norm_ci_width = ci_width / (jnp.abs(mean) + epsilon)
            uncertainty[f"{key}_norm_ci_width"] = norm_ci_width

        # For velocity specifically, compute additional measures
        if "velocity" in velocity_samples:
            velocity = velocity_samples["velocity"]

            # Convert to JAX array if needed
            if isinstance(velocity, np.ndarray):
                velocity = jnp.array(velocity)
            elif isinstance(velocity, torch.Tensor):
                velocity = jnp.array(velocity.detach().cpu().numpy())

            # Compute probability of positive velocity
            prob_positive = jnp.mean(velocity > 0, axis=0)
            uncertainty["velocity_prob_positive"] = prob_positive

            # Compute velocity confidence (1 - 2*|0.5 - prob_positive|)
            # This is 1 when prob_positive is 0 or 1 (certain) and 0 when prob_positive is 0.5 (uncertain)
            velocity_confidence = 1 - 2 * jnp.abs(0.5 - prob_positive)
            uncertainty["velocity_confidence"] = velocity_confidence
    else:
        # Array input - treat as velocity samples
        # Convert to JAX array if needed
        samples = velocity_samples
        if isinstance(samples, np.ndarray):
            samples = jnp.array(samples)
        elif isinstance(samples, torch.Tensor):
            samples = jnp.array(samples.detach().cpu().numpy())

        # Ensure samples has at least 2 dimensions
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        # Compute mean across samples (first dimension)
        mean = jnp.mean(samples, axis=0)
        uncertainty["velocity_mean"] = mean

        # Compute standard deviation across samples
        std = jnp.std(samples, axis=0)
        uncertainty["velocity_std"] = std

        # Compute coefficient of variation (CV = std/mean)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        cv = std / (jnp.abs(mean) + epsilon)
        uncertainty["velocity_cv"] = cv

        # Compute quantiles
        q10 = jnp.quantile(samples, 0.1, axis=0)
        q90 = jnp.quantile(samples, 0.9, axis=0)
        uncertainty["velocity_q10"] = q10
        uncertainty["velocity_q90"] = q90

        # Compute credible interval width
        ci_width = q90 - q10
        uncertainty["velocity_ci_width"] = ci_width

        # Compute normalized credible interval width
        norm_ci_width = ci_width / (jnp.abs(mean) + epsilon)
        uncertainty["velocity_norm_ci_width"] = norm_ci_width

        # Compute probability of positive velocity
        prob_positive = jnp.mean(samples > 0, axis=0)
        uncertainty["velocity_prob_positive"] = prob_positive

        # Compute velocity confidence (1 - 2*|0.5 - prob_positive|)
        # This is 1 when prob_positive is 0 or 1 (certain) and 0 when prob_positive is 0.5 (uncertain)
        velocity_confidence = 1 - 2 * jnp.abs(0.5 - prob_positive)
        uncertainty["velocity_confidence"] = velocity_confidence

    return uncertainty


@beartype
def analyze_posterior(
    inference_state: InferenceState,
    model: Union[Callable, Dict[str, Any], ModelConfig],
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    dynamics_fn: Callable = standard_dynamics_model,
    num_samples: int = 1000,
    key: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """Analyze posterior samples from either SVI or MCMC.

    This function analyzes posterior samples from an inference state, computes velocity,
    uncertainty, and generates posterior predictive samples. It supports both direct model
    functions and model configurations through the factory system.

    Args:
        inference_state: Inference state containing posterior samples
        model: Either a NumPyro model function or a model configuration dictionary
        args: Positional arguments for the model (optional)
        kwargs: Keyword arguments for the model (optional)
        dynamics_fn: Dynamics function to use for velocity computation
        num_samples: Number of posterior samples to use
        key: JAX random key (optional)

    Returns:
        Dictionary of analysis results containing:
        - posterior_samples: Samples from the posterior distribution
        - posterior_predictive: Posterior predictive samples
        - velocity: Computed velocity samples
        - uncertainty: Uncertainty measures for velocity
        - inference_data: ArviZ InferenceData object
        - diagnostics: Inference diagnostics (if available)
    """
    # Generate random key if not provided
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize kwargs if None
    if kwargs is None:
        kwargs = {}

    # Split key for different operations
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Sample from the posterior
    posterior_samples = sample_posterior(
        inference_state=inference_state,
        num_samples=num_samples,
        key=subkey1,
    )

    # Handle model configuration by creating a model function
    model_fn = model
    if isinstance(model, (dict, ModelConfig)):
        model_fn = create_model(model)

    # Generate posterior predictive samples
    posterior_predictive_samples = posterior_predictive(
        model=model_fn,
        posterior_samples=posterior_samples,
        args=args,
        kwargs=kwargs,
        num_samples=num_samples,
        key=subkey2,
    )

    # Compute velocity
    velocity_samples = compute_velocity(
        posterior_samples=posterior_samples,
        dynamics_fn=dynamics_fn,
    )

    # Compute uncertainty
    uncertainty = compute_uncertainty(
        velocity_samples=velocity_samples,
    )

    # Create inference data for ArviZ
    observed_data = {}
    if args and len(args) > 0:
        observed_data["u_obs"] = args[0]
    if args and len(args) > 1:
        observed_data["s_obs"] = args[1]

    # Override with kwargs if provided
    if "u_obs" in kwargs:
        observed_data["u_obs"] = kwargs["u_obs"]
    if "s_obs" in kwargs:
        observed_data["s_obs"] = kwargs["s_obs"]

    inference_data = create_inference_data(
        posterior_samples=posterior_samples,
        posterior_predictive_samples=posterior_predictive_samples,
        observed_data=observed_data,
    )

    # Combine all results
    results = {
        "posterior_samples": posterior_samples,
        "posterior_predictive": posterior_predictive_samples,
        "velocity": velocity_samples,
        "uncertainty": uncertainty,
        "inference_data": inference_data,
    }

    # Add diagnostics if available
    if inference_state.diagnostics is not None:
        results["diagnostics"] = inference_state.diagnostics

    return results


@beartype
def create_inference_data(
    posterior_samples: Dict[str, jnp.ndarray],
    posterior_predictive_samples: Optional[Dict[str, jnp.ndarray]] = None,
    observed_data: Optional[Dict[str, jnp.ndarray]] = None,
) -> Any:  # Using Any instead of az.InferenceData to avoid beartype issues
    """Create ArviZ InferenceData object from posterior samples.

    Args:
        posterior_samples: Dictionary of posterior samples
        posterior_predictive_samples: Optional dictionary of posterior predictive samples
        observed_data: Optional dictionary of observed data

    Returns:
        ArviZ InferenceData object
    """
    # Convert JAX arrays to NumPy arrays
    posterior_dict = {k: jnp.array(v) for k, v in posterior_samples.items()}

    # Create dictionary for InferenceData
    inference_dict = {}

    # Add posterior samples
    inference_dict["posterior"] = posterior_dict

    # Add posterior predictive samples if provided
    if posterior_predictive_samples is not None:
        pp_dict = {
            k: jnp.array(v) for k, v in posterior_predictive_samples.items()
        }
        inference_dict["posterior_predictive"] = pp_dict

    # Add observed data if provided
    if observed_data is not None:
        obs_dict = {k: jnp.array(v) for k, v in observed_data.items()}
        inference_dict["observed_data"] = obs_dict

    # Create InferenceData object
    return az.from_dict(**inference_dict)


@beartype
def format_anndata_output(
    adata: anndata.AnnData,
    results: Dict[str, Any],
    model_name: str = "velocity_model",
) -> anndata.AnnData:
    """Format results into AnnData object compatible with src/pyrovelocity/plots.

    This function takes the results from analyze_posterior and formats them into an
    AnnData object for visualization and further analysis. It handles different shapes
    and formats of the input data, ensuring compatibility with the plotting functions.

    Args:
        adata: AnnData object with cells as rows and genes as columns
        results: Dictionary of results from analyze_posterior
        model_name: Name of the model to use as a prefix for annotations

    Returns:
        Updated AnnData object with velocity information
    """
    # Create a copy of the AnnData object to avoid modifying the original
    adata_copy = adata.copy()

    # Validate AnnData structure
    if adata_copy.n_obs == 0 or adata_copy.n_vars == 0:
        raise ValueError("AnnData object is empty")

    # Extract uncertainty results
    uncertainty = results.get("uncertainty", {})
    if not uncertainty:
        raise ValueError("Results dictionary does not contain uncertainty measures")

    # Extract posterior samples
    posterior_samples = results.get("posterior_samples", {})
    if not posterior_samples:
        raise ValueError("Results dictionary does not contain posterior samples")

    # Extract velocity samples
    velocity_samples = results.get("velocity", {})
    if not velocity_samples:
        # Try to compute velocity if not provided
        try:
            from pyrovelocity.models.jax.registry.dynamics import get_dynamics

            # Get the standard dynamics function from registry
            dynamics_fn = get_dynamics("piecewise_activation")
            if dynamics_fn is None:
                raise ValueError("Standard dynamics function 'piecewise_activation' not found in registry")

            velocity_samples = compute_velocity(
                posterior_samples=posterior_samples,
                dynamics_fn=dynamics_fn,
            )
        except Exception as e:
            raise ValueError(f"Could not compute velocity: {e}")

    # Get dimensions
    n_cells = adata_copy.n_obs
    n_genes = adata_copy.n_vars

    # Add mean velocity to cell-specific annotations
    if "velocity_mean" in uncertainty:
        # Handle both 1D and 2D cases
        velocity = jnp.array(uncertainty["velocity_mean"])

        # Check shape and reshape if necessary
        if len(velocity.shape) == 1:
            # If 1D, reshape to (1, n_genes)
            velocity = velocity.reshape(1, -1)
        elif velocity.shape[0] == n_genes and velocity.shape[1] == n_cells:
            # If shape is (n_genes, n_cells), transpose to (n_cells, n_genes)
            velocity = velocity.T
        elif velocity.shape[0] == 1 and velocity.shape[1] == n_genes:
            # If shape is (1, n_genes), broadcast to (n_cells, n_genes)
            velocity = jnp.broadcast_to(velocity, (n_cells, n_genes))

        # Ensure the shape matches AnnData dimensions
        if velocity.shape[1] != n_genes:
            # If the number of genes doesn't match, try to reshape
            if velocity.size == n_genes:
                velocity = velocity.reshape(1, n_genes)
                velocity = jnp.broadcast_to(velocity, (n_cells, n_genes))
            else:
                raise ValueError(
                    f"Velocity shape {velocity.shape} does not match AnnData dimensions "
                    f"({n_cells}, {n_genes})"
                )

        # Store in layers (convert JAX array to numpy for AnnData compatibility)
        adata_copy.layers[f"{model_name}_velocity"] = np.array(velocity)

    # Add velocity confidence to cell-specific annotations
    if "velocity_confidence" in uncertainty:
        # Handle both 1D and 2D cases
        confidence = jnp.array(uncertainty["velocity_confidence"])

        # Check shape and reshape if necessary
        if len(confidence.shape) > 1:
            if confidence.shape[0] == 1:
                # If shape is (1, n_cells), reshape to (n_cells,)
                confidence = confidence.reshape(-1)
            elif confidence.shape[1] == 1:
                # If shape is (n_cells, 1), reshape to (n_cells,)
                confidence = confidence.reshape(-1)
            elif confidence.shape[0] == n_genes and confidence.shape[1] == n_cells:
                # If shape is (n_genes, n_cells), take mean across genes
                confidence = jnp.mean(confidence, axis=0)
            elif confidence.shape[0] == n_cells and confidence.shape[1] == n_genes:
                # If shape is (n_cells, n_genes), take mean across genes
                confidence = jnp.mean(confidence, axis=1)

        # Ensure the shape matches AnnData dimensions
        if len(confidence.shape) == 1 and confidence.shape[0] != n_cells:
            # If the number of cells doesn't match, try to reshape or broadcast
            if confidence.size == 1:
                # If scalar, broadcast to all cells
                confidence = jnp.broadcast_to(confidence, (n_cells,))
            elif confidence.size == n_genes:
                # If gene-wise, take mean
                confidence = jnp.mean(confidence) * jnp.ones(n_cells)
            else:
                raise ValueError(
                    f"Confidence shape {confidence.shape} does not match AnnData dimensions "
                    f"({n_cells}, {n_genes})"
                )

        # Store in obs (convert JAX array to numpy for AnnData compatibility)
        adata_copy.obs[f"{model_name}_velocity_confidence"] = np.array(confidence)

    # Add velocity probability to cell-specific annotations
    if "velocity_prob_positive" in uncertainty:
        # Handle both 1D and 2D cases
        probability = jnp.array(uncertainty["velocity_prob_positive"])

        # Check shape and reshape if necessary
        if len(probability.shape) > 1:
            if probability.shape[0] == 1:
                # If shape is (1, n_cells), reshape to (n_cells,)
                probability = probability.reshape(-1)
            elif probability.shape[1] == 1:
                # If shape is (n_cells, 1), reshape to (n_cells,)
                probability = probability.reshape(-1)
            elif probability.shape[0] == n_genes and probability.shape[1] == n_cells:
                # If shape is (n_genes, n_cells), take mean across genes
                probability = jnp.mean(probability, axis=0)
            elif probability.shape[0] == n_cells and probability.shape[1] == n_genes:
                # If shape is (n_cells, n_genes), take mean across genes
                probability = jnp.mean(probability, axis=1)

        # Ensure the shape matches AnnData dimensions
        if len(probability.shape) == 1 and probability.shape[0] != n_cells:
            # If the number of cells doesn't match, try to reshape or broadcast
            if probability.size == 1:
                # If scalar, broadcast to all cells
                probability = jnp.broadcast_to(probability, (n_cells,))
            elif probability.size == n_genes:
                # If gene-wise, take mean
                probability = jnp.mean(probability) * jnp.ones(n_cells)
            else:
                raise ValueError(
                    f"Probability shape {probability.shape} does not match AnnData dimensions "
                    f"({n_cells}, {n_genes})"
                )

        # Store in obs (convert JAX array to numpy for AnnData compatibility)
        adata_copy.obs[f"{model_name}_velocity_probability"] = np.array(probability)

    # Add expected unspliced and spliced counts to layers
    if "u_expected_mean" in uncertainty and "s_expected_mean" in uncertainty:
        # Handle both 1D and 2D cases for u_expected
        u_expected = jnp.array(uncertainty["u_expected_mean"])
        s_expected = jnp.array(uncertainty["s_expected_mean"])

        # Check shape and reshape if necessary for u_expected
        if len(u_expected.shape) == 1:
            # If 1D, reshape to (1, n_genes)
            u_expected = u_expected.reshape(1, -1)
        elif u_expected.shape[0] == n_genes and u_expected.shape[1] == n_cells:
            # If shape is (n_genes, n_cells), transpose to (n_cells, n_genes)
            u_expected = u_expected.T
        elif u_expected.shape[0] == 1 and u_expected.shape[1] == n_genes:
            # If shape is (1, n_genes), broadcast to (n_cells, n_genes)
            u_expected = jnp.broadcast_to(u_expected, (n_cells, n_genes))

        # Check shape and reshape if necessary for s_expected
        if len(s_expected.shape) == 1:
            # If 1D, reshape to (1, n_genes)
            s_expected = s_expected.reshape(1, -1)
        elif s_expected.shape[0] == n_genes and s_expected.shape[1] == n_cells:
            # If shape is (n_genes, n_cells), transpose to (n_cells, n_genes)
            s_expected = s_expected.T
        elif s_expected.shape[0] == 1 and s_expected.shape[1] == n_genes:
            # If shape is (1, n_genes), broadcast to (n_cells, n_genes)
            s_expected = jnp.broadcast_to(s_expected, (n_cells, n_genes))

        # Ensure the shapes match AnnData dimensions
        if u_expected.shape[1] != n_genes or s_expected.shape[1] != n_genes:
            # If the number of genes doesn't match, try to reshape
            if u_expected.size == n_genes and s_expected.size == n_genes:
                u_expected = u_expected.reshape(1, n_genes)
                s_expected = s_expected.reshape(1, n_genes)
                u_expected = jnp.broadcast_to(u_expected, (n_cells, n_genes))
                s_expected = jnp.broadcast_to(s_expected, (n_cells, n_genes))
            else:
                raise ValueError(
                    f"Expected counts shapes {u_expected.shape}, {s_expected.shape} do not match "
                    f"AnnData dimensions ({n_cells}, {n_genes})"
                )

        # Store in layers (convert JAX arrays to numpy for AnnData compatibility)
        adata_copy.layers[f"{model_name}_u_expected"] = np.array(u_expected)
        adata_copy.layers[f"{model_name}_s_expected"] = np.array(s_expected)

    # Add uncertainty measures to var annotations
    for key in uncertainty:
        if key.endswith("_cv") or key.endswith("_norm_ci_width"):
            # These are gene-specific measures
            value = jnp.array(uncertainty[key])

            # Handle both 1D and 2D cases
            if len(value.shape) > 1:
                if value.shape[0] == 1:
                    # If shape is (1, n_genes), reshape to (n_genes,)
                    value = value[0]
                elif value.shape[1] == 1:
                    # If shape is (n_genes, 1), reshape to (n_genes,)
                    value = value.reshape(-1)
                elif value.shape[0] == n_cells and value.shape[1] == n_genes:
                    # If shape is (n_cells, n_genes), take mean across cells
                    value = jnp.mean(value, axis=0)
                elif value.shape[0] == n_genes and value.shape[1] == n_cells:
                    # If shape is (n_genes, n_cells), take mean across cells
                    value = jnp.mean(value, axis=1)

            # Ensure the shape matches AnnData dimensions
            if len(value.shape) == 1 and value.shape[0] != n_genes:
                # If the number of genes doesn't match, try to reshape or broadcast
                if value.size == 1:
                    # If scalar, broadcast to all genes
                    value = jnp.broadcast_to(value, (n_genes,))
                elif value.size == n_cells:
                    # If cell-wise, take mean
                    value = jnp.mean(value) * jnp.ones(n_genes)
                else:
                    raise ValueError(
                        f"Uncertainty measure {key} shape {value.shape} does not match "
                        f"AnnData dimensions ({n_cells}, {n_genes})"
                    )

            # Store in var (convert JAX array to numpy for AnnData compatibility)
            adata_copy.var[f"{model_name}_{key}"] = np.array(value)

    # Add model parameters to var annotations
    # Handle both legacy and piecewise activation parameter names
    legacy_params = ["alpha", "beta", "gamma"]
    piecewise_params = ["R_on", "gamma_star", "t_on_star", "delta_star", "U_0i"]
    
    # Check which parameter set we have
    has_legacy = any(param in posterior_samples for param in legacy_params)
    has_piecewise = any(param in posterior_samples for param in piecewise_params)
    
    if has_piecewise:
        param_names = piecewise_params
    elif has_legacy:
        param_names = legacy_params
    else:
        param_names = []
    
    for param_name in param_names:
        if param_name in posterior_samples:
            # Compute mean across samples
            param_samples = jnp.array(posterior_samples[param_name])
            param_mean = jnp.mean(param_samples, axis=0)

            # Handle both scalar and vector cases
            if not isinstance(param_mean, jnp.ndarray) or param_mean.ndim == 0:
                # If scalar, broadcast to all genes
                param_mean = jnp.array([param_mean])
                param_mean = jnp.broadcast_to(param_mean, (n_genes,))
            elif param_mean.shape[0] != n_genes:
                # If the number of genes doesn't match, try to reshape or broadcast
                if param_mean.size == 1:
                    # If scalar, broadcast to all genes
                    param_mean = jnp.broadcast_to(param_mean, (n_genes,))
                elif param_mean.size == n_cells:
                    # If cell-wise, take mean
                    param_mean = jnp.mean(param_mean) * jnp.ones(n_genes)
                else:
                    raise ValueError(
                        f"Parameter {param_name} shape {param_mean.shape} does not match "
                        f"AnnData dimensions ({n_cells}, {n_genes})"
                    )

            # Store in var (convert JAX array to numpy for AnnData compatibility)
            adata_copy.var[f"{model_name}_{param_name}"] = np.array(param_mean)

    # Add latent time to obs annotations
    # Handle both tau (legacy) and t_star (piecewise activation) time coordinates
    time_param_name = None
    for time_key in ["t_star", "tau"]:
        if time_key in posterior_samples:
            time_param_name = time_key
            break
    
    if time_param_name is not None:
        # Compute mean across samples
        time_samples = jnp.array(posterior_samples[time_param_name])
        time_mean = jnp.mean(time_samples, axis=0)

        # Handle both scalar and vector cases
        if not isinstance(time_mean, jnp.ndarray) or time_mean.ndim == 0:
            # If scalar, broadcast to all cells
            time_mean = jnp.array([time_mean])
            time_mean = jnp.broadcast_to(time_mean, (n_cells,))
        elif time_mean.shape[0] != n_cells:
            # If the number of cells doesn't match, try to reshape or broadcast
            if time_mean.size == 1:
                # If scalar, broadcast to all cells
                time_mean = jnp.broadcast_to(time_mean, (n_cells,))
            elif time_mean.size == n_genes:
                # If gene-wise, take mean
                time_mean = jnp.mean(time_mean) * jnp.ones(n_cells)
            else:
                raise ValueError(
                    f"Latent time shape {time_mean.shape} does not match "
                    f"AnnData dimensions ({n_cells}, {n_genes})"
                )

        # Store in obs (convert JAX array to numpy for AnnData compatibility)
        adata_copy.obs[f"{model_name}_latent_time"] = np.array(time_mean)

    # Store the model name in uns
    # Initialize uns if it doesn't exist or is None
    if not hasattr(adata_copy, "uns") or adata_copy.uns is None:
        adata_copy.uns = {}

    # Initialize velocity_models if it doesn't exist
    if "velocity_models" not in adata_copy.uns:
        adata_copy.uns["velocity_models"] = []

    # Add model_name to velocity_models if it's not already there
    if model_name not in adata_copy.uns["velocity_models"]:
        adata_copy.uns["velocity_models"].append(model_name)

    # Store additional model information
    adata_copy.uns[f"{model_name}_model_type"] = "jax_numpyro"

    # Store model parameters in uns
    # Handle both legacy and piecewise activation parameters
    stored_params = {}
    for param_name in param_names:
        if param_name in posterior_samples:
            # Convert JAX array to numpy then to list for JSON serialization
            stored_params[param_name] = np.array(posterior_samples[param_name]).tolist()
    
    adata_copy.uns[f"{model_name}_params"] = stored_params

    return adata_copy
