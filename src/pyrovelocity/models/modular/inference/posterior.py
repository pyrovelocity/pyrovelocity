"""
Posterior analysis utilities for PyroVelocity PyTorch/Pyro modular implementation.

This module contains posterior analysis utilities, including:

- sample_posterior: Sample from the posterior
- compute_velocity: Compute RNA velocity from posterior samples
- compute_uncertainty: Compute uncertainty in RNA velocity
- analyze_posterior: Analyze posterior samples from either SVI or MCMC
- create_inference_data: Create ArviZ InferenceData object from posterior samples
- format_anndata_output: Format results into AnnData object
"""

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Import PyroVelocityModel using TYPE_CHECKING to avoid circular import
from typing import TYPE_CHECKING

import numpy as np
import pyro
import pyro.distributions as dist
import pyro.infer as infer
import torch
from anndata import AnnData
from beartype import beartype
from jaxtyping import Array, Float

from pyrovelocity.models.modular.constants import CELLS_DIM, GENES_DIM
from pyrovelocity.models.modular.inference.unified import (
    InferenceState,
    extract_posterior_samples,
    posterior_predictive,
)

if TYPE_CHECKING:
    from pyrovelocity.models.modular.model import PyroVelocityModel


@beartype
def sample_posterior(
    model: Union[Callable, "PyroVelocityModel"],
    state: InferenceState,
    num_samples: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Sample from the posterior.

    Args:
        model: Pyro model function or PyroVelocityModel
        state: Inference state
        num_samples: Number of posterior samples
        seed: Random seed

    Returns:
        Dictionary of posterior samples
    """
    # Extract posterior samples from inference state
    return extract_posterior_samples(state, num_samples)


@beartype
def compute_velocity(
    model: "PyroVelocityModel",
    posterior_samples: Dict[str, Union[torch.Tensor, np.ndarray]],
    adata: Optional[AnnData] = None,
    use_mean: bool = False,
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Compute RNA velocity from posterior samples using the dynamics component.

    This function leverages the dynamics component from the PyroVelocityModel to compute
    velocity, ensuring consistency with the model's differential equations.

    Args:
        model: PyroVelocityModel with dynamics component
        posterior_samples: Posterior samples from inference
        adata: AnnData object (unused but kept for API compatibility)
        use_mean: Whether to use mean of posterior samples

    Returns:
        Dictionary containing velocity results
    """
    # All modular models must have a dynamics component that follows the protocol
    if not hasattr(model, 'dynamics_model'):
        raise ValueError("Model must have a dynamics_model component")

    dynamics_model = model.dynamics_model
    if not hasattr(dynamics_model, 'compute_velocity'):
        raise ValueError("Dynamics model must implement compute_velocity method")

    print(f"Using dynamics component: {dynamics_model.__class__.__name__}")

    # Print all keys in posterior_samples for debugging
    print(f"compute_velocity - posterior_samples keys: {list(posterior_samples.keys())}")

    # Extract required parameters from posterior samples
    alpha = posterior_samples.get("alpha")
    beta = posterior_samples.get("beta")
    gamma = posterior_samples.get("gamma")
    ut = posterior_samples.get("ut")
    st = posterior_samples.get("st")
    u_scale = posterior_samples.get("u_scale")
    s_scale = posterior_samples.get("s_scale")

    # Validate that we have the required parameters
    if alpha is None or beta is None or gamma is None:
        raise ValueError("Posterior samples must contain alpha, beta, and gamma")

    if ut is None or st is None:
        # Fallback to observed counts if latent counts are not available
        ut = posterior_samples.get("u")
        st = posterior_samples.get("s")

        if ut is None or st is None:
            raise ValueError("Posterior samples must contain either ut/st (latent RNA counts) or u/s (observed RNA counts)")

        print("Warning: Using observed RNA counts (u/s) instead of latent counts (ut/st) for velocity computation")


    # Convert numpy arrays to torch tensors if needed
    def ensure_tensor(x):
        if x is not None and isinstance(x, np.ndarray):
            return torch.tensor(x)
        return x

    ut = ensure_tensor(ut)
    st = ensure_tensor(st)
    alpha = ensure_tensor(alpha)
    beta = ensure_tensor(beta)
    gamma = ensure_tensor(gamma)
    u_scale = ensure_tensor(u_scale)
    s_scale = ensure_tensor(s_scale)

    # Prepare kwargs for scaling factors
    velocity_kwargs = {}
    if u_scale is not None:
        velocity_kwargs["u_scale"] = u_scale
    if s_scale is not None:
        velocity_kwargs["s_scale"] = s_scale

    # Compute velocity using the dynamics component
    if use_mean:
        # Take mean across samples first
        alpha_mean = alpha.mean(dim=0) if alpha.dim() > 1 else alpha
        beta_mean = beta.mean(dim=0) if beta.dim() > 1 else beta
        gamma_mean = gamma.mean(dim=0) if gamma.dim() > 1 else gamma
        ut_mean = ut.mean(dim=0) if ut.dim() > 2 else ut
        st_mean = st.mean(dim=0) if st.dim() > 2 else st

        # Update kwargs with mean values
        if u_scale is not None:
            velocity_kwargs["u_scale"] = u_scale.mean(dim=0) if u_scale.dim() > 1 else u_scale
        if s_scale is not None:
            velocity_kwargs["s_scale"] = s_scale.mean(dim=0) if s_scale.dim() > 1 else s_scale

        velocity = dynamics_model.compute_velocity(
            ut=ut_mean,
            st=st_mean,
            alpha=alpha_mean,
            beta=beta_mean,
            gamma=gamma_mean,
            **velocity_kwargs
        )

        # Compute steady state
        u_ss, s_ss = dynamics_model.steady_state(alpha_mean, beta_mean, gamma_mean)

        return {
            "velocity": velocity,
            "alpha": alpha_mean,
            "beta": beta_mean,
            "gamma": gamma_mean,
            "u_ss": u_ss,
            "s_ss": s_ss,
        }
    else:
        # Compute velocity for all samples
        velocity = dynamics_model.compute_velocity(
            ut=ut,
            st=st,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            **velocity_kwargs
        )

        # Compute steady state
        u_ss, s_ss = dynamics_model.steady_state(alpha, beta, gamma)

        return {
            "velocity": velocity,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "u_ss": u_ss,
            "s_ss": s_ss,
        }


@beartype
def compute_uncertainty(
    velocity_samples: Union[torch.Tensor, Dict[str, torch.Tensor]],
    method: str = "std",
) -> torch.Tensor:
    """
    Compute uncertainty in RNA velocity.

    Args:
        velocity_samples: Velocity samples from posterior, either a tensor or a dictionary with a 'velocity' key
        method: Method for computing uncertainty ("std", "quantile", "entropy")

    Returns:
        Uncertainty in RNA velocity
    """
    # Handle dictionary input
    if isinstance(velocity_samples, dict):
        if 'velocity' not in velocity_samples:
            raise ValueError("velocity_samples dictionary must contain a 'velocity' key")
        velocity_samples = velocity_samples['velocity']

    if method == "std":
        # Standard deviation across samples
        return velocity_samples.std(dim=0)
    elif method == "quantile":
        # Interquartile range across samples
        q75 = torch.quantile(velocity_samples, 0.75, dim=0)
        q25 = torch.quantile(velocity_samples, 0.25, dim=0)
        return q75 - q25
    elif method == "entropy":
        # Approximate entropy using histogram
        # This is a simplified version and may not be accurate for all distributions
        try:
            import numpy as np
            from scipy.stats import entropy

            # Convert to numpy for scipy
            velocity_np = velocity_samples.detach().cpu().numpy()
            # Compute entropy for each gene
            uncertainties = []
            for gene_idx in range(velocity_np.shape[1]):
                hist, _ = np.histogram(
                    velocity_np[:, gene_idx], bins=20, density=True
                )
                uncertainties.append(entropy(hist))
            return torch.tensor(uncertainties)
        except ImportError:
            warnings.warn("scipy not available, falling back to std method")
            return velocity_samples.std(dim=0)
    else:
        raise ValueError(f"Unsupported uncertainty method: {method}")


@beartype
def analyze_posterior(
    state: InferenceState,
    model: Union[Callable, "PyroVelocityModel"],
    adata: Optional[AnnData] = None,
    num_samples: int = 1000,
    compute_velocity_flag: bool = True,
    compute_uncertainty_flag: bool = True,
    uncertainty_method: str = "std",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyze posterior samples from either SVI or MCMC.

    Args:
        state: Inference state
        model: Pyro model function or PyroVelocityModel
        adata: AnnData object
        num_samples: Number of posterior samples
        compute_velocity_flag: Whether to compute velocity
        compute_uncertainty_flag: Whether to compute uncertainty
        uncertainty_method: Method for computing uncertainty
        seed: Random seed

    Returns:
        Dictionary containing analysis results
    """
    # Sample from posterior
    posterior_samples = sample_posterior(model, state, num_samples, seed)

    # Initialize results dictionary
    results = {"posterior_samples": posterior_samples}

    # Compute velocity if requested
    if compute_velocity_flag:
        velocity_results = compute_velocity(model, posterior_samples, adata)
        # Update results dictionary with velocity results
        for k, v in velocity_results.items():
            results[k] = v

        # Compute uncertainty if requested
        if compute_uncertainty_flag and "velocity" in velocity_results:
            velocity = velocity_results["velocity"]
            # Convert numpy array to tensor if needed
            if isinstance(velocity, np.ndarray):
                velocity = torch.tensor(velocity)
            uncertainty = compute_uncertainty(velocity, uncertainty_method)
            results["uncertainty"] = uncertainty

    return results


@beartype
def create_inference_data(
    posterior_samples: Dict[str, torch.Tensor],
    observed_data: Optional[Dict[str, torch.Tensor]] = None,
    prior_samples: Optional[Dict[str, torch.Tensor]] = None,
) -> Any:
    """
    Create ArviZ InferenceData object from posterior samples.

    Args:
        posterior_samples: Posterior samples from inference
        observed_data: Observed data
        prior_samples: Prior samples

    Returns:
        ArviZ InferenceData object
    """
    try:
        import arviz as az
    except ImportError:
        warnings.warn("arviz not available, returning samples as dictionary")
        return {
            "posterior": posterior_samples,
            "observed_data": observed_data,
            "prior": prior_samples,
        }

    # Convert torch tensors to numpy arrays
    posterior_np = {
        k: v.detach().cpu().numpy() for k, v in posterior_samples.items()
    }
    observed_np = (
        {k: v.detach().cpu().numpy() for k, v in observed_data.items()}
        if observed_data
        else None
    )
    prior_np = (
        {k: v.detach().cpu().numpy() for k, v in prior_samples.items()}
        if prior_samples
        else None
    )

    # Create InferenceData object
    return az.from_dict(
        posterior=posterior_np,
        observed_data=observed_np,
        prior=prior_np,
    )


@beartype
def format_anndata_output(
    adata: AnnData,
    results: Dict[str, Any],
    model_name: str = "pyrovelocity",
) -> AnnData:
    """
    Format results into AnnData object.

    Args:
        adata: AnnData object
        results: Results from analyze_posterior
        model_name: Name of the model

    Returns:
        AnnData object with results added
    """
    # Create a copy of the AnnData object
    adata = adata.copy()

    # Add posterior samples to AnnData object
    if "posterior_samples" in results:
        for key, value in results["posterior_samples"].items():
            # Convert to numpy array
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()

            # Special handling for alpha, beta, gamma - store in var
            if key in ["alpha", "beta", "gamma"]:
                # If value is multi-dimensional, take the mean across samples
                if value.ndim > 1:
                    value_mean = value.mean(axis=0)
                else:
                    value_mean = value

                # Ensure the value has the right shape for var
                if value_mean.shape[0] != adata.n_vars:
                    # Transpose if needed
                    if value_mean.shape[0] == adata.n_obs:
                        value_mean = value_mean.T
                    else:
                        # Reshape if possible
                        try:
                            value_mean = value_mean.reshape(adata.n_vars)
                        except ValueError:
                            warnings.warn(f"Could not reshape {key} to match var dimensions")
                            # Still store the original in uns
                            adata.uns[f"{model_name}_{key}"] = value
                            continue

                # Store in var dataframe
                adata.var[f"{model_name}_{key}"] = value_mean

            # Store original in uns for all parameters
            adata.uns[f"{model_name}_{key}"] = value

    # Add velocity to AnnData object
    if "velocity" in results:
        velocity = results["velocity"]
        if isinstance(velocity, torch.Tensor):
            velocity = velocity.detach().cpu().numpy()
        # Reshape velocity to match AnnData dimensions
        if velocity.ndim == 1:
            velocity = velocity.reshape(1, -1)
            # Transpose if needed to match AnnData dimensions
            if (
                velocity.shape[1] == adata.n_obs
                and velocity.shape[0] != adata.n_vars
            ):
                velocity = velocity.T
        adata.layers[f"{model_name}_velocity"] = velocity

    # Add uncertainty to AnnData object
    if "uncertainty" in results:
        uncertainty = results["uncertainty"]
        if isinstance(uncertainty, torch.Tensor):
            uncertainty = uncertainty.detach().cpu().numpy()
        adata.var[f"{model_name}_uncertainty"] = uncertainty

    # Add latent time to AnnData object
    if "latent_time" in results:
        latent_time = results["latent_time"]
        if isinstance(latent_time, torch.Tensor):
            latent_time = latent_time.detach().cpu().numpy()
        adata.obs[f"{model_name}_latent_time"] = latent_time

    # Add steady state to AnnData object
    if "u_ss" in results and "s_ss" in results:
        u_ss = results["u_ss"]
        s_ss = results["s_ss"]
        if isinstance(u_ss, torch.Tensor):
            u_ss = u_ss.detach().cpu().numpy()
        if isinstance(s_ss, torch.Tensor):
            s_ss = s_ss.detach().cpu().numpy()
        # Reshape steady states to match AnnData dimensions
        if u_ss.ndim == 1:
            u_ss = u_ss.reshape(1, -1)
            # Transpose if needed to match AnnData dimensions
            if u_ss.shape[1] == adata.n_obs and u_ss.shape[0] != adata.n_vars:
                u_ss = u_ss.T
        if s_ss.ndim == 1:
            s_ss = s_ss.reshape(1, -1)
            # Transpose if needed to match AnnData dimensions
            if s_ss.shape[1] == adata.n_obs and s_ss.shape[0] != adata.n_vars:
                s_ss = s_ss.T
        adata.layers[f"{model_name}_u_ss"] = u_ss
        adata.layers[f"{model_name}_s_ss"] = s_ss

    # Add parameters to AnnData object
    for param in ["alpha", "beta", "gamma"]:
        if param in results:
            value = results[param]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            # If value is multi-dimensional, take the mean across samples
            if value.ndim > 1:
                value = value.mean(axis=0)
            # Ensure the value has the right shape for var
            if value.shape[0] != adata.n_vars:
                # Transpose if needed
                if value.shape[0] == adata.n_obs:
                    value = value.T
                else:
                    # Reshape if possible
                    try:
                        value = value.reshape(adata.n_vars)
                    except ValueError:
                        warnings.warn(f"Could not reshape {param} to match var dimensions")
                        continue
            adata.var[f"{model_name}_{param}"] = value

    return adata
