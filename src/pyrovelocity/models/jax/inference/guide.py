"""
Variational guide implementations for PyroVelocity JAX/NumPyro implementation.

This module contains variational guide implementations, including:

- auto_normal_guide: AutoNormal guide for variational inference
- auto_delta_guide: AutoDelta guide for MAP inference
- custom_guide: Custom guide for specialized inference
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, Float, PyTree
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoGuide,
    AutoNormal,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
    AutoBNAFNormal,
)


@beartype
def auto_normal_guide(
    model: Callable,
    init_loc_fn: Optional[Callable] = None,
) -> AutoNormal:
    """Create an AutoNormal guide for variational inference.

    Args:
        model: NumPyro model function
        init_loc_fn: Function to initialize location parameters

    Returns:
        AutoNormal guide
    """
    # Define a custom initialization function if none is provided
    if init_loc_fn is None:
        # Use a simpler initialization approach with init_to_median
        return numpyro.infer.autoguide.AutoNormal(
            model, init_loc_fn=numpyro.infer.autoguide.init_to_median
        )

        # Create AutoNormal guide with robust initialization
        return numpyro.infer.autoguide.AutoNormal(
            model, init_loc_fn=robust_init_fn
        )
    else:
        # Create AutoNormal guide with custom initialization
        return numpyro.infer.autoguide.AutoNormal(
            model, init_loc_fn=init_loc_fn
        )


@beartype
def auto_delta_guide(
    model: Callable,
    init_loc_fn: Optional[Callable] = None,
) -> AutoDelta:
    """Create an AutoDelta guide for MAP inference.

    Args:
        model: NumPyro model function
        init_loc_fn: Function to initialize location parameters

    Returns:
        AutoDelta guide
    """
    # Use default initialization if not provided
    if init_loc_fn is None:
        # Create AutoDelta guide with default initialization
        return numpyro.infer.autoguide.AutoDelta(model)
    else:
        # Create AutoDelta guide with custom initialization
        return numpyro.infer.autoguide.AutoDelta(model, init_loc_fn=init_loc_fn)


@beartype
def auto_diagonal_normal_guide(
    model: Callable,
    init_loc_fn: Optional[Callable] = None,
) -> AutoDiagonalNormal:
    """Create an AutoDiagonalNormal guide for variational inference.

    Args:
        model: NumPyro model function
        init_loc_fn: Function to initialize location parameters

    Returns:
        AutoDiagonalNormal guide
    """
    # Use default initialization if not provided
    if init_loc_fn is None:
        return numpyro.infer.autoguide.AutoDiagonalNormal(
            model, init_loc_fn=numpyro.infer.autoguide.init_to_median
        )
    else:
        return numpyro.infer.autoguide.AutoDiagonalNormal(
            model, init_loc_fn=init_loc_fn
        )


@beartype
def auto_multivariate_normal_guide(
    model: Callable,
    init_loc_fn: Optional[Callable] = None,
) -> AutoMultivariateNormal:
    """Create an AutoMultivariateNormal guide for variational inference.

    Args:
        model: NumPyro model function
        init_loc_fn: Function to initialize location parameters

    Returns:
        AutoMultivariateNormal guide
    """
    # Use default initialization if not provided
    if init_loc_fn is None:
        return numpyro.infer.autoguide.AutoMultivariateNormal(
            model, init_loc_fn=numpyro.infer.autoguide.init_to_median
        )
    else:
        return numpyro.infer.autoguide.AutoMultivariateNormal(
            model, init_loc_fn=init_loc_fn
        )


@beartype
def auto_lowrank_multivariate_normal_guide(
    model: Callable,
    init_loc_fn: Optional[Callable] = None,
    rank: Optional[int] = None,
) -> AutoLowRankMultivariateNormal:
    """Create an AutoLowRankMultivariateNormal guide for variational inference.

    Args:
        model: NumPyro model function
        init_loc_fn: Function to initialize location parameters
        rank: Rank of the low-rank approximation (None for automatic selection based on sqrt(latent_dim))

    Returns:
        AutoLowRankMultivariateNormal guide
    """
    # Use default initialization if not provided
    if init_loc_fn is None:
        return numpyro.infer.autoguide.AutoLowRankMultivariateNormal(
            model, init_loc_fn=numpyro.infer.autoguide.init_to_median, rank=rank
        )
    else:
        return numpyro.infer.autoguide.AutoLowRankMultivariateNormal(
            model, init_loc_fn=init_loc_fn, rank=rank
        )


@beartype
def auto_iaf_normal_guide(
    model: Callable,
    num_flows: int = 1,  # Reduced from 3 to 1 for stability
    hidden_dims: Optional[List[int]] = None,
    init_loc_fn: Optional[Callable] = None,
) -> AutoIAFNormal:
    """Create an AutoIAFNormal guide with normalizing flows for flexible posterior approximation.
    
    This guide uses Inverse Autoregressive Flows (IAF) to capture complex posterior
    geometries that cannot be represented by simple Gaussian distributions.
    
    Args:
        model: NumPyro model function
        num_flows: Number of normalizing flow transformations (default: 3)
        hidden_dims: Hidden layer dimensions for flow transformations
        init_loc_fn: Function to initialize location parameters
        
    Returns:
        AutoIAFNormal guide
    """
    # Note: IAF will use default hidden_dims if None is provided
    # Default is [latent_dim, latent_dim] which ensures compatibility
    # If we provide custom hidden_dims, they must all be >= latent_dim
        
    # Use default initialization if not provided
    if init_loc_fn is None:
        return numpyro.infer.autoguide.AutoIAFNormal(
            model, 
            num_flows=num_flows,
            hidden_dims=hidden_dims,
            init_loc_fn=numpyro.infer.autoguide.init_to_median
        )
    else:
        return numpyro.infer.autoguide.AutoIAFNormal(
            model,
            num_flows=num_flows,
            hidden_dims=hidden_dims,
            init_loc_fn=init_loc_fn
        )


@beartype
def auto_bnaf_normal_guide(
    model: Callable,
    num_flows: int = 1,
    hidden_factors: Optional[List[int]] = None,
    init_loc_fn: Optional[Callable] = None,
) -> AutoBNAFNormal:
    """Create an AutoBNAFNormal guide with Block Neural Autoregressive Flows.
    
    This guide uses Block Neural Autoregressive Flows (BNAF) which are more
    flexible than IAF but computationally more expensive.
    
    Args:
        model: NumPyro model function
        num_flows: Number of BNAF transformations (default: 1)
        hidden_factors: Hidden layer factors for flow transformations
        init_loc_fn: Function to initialize location parameters
        
    Returns:
        AutoBNAFNormal guide
    """
    if hidden_factors is None:
        hidden_factors = [8]
        
    # Use default initialization if not provided
    if init_loc_fn is None:
        return numpyro.infer.autoguide.AutoBNAFNormal(
            model,
            num_flows=num_flows,
            hidden_factors=hidden_factors,
            init_loc_fn=numpyro.infer.autoguide.init_to_median
        )
    else:
        return numpyro.infer.autoguide.AutoBNAFNormal(
            model,
            num_flows=num_flows,
            hidden_factors=hidden_factors,
            init_loc_fn=init_loc_fn
        )


@beartype
def custom_guide(
    model: Callable,
    init_params: Optional[Dict[str, jnp.ndarray]] = None,
) -> Callable:
    """Create a custom guide for specialized inference.

    Args:
        model: NumPyro model function
        init_params: Initial parameter values

    Returns:
        Custom guide function
    """

    # Define a custom guide function
    def guide_fn(*args, **kwargs):
        # Get model parameters
        # Handle the case when no data is provided (e.g., during testing)
        data = kwargs.get("u_obs", args[0] if args else jnp.ones((10, 1)))
        if len(data.shape) < 2:
            # If data is just a key, use default dimensions
            num_cells, num_genes = 10, 1
        else:
            num_cells, num_genes = data.shape

        # Sample alpha, beta, gamma with Normal distributions
        with numpyro.plate("genes", num_genes, dim=-1):
            # Use initial values if provided, otherwise use defaults
            if init_params is not None and "alpha_loc" in init_params:
                alpha_loc = init_params["alpha_loc"]
                alpha_scale = init_params.get(
                    "alpha_scale", jnp.ones_like(alpha_loc) * 0.1
                )
            else:
                alpha_loc = jnp.zeros(num_genes)
                alpha_scale = jnp.ones(num_genes) * 0.1

            if init_params is not None and "beta_loc" in init_params:
                beta_loc = init_params["beta_loc"]
                beta_scale = init_params.get(
                    "beta_scale", jnp.ones_like(beta_loc) * 0.1
                )
            else:
                beta_loc = jnp.zeros(num_genes)
                beta_scale = jnp.ones(num_genes) * 0.1

            if init_params is not None and "gamma_loc" in init_params:
                gamma_loc = init_params["gamma_loc"]
                gamma_scale = init_params.get(
                    "gamma_scale", jnp.ones_like(gamma_loc) * 0.1
                )
            else:
                gamma_loc = jnp.zeros(num_genes)
                gamma_scale = jnp.ones(num_genes) * 0.1

            # Sample parameters using amortized variational distributions
            alpha = numpyro.sample(
                "alpha", dist.LogNormal(alpha_loc, alpha_scale)
            )
            beta = numpyro.sample("beta", dist.LogNormal(beta_loc, beta_scale))
            gamma = numpyro.sample(
                "gamma", dist.LogNormal(gamma_loc, gamma_scale)
            )

        # Sample latent time if needed
        if "tau" in model.__code__.co_varnames:
            with numpyro.plate("cells", num_cells, dim=-2):
                # Use initial values if provided, otherwise use defaults
                if init_params is not None and "tau_loc" in init_params:
                    tau_loc = init_params["tau_loc"]
                    tau_scale = init_params.get(
                        "tau_scale", jnp.ones_like(tau_loc) * 0.1
                    )
                else:
                    tau_loc = jnp.zeros(num_cells)
                    tau_scale = jnp.ones(num_cells) * 0.1

                # Sample latent time
                numpyro.sample("tau", dist.Normal(tau_loc, tau_scale))

    return guide_fn


@beartype
def create_guide(
    model: Callable, guide_type: str = "auto_normal", **kwargs
) -> Union[AutoGuide, Callable]:
    """Create a guide based on the specified type.

    Args:
        model: NumPyro model function
        guide_type: Type of guide ("auto_normal", "auto_diagonal_normal", 
                   "auto_multivariate_normal", "auto_lowrank_multivariate_normal",
                   "auto_iaf_normal", "auto_bnaf_normal", "auto_delta", or "custom")
        **kwargs: Additional guide parameters

    Returns:
        Guide object or function
    """
    if guide_type == "auto_normal":
        return auto_normal_guide(model, **kwargs)
    elif guide_type == "auto_diagonal_normal":
        return auto_diagonal_normal_guide(model, **kwargs)
    elif guide_type == "auto_multivariate_normal":
        return auto_multivariate_normal_guide(model, **kwargs)
    elif guide_type == "auto_lowrank_multivariate_normal":
        return auto_lowrank_multivariate_normal_guide(model, **kwargs)
    elif guide_type == "auto_iaf_normal":
        return auto_iaf_normal_guide(model, **kwargs)
    elif guide_type == "auto_bnaf_normal":
        return auto_bnaf_normal_guide(model, **kwargs)
    elif guide_type == "auto_delta":
        return auto_delta_guide(model, **kwargs)
    elif guide_type == "custom":
        return custom_guide(model, **kwargs)
    else:
        raise ValueError(f"Unknown guide type: {guide_type}")
