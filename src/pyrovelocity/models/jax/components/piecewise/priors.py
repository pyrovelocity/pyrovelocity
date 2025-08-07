"""
Standard prior functions for PyroVelocity JAX/NumPyro implementation.

This module registers standard prior functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Dict, Optional

import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.jax.registry import register_prior


@jaxtyped(typechecker=beartype)
def piecewise_activation_prior_function(
    num_genes: int,
    prior_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Float[Array, "..."]]:
    """
    Piecewise activation hierarchical prior function for NumPyro.
    
    This function implements the complete hierarchical prior structure for the 
    piecewise activation RNA velocity model, following the JAX functional pattern
    where dimensions are passed via prior_params. NumPyro handles randomness
    automatically through numpyro.sample() calls.
    
    Mathematical Structure:
    
    Global Time:
        T_M_star ~ Gamma(α=1.0, β=0.25)                   # Skeptical prior: mode=0, mean=4.0
                                                          # Forces data to justify large T_M values
        
    Cell Temporal Coordinates:
        t_star_normalized ~ Uniform(0.0, 1.0)             # Simple uniform prior
        t_star = T_M_star × t_star_normalized             # Scaled dimensionless time
        lambda_j ~ LogNormal(loc=0.0, scale=0.2)         # Cell capture efficiency
        
    Gene Piecewise Activation Parameters:
        α*_off = 1.0 (fixed reference)                   # Eliminates scaling redundancy
        R_on ~ LogNormal(loc=0.916, scale=0.4)           # Activation fold-change (≈2.5, calibrated)
        γ* ~ LogNormal(loc=-0.405, scale=0.5)            # Relative degradation rate
        t*_on ~ Normal(loc=1.5, scale=2.296)             # Activation onset (allows negative)
        δ* ~ LogNormal(loc=0.48, scale=0.464)            # Activation duration
        U_0i ~ LogNormal(loc=2.3, scale=0.4)             # Gene concentration scale

    Args:
        num_genes: Number of genes in the dataset
        prior_params: Dictionary containing n_cells and hyperparameter overrides
        
    Returns:
        Dictionary containing all sampled parameters with appropriate shapes:
            - Scalar: T_M_star
            - Cell arrays [n_cells]: t_star, lambda_j
            - Gene arrays [num_genes]: R_on, gamma_star, t_on_star, delta_star, U_0i
    """
    if prior_params is None:
        prior_params = {}
    
    # Extract number of cells from prior_params (passed by model)
    n_cells = prior_params.get("n_cells")
    if n_cells is None:
        raise ValueError("n_cells must be provided in prior_params for piecewise activation model")
    
    # Global temporal structure with skeptical prior
    # Use a skeptical prior that puts mode at 0 and requires data to justify large T_M values
    T_M_star = numpyro.sample(
        "T_M_star",
        dist.Gamma(
            concentration=prior_params.get("T_M_star_alpha", 1.0),  # Mode at 0 when alpha=1
            rate=prior_params.get("T_M_star_beta", 0.25)  # Mean = alpha/beta = 4.0
        )
    )
    
    # Cell-specific temporal coordinates with simple uniform prior
    with numpyro.plate("cells", n_cells, dim=-2):
        # Simple uniform prior for temporal coordinates
        t_star_normalized = numpyro.sample(
            "t_star_normalized",
            dist.Uniform(0.0, 1.0)
        )
        
        # Scaled dimensionless time coordinates
        t_star = numpyro.deterministic("t_star", T_M_star * t_star_normalized)
        
        # Cell capture efficiency 
        lambda_j = numpyro.sample(
            "lambda_j",
            dist.LogNormal(
                loc=prior_params.get("lambda_loc", 0.0),
                scale=prior_params.get("lambda_scale", 0.2)
            )
        )
    
    # Gene-specific piecewise activation parameters
    with numpyro.plate("genes", num_genes, dim=-1):
        # Activation fold-change (target mean ≈ 2.0, aligned with modular)
        R_on = numpyro.sample(
            "R_on",
            dist.LogNormal(
                loc=prior_params.get("R_on_loc", 0.693),    # log(2.0) ≈ 0.693
                scale=prior_params.get("R_on_scale", 0.35)
            )
        )
        
        # Relative degradation rate γ*
        gamma_star = numpyro.sample(
            "gamma_star", 
            dist.LogNormal(
                loc=prior_params.get("gamma_star_loc", -0.405),
                scale=prior_params.get("gamma_star_scale", 0.5)
            )
        )
        
        # Activation onset time (allows negative for pre-activation)
        t_on_star = numpyro.sample(
            "t_on_star",
            dist.Normal(
                loc=prior_params.get("t_on_star_loc", 1.5),
                scale=prior_params.get("t_on_star_scale", 2.296)
            )
        )
        
        # Activation duration (positive)
        delta_star = numpyro.sample(
            "delta_star",
            dist.LogNormal(
                loc=prior_params.get("delta_star_loc", 0.48),
                scale=prior_params.get("delta_star_scale", 0.464)
            )
        )
        
        # Gene-specific concentration scale
        U_0i = numpyro.sample(
            "U_0i",
            dist.LogNormal(
                loc=prior_params.get("U_0i_loc", 2.3),
                scale=prior_params.get("U_0i_scale", 0.4)
            )
        )
    
    return {
        # Global parameters
        "T_M_star": T_M_star,
        # Cell parameters  
        "t_star": t_star,
        "lambda_j": lambda_j,
        # Gene parameters
        "R_on": R_on,
        "gamma_star": gamma_star, 
        "t_on_star": t_on_star,
        "delta_star": delta_star,
        "U_0i": U_0i
    }


def register_standard_priors():
    """Register standard prior functions."""
    register_prior("piecewise_activation", piecewise_activation_prior_function)
    register_prior("piecewise_activation_prior", piecewise_activation_prior_function)  # Alias for tests
