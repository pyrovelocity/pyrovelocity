"""
Standard prior functions for PyroVelocity JAX/NumPyro implementation.

This module registers standard prior functions for the JAX implementation of PyroVelocity.
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
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
        T_M_star ~ Gamma(α=5.0, β=1.0)                    # Maximum dimensionless time
        boundary_concentration ~ Gamma(α=2.0, β=1.0)      # Temporal boundary control (calibrated)
        
    Cell Temporal Coordinates:
        κ = boundary_concentration
        t_star_normalized ~ Beta(α=1/κ, β=1/κ)           # Beta boundary concentration
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
            - Scalar: T_M_star, boundary_concentration
            - Cell arrays [n_cells]: t_star, lambda_j
            - Gene arrays [num_genes]: R_on, gamma_star, t_on_star, delta_star, U_0i
    """
    if prior_params is None:
        prior_params = {}
    
    # Extract number of cells from prior_params (passed by model)
    n_cells = prior_params.get("n_cells")
    if n_cells is None:
        raise ValueError("n_cells must be provided in prior_params for piecewise activation model")
    
    # Global temporal structure
    T_M_star = numpyro.sample(
        "T_M_star",
        dist.Gamma(
            concentration=prior_params.get("T_M_star_alpha", 5.0),
            rate=prior_params.get("T_M_star_beta", 1.0)
        )
    )
    
    boundary_concentration = numpyro.sample(
        "boundary_concentration", 
        dist.Gamma(
            concentration=prior_params.get("boundary_conc_alpha", 2.0),
            rate=prior_params.get("boundary_conc_beta", 1.0)
        )
    )
    
    # Cell-specific temporal coordinates with boundary concentration
    with numpyro.plate("cells", n_cells):
        # Beta distribution concentration parameter 
        kappa = 1.0 / boundary_concentration
        
        # Normalized temporal coordinates using Beta boundary concentration
        t_star_normalized = numpyro.sample(
            "t_star_normalized",
            dist.Beta(
                concentration1=kappa,
                concentration0=kappa  
            )
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
    with numpyro.plate("genes", num_genes):
        # Activation fold-change (target mean ≈ 2.5)
        R_on = numpyro.sample(
            "R_on",
            dist.LogNormal(
                loc=prior_params.get("R_on_loc", 0.916),    # log(2.5) ≈ 0.916
                scale=prior_params.get("R_on_scale", 0.4)
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
        "boundary_concentration": boundary_concentration,
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
