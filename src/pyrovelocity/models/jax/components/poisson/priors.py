"""
Poisson-only prior functions for PyroVelocity JAX/NumPyro implementation.

This module registers Poisson-only prior functions for the JAX implementation of PyroVelocity.
The Poisson prior model implements a simplified hierarchical prior structure focusing
on library size normalization and basic scaling parameters.
"""

from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from pyrovelocity.models.jax.registry import register_prior


@jaxtyped(typechecker=beartype)
def poisson_prior_function(
    num_genes: int,
    prior_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Float[Array, "..."]]:
    """
    Poisson-only hierarchical prior function for NumPyro.

    This function implements a simplified hierarchical prior structure for the
    Poisson-only RNA velocity model, focusing on library size normalization
    and basic scaling parameters without temporal dynamics.

    Mathematical Structure:

    Cell Library Size Scaling:
        lambda_j ~ LogNormal(loc=0.0, scale=0.5)      # Cell capture efficiency
        t_star ~ Delta(0.5)                           # Constant temporal coordinates (compatibility)

    Gene Concentration Scaling:
        U_0i ~ LogNormal(loc=0.0, scale=0.5)          # Gene-specific scale
        S_0i ~ LogNormal(loc=0.0, scale=0.5)          # Gene-specific scale (spliced)

    Args:
        num_genes: Number of genes in the dataset
        prior_params: Dictionary containing n_cells and hyperparameter overrides

    Returns:
        Dictionary containing all sampled parameters with appropriate shapes:
            - Cell arrays [n_cells]: lambda_j, t_star
            - Gene arrays [num_genes]: U_0i, S_0i
    """
    if prior_params is None:
        prior_params = {}

    # Extract number of cells from prior_params (passed by model)
    n_cells = prior_params.get("n_cells")
    if n_cells is None:
        raise ValueError("n_cells must be provided in prior_params for Poisson model")

    # Cell-specific library size scaling
    with numpyro.plate("cells", n_cells, dim=-2):
        # Cell capture efficiency/library size
        lambda_j = numpyro.sample(
            "lambda_j",
            dist.LogNormal(
                loc=prior_params.get("lambda_loc", 0.0),
                scale=prior_params.get("lambda_scale", 0.5)
            )
        )

        # COMPATIBILITY: Add neutral temporal coordinates for non-temporal model
        # Constant 0.5 indicates no temporal structure (vs random values that suggest structure)
        t_star = numpyro.sample(
            "t_star",
            dist.Delta(0.5)  # numpyro may broadcast across cells dimension
        )

    # Gene-specific concentration scaling
    with numpyro.plate("genes", num_genes, dim=-1):
        # Unspliced RNA concentration scale
        U_0i = numpyro.sample(
            "U_0i",
            dist.LogNormal(
                loc=prior_params.get("U_0i_loc", 0.0),
                scale=prior_params.get("U_0i_scale", 0.5)
            )
        )

        # Spliced RNA concentration scale
        S_0i = numpyro.sample(
            "S_0i",
            dist.LogNormal(
                loc=prior_params.get("S_0i_loc", 0.0),
                scale=prior_params.get("S_0i_scale", 0.5)
            )
        )

    return {
        # Cell parameters
        "lambda_j": lambda_j,
        "t_star": t_star,  # Required for factory layer compatibility
        # Gene parameters
        "U_0i": U_0i,
        "S_0i": S_0i,
    }


def register_poisson_priors():
    """Register Poisson-only prior functions."""
    register_prior("poisson", poisson_prior_function)
    register_prior("poisson_prior", poisson_prior_function)  # Alias
