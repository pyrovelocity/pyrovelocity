"""
Fixtures for BDD testing of PyroVelocity's modular implementation.

This module provides fixtures that are specific to BDD testing, complementing
the fixtures defined in the main conftest.py file.
"""

import numpy as np
import pyro
import pytest
import torch
from anndata import AnnData

from pyrovelocity.models.modular.components import (
    AutoGuideFactory,
    PiecewiseActivationDynamicsModel,
    PiecewiseActivationPoissonLikelihoodModel,
    PiecewiseActivationPriorModel,
)
from pyrovelocity.models.modular.model import PyroVelocityModel


@pytest.fixture
def bdd_simple_data():
    """Create simple data for BDD testing."""
    # Create random data for testing
    torch.manual_seed(42)
    n_cells = 10
    n_genes = 5

    # Generate random data
    u_obs = torch.abs(torch.randn((n_cells, n_genes)))
    s_obs = torch.abs(torch.randn((n_cells, n_genes)))

    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture
def bdd_model_parameters():
    """Create legacy model parameters for BDD testing."""
    torch.manual_seed(42)
    n_genes = 5

    # Generate random parameters for legacy models
    alpha = torch.abs(torch.randn(n_genes))
    beta = torch.abs(torch.randn(n_genes))
    gamma = torch.abs(torch.randn(n_genes))

    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }


@pytest.fixture
def bdd_piecewise_model_parameters():
    """Create piecewise activation model parameters for BDD testing."""
    torch.manual_seed(42)
    n_genes = 5

    # Generate random parameters for piecewise activation models
    alpha_off = torch.abs(torch.randn(n_genes)) * 0.5 + 0.1  # [0.1, 0.6]
    alpha_on = torch.abs(torch.randn(n_genes)) * 2.0 + 1.0   # [1.0, 3.0]
    gamma_star = torch.abs(torch.randn(n_genes)) * 1.0 + 0.5 # [0.5, 1.5]
    t_on_star = torch.abs(torch.randn(n_genes)) * 0.3 + 0.2  # [0.2, 0.5]
    delta_star = torch.abs(torch.randn(n_genes)) * 0.4 + 0.3 # [0.3, 0.7]
    t_star = torch.abs(torch.randn(n_genes)) * 0.5 + 0.1     # [0.1, 0.6]

    return {
        "alpha_off": alpha_off,
        "alpha_on": alpha_on,
        "gamma_star": gamma_star,
        "t_on_star": t_on_star,
        "delta_star": delta_star,
        "t_star": t_star,
    }


@pytest.fixture
def bdd_standard_dynamics_model():
    """Create a PiecewiseActivationDynamicsModel for BDD testing."""
    return PiecewiseActivationDynamicsModel()






@pytest.fixture
def bdd_poisson_likelihood_model():
    """Create a PiecewiseActivationPoissonLikelihoodModel for BDD testing."""
    return PiecewiseActivationPoissonLikelihoodModel()






@pytest.fixture
def bdd_auto_guide_factory():
    """Create an AutoGuideFactory for BDD testing."""
    return AutoGuideFactory(guide_type="AutoNormal")




@pytest.fixture
def bdd_piecewise_activation_prior_model():
    """Create a PiecewiseActivationPriorModel for BDD testing."""
    return PiecewiseActivationPriorModel()


@pytest.fixture
def bdd_piecewise_pyro_velocity_model(
    bdd_standard_dynamics_model,
    bdd_piecewise_activation_prior_model,
    bdd_poisson_likelihood_model,
    bdd_auto_guide_factory,
):
    """Create a PyroVelocityModel for BDD testing using piecewise activation components."""
    return PyroVelocityModel(
        dynamics_model=bdd_standard_dynamics_model,
        prior_model=bdd_piecewise_activation_prior_model,
        likelihood_model=bdd_poisson_likelihood_model,
        guide_model=bdd_auto_guide_factory,
    )




@pytest.fixture
def bdd_anndata():
    """Create a simple AnnData object for BDD testing."""
    # Create random data
    np.random.seed(42)
    n_cells = 10
    n_genes = 5
    
    X = np.random.rand(n_cells, n_genes)
    layers = {
        "spliced": np.random.rand(n_cells, n_genes),
        "unspliced": np.random.rand(n_cells, n_genes),
    }
    
    # Create AnnData object
    adata = AnnData(X=X, layers=layers)
    
    return adata


@pytest.fixture(autouse=True)
def clear_pyro_param_store():
    """Clear Pyro's parameter store before and after each test."""
    pyro.clear_param_store()
    yield
    pyro.clear_param_store()
