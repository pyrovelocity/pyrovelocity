"""
Component integration tests for the PyroVelocity modular implementation.

This module tests the interactions between different components in the
modular architecture, ensuring they work correctly together.
"""

import numpy as np
import pyro
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyrovelocity.models.modular.components.dynamics import (
    PiecewiseActivationDynamicsModel,
)
from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
)
from pyrovelocity.models.modular.components.likelihoods import (
    PiecewiseActivationPoissonLikelihoodModel,
)
from pyrovelocity.models.modular.components.priors import PiecewiseActivationPriorModel
from pyrovelocity.models.modular.model import ModelState, PyroVelocityModel
from pyrovelocity.models.modular.registry import register_standard_components


@pytest.fixture
def simple_data():
    """Create a simple dataset for testing."""
    # Create a small dataset with random values
    n_cells = 20
    n_genes = 5

    u_obs = torch.poisson(torch.rand(n_cells, n_genes) * 5).float()
    s_obs = torch.poisson(torch.rand(n_cells, n_genes) * 5).float()

    # Create a DataLoader for batch processing
    dataset = TensorDataset(u_obs, s_obs)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
        "dataloader": dataloader,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture(scope="module", autouse=True)
def setup_registries():
    """Register standard components in registries before running tests."""
    register_standard_components()
    return


class TestComponentIntegration:
    """Tests for component interactions."""

    def test_dynamics_prior_integration(self):
        """Test that dynamics model and prior model work together."""
        # Create component models - use piecewise activation components
        dynamics_model = PiecewiseActivationDynamicsModel()
        prior_model = PiecewiseActivationPriorModel()

        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Sample parameters from prior
        n_genes = 5
        prior_params = prior_model.sample_parameters(n_genes=n_genes)

        # Verify that prior produces expected parameters for piecewise activation
        assert "alpha_off" in prior_params
        assert "gamma_star" in prior_params
        assert "U_0i" in prior_params  # Characteristic concentration scale
        assert "lambda_j" in prior_params  # Capture efficiency

        # Check shape of prior samples
        assert prior_params["alpha_off"].shape == (n_genes,)
        assert prior_params["gamma_star"].shape == (n_genes,)

        # Test that dynamics model can compute steady state with prior parameters
        u_ss, s_ss = dynamics_model.steady_state(
            prior_params["alpha_off"],
            prior_params["gamma_star"],
        )

        # Check shapes of steady state results
        assert u_ss.shape == (n_genes,)
        assert s_ss.shape == (n_genes,)

        # Check that steady state values are positive (biological constraint)
        assert torch.all(u_ss > 0)
        assert torch.all(s_ss > 0)

        # Test that dynamics model can be used in a forward pass context
        # Create a minimal context for testing forward pass
        n_cells = 10
        context = {
            "u_obs": torch.poisson(torch.rand(n_cells, n_genes) * 5).float(),
            "s_obs": torch.poisson(torch.rand(n_cells, n_genes) * 5).float(),
            "R_on": prior_params["R_on"],
            "gamma_star": prior_params["gamma_star"],
            "t_on_star": prior_params["t_on_star"],
            "delta_star": prior_params["delta_star"],
            "t_star": prior_params["t_star"][:n_cells],  # Use only n_cells from t_star
        }

        # Test forward pass
        result_context = dynamics_model.forward(context)

        # Check that forward pass produces expected outputs
        assert "u_expected" in result_context
        assert "s_expected" in result_context
        # Allow for sample dimension in the results
        assert result_context["u_expected"].shape[-2:] == (n_cells, n_genes)
        assert result_context["s_expected"].shape[-2:] == (n_cells, n_genes)

    def test_dynamics_likelihood_integration(self, simple_data):
        """Test that dynamics model and likelihood model work together."""
        # Create component models
        dynamics_model = PiecewiseActivationDynamicsModel()
        likelihood_model = PiecewiseActivationPoissonLikelihoodModel()

        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Create parameters for piecewise activation model
        n_genes = simple_data["n_genes"]
        alpha_off = torch.rand(n_genes) * 0.5 + 0.1  # [0.1, 0.6] - basal transcription
        gamma_star = torch.rand(n_genes) * 1 + 0.2  # [0.2, 1.2] - relative degradation

        # Compute steady state using piecewise parameters
        u_ss, s_ss = dynamics_model.steady_state(alpha_off, gamma_star)

        # Create expected values using steady state - use a small batch size to match test
        batch_size = 10  # Use a smaller batch size to test
        u_expected = u_ss.unsqueeze(0).expand(batch_size, -1)
        s_expected = s_ss.unsqueeze(0).expand(batch_size, -1)

        # Create subset of observations for testing
        u_obs = simple_data["u_obs"][:batch_size]
        s_obs = simple_data["s_obs"][:batch_size]

        # Use log_prob directly instead of the __call__ interface that uses plates
        # Convert numpy arrays to torch tensors to satisfy jaxtyping constraints
        u_log_prob = likelihood_model.log_prob(
            observations=u_obs,  # Keep as torch tensor
            predictions=u_expected,  # Keep as torch tensor
        )
        s_log_prob = likelihood_model.log_prob(
            observations=s_obs,  # Keep as torch tensor
            predictions=s_expected,  # Keep as torch tensor
        )

        # Verify that log probabilities have the right shape
        assert u_log_prob.shape == (batch_size,)
        assert s_log_prob.shape == (batch_size,)

    def test_full_model_integration(self, simple_data):
        """Test that all components work together in the full model."""
        # Create compatible components - use piecewise activation combination
        dynamics_model = PiecewiseActivationDynamicsModel()
        prior_model = PiecewiseActivationPriorModel()
        likelihood_model = PiecewiseActivationPoissonLikelihoodModel()
        guide_model = AutoGuideFactory()

        # Create the full model
        model = PyroVelocityModel(
            dynamics_model=dynamics_model,
            prior_model=prior_model,
            likelihood_model=likelihood_model,
            guide_model=guide_model,
        )

        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Create a simple dataset with fixed dimensions
        n_cells = 5
        n_genes = 5
        u_batch = torch.rand(n_cells, n_genes) * 5
        s_batch = torch.rand(n_cells, n_genes) * 5

        # Create parameters for the model - compatible with PiecewiseActivationDynamicsModel
        alpha_off = torch.rand(n_genes) * 0.5 + 0.1  # [0.1, 0.6] - basal transcription
        gamma_star = torch.rand(n_genes) * 1 + 0.2  # [0.2, 1.2] - relative degradation

        # Compute steady state values using correct parameter interface
        u_ss, s_ss = dynamics_model.steady_state(alpha_off, gamma_star)

        # Expand to match batch size
        u_expected = u_ss.unsqueeze(0).expand(n_cells, -1)
        s_expected = s_ss.unsqueeze(0).expand(n_cells, -1)

        # Create a context dictionary with the data
        context = {
            "u_obs": u_batch,
            "s_obs": s_batch,
            "u_expected": u_expected,
            "s_expected": s_expected,
            "alpha_off": alpha_off,
            "gamma_star": gamma_star,
        }

        # Call the model directly with the context
        result = model.forward(**context)

        # Verify that the model produced expected outputs
        assert "u_dist" in result
        assert "s_dist" in result

    def test_different_guides(self, simple_data):
        """Test that different guide implementations work with the same model."""
        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

        # Create compatible common components - use piecewise activation combination
        dynamics_model = PiecewiseActivationDynamicsModel()
        prior_model = PiecewiseActivationPriorModel()
        likelihood_model = PiecewiseActivationPoissonLikelihoodModel()

        # Create a simple dataset with fixed dimensions
        n_cells = 5
        n_genes = 5
        u_batch = torch.rand(n_cells, n_genes) * 5
        s_batch = torch.rand(n_cells, n_genes) * 5

        # Create parameters for the model - compatible with PiecewiseActivationDynamicsModel
        alpha_off = torch.rand(n_genes) * 0.5 + 0.1  # [0.1, 0.6] - basal transcription
        gamma_star = torch.rand(n_genes) * 1 + 0.2  # [0.2, 1.2] - relative degradation

        # Compute steady state values using correct parameter interface
        u_ss, s_ss = dynamics_model.steady_state(alpha_off, gamma_star)

        # Expand to match batch size
        u_expected = u_ss.unsqueeze(0).expand(n_cells, -1)
        s_expected = s_ss.unsqueeze(0).expand(n_cells, -1)

        # Create a context dictionary with the data
        context = {
            "u_obs": u_batch,
            "s_obs": s_batch,
            "u_expected": u_expected,
            "s_expected": s_expected,
            "alpha_off": alpha_off,
            "gamma_star": gamma_star,
        }

        # Create different guide types to test - use compatible guides
        guide_types = [
            AutoGuideFactory(guide_type="AutoNormal"),
            AutoGuideFactory(guide_type="AutoDelta"),
        ]

        # Test each guide
        for guide_model in guide_types:
            # Create the full model
            model = PyroVelocityModel(
                dynamics_model=dynamics_model,
                prior_model=prior_model,
                likelihood_model=likelihood_model,
                guide_model=guide_model,
            )

            # Call the model directly with the context
            result = model.forward(**context)

            # Verify that the model produced expected outputs
            assert "u_dist" in result
            assert "s_dist" in result
