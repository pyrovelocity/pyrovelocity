"""
Tests for the piecewise activation model factory function.

This module tests the factory function that creates a complete PyroVelocityModel
with piecewise activation components for parameter recovery validation.
"""

import pyro
import pytest
import torch

from pyrovelocity.models.modular.components.dynamics import (
    PiecewiseActivationDynamicsModel,
)
from pyrovelocity.models.modular.components.guides import AutoGuideFactory
from pyrovelocity.models.modular.components.likelihoods import (
    PiecewiseActivationPoissonLikelihoodModel,
)
from pyrovelocity.models.modular.components.priors import (
    PiecewiseActivationPriorModel,
)
from pyrovelocity.models.modular.factory import (
    create_piecewise_activation_model,
)
from pyrovelocity.models.modular.model import PyroVelocityModel


class TestPiecewiseActivationFactory:
    """Test suite for the piecewise activation model factory."""

    def test_create_piecewise_activation_model_returns_pyrovelocity_model(self):
        """Test that the factory returns a PyroVelocityModel instance."""
        model = create_piecewise_activation_model()
        assert isinstance(model, PyroVelocityModel)

    def test_create_piecewise_activation_model_has_correct_components(self):
        """Test that the factory creates a model with the correct component types."""
        model = create_piecewise_activation_model()

        # Check that the model has the expected component types
        assert isinstance(model.dynamics_model, PiecewiseActivationDynamicsModel)
        assert isinstance(model.prior_model, PiecewiseActivationPriorModel)
        assert isinstance(model.likelihood_model, PiecewiseActivationPoissonLikelihoodModel)
        assert isinstance(model.guide_model, AutoGuideFactory)

    def test_create_piecewise_activation_model_components_have_correct_names(self):
        """Test that the components have the expected names."""
        model = create_piecewise_activation_model()

        # Check component names (using actual names from the implementation)
        assert model.dynamics_model.name == "piecewise_dynamics_model"
        assert model.prior_model.name == "piecewise_activation"
        assert model.likelihood_model.name == "piecewise_activation_poisson_likelihood"
        assert model.guide_model.name == "inference_guide"

    def test_create_piecewise_activation_model_forward_pass(self):
        """Test that the model can perform a forward pass with synthetic data."""
        # Clear any existing Pyro state
        pyro.clear_param_store()
        
        model = create_piecewise_activation_model()
        
        # Create simple synthetic data (2 genes, 10 cells)
        n_cells, n_genes = 10, 2
        u_obs = torch.randint(0, 20, (n_cells, n_genes)).float()
        s_obs = torch.randint(0, 20, (n_cells, n_genes)).float()
        
        # Test forward pass
        with torch.no_grad():
            result = model.forward(u_obs=u_obs, s_obs=s_obs)
        
        # Check that the result contains expected keys
        expected_keys = [
            "u_obs", "s_obs", "u_expected", "s_expected",
            "alpha_off", "alpha_on", "gamma_star", "t_on_star", "delta_star", "t_star"
        ]
        
        for key in expected_keys:
            assert key in result, f"Expected key '{key}' not found in result"
        
        # Check tensor shapes
        assert result["u_expected"].shape == (n_cells, n_genes)
        assert result["s_expected"].shape == (n_cells, n_genes)
        assert result["alpha_off"].shape == (n_genes,)
        assert result["alpha_on"].shape == (n_genes,)
        assert result["gamma_star"].shape == (n_genes,)
        assert result["t_on_star"].shape == (n_genes,)
        assert result["delta_star"].shape == (n_genes,)
        assert result["t_star"].shape == (n_cells,)

    def test_create_piecewise_activation_model_guide_creation(self):
        """Test that the model can create a guide for variational inference."""
        # Clear any existing Pyro state
        pyro.clear_param_store()
        
        model = create_piecewise_activation_model()
        
        # Create the guide
        guide = model.guide_model.create_guide(model.forward)
        
        # Check that the guide was created
        assert guide is not None
        assert callable(guide)

    def test_create_piecewise_activation_model_integration(self):
        """Test basic integration between components."""
        # Clear any existing Pyro state
        pyro.clear_param_store()
        
        model = create_piecewise_activation_model()
        
        # Create simple synthetic data
        n_cells, n_genes = 5, 2
        u_obs = torch.randint(1, 10, (n_cells, n_genes)).float()
        s_obs = torch.randint(1, 10, (n_cells, n_genes)).float()
        
        # Test that prior model can sample parameters
        prior_params = model.prior_model.sample_parameters(n_genes=n_genes, n_cells=n_cells)
        
        # Check that prior parameters have correct keys and shapes
        expected_param_keys = [
            "alpha_off", "alpha_on", "gamma_star", "t_on_star", "delta_star", "t_star"
        ]
        
        for key in expected_param_keys:
            assert key in prior_params, f"Expected parameter '{key}' not found"
        
        # Test that dynamics model can process prior parameters
        context = {
            "u_obs": u_obs,
            "s_obs": s_obs,
            **prior_params
        }
        
        dynamics_result = model.dynamics_model.forward(context)
        
        # Check that dynamics model produces expected outputs
        assert "u_expected" in dynamics_result
        assert "s_expected" in dynamics_result
        assert dynamics_result["u_expected"].shape == (n_cells, n_genes)
        assert dynamics_result["s_expected"].shape == (n_cells, n_genes)

    def test_create_piecewise_activation_model_reproducible(self):
        """Test that the factory creates reproducible models."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        pyro.set_rng_seed(42)
        
        model1 = create_piecewise_activation_model()
        
        torch.manual_seed(42)
        pyro.set_rng_seed(42)
        
        model2 = create_piecewise_activation_model()
        
        # Models should have the same component types
        assert type(model1.dynamics_model) == type(model2.dynamics_model)
        assert type(model1.prior_model) == type(model2.prior_model)
        assert type(model1.likelihood_model) == type(model2.likelihood_model)
        assert type(model1.guide_model) == type(model2.guide_model)

    def test_create_piecewise_activation_model_basic_training(self):
        """Test that the model can perform basic training with SVI."""
        # Clear any existing Pyro state
        pyro.clear_param_store()

        model = create_piecewise_activation_model()

        # Create simple synthetic data (2 genes, 10 cells)
        n_cells, n_genes = 10, 2
        u_obs = torch.randint(1, 20, (n_cells, n_genes)).float()
        s_obs = torch.randint(1, 20, (n_cells, n_genes)).float()

        # Create the guide
        guide = model.guide_model.create_guide(model.forward)

        # Set up SVI
        from pyro.infer import SVI, Trace_ELBO
        from pyro.optim import Adam

        optimizer = Adam({"lr": 0.01})
        svi = SVI(model.forward, guide, optimizer, loss=Trace_ELBO())

        # Run a few training steps
        losses = []
        for step in range(5):  # Just a few steps for testing
            loss = svi.step(u_obs=u_obs, s_obs=s_obs)
            losses.append(loss)

        # Check that training ran without errors
        assert len(losses) == 5
        assert all(isinstance(loss, (int, float)) for loss in losses)
        assert all(loss >= 0 for loss in losses)  # Losses should be non-negative
