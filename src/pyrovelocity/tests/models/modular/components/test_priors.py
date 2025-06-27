"""Tests for prior models."""

import pytest

from pyrovelocity.models.modular.components.priors import (
    PiecewiseActivationPriorModel,
)
from pyrovelocity.models.modular.registry import PriorModelRegistry


@pytest.fixture(scope="module", autouse=True)
def register_prior_models():
    """Register prior models for testing."""
    # Save original registry state
    original_registry = dict(PriorModelRegistry._registry)

    # Clear registry and register test components
    PriorModelRegistry.clear()
    PriorModelRegistry._registry["piecewise_activation"] = PiecewiseActivationPriorModel

    yield

    # Restore original registry state
    PriorModelRegistry._registry = original_registry


import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.nn import PyroModule

from pyrovelocity.models.modular.components.priors import (
    PiecewiseActivationPriorModel,
)
from pyrovelocity.models.modular.registry import PriorModelRegistry


def test_registry_create():
    """Test creating models through the registry."""
    # Create models through the registry
    piecewise_model = PriorModelRegistry.create("piecewise_activation")

    # Check that the models are of the correct type
    assert isinstance(piecewise_model, PiecewiseActivationPriorModel)

    # Check that the models have the correct names
    assert piecewise_model.name == "piecewise_activation"

    # Test that registry only contains piecewise activation models
    available_models = PriorModelRegistry.list_available()
    assert "piecewise_activation" in available_models
    assert len(available_models) == 1


class TestPiecewiseActivationPriorModel:
    """Test suite for PiecewiseActivationPriorModel."""

    @pytest.fixture
    def prior_model(self):
        """Create a PiecewiseActivationPriorModel instance for testing."""
        return PiecewiseActivationPriorModel()

    @pytest.fixture
    def test_context(self):
        """Create a test context with synthetic data."""
        n_cells = 50
        n_genes = 2

        # Create synthetic observed data
        u_obs = torch.ones(n_cells, n_genes) * 10.0
        s_obs = torch.ones(n_cells, n_genes) * 5.0

        return {
            "u_obs": u_obs,
            "s_obs": s_obs,
            "include_prior": True,
        }

    def test_initialization(self, prior_model):
        """Test that the prior model initializes correctly."""
        assert prior_model.name == "piecewise_activation"
        assert hasattr(prior_model, "T_M_alpha")
        assert hasattr(prior_model, "T_M_beta")
        assert hasattr(prior_model, "R_on_loc")
        assert hasattr(prior_model, "R_on_scale")
        assert hasattr(prior_model, "gamma_star_loc")
        assert hasattr(prior_model, "gamma_star_scale")
        assert hasattr(prior_model, "t_on_star_loc")
        assert hasattr(prior_model, "t_on_star_scale")
        assert hasattr(prior_model, "delta_star_loc")
        assert hasattr(prior_model, "delta_star_scale")
        assert hasattr(prior_model, "lambda_loc")
        assert hasattr(prior_model, "lambda_scale")

    def test_sample_parameters_shapes(self, prior_model):
        """Test that sample_parameters returns correct shapes."""
        n_genes = 3
        n_cells = 20

        params = prior_model.sample_parameters(n_genes=n_genes, n_cells=n_cells)

        # Check hierarchical time parameters (scalars)
        assert params["T_M_star"].shape == torch.Size([])
        assert params["boundary_concentration"].shape == torch.Size([])
        assert params["t_star_normalized"].shape == torch.Size([n_cells])

        # Check cell-specific time (n_cells,)
        assert params["t_star"].shape == torch.Size([n_cells])

        # Check gene-specific parameters (n_genes,)
        assert params["alpha_off"].shape == torch.Size([n_genes])
        assert params["alpha_on"].shape == torch.Size([n_genes])
        assert params["R_on"].shape == torch.Size([n_genes])
        assert params["gamma_star"].shape == torch.Size([n_genes])
        assert params["t_on_star"].shape == torch.Size([n_genes])
        assert params["delta_star"].shape == torch.Size([n_genes])

        # Check cell-specific capture efficiency (n_cells,)
        assert params["lambda_j"].shape == torch.Size([n_cells])

    def test_forward_method_with_context(self, prior_model, test_context):
        """Test the forward method with a proper context."""
        pyro.clear_param_store()

        with pyro.poutine.trace() as trace:
            result_context = prior_model.forward(test_context)

        # Check that all expected parameters are in the result
        expected_params = [
            "T_M_star", "boundary_concentration", "t_star", "t_star_normalized",
            "alpha_off", "alpha_on", "R_on", "gamma_star", "t_on_star", "delta_star",
            "lambda_j"
        ]

        for param in expected_params:
            assert param in result_context

        # Check that original context is preserved
        assert "u_obs" in result_context
        assert "s_obs" in result_context

        # Check that Pyro trace contains the expected sample sites
        trace_sites = list(trace.trace.nodes.keys())

        # For PiecewiseActivationPriorModel, t_star is computed deterministically
        expected_trace_params = [
            "T_M_star", "boundary_concentration", "t_star_normalized",
            "R_on", "gamma_star", "t_on_star", "delta_star",
            "lambda_j"
        ]

        for param in expected_trace_params:
            assert param in trace_sites, f"Expected parameter '{param}' not found in trace sites: {trace_sites}"

    def test_integration_with_dynamics_model(self, prior_model):
        """Test that sampled parameters work with the dynamics model."""
        from pyrovelocity.models.modular.components.dynamics import (
            PiecewiseActivationDynamicsModel,
        )

        # Create dynamics model
        dynamics_model = PiecewiseActivationDynamicsModel()

        # Sample parameters
        params = prior_model.sample_parameters(n_genes=2, n_cells=10)

        # Create context for dynamics model
        u_obs = torch.ones(10, 2) * 10.0
        s_obs = torch.ones(10, 2) * 5.0

        context = {
            "u_obs": u_obs,
            "s_obs": s_obs,
            **params
        }

        # Test that dynamics model can process the parameters
        try:
            result = dynamics_model.forward(context)
            assert "u_expected" in result
            assert "s_expected" in result
            assert "ut" in result
            assert "st" in result
        except Exception as e:
            pytest.fail(f"Dynamics model failed with prior parameters: {e}")


def test_piecewise_activation_prior_model_registration():
    """Test that PiecewiseActivationPriorModel is properly registered."""
    model_class = PriorModelRegistry.get("piecewise_activation")
    assert model_class == PiecewiseActivationPriorModel
    assert model_class.name == "piecewise_activation"
    assert "piecewise_activation" in PriorModelRegistry.list_available()
