"""
Shared fixtures for model selection and comparison tests.

This module provides fixtures that are shared between test_comparison.py and test_selection.py,
facilitating the testing of model selection and comparison functionality in the PyroVelocity framework.
"""

from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from pyrovelocity.models.modular.interfaces import (
    DynamicsModel,
    InferenceGuide,
    LikelihoodModel,
    ObservationModel,
    PriorModel,
)
from pyrovelocity.models.modular.model import PyroVelocityModel
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    InferenceGuideRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    PriorModelRegistry,
    register_standard_components,
)


# Mock implementations for testing
class MockDynamicsModel:
    """Mock dynamics model for testing."""

    def __init__(self, name="mock_dynamics_model"):
        self.name = name
        self.state = {}

    def forward(self, context):
        """Forward pass that adds predictions to the context."""
        # Extract parameters from context
        parameters = context.get("parameters", {})

        # Add predictions to context
        context["predictions"] = torch.ones((10, 5)) * parameters.get(
            "alpha", 1.0
        )

        return context

    def _forward_impl(
        self,
        u: torch.Tensor,
        s: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        scaling: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation of the forward method."""
        # Simple implementation for testing
        u_pred = torch.ones_like(u) * alpha
        s_pred = torch.ones_like(s) * alpha
        return u_pred, s_pred

    def _steady_state_impl(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        scaling: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation of the steady state calculation for testing."""
        # For testing, just return ones
        u_ss = torch.ones_like(alpha)
        s_ss = torch.ones_like(alpha)

        # Apply scaling if provided
        if scaling is not None:
            u_ss = u_ss * scaling
            s_ss = s_ss * scaling

        return u_ss, s_ss

    def _predict_future_states_impl(
        self,
        current_state: Tuple[torch.Tensor, torch.Tensor],
        time_delta: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        scaling: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Implementation of the predict_future_states method."""
        # Simple implementation for testing
        u_current, s_current = current_state
        u_future = u_current + alpha * time_delta
        s_future = s_current + beta * time_delta
        return u_future, s_future

    def compute_velocity(
        self,
        ut: torch.Tensor,
        st: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute RNA velocity for testing."""
        # Simple velocity computation for testing
        # velocity = alpha - beta * ut
        return alpha - beta * ut


class MockLikelihoodModel:
    """Mock likelihood model for testing."""

    def __init__(self, name="mock_likelihood_model", use_observed_lib_size=True):
        self.name = name
        self.use_observed_lib_size = use_observed_lib_size
        self.state = {}

    def forward(self, context):
        """Forward pass that just returns the input context."""
        return context

    def _log_prob_impl(self, observations, predictions, scale_factors=None):
        """Implementation of log probability calculation."""
        # Simple log probability calculation for testing
        # Make sure we return a PyTorch tensor with shape [batch_size]
        return -torch.sum((observations - predictions) ** 2, dim=1)

    def _sample_impl(self, predictions, scale_factors=None):
        """Implementation of sampling."""
        # Return the predictions with some noise for testing
        return predictions + torch.randn_like(predictions) * 0.1


class MockPriorModel:
    """Mock prior model for testing."""

    def __init__(self, name="mock_prior_model"):
        self.name = name
        self.state = {}

    def forward(self, context):
        """Forward pass that adds parameters to the context."""
        # Add parameters to context
        context["alpha"] = 1.0
        context["beta"] = 1.0
        context["gamma"] = 1.0

        return context

    def _register_priors_impl(self, prefix=""):
        """Implementation of prior registration."""
        # No-op for testing
        pass

    def _sample_parameters_impl(self, prefix="", n_genes=None, **kwargs):
        """Implementation of parameter sampling."""
        # Return empty dict for testing
        return {}


class MockObservationModel:
    """Mock observation model for testing."""

    def __init__(self, name="mock_observation_model"):
        self.name = name
        self.state = {}

    def forward(self, context):
        """Forward pass that adds observations to the context."""
        # Add observations to context
        context["observations"] = context.get("x", torch.ones((10, 5)))

        return context

    def _forward_impl(self, u_obs=None, s_obs=None, **kwargs):
        """Implementation of the forward method."""
        # Create a context dictionary
        context = {}

        # Add u and s to context
        if u_obs is not None:
            context["u"] = u_obs
        if s_obs is not None:
            context["s"] = s_obs

        return context

    def _prepare_data_impl(self, adata, **kwargs):
        """Implementation of data preparation."""
        # Return empty dict for testing
        return {}

    def _create_dataloaders_impl(self, data, **kwargs):
        """Implementation of dataloader creation."""
        # Return empty dict for testing
        return {}

    def _preprocess_batch_impl(self, batch):
        """Implementation of batch preprocessing."""
        # Return the input batch for testing
        return batch


class MockGuideModel:
    """Mock guide model for testing."""

    def __init__(self, name="mock_guide_model"):
        self.name = name
        self.state = {}

    def forward(self, context):
        """Forward pass that just returns the input context."""
        return context

    def __call__(self, model, *args, **kwargs):
        """Create a guide function for the given model."""

        def guide(*args, **kwargs):
            """Mock guide function."""
            return {}

        return guide

    def _setup_guide_impl(self, model, **kwargs):
        """Implementation of guide setup."""
        # No-op for testing
        pass

    def sample_posterior(self, **kwargs):
        """Sample from the posterior distribution."""
        # Return empty dict for testing
        return {}


@pytest.fixture(autouse=True, scope="function")
def setup_registries():
    """
    Set up component registries for each test with proper isolation.

    This fixture ensures that each test starts with a clean registry state
    and that any registrations made during the test don't affect other tests.
    """
    # Store original registry state
    original_dynamics = DynamicsModelRegistry._registry.copy()
    original_priors = PriorModelRegistry._registry.copy()
    original_likelihoods = LikelihoodModelRegistry._registry.copy()
    original_observations = ObservationModelRegistry._registry.copy()
    original_guides = InferenceGuideRegistry._registry.copy()

    # Clear all registries
    DynamicsModelRegistry.clear()
    PriorModelRegistry.clear()
    LikelihoodModelRegistry.clear()
    ObservationModelRegistry.clear()
    InferenceGuideRegistry.clear()

    # Register standard components for the test
    register_standard_components()

    yield

    # Restore original registry state
    DynamicsModelRegistry._registry = original_dynamics
    PriorModelRegistry._registry = original_priors
    LikelihoodModelRegistry._registry = original_likelihoods
    ObservationModelRegistry._registry = original_observations
    InferenceGuideRegistry._registry = original_guides


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create random data for testing
    torch.manual_seed(42)
    n_cells = 10
    n_genes = 5

    # Generate random data
    x = torch.randn((n_cells, n_genes))
    time_points = torch.linspace(0, 1, 3)
    observations = torch.abs(torch.randn((n_cells, n_genes)))

    return {
        "x": x,
        "time_points": time_points,
        "observations": observations,
        "n_cells": n_cells,
        "n_genes": n_genes,
    }


@pytest.fixture
def mock_model(sample_data):
    """Create a mock PyroVelocityModel for testing."""
    dynamics_model = MockDynamicsModel()
    prior_model = MockPriorModel()
    likelihood_model = MockLikelihoodModel()
    guide_model = MockGuideModel()

    return PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        guide_model=guide_model,
    )


@pytest.fixture
def mock_posterior_samples():
    """Create mock posterior samples for testing."""
    torch.manual_seed(42)

    # Create random posterior samples
    return {
        "alpha": torch.abs(torch.randn(100, 5)),
        "beta": torch.abs(torch.randn(100, 5)),
        "gamma": torch.abs(torch.randn(100, 5)),
    }


@pytest.fixture
def multiple_models(sample_data):
    """Create multiple mock PyroVelocityModel instances for testing."""
    models = {}

    # Create three different models with slightly different names
    for i in range(3):
        dynamics_model = MockDynamicsModel(name=f"mock_dynamics_model_{i}")
        prior_model = MockPriorModel(name=f"mock_prior_model_{i}")
        likelihood_model = MockLikelihoodModel(
            name=f"mock_likelihood_model_{i}"
        )
        guide_model = MockGuideModel(name=f"mock_guide_model_{i}")

        models[f"model{i+1}"] = PyroVelocityModel(
            dynamics_model=dynamics_model,
            prior_model=prior_model,
            likelihood_model=likelihood_model,
            guide_model=guide_model,
        )

    return models


@pytest.fixture
def multiple_posterior_samples(mock_posterior_samples):
    """Create posterior samples for multiple models."""
    samples = {}

    # Create slightly different posterior samples for each model
    for i in range(3):
        # Add small offsets to make samples different for each model
        offset = (i + 1) * 0.1
        samples[f"model{i+1}"] = {
            "alpha": mock_posterior_samples["alpha"] + offset,
            "beta": mock_posterior_samples["beta"] + offset,
            "gamma": mock_posterior_samples["gamma"] + offset,
        }

    return samples


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object for testing cross-validation."""
    try:
        import anndata as ad
        import pandas as pd

        # Create a simple AnnData object
        n_cells = 10
        n_genes = 5
        X = np.random.randn(n_cells, n_genes)

        # Create cell metadata with a stratification column
        obs = pd.DataFrame(
            {
                "cell_type": np.random.choice(["A", "B", "C"], size=n_cells),
                "timepoint": np.random.choice([0, 1, 2], size=n_cells),
            }
        )

        return ad.AnnData(X=X, obs=obs)
    except ImportError:
        # If anndata isn't available, return a mock object
        mock_adata = MagicMock()
        mock_adata.obs = {"cell_type": np.array(["A"] * 5 + ["B"] * 5)}
        return mock_adata
