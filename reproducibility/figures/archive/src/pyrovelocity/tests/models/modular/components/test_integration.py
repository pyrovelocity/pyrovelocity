"""Integration tests for component implementations."""

import pyro
import pytest
import torch

from pyrovelocity.models.modular.components.dynamics import (
    PiecewiseActivationDynamicsModel,
)
from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
)
from pyrovelocity.models.modular.components.likelihoods import (
    PiecewiseActivationPoissonLikelihoodModel,
)
from pyrovelocity.models.modular.components.priors import (
    PiecewiseActivationPriorModel,
)
from pyrovelocity.models.modular.model import PyroVelocityModel


@pytest.fixture
def simple_data():
    """Create simple data for testing."""
    # Create random data
    batch_size = 3
    n_genes = 4
    u_obs = torch.randint(0, 10, (batch_size, n_genes)).float()
    s_obs = torch.randint(0, 10, (batch_size, n_genes)).float()

    return {
        "u_obs": u_obs,
        "s_obs": s_obs,
    }


def test_components_integration(simple_data):
    """Test that all components work together."""
    # Import the correct prior model for piecewise activation
    from pyrovelocity.models.modular.components.priors import (
        PiecewiseActivationPriorModel,
    )

    # Create components - use compatible model combinations
    dynamics_model = PiecewiseActivationDynamicsModel()
    prior_model = PiecewiseActivationPriorModel()  # Use piecewise prior for piecewise dynamics
    likelihood_model = PiecewiseActivationPoissonLikelihoodModel()
    guide_model = AutoGuideFactory(guide_type="AutoNormal")

    # Create context
    context = {
        "u_obs": simple_data["u_obs"],
        "s_obs": simple_data["s_obs"],
    }

    # Initialize Pyro
    pyro.clear_param_store()

    # Call each component's forward method
    # Note: observation functionality is now in likelihood model
    context = prior_model.forward(context)
    context = dynamics_model.forward(context)
    context = likelihood_model.forward(context)

    # Check that the context contains all expected keys
    assert "u_obs" in context
    assert "s_obs" in context

    # Check for piecewise activation parameters (not legacy parameters)
    assert "alpha_off" in context
    assert "alpha_on" in context
    assert "gamma_star" in context
    assert "t_on_star" in context
    assert "delta_star" in context
    assert "t_star" in context
    assert "lambda_j" in context

    # Check for dynamics outputs
    assert "u_expected" in context
    assert "s_expected" in context
    assert "ut" in context
    assert "st" in context

    # Check for likelihood outputs
    assert "u_dist" in context
    assert "s_dist" in context


def test_components_with_model(simple_data):
    """Test that components work with PyroVelocityModel."""
    # Import the correct prior model for piecewise activation
    from pyrovelocity.models.modular.components.priors import (
        PiecewiseActivationPriorModel,
    )

    # Create components - use compatible model combinations
    dynamics_model = PiecewiseActivationDynamicsModel()
    prior_model = PiecewiseActivationPriorModel()  # Use piecewise prior for piecewise dynamics
    likelihood_model = PiecewiseActivationPoissonLikelihoodModel()
    guide_model = AutoGuideFactory(guide_type="AutoNormal")

    # Create the full model
    model = PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        guide_model=guide_model,
    )

    # Create context
    context = {
        "u_obs": simple_data["u_obs"],
        "s_obs": simple_data["s_obs"],
    }

    # Call the model directly with the context (it will call prior.forward internally)
    result = model.forward(**context)

    # Verify that the model produced expected outputs
    assert "u_dist" in result
    assert "s_dist" in result
    assert "u_expected" in result
    assert "s_expected" in result

    # Test compatible model combinations only
    # Each tuple contains (dynamics, prior, likelihood, guide) that are compatible
    compatible_combinations = [
        # Piecewise combination
        (
            PiecewiseActivationDynamicsModel(),
            PiecewiseActivationPriorModel(),
            PiecewiseActivationPoissonLikelihoodModel(),
            AutoGuideFactory(guide_type="AutoNormal"),
        ),
    ]

    # Test each compatible combination
    for dynamics, prior, likelihood, guide in compatible_combinations:
        # Create the model with this combination
        model = PyroVelocityModel(
            dynamics_model=dynamics,
            prior_model=prior,
            likelihood_model=likelihood,
            guide_model=guide,
        )

        # Clear Pyro's param store between runs
        pyro.clear_param_store()

        # Call the model
        result = model.forward(**context)

        # Verify outputs
        assert "u_dist" in result
        assert "s_dist" in result
        assert "u_expected" in result
        assert "s_expected" in result
