"""
Test the string representation of PyroVelocityModel.
"""

import pyro
import pytest
import torch

from pyrovelocity.models.modular.components import (
    AutoGuideFactory,
    PiecewiseActivationDynamicsModel,
    PiecewiseActivationPoissonLikelihoodModel,
    PiecewiseActivationPriorModel,
)
from pyrovelocity.models.modular.factory import (
    create_piecewise_activation_model,
)
from pyrovelocity.models.modular.model import PyroVelocityModel


def test_model_repr_untrained():
    """Test the string representation of an untrained PyroVelocityModel."""
    # Create a model with piecewise activation components
    model = create_piecewise_activation_model()

    # Get the string representation
    repr_str = repr(model)

    # Check that the representation contains expected elements
    assert "PyroVelocityModel (Untrained)" in repr_str
    assert "Components:" in repr_str
    assert "Dynamics:" in repr_str
    assert "Prior:" in repr_str
    assert "Likelihood:" in repr_str
    assert "Guide:" in repr_str

    # Check that __str__ returns the same as __repr__
    assert str(model) == repr(model)


def test_model_repr_custom_components():
    """Test the string representation of a model with custom components."""
    # Create a model with custom components
    dynamics_model = PiecewiseActivationDynamicsModel()
    prior_model = PiecewiseActivationPriorModel()
    likelihood_model = PiecewiseActivationPoissonLikelihoodModel()
    guide_model = AutoGuideFactory()

    # Add names and descriptions to components
    dynamics_model.name = "CustomDynamics"
    dynamics_model.description = "Custom dynamics model for testing"
    prior_model.name = "CustomPrior"
    prior_model.description = "Custom prior model for testing"

    # Create model
    model = PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        guide_model=guide_model
    )

    # Get the string representation
    repr_str = repr(model)

    # Check that the representation contains custom names and descriptions
    assert "CustomDynamics" in repr_str
    assert "Custom dynamics model for testing" in repr_str
    assert "CustomPrior" in repr_str
    assert "Custom prior model for testing" in repr_str


def test_model_repr_with_config():
    """Test the string representation of a model with component configurations."""
    # Create a model with custom components that have config attributes
    dynamics_model = PiecewiseActivationDynamicsModel()
    dynamics_model.config = {"alpha_off": 0.1, "alpha_on": 2.0}

    prior_model = PiecewiseActivationPriorModel()
    prior_model.config = {"alpha_off_prior": "LogNormal", "alpha_on_prior": "LogNormal"}

    likelihood_model = PiecewiseActivationPoissonLikelihoodModel()
    guide_model = AutoGuideFactory()

    # Create model
    model = PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        guide_model=guide_model
    )

    # Get the string representation
    repr_str = repr(model)

    # Check that the representation contains configuration information
    assert "alpha_off=0.1" in repr_str
    assert "alpha_on=2.0" in repr_str
    assert "alpha_off_prior=LogNormal" in repr_str
    assert "alpha_on_prior=LogNormal" in repr_str


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    pyro.set_rng_seed(42)
    
    # Run tests
    test_model_repr_untrained()
    test_model_repr_custom_components()
    test_model_repr_with_config()
    
    print("All tests passed!")
