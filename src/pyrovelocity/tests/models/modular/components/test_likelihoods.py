"""Tests for likelihood models."""

import numpy as np
import pyro.distributions
import pytest
import torch
from anndata._core.anndata import AnnData

from pyrovelocity.models.modular.components.likelihoods import (
    PiecewiseActivationPoissonLikelihoodModel,
)
from pyrovelocity.models.modular.registry import LikelihoodModelRegistry


@pytest.fixture(scope="module", autouse=True)
def register_likelihood_models():
    """Register likelihood models for testing."""
    # Save original registry state
    original_registry = dict(LikelihoodModelRegistry._registry)

    # Clear registry and register test components
    LikelihoodModelRegistry.clear()
    LikelihoodModelRegistry._registry["piecewise_activation_poisson"] = PiecewiseActivationPoissonLikelihoodModel

    yield

    # Restore original registry state
    LikelihoodModelRegistry._registry = original_registry


@pytest.fixture
def test_adata():
    """Create a test AnnData object."""
    # Create a simple AnnData object for testing
    n_cells = 10
    n_genes = 20
    X = np.random.poisson(lam=5, size=(n_cells, n_genes))
    adata = AnnData(X=X)
    # Add X to layers as well, which is required by the likelihood models
    adata.layers["X"] = X.copy()
    return adata


def test_likelihood_registry():
    """Test likelihood model registry."""
    # Check that piecewise activation model is registered
    available_models = LikelihoodModelRegistry.list_available()
    assert "piecewise_activation_poisson" in available_models
    assert len(available_models) == 1  # Only piecewise activation model

    # Check retrieved models
    piecewise_class = LikelihoodModelRegistry.get("piecewise_activation_poisson")
    assert piecewise_class == PiecewiseActivationPoissonLikelihoodModel

    # Create instances using the registry
    piecewise_model = LikelihoodModelRegistry.create("piecewise_activation_poisson")
    assert isinstance(piecewise_model, PiecewiseActivationPoissonLikelihoodModel)
