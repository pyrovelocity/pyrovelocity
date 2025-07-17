"""
Tests for the unified inference interface.
"""

import pyro
import pyro.distributions as dist
import pytest
import torch

from pyrovelocity.models.modular.inference.config import create_inference_config
from pyrovelocity.models.modular.inference.svi import TrainingState
from pyrovelocity.models.modular.inference.unified import (
    InferenceState,
    create_inference_state,
    extract_posterior_samples,
    posterior_predictive,
    run_inference,
)


# Define a simple model for testing
def simple_model(x=None):
    # Sample parameters from prior
    alpha = pyro.sample("alpha", dist.LogNormal(0.0, 1.0))
    beta = pyro.sample("beta", dist.LogNormal(0.0, 1.0))
    gamma = pyro.sample("gamma", dist.LogNormal(0.0, 1.0))

    # Generate data
    if x is not None:
        # Use plate to handle batched data
        with pyro.plate("data", len(x)):
            # Sample observations
            pyro.sample("obs", dist.Normal(alpha * x, beta), obs=x * gamma)

    return {"alpha": alpha, "beta": beta, "gamma": gamma}


# Define a simple guide for testing
def simple_guide(x=None):  # x is unused but required for API compatibility
    # Variational parameters
    alpha_loc = pyro.param("alpha_loc", torch.tensor(0.0))
    alpha_scale = pyro.param(
        "alpha_scale", torch.tensor(1.0), constraint=dist.constraints.positive
    )
    beta_loc = pyro.param("beta_loc", torch.tensor(0.0))
    beta_scale = pyro.param(
        "beta_scale", torch.tensor(1.0), constraint=dist.constraints.positive
    )
    gamma_loc = pyro.param("gamma_loc", torch.tensor(0.0))
    gamma_scale = pyro.param(
        "gamma_scale", torch.tensor(1.0), constraint=dist.constraints.positive
    )

    # Sample from variational distributions
    alpha = pyro.sample("alpha", dist.LogNormal(alpha_loc, alpha_scale))
    beta = pyro.sample("beta", dist.LogNormal(beta_loc, beta_scale))
    gamma = pyro.sample("gamma", dist.LogNormal(gamma_loc, gamma_scale))

    return {"alpha": alpha, "beta": beta, "gamma": gamma}


class TestUnified:
    """Tests for unified inference interface."""

    def test_create_inference_state(self):
        """Test creating an inference state."""
        # Test creating inference state with minimal parameters
        state = create_inference_state(method="svi")
        assert state.method == "svi"
        assert state.params == {}
        assert state.posterior_samples == {}
        assert state.training_state is None
        assert state.mcmc is None
        assert state.svi is None
        assert state.diagnostics == {}

        # Test creating inference state with all parameters
        params = {"alpha": torch.tensor(1.0)}
        posterior_samples = {"alpha": torch.tensor([1.0, 2.0])}
        training_state = TrainingState()  # Create a proper TrainingState object
        mcmc = None  # Skip MCMC object for now
        svi = None  # Skip SVI object for now
        diagnostics = {"loss": 1.0}
        state = create_inference_state(
            method="mcmc",
            params=params,
            posterior_samples=posterior_samples,
            training_state=training_state,
            mcmc=mcmc,
            svi=svi,
            diagnostics=diagnostics,
        )
        assert state.method == "mcmc"
        assert state.params == params
        assert state.posterior_samples == posterior_samples
        assert state.training_state is training_state
        assert state.mcmc is mcmc
        assert state.svi is svi
        assert state.diagnostics == diagnostics

    def test_run_inference_svi(self):
        """Test running inference with SVI."""
        # Generate data
        x = torch.randn(10)

        # Create inference config
        config = create_inference_config(
            method="svi",
            num_epochs=10,
            num_samples=10,
            learning_rate=0.01,
            optimizer="adam",
        )

        # Run inference
        state = run_inference(
            simple_model,
            simple_guide,
            args=(x,),
            config=config,
        )

        # Check results
        assert isinstance(state, InferenceState)
        assert state.method == "svi"
        assert state.params != {}
        assert state.posterior_samples != {}
        assert state.training_state is not None
        assert state.mcmc is None
        assert state.svi is None
        assert "alpha" in state.posterior_samples
        assert "beta" in state.posterior_samples
        assert "gamma" in state.posterior_samples
        assert state.posterior_samples["alpha"].shape[0] == 10

    def test_run_inference_mcmc(self):
        """Test running inference with MCMC."""
        # Generate data
        x = torch.randn(10)

        # Create inference config
        config = create_inference_config(
            method="mcmc",
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        # Run inference
        state = run_inference(
            simple_model,
            args=(x,),
            config=config,
        )

        # Check results
        assert isinstance(state, InferenceState)
        assert state.method == "mcmc"
        assert state.params == {}
        assert state.posterior_samples != {}
        assert state.training_state is None
        assert state.mcmc is not None
        assert state.svi is None
        assert "alpha" in state.posterior_samples
        assert "beta" in state.posterior_samples
        assert "gamma" in state.posterior_samples
        assert state.posterior_samples["alpha"].shape[0] == 10

    def test_extract_posterior_samples(self):
        """Test extracting posterior samples."""
        # Generate data
        x = torch.randn(10)

        # Create inference config
        config = create_inference_config(
            method="svi",
            num_epochs=10,
            num_samples=10,
            learning_rate=0.01,
            optimizer="adam",
        )

        # Run inference
        state = run_inference(
            simple_model,
            simple_guide,
            args=(x,),
            config=config,
        )

        # Extract posterior samples
        samples = extract_posterior_samples(state)
        assert isinstance(samples, dict)
        assert "alpha" in samples
        assert "beta" in samples
        assert "gamma" in samples
        assert samples["alpha"].shape[0] == 10

        # Extract subset of posterior samples
        samples = extract_posterior_samples(state, num_samples=5)
        assert isinstance(samples, dict)
        assert "alpha" in samples
        assert "beta" in samples
        assert "gamma" in samples
        assert samples["alpha"].shape[0] == 5

    def test_posterior_predictive(self):
        """Test generating posterior predictive samples."""
        # Generate data
        x = torch.randn(10)

        # Create inference config
        config = create_inference_config(
            method="svi",
            num_epochs=10,
            num_samples=10,
            learning_rate=0.01,
            optimizer="adam",
        )

        # Run inference
        state = run_inference(
            simple_model,
            simple_guide,
            args=(x,),
            config=config,
        )

        # Generate posterior predictive samples
        predictive_samples = posterior_predictive(
            simple_model,
            state.posterior_samples,
            args=(x,),
        )
        assert isinstance(predictive_samples, dict)
        assert "alpha" in predictive_samples
        assert "beta" in predictive_samples
        assert "gamma" in predictive_samples
        assert "obs" in predictive_samples
        assert predictive_samples["alpha"].shape[0] == 10

        # Generate subset of posterior predictive samples
        predictive_samples = posterior_predictive(
            simple_model,
            state.posterior_samples,
            args=(x,),
            num_samples=5,
        )
        assert isinstance(predictive_samples, dict)
        assert "alpha" in predictive_samples
        assert "beta" in predictive_samples
        assert "gamma" in predictive_samples
        assert "obs" in predictive_samples
        assert predictive_samples["alpha"].shape[0] == 5
