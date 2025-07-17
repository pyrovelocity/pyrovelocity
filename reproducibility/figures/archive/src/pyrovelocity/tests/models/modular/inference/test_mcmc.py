"""
Tests for the MCMC inference components.
"""

import torch
import pyro
import pyro.distributions as dist
import pytest

from pyrovelocity.models.modular.inference.config import (
    InferenceConfig,
    create_inference_config,
    validate_config,
)
from pyrovelocity.models.modular.inference.mcmc import (
    create_mcmc,
    run_mcmc_inference,
    mcmc_diagnostics,
    extract_posterior_samples,
)


# Define a simple model for testing
def simple_model(x=None):
    # Sample parameters from prior
    alpha = pyro.sample("alpha", dist.LogNormal(0.0, 1.0))
    beta = pyro.sample("beta", dist.LogNormal(0.0, 1.0))
    gamma = pyro.sample("gamma", dist.LogNormal(0.0, 1.0))

    # Generate data
    if x is not None:
        # Sample observations
        pyro.sample("obs", dist.Normal(alpha * x, beta), obs=x * gamma)

    return {"alpha": alpha, "beta": beta, "gamma": gamma}


class TestMCMC:
    """Tests for MCMC inference."""

    def test_create_mcmc(self):
        """Test creating an MCMC object."""
        # Test creating MCMC with default kernel
        mcmc = create_mcmc(simple_model, num_warmup=10, num_samples=10)
        assert mcmc is not None

        # Test creating MCMC with custom kernel
        kernel = pyro.infer.NUTS(simple_model)
        mcmc = create_mcmc(
            simple_model, kernel=kernel, num_warmup=10, num_samples=10
        )
        assert mcmc is not None

    def test_run_mcmc_inference(self):
        """Test running MCMC inference."""
        # Generate data
        x = torch.randn(10)

        # Create inference config
        config = create_inference_config(
            method="mcmc",
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        # Run MCMC inference
        mcmc, samples = run_mcmc_inference(
            simple_model,
            args=(x,),
            config=config,
        )

        # Check results
        assert mcmc is not None
        assert isinstance(samples, dict)
        assert "alpha" in samples
        assert "beta" in samples
        assert "gamma" in samples
        assert samples["alpha"].shape[0] == 10

    def test_mcmc_diagnostics(self):
        """Test computing MCMC diagnostics."""
        # Generate data
        x = torch.randn(10)

        # Create inference config
        config = create_inference_config(
            method="mcmc",
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        # Run MCMC inference
        mcmc, _ = run_mcmc_inference(
            simple_model,
            args=(x,),
            config=config,
        )

        # Compute diagnostics
        diagnostics = mcmc_diagnostics(mcmc)
        assert isinstance(diagnostics, dict)
        assert "summary" in diagnostics

    def test_extract_posterior_samples(self):
        """Test extracting posterior samples."""
        # Generate data
        x = torch.randn(10)

        # Create inference config
        config = create_inference_config(
            method="mcmc",
            num_warmup=10,
            num_samples=10,
            num_chains=1,
        )

        # Run MCMC inference
        mcmc, _ = run_mcmc_inference(
            simple_model,
            args=(x,),
            config=config,
        )

        # Extract posterior samples
        samples = extract_posterior_samples(mcmc)
        assert isinstance(samples, dict)
        assert "alpha" in samples
        assert "beta" in samples
        assert "gamma" in samples
        assert samples["alpha"].shape[0] == 10

        # Extract subset of posterior samples
        samples = extract_posterior_samples(mcmc, num_samples=5)
        assert isinstance(samples, dict)
        assert "alpha" in samples
        assert "beta" in samples
        assert "gamma" in samples
        assert samples["alpha"].shape[0] == 5
