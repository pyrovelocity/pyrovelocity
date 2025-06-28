"""
Tests for the posterior analysis utilities.
"""

import anndata
import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from pyrovelocity.models.jax.registry.dynamics import get_dynamics
from pyrovelocity.models.jax.core.model import velocity_model


def get_test_dynamics():
    """Get dynamics function for testing."""
    dynamics_fn = get_dynamics("piecewise_activation")
    if dynamics_fn is None:
        raise ValueError("Test dynamics function 'piecewise_activation' not found in registry")
    return dynamics_fn
from pyrovelocity.models.jax.core.state import InferenceState
from pyrovelocity.models.jax.inference.posterior import (
    analyze_posterior,
    compute_uncertainty,
    compute_velocity,
    create_inference_data,
    format_anndata_output,
    posterior_predictive,
    sample_posterior,
)


# Simple model for testing
def simple_model(u_obs=None, s_obs=None):
    # Sample parameters
    alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
    beta = numpyro.sample("beta", dist.LogNormal(0.0, 1.0))
    gamma = numpyro.sample("gamma", dist.LogNormal(0.0, 1.0))

    # Sample latent time
    with numpyro.plate("cell", 10):
        tau = numpyro.sample("tau", dist.Normal(0.0, 1.0))

    # Compute expected values
    u_expected = alpha * jnp.exp(-beta * tau)
    s_expected = (
        alpha
        * beta
        / (gamma - beta + 1e-6)
        * (jnp.exp(-beta * tau) - jnp.exp(-gamma * tau))
    )

    # Sample observations
    with numpyro.plate("cell_gene", 10):
        if u_obs is not None:
            numpyro.sample(
                "u_obs", dist.Poisson(u_expected.reshape(-1)), obs=u_obs
            )
        if s_obs is not None:
            numpyro.sample(
                "s_obs", dist.Poisson(s_expected.reshape(-1)), obs=s_obs
            )

    # Return expected values
    return {
        "u_expected": u_expected,
        "s_expected": s_expected,
    }


@pytest.fixture
def test_data():
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(0)

    # Generate synthetic data
    alpha = jnp.exp(jnp.array(0.0))
    beta = jnp.exp(jnp.array(0.0))
    gamma = jnp.exp(jnp.array(0.0))
    tau = jnp.linspace(0.0, 1.0, 10)

    # Compute expected values
    u_expected = alpha * jnp.exp(-beta * tau)
    s_expected = (
        alpha
        * beta
        / (gamma - beta + 1e-6)
        * (jnp.exp(-beta * tau) - jnp.exp(-gamma * tau))
    )

    # Sample observations
    key, subkey1, subkey2 = jax.random.split(key, 3)
    u_obs = jax.random.poisson(subkey1, u_expected)
    s_obs = jax.random.poisson(subkey2, s_expected)

    return u_obs, s_obs


@pytest.fixture
def test_inference_state():
    # Create posterior samples using correct piecewise activation parameter names
    key = jax.random.PRNGKey(0)
    key, subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 6)

    posterior_samples = {
        "R_on": jax.random.lognormal(subkey1, shape=(10, 10)),  # (samples, genes)
        "gamma_star": jax.random.lognormal(subkey2, shape=(10, 10)),  # (samples, genes)
        "t_on_star": jax.random.normal(subkey3, shape=(10, 10)),  # (samples, genes)
        "delta_star": jax.random.lognormal(subkey4, shape=(10, 10)),  # (samples, genes)
        "t_star": jax.random.normal(subkey5, shape=(10, 1)),  # (samples, cells)
    }

    # Create inference state
    inference_state = InferenceState(
        posterior_samples=posterior_samples,
    )

    return inference_state


def test_sample_posterior(test_inference_state):
    """Test sampling from the posterior."""
    # Sample from the posterior
    key = jax.random.PRNGKey(0)
    posterior_samples = sample_posterior(
        inference_state=test_inference_state,
        num_samples=5,
        key=key,
    )

    # Check that posterior samples are present
    assert "R_on" in posterior_samples
    assert "gamma_star" in posterior_samples
    assert "t_on_star" in posterior_samples
    assert "delta_star" in posterior_samples
    assert "t_star" in posterior_samples

    # Check that posterior samples have the correct shape
    assert posterior_samples["R_on"].shape == (5, 10)
    assert posterior_samples["gamma_star"].shape == (5, 10)
    assert posterior_samples["t_on_star"].shape == (5, 10)
    assert posterior_samples["delta_star"].shape == (5, 10)
    assert posterior_samples["t_star"].shape == (5, 1)


def test_posterior_predictive(test_data, test_inference_state):
    """Test generating posterior predictive samples."""
    # Get test data
    u_obs, s_obs = test_data

    # Sample from the posterior
    key = jax.random.PRNGKey(0)
    posterior_samples = sample_posterior(
        inference_state=test_inference_state,
        num_samples=5,
        key=key,
    )

    # Generate posterior predictive samples
    key, subkey = jax.random.split(key)
    posterior_predictive_samples = posterior_predictive(
        model=simple_model,
        posterior_samples=posterior_samples,
        args=(),
        kwargs={},
        num_samples=5,
        key=subkey,
    )

    # Check that posterior predictive samples are present
    assert "alpha" in posterior_predictive_samples
    assert "beta" in posterior_predictive_samples
    assert "gamma" in posterior_predictive_samples
    assert "tau" in posterior_predictive_samples

    # Check that posterior predictive samples have the correct shape
    assert posterior_predictive_samples["alpha"].shape == (5,)
    assert posterior_predictive_samples["beta"].shape == (5,)
    assert posterior_predictive_samples["gamma"].shape == (5,)
    assert posterior_predictive_samples["tau"].shape == (5, 10)


def test_compute_velocity(test_inference_state):
    """Test computing RNA velocity."""
    # Sample from the posterior
    key = jax.random.PRNGKey(0)
    posterior_samples = sample_posterior(
        inference_state=test_inference_state,
        num_samples=5,
        key=key,
    )

    # Compute velocity
    velocity_samples = compute_velocity(
        posterior_samples=posterior_samples,
        dynamics_fn=get_test_dynamics(),
    )

    # Check that velocity samples are present
    assert "u_expected" in velocity_samples
    assert "s_expected" in velocity_samples
    assert "velocity" in velocity_samples
    assert "acceleration" in velocity_samples

    # Check that velocity samples have the correct shape
    assert velocity_samples["u_expected"].shape == (5, 1, 10)
    assert velocity_samples["s_expected"].shape == (5, 1, 10)
    assert velocity_samples["velocity"].shape == (5, 1, 10)
    assert velocity_samples["acceleration"].shape == (5, 1, 10)


def test_compute_uncertainty(test_inference_state):
    """Test computing uncertainty in RNA velocity."""
    # Sample from the posterior
    key = jax.random.PRNGKey(0)
    posterior_samples = sample_posterior(
        inference_state=test_inference_state,
        num_samples=5,
        key=key,
    )

    # Compute velocity
    velocity_samples = compute_velocity(
        posterior_samples=posterior_samples,
        dynamics_fn=get_test_dynamics(),
    )

    # Compute uncertainty
    uncertainty = compute_uncertainty(
        velocity_samples=velocity_samples,
    )

    # Check that uncertainty measures are present
    assert "u_expected_mean" in uncertainty
    assert "s_expected_mean" in uncertainty
    assert "velocity_mean" in uncertainty
    assert "acceleration_mean" in uncertainty
    assert "u_expected_std" in uncertainty
    assert "s_expected_std" in uncertainty
    assert "velocity_std" in uncertainty
    assert "acceleration_std" in uncertainty
    assert "velocity_prob_positive" in uncertainty
    assert "velocity_confidence" in uncertainty

    # Check that uncertainty measures have the correct shape
    assert uncertainty["u_expected_mean"].shape == (1, 10)
    assert uncertainty["s_expected_mean"].shape == (1, 10)
    assert uncertainty["velocity_mean"].shape == (1, 10)
    assert uncertainty["acceleration_mean"].shape == (1, 10)
    assert uncertainty["u_expected_std"].shape == (1, 10)
    assert uncertainty["s_expected_std"].shape == (1, 10)
    assert uncertainty["velocity_std"].shape == (1, 10)
    assert uncertainty["acceleration_std"].shape == (1, 10)
    assert uncertainty["velocity_prob_positive"].shape == (1, 10)
    assert uncertainty["velocity_confidence"].shape == (1, 10)


def test_create_inference_data(test_inference_state):
    """Test creating an ArviZ InferenceData object."""
    # Sample from the posterior
    key = jax.random.PRNGKey(0)
    posterior_samples = sample_posterior(
        inference_state=test_inference_state,
        num_samples=5,
        key=key,
    )

    # Create inference data
    inference_data = create_inference_data(
        posterior_samples=posterior_samples,
    )

    # Check that inference data is an ArviZ InferenceData object
    assert isinstance(inference_data, az.InferenceData)

    # Check that posterior samples are present
    assert hasattr(inference_data, "posterior")
    assert "alpha" in inference_data.posterior
    assert "beta" in inference_data.posterior
    assert "gamma" in inference_data.posterior
    assert "tau" in inference_data.posterior


def test_analyze_posterior_with_model_config(test_inference_state):
    """Test analyzing posterior samples with a model configuration."""
    # Create test data with the correct shape and type
    u_obs = jnp.ones((10,), dtype=jnp.float32)  # (cells,)
    s_obs = jnp.ones((10,), dtype=jnp.float32)  # (cells,)

    # Create a simple model function
    def simple_model(u_obs=None, s_obs=None):
        # Sample parameters
        alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 1.0))
        beta = numpyro.sample("beta", dist.LogNormal(0.0, 1.0))
        gamma = numpyro.sample("gamma", dist.LogNormal(0.0, 1.0))

        # Sample latent time
        with numpyro.plate("cell", 10):
            tau = numpyro.sample("tau", dist.Normal(0.0, 1.0))

        # Compute expected values
        u_expected = alpha * jnp.exp(-beta * tau)
        s_expected = (
            alpha
            * beta
            / (gamma - beta + 1e-6)
            * (jnp.exp(-beta * tau) - jnp.exp(-gamma * tau))
        )

        # Sample observations
        with numpyro.plate("cell_gene", 10):
            if u_obs is not None:
                numpyro.sample(
                    "u_obs", dist.Poisson(u_expected.reshape(-1)), obs=u_obs
                )
            if s_obs is not None:
                numpyro.sample(
                    "s_obs", dist.Poisson(s_expected.reshape(-1)), obs=s_obs
                )

        # Return expected values
        return {
            "u_expected": u_expected,
            "s_expected": s_expected,
        }

    # Generate random key
    key = jax.random.PRNGKey(0)

    # Analyze posterior
    results = analyze_posterior(
        inference_state=test_inference_state,
        model=simple_model,
        kwargs={"u_obs": u_obs, "s_obs": s_obs},
        num_samples=5,
        key=key,
    )

    # Check that results contain expected keys
    assert "posterior_samples" in results
    assert "posterior_predictive" in results
    assert "velocity" in results
    assert "uncertainty" in results
    assert "inference_data" in results

    # Check that posterior samples have the correct shape
    assert results["posterior_samples"]["R_on"].shape == (5, 10)
    assert results["posterior_samples"]["gamma_star"].shape == (5, 10)
    assert results["posterior_samples"]["t_on_star"].shape == (5, 10)
    assert results["posterior_samples"]["delta_star"].shape == (5, 10)
    assert results["posterior_samples"]["t_star"].shape == (5, 1)

    # Check that velocity samples have the correct shape
    assert "u_expected" in results["velocity"]
    assert "s_expected" in results["velocity"]
    assert "velocity" in results["velocity"]
    assert "acceleration" in results["velocity"]

    # Check that uncertainty measures have the correct shape
    assert "velocity_mean" in results["uncertainty"]
    assert "velocity_std" in results["uncertainty"]
    assert "velocity_prob_positive" in results["uncertainty"]
    assert "velocity_confidence" in results["uncertainty"]


def test_format_anndata_output(test_inference_state):
    """Test formatting results into an AnnData object."""
    # Sample from the posterior
    key = jax.random.PRNGKey(0)
    posterior_samples = sample_posterior(
        inference_state=test_inference_state,
        num_samples=5,
        key=key,
    )

    # Create mock velocity samples with shape (num_samples, num_cells, num_genes)
    n_samples = 5
    n_cells = 10
    n_genes = 10
    mock_velocity_samples = {
        "u_expected": jnp.ones((n_samples, n_genes, n_cells)),  # 5 samples, 10 genes, 10 cells
        "s_expected": jnp.ones((n_samples, n_genes, n_cells)),
        "velocity": jnp.ones((n_samples, n_genes, n_cells)),
        "acceleration": jnp.ones((n_samples, n_genes, n_cells)),
    }

    # Compute uncertainty from mock data
    uncertainty = compute_uncertainty(
        velocity_samples=mock_velocity_samples,
    )

    # Create results dictionary
    results = {
        "posterior_samples": posterior_samples,
        "velocity": mock_velocity_samples,
        "uncertainty": uncertainty,
    }

    # Create AnnData object with the correct dimensions
    adata = anndata.AnnData(
        X=np.random.rand(n_cells, n_genes),
        obs={"cell_type": [f"cell_{i}" for i in range(n_cells)]},
        var={"gene_name": [f"gene_{i}" for i in range(n_genes)]},
    )

    # Format results into AnnData object
    adata_out = format_anndata_output(
        adata=adata,
        results=results,
        model_name="test_model",
    )

    # Check that the output is an AnnData object
    assert isinstance(adata_out, anndata.AnnData)

    # Check that the output has the expected layers
    assert "test_model_velocity" in adata_out.layers
    assert "test_model_u_expected" in adata_out.layers
    assert "test_model_s_expected" in adata_out.layers

    # Check that the output has the expected obs
    assert "test_model_velocity_confidence" in adata_out.obs
    assert "test_model_velocity_probability" in adata_out.obs

    # Check that the output has the expected var
    assert "test_model_velocity_cv" in adata_out.var
    assert "test_model_R_on" in adata_out.var
    assert "test_model_gamma_star" in adata_out.var
    assert "test_model_t_on_star" in adata_out.var

    # Check that the output has the expected uns
    assert "velocity_models" in adata_out.uns
    assert "test_model" in adata_out.uns["velocity_models"]
    assert "test_model_model_type" in adata_out.uns
    assert adata_out.uns["test_model_model_type"] == "jax_numpyro"
    assert "test_model_params" in adata_out.uns

    # Check that the shapes are correct
    assert adata_out.layers["test_model_velocity"].shape == (n_cells, n_genes)
    assert adata_out.layers["test_model_u_expected"].shape == (n_cells, n_genes)
    assert adata_out.layers["test_model_s_expected"].shape == (n_cells, n_genes)
    assert adata_out.obs["test_model_velocity_confidence"].shape == (n_cells,)
    assert adata_out.obs["test_model_velocity_probability"].shape == (n_cells,)
    assert adata_out.var["test_model_velocity_cv"].shape == (n_genes,)
    assert adata_out.var["test_model_R_on"].shape == (n_genes,)
    assert adata_out.var["test_model_gamma_star"].shape == (n_genes,)
    assert adata_out.var["test_model_t_on_star"].shape == (n_genes,)
