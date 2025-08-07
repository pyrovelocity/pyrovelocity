"""Unit tests for JAX scale factor handling."""

import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import pytest
from numpyro.infer import Predictive

from pyrovelocity.models.jax.factory.factory import create_piecewise_activation_model
from pyrovelocity.models.jax.components.dynamics import piecewise_activation_dynamics_function


class TestScaleFactors:
    """Test scale factor application in JAX implementation."""
    
    def test_dynamics_output_scale(self):
        """Test that dynamics function outputs are dimensionless."""
        # Create test inputs
        batch_size = 1
        n_cells = 10
        n_genes = 5
        
        t_star = jnp.ones((batch_size, n_cells)) * 2.0  # Shape: (batch_size, n_cells)
        u0_star = jnp.ones((batch_size, n_cells, n_genes))
        
        params = {
            "R_on": jnp.ones((n_genes,)) * 2.5,
            "gamma_star": jnp.ones((n_genes,)) * 0.7,
            "t_on_star": jnp.ones((n_genes,)) * 1.0,
            "delta_star": jnp.ones((n_genes,)) * 1.5,
        }
        
        # Call dynamics function
        u_star, s_star = piecewise_activation_dynamics_function(t_star, u0_star, params)
        
        # Check outputs are positive and in reasonable dimensionless range
        assert jnp.all(u_star > 0), "u_star has non-positive values"
        assert jnp.all(s_star > 0), "s_star has non-positive values"
        
        # Dimensionless values should typically be O(1)
        assert jnp.max(u_star) < 10, f"u_star max {jnp.max(u_star)} too large for dimensionless"
        assert jnp.max(s_star) < 20, f"s_star max {jnp.max(s_star)} too large for dimensionless"
    
    def test_likelihood_scale_application(self):
        """Test that likelihood correctly applies scale factors once."""
        model = create_piecewise_activation_model()
        
        rng_key = jax.random.PRNGKey(42)
        
        # Generate prior samples to understand scale
        predictive = Predictive(model, num_samples=10)
        samples = predictive(rng_key, u_obs=None, s_obs=None,
                          num_cells=20, num_genes=10)
        
        # Extract scale parameters
        lambda_j = samples["lambda_j"]  # Shape: (10, 20)
        U_0i = samples["U_0i"]  # Shape: (10, 10)
        
        # Extract dimensionless concentrations from dynamics
        u_star = samples["u_star"]  # Shape: (10, 1, 20, 10)
        s_star = samples["s_star"]
        
        # Compute expected values by applying scaling (as done in likelihood now)
        # u_expected = lambda_j * U_0i * u_star
        # Need to handle dimensions properly:
        # lambda_j: (10, 20) -> (10, 20, 1) for broadcasting with u_star (10, 1, 20, 10)
        # U_0i: (10, 10) -> (10, 1, 1, 10) for broadcasting with u_star (10, 1, 20, 10)
        lambda_j_expanded = lambda_j[:, jnp.newaxis, :, jnp.newaxis]  # (10, 1, 20, 1)
        U_0i_expanded = U_0i[:, jnp.newaxis, jnp.newaxis, :]  # (10, 1, 1, 10)
        u_expected = lambda_j_expanded * U_0i_expanded * u_star  # (10, 1, 20, 10)
        s_expected = lambda_j_expanded * U_0i_expanded * s_star
        
        # Extract observed values
        u_obs = samples["u_obs"]
        s_obs = samples["s_obs"]
        
        # Check that observed values are Poisson samples with rates = expected values
        # The expected values should already include lambda_j * U_0i scaling
        
        # Mean of Poisson(rate) = rate
        # So mean observed should approximately equal expected
        mean_u_obs = jnp.mean(u_obs, axis=(0, 1, 2))
        mean_u_expected = jnp.mean(u_expected, axis=(0, 1, 2))
        
        # Check they're in the same order of magnitude
        ratio = mean_u_obs / mean_u_expected
        assert jnp.all((0.5 < ratio) & (ratio < 2.0)), f"Observed/expected ratio {ratio} suggests scaling issue"
    
    def test_no_double_scaling(self):
        """Test that scale factors are not applied twice."""
        model = create_piecewise_activation_model()
        
        rng_key = jax.random.PRNGKey(42)
        
        # Set up controlled test
        num_cells = 10
        num_genes = 5
        
        # Generate prior samples
        predictive = Predictive(model, num_samples=1)
        samples = predictive(rng_key, u_obs=None, s_obs=None,
                          num_cells=num_cells, num_genes=num_genes)
        
        # Get scale parameters
        lambda_j = samples["lambda_j"][0]  # Shape: (10,)
        U_0i = samples["U_0i"][0]  # Shape: (5,)
        
        # Get dimensionless concentrations 
        u_star = samples["u_star"][0, 0]  # Shape: (10, 5)
        
        # Now compute expected values as likelihood does: lambda_j * U_0i * u_star
        scale_matrix = lambda_j[:, np.newaxis] * U_0i[np.newaxis, :]
        u_expected = scale_matrix * u_star
        
        # Check that u_star (dimensionless concentrations) are O(1)
        dynamics_implied = u_star  # These should be dimensionless
        
        # Check implied dynamics are in dimensionless range
        assert np.all(dynamics_implied > 0.01), "Implied dynamics too small"
        assert np.all(dynamics_implied < 10), "Implied dynamics too large, possible double scaling"
        
        # Check mean is reasonable
        mean_dynamics = np.mean(dynamics_implied)
        assert 0.1 < mean_dynamics < 5, f"Mean implied dynamics {mean_dynamics} outside expected range"
    
    def test_prior_posterior_scale_consistency(self):
        """Test that scale remains consistent between prior and posterior."""
        model = create_piecewise_activation_model()
        
        rng_key = jax.random.PRNGKey(42)
        
        # Generate prior predictive data
        prior_pred = Predictive(model, num_samples=1)
        prior_samples = prior_pred(rng_key, u_obs=None, s_obs=None,
                                 num_cells=30, num_genes=15)
        
        # Use prior samples as observed data
        # Remove num_samples dimension to match expected shape
        u_obs = prior_samples["u_obs"][0]  # Shape: (batch_size, n_cells, n_genes)
        s_obs = prior_samples["s_obs"][0]  # Shape: (batch_size, n_cells, n_genes)
        
        # Generate posterior predictive (without inference, just for scale check)
        post_pred = Predictive(model, num_samples=1)
        post_samples = post_pred(rng_key, u_obs=u_obs, s_obs=s_obs)
        
        # Compare scale of dimensionless concentrations 
        prior_u_star = prior_samples["u_star"]
        post_u_star = post_samples["u_star"]
        
        # They should be in similar scale ranges (both dimensionless)
        prior_mean = np.mean(prior_u_star)
        post_mean = np.mean(post_u_star)
        
        ratio = post_mean / prior_mean
        assert 0.1 < ratio < 10, f"Prior/posterior scale ratio {ratio} suggests inconsistency"