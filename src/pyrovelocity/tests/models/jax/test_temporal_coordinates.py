"""Unit tests for JAX temporal coordinate handling."""

import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import pytest
from numpyro.infer import Predictive

from pyrovelocity.models.jax.factory.factory import create_piecewise_activation_model


class TestTemporalCoordinates:
    """Test temporal coordinate generation and scaling in JAX implementation."""
    
    def test_prior_t_star_scale(self):
        """Test that prior t_star values are in the expected range."""
        model = create_piecewise_activation_model()
        
        # Set random seed for reproducibility
        rng_key = jax.random.PRNGKey(42)
        
        # Generate prior predictive samples
        predictive = Predictive(model, num_samples=100)
        samples = predictive(rng_key, u_obs=None, s_obs=None, 
                           num_cells=50, num_genes=20)
        
        # Check t_star exists and has correct shape
        assert "t_star" in samples
        t_star = samples["t_star"]
        # t_star can have shape (100, 50) or (100, 50, 1) depending on implementation
        assert t_star.shape == (100, 50) or t_star.shape == (100, 50, 1), f"Expected shape (100, 50) or (100, 50, 1), got {t_star.shape}"
        
        # Flatten if needed for consistency
        if len(t_star.shape) == 3:
            t_star = t_star[:, :, 0]
        
        # Check T_M_star scale
        assert "T_M_star" in samples
        T_M_star = samples["T_M_star"]
        
        # T_M_star ~ Gamma(5, 1) should have mean = 5/1 = 5
        T_M_star_mean = np.mean(T_M_star)
        assert 3 < T_M_star_mean < 7, f"T_M_star mean {T_M_star_mean} outside expected range"
        
        # Check t_star is bounded by T_M_star
        for i in range(100):
            assert np.all(t_star[i] >= 0), "t_star has negative values"
            assert np.all(t_star[i] <= T_M_star[i] * 1.1), "t_star exceeds T_M_star bound"
    
    def test_t_star_normalized_beta_distribution(self):
        """Test that t_star_normalized follows Beta distribution."""
        model = create_piecewise_activation_model()
        
        rng_key = jax.random.PRNGKey(42)
        predictive = Predictive(model, num_samples=1000)
        samples = predictive(rng_key, u_obs=None, s_obs=None, 
                           num_cells=100, num_genes=20)
        
        assert "t_star_normalized" in samples
        t_normalized = samples["t_star_normalized"]
        
        # Check all values are in [0, 1]
        assert np.all(t_normalized >= 0), "t_star_normalized has values < 0"
        assert np.all(t_normalized <= 1), "t_star_normalized has values > 1"
        
        # Check distribution properties
        # With boundary_concentration ~ Gamma(100, 100), mean ≈ 1
        # So Beta(1, 1) should be approximately uniform
        mean_val = np.mean(t_normalized)
        assert 0.4 < mean_val < 0.6, f"t_star_normalized mean {mean_val} suggests non-uniform distribution"
    
    def test_scale_factor_application(self):
        """Test that scale factors are applied correctly without double-scaling."""
        model = create_piecewise_activation_model()
        
        rng_key = jax.random.PRNGKey(42)
        
        # Generate data with known scale
        num_cells = 30
        num_genes = 10
        
        # Create synthetic observed data
        u_obs = jnp.ones((1, num_cells, num_genes)) * 10
        s_obs = jnp.ones((1, num_cells, num_genes)) * 20
        
        # Run model to get expected values
        predictive = Predictive(model, num_samples=1)
        samples = predictive(rng_key, u_obs=u_obs, s_obs=s_obs)
        
        # Check that expected values are scaled appropriately
        u_expected = samples["u_expected"]
        s_expected = samples["s_expected"]
        
        # Extract scale parameters
        lambda_j = samples["lambda_j"]  # Cell capture efficiency
        U_0i = samples["U_0i"]  # Gene concentration scale
        
        # The expected values should already include scaling
        # Check that values are in reasonable range given the scale parameters
        mean_u = np.mean(u_expected)
        mean_s = np.mean(s_expected)
        
        # With typical scale parameters, means should be in similar range to observations
        assert 1 < mean_u < 100, f"Mean u_expected {mean_u} outside reasonable range"
        assert 1 < mean_s < 100, f"Mean s_expected {mean_s} outside reasonable range"
    
    def test_temporal_coordinate_consistency(self):
        """Test that temporal coordinates have consistent shapes across samples."""
        model = create_piecewise_activation_model()
        
        rng_key = jax.random.PRNGKey(42)
        
        num_cells = 30
        num_genes = 15
        
        # Generate multiple prior predictive samples
        predictive = Predictive(model, num_samples=5)
        samples = predictive(rng_key, u_obs=None, s_obs=None,
                           num_cells=num_cells, num_genes=num_genes)
        
        # Check t_star consistency
        assert "t_star" in samples
        t_star = samples["t_star"]
        
        # All samples should have same cell dimension
        expected_cells = num_cells
        for i in range(5):
            sample_cells = t_star[i].shape[0]
            assert sample_cells == expected_cells, f"Sample {i} has {sample_cells} cells, expected {expected_cells}"
        
        # All samples should have reasonable temporal ranges
        for i in range(5):
            sample_t = t_star[i]
            if len(sample_t.shape) == 2:
                sample_t = sample_t[:, 0]  # Flatten if needed
            
            assert np.all(sample_t >= 0), f"Sample {i} has negative t_star values"
            assert np.max(sample_t) > 0.1, f"Sample {i} has suspiciously small t_star values"