"""
Tests for parameter conditioning in modular model predictive sampling.
"""

import pytest
import torch
import pyro
import numpy as np
from pyrovelocity.models.modular.factory import create_piecewise_activation_model


class TestParameterConditioning:
    """Test parameter conditioning functionality in modular models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Set seeds for reproducibility
        torch.manual_seed(42)
        pyro.set_rng_seed(42)
        np.random.seed(42)
        
        self.model = create_piecewise_activation_model()
        self.num_cells = 50
        self.num_genes = 20
        
    def test_t_m_star_conditioning_dict_output(self):
        """Test that T_M_star conditioning works with dict output format."""
        target_t_m_star = 7.0
        
        # Generate with conditioning
        conditioned_samples = self.model.generate_predictive_samples(
            num_cells=self.num_cells,
            num_genes=self.num_genes,
            num_samples=1,
            return_format="dict",
            condition_values={"T_M_star": target_t_m_star}
        )
        
        # Check that T_M_star is conditioned correctly
        assert "T_M_star" in conditioned_samples
        t_m_star_value = conditioned_samples["T_M_star"]
        
        # Extract scalar value
        if hasattr(t_m_star_value, 'item'):
            actual_value = float(t_m_star_value.item())
        elif hasattr(t_m_star_value, '__iter__') and not isinstance(t_m_star_value, str):
            actual_value = float(t_m_star_value.flatten()[0])
        else:
            actual_value = float(t_m_star_value)
            
        assert abs(actual_value - target_t_m_star) < 1e-6, f"Expected {target_t_m_star}, got {actual_value}"
    
    def test_t_m_star_conditioning_anndata_output(self):
        """Test that T_M_star conditioning works with AnnData output format."""
        target_t_m_star = 5.5
        
        # Generate with conditioning
        conditioned_adata = self.model.generate_predictive_samples(
            num_cells=self.num_cells,
            num_genes=self.num_genes,
            num_samples=1,
            return_format="anndata",
            condition_values={"T_M_star": target_t_m_star}
        )
        
        # Check that T_M_star is stored correctly in AnnData
        assert "true_parameters" in conditioned_adata.uns
        assert "T_M_star" in conditioned_adata.uns["true_parameters"]
        
        t_m_star_value = conditioned_adata.uns["true_parameters"]["T_M_star"]
        
        # Extract scalar value
        if isinstance(t_m_star_value, (torch.Tensor, np.ndarray)):
            actual_value = float(t_m_star_value.item() if hasattr(t_m_star_value, 'item') else t_m_star_value.flatten()[0])
        else:
            actual_value = float(t_m_star_value)
            
        assert abs(actual_value - target_t_m_star) < 1e-6, f"Expected {target_t_m_star}, got {actual_value}"
    
    def test_conditioning_vs_no_conditioning(self):
        """Test that conditioning produces different results than no conditioning."""
        target_t_m_star = 8.0
        
        # Generate without conditioning
        unconditioned_samples = self.model.generate_predictive_samples(
            num_cells=self.num_cells,
            num_genes=self.num_genes,
            num_samples=1,
            return_format="dict"
        )
        
        # Generate with conditioning
        conditioned_samples = self.model.generate_predictive_samples(
            num_cells=self.num_cells,
            num_genes=self.num_genes,
            num_samples=1,
            return_format="dict",
            condition_values={"T_M_star": target_t_m_star}
        )
        
        # Extract T_M_star values
        unconditioned_t_m = float(unconditioned_samples["T_M_star"].item())
        conditioned_t_m = float(conditioned_samples["T_M_star"].item())
        
        # Should be different
        assert abs(unconditioned_t_m - conditioned_t_m) > 1e-6, "Conditioning should produce different results"
        
        # Conditioned value should match target
        assert abs(conditioned_t_m - target_t_m_star) < 1e-6, f"Expected {target_t_m_star}, got {conditioned_t_m}"
    
    def test_multiple_parameter_conditioning(self):
        """Test conditioning on multiple parameters simultaneously."""
        target_t_m_star = 6.0
        target_lambda_j = torch.ones(self.num_cells) * 0.5
        
        # Generate with multiple conditioning
        conditioned_samples = self.model.generate_predictive_samples(
            num_cells=self.num_cells,
            num_genes=self.num_genes,
            num_samples=1,
            return_format="dict",
            condition_values={
                "T_M_star": target_t_m_star,
                "lambda_j": target_lambda_j
            }
        )
        
        # Check T_M_star conditioning
        t_m_star_value = float(conditioned_samples["T_M_star"].item())
        assert abs(t_m_star_value - target_t_m_star) < 1e-6
        
        # Check lambda_j conditioning
        lambda_j_values = conditioned_samples["lambda_j"].flatten()
        expected_lambda_j = target_lambda_j.flatten()
        assert torch.allclose(lambda_j_values, expected_lambda_j, atol=1e-6)
    
    def test_conditioning_preserves_other_parameters(self):
        """Test that conditioning doesn't break generation of other parameters."""
        target_t_m_star = 4.5
        
        conditioned_samples = self.model.generate_predictive_samples(
            num_cells=self.num_cells,
            num_genes=self.num_genes,
            num_samples=1,
            return_format="dict",
            condition_values={"T_M_star": target_t_m_star}
        )
        
        # Check that expected parameters are present
        expected_params = ["u_obs", "s_obs", "R_on", "gamma_star", "t_on_star", "delta_star", "U_0i"]
        for param in expected_params:
            assert param in conditioned_samples, f"Parameter {param} missing from conditioned samples"
            
        # Check that observations have correct shape (with batch dimension)
        assert conditioned_samples["u_obs"].shape == (1, self.num_cells, self.num_genes)
        assert conditioned_samples["s_obs"].shape == (1, self.num_cells, self.num_genes)