"""
Test parameter categorization system for PyroVelocity plotting functions.

This module tests the new parameter categorization infrastructure that allows
model-agnostic parameter selection for plotting functions.
"""

import pytest
from typing import Dict, Any

from pyrovelocity.plots.parameter_metadata import (
    get_parameters_for_plot,
    infer_component_name_from_parameters
)
from pyrovelocity.models.jax.components.poisson.metadata import create_poisson_prior_metadata
from pyrovelocity.models.jax.components.piecewise.metadata import create_piecewise_activation_prior_metadata


class TestPoissonModelCategorization:
    """Test parameter categorization for poisson model."""
    
    def test_poisson_parameter_categories(self):
        """Test that poisson model metadata has correct parameter categories."""
        metadata = create_poisson_prior_metadata()
        
        # Test that categories are properly assigned
        gene_expression_params = metadata.get_parameters_by_category("gene_expression")
        technical_scaling_params = metadata.get_parameters_by_category("technical_scaling")
        temporal_dynamics_params = metadata.get_parameters_by_category("temporal_dynamics")
        
        # Expected categorization for poisson model
        assert set(gene_expression_params) == {"r_u_i", "r_s_i"}
        assert set(technical_scaling_params) == {"lambda_j", "U_0i"}
        assert set(temporal_dynamics_params) == {"t_star"}
    
    def test_poisson_component_detection(self):
        """Test component name detection for poisson parameters."""
        # Typical poisson parameters
        poisson_params = {
            "lambda_j": [0.5, 1.2, 0.8],
            "t_star": [0.5, 0.5, 0.5], 
            "U_0i": [10.0, 25.0, 15.0],
            "r_u_i": [1.2, 0.8, 1.5],
            "r_s_i": [0.9, 1.1, 1.3]
        }
        
        component_name = infer_component_name_from_parameters(poisson_params)
        assert component_name == "poisson_prior"
    
    def test_get_parameters_for_plot_poisson(self):
        """Test get_parameters_for_plot with poisson model parameters."""
        poisson_params = {
            "lambda_j": [0.5, 1.2, 0.8],
            "t_star": [0.5, 0.5, 0.5], 
            "U_0i": [10.0, 25.0, 15.0],
            "r_u_i": [1.2, 0.8, 1.5],
            "r_s_i": [0.9, 1.1, 1.3]
        }
        
        # Test gene expression parameters
        gene_expr_params = get_parameters_for_plot(
            category="gene_expression",
            available_parameters=poisson_params
        )
        assert gene_expr_params == ["r_u_i", "r_s_i"]  # Should be ordered by plot_order
        
        # Test technical scaling parameters
        tech_params = get_parameters_for_plot(
            category="technical_scaling",
            available_parameters=poisson_params
        )
        assert tech_params == ["lambda_j", "U_0i"]  # Should be ordered by plot_order
        
        # Test temporal dynamics parameters
        temporal_params = get_parameters_for_plot(
            category="temporal_dynamics",
            available_parameters=poisson_params
        )
        assert temporal_params == ["t_star"]


class TestPiecewiseModelCategorization:
    """Test parameter categorization for piecewise model."""
    
    def test_piecewise_parameter_categories(self):
        """Test that piecewise model metadata has correct parameter categories."""
        metadata = create_piecewise_activation_prior_metadata()
        
        # Test key parameter categories
        gene_expression_params = metadata.get_parameters_by_category("gene_expression")
        technical_scaling_params = metadata.get_parameters_by_category("technical_scaling")
        temporal_dynamics_params = metadata.get_parameters_by_category("temporal_dynamics")
        
        # Expected categorization for piecewise model (key parameters only)
        assert "R_on" in gene_expression_params
        assert "gamma_star" in gene_expression_params
        assert "t_on_star" in gene_expression_params
        assert "delta_star" in gene_expression_params
        
        assert "lambda_j" in technical_scaling_params
        assert "U_0i" in technical_scaling_params
        
        assert "T_M_star" in temporal_dynamics_params
        assert "t_star" in temporal_dynamics_params
    
    def test_piecewise_component_detection(self):
        """Test component name detection for piecewise parameters."""
        # Typical piecewise parameters
        piecewise_params = {
            "R_on": [2.0, 3.0, 2.5],
            "gamma_star": [1.0, 1.5, 0.8],
            "t_on_star": [0.3, 0.5, 0.4],
            "delta_star": [0.4, 0.6, 0.5],
            "lambda_j": [0.5, 1.2, 0.8],
            "U_0i": [10.0, 25.0, 15.0],
            "t_star": [0.2, 0.8, 0.5]
        }
        
        component_name = infer_component_name_from_parameters(piecewise_params)
        assert component_name == "piecewise_activation_prior"
    
    def test_get_parameters_for_plot_piecewise(self):
        """Test get_parameters_for_plot with piecewise model parameters."""
        piecewise_params = {
            "R_on": [2.0, 3.0, 2.5],
            "gamma_star": [1.0, 1.5, 0.8],
            "t_on_star": [0.3, 0.5, 0.4],
            "delta_star": [0.4, 0.6, 0.5],
            "lambda_j": [0.5, 1.2, 0.8],
            "U_0i": [10.0, 25.0, 15.0]
        }
        
        # Test gene expression parameters
        gene_expr_params = get_parameters_for_plot(
            category="gene_expression",
            available_parameters=piecewise_params
        )
        expected_gene_params = ["R_on", "gamma_star", "t_on_star", "delta_star"]
        assert all(param in gene_expr_params for param in expected_gene_params)
        
        # Test technical scaling parameters  
        tech_params = get_parameters_for_plot(
            category="technical_scaling",
            available_parameters=piecewise_params
        )
        expected_tech_params = ["lambda_j", "U_0i"]
        assert all(param in tech_params for param in expected_tech_params)


class TestBackwardCompatibility:
    """Test that explicit parameter lists still work."""
    
    def test_fallback_behavior(self):
        """Test fallback when categories not defined."""
        # Parameters from unknown/unsupported model
        unknown_params = {
            "custom_param_1": [1.0, 2.0],
            "custom_param_2": [3.0, 4.0]
        }
        
        # Should return empty list when no component detected
        result = get_parameters_for_plot(
            category="gene_expression",
            available_parameters=unknown_params
        )
        assert result == []
    
    def test_partial_parameter_availability(self):
        """Test behavior when only some parameters from a category are available."""
        # Only subset of poisson parameters available
        partial_params = {
            "lambda_j": [0.5, 1.2, 0.8],
            "r_u_i": [1.2, 0.8, 1.5]
            # Missing: t_star, U_0i, r_s_i
        }
        
        # Should return only available parameters from the category
        gene_expr_params = get_parameters_for_plot(
            category="gene_expression",
            available_parameters=partial_params
        )
        assert gene_expr_params == ["r_u_i"]  # Only r_u_i is available from gene_expression
        
        tech_params = get_parameters_for_plot(
            category="technical_scaling", 
            available_parameters=partial_params
        )
        assert tech_params == ["lambda_j"]  # Only lambda_j is available from technical_scaling
    
    def test_empty_category(self):
        """Test behavior when requesting non-existent category."""
        poisson_params = {
            "lambda_j": [0.5, 1.2, 0.8],
            "r_u_i": [1.2, 0.8, 1.5]
        }
        
        # Request non-existent category
        result = get_parameters_for_plot(
            category="nonexistent_category",
            available_parameters=poisson_params
        )
        assert result == []


class TestIntegration:
    """Integration tests for the complete parameter categorization system."""
    
    def test_expected_plot_outputs(self):
        """Test that the system produces expected parameter selections for plotting."""
        
        # Test Case 1: Poisson model should show different parameters in each plot
        poisson_params = {
            "lambda_j": [0.5, 1.2, 0.8],
            "t_star": [0.5, 0.5, 0.5], 
            "U_0i": [10.0, 25.0, 15.0],
            "r_u_i": [1.2, 0.8, 1.5],
            "r_s_i": [0.9, 1.1, 1.3]
        }
        
        # Plot 07 should show gene expression parameters
        plot_07_params = get_parameters_for_plot(
            category="gene_expression",
            available_parameters=poisson_params
        )
        
        # Plot 13 should show technical scaling parameters
        plot_13_params = get_parameters_for_plot(
            category="technical_scaling",
            available_parameters=poisson_params
        )
        
        # Verify expected outputs
        assert plot_07_params == ["r_u_i", "r_s_i"]
        assert plot_13_params == ["lambda_j", "U_0i"]
        
        # Verify no duplication between plots
        assert set(plot_07_params).isdisjoint(set(plot_13_params))
        
        # Test Case 2: Piecewise model should maintain backward compatibility
        piecewise_params = {
            "R_on": [2.0, 3.0, 2.5],
            "gamma_star": [1.0, 1.5, 0.8], 
            "t_on_star": [0.3, 0.5, 0.4],
            "delta_star": [0.4, 0.6, 0.5],
            "lambda_j": [0.5, 1.2, 0.8],
            "U_0i": [10.0, 25.0, 15.0]
        }
        
        # Plot 07 should show gene expression parameters (should include the original hardcoded ones)
        piecewise_plot_07 = get_parameters_for_plot(
            category="gene_expression",
            available_parameters=piecewise_params
        )
        
        # Plot 13 should show technical scaling parameters
        piecewise_plot_13 = get_parameters_for_plot(
            category="technical_scaling",
            available_parameters=piecewise_params
        )
        
        # Verify backward compatibility - original parameters should be included
        expected_piecewise_gene_expr = {"R_on", "gamma_star", "t_on_star", "delta_star"}
        assert expected_piecewise_gene_expr.issubset(set(piecewise_plot_07))
        
        expected_piecewise_tech = {"lambda_j", "U_0i"}
        assert expected_piecewise_tech.issubset(set(piecewise_plot_13))


if __name__ == "__main__":
    pytest.main([__file__])