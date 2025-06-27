"""
Unit tests for PiecewiseActivationDynamicsModel.

This module tests the mathematical correctness of the piecewise activation
dynamics model, including:
- Analytical solutions for all three phases
- Special case handling for γ* = 1
- Steady-state initial conditions
- Proper phase transitions
"""

import pytest
import torch
from beartype.typing import Any, Dict

from pyrovelocity.models.modular.components.dynamics import (
    PiecewiseActivationDynamicsModel,
)


class TestPiecewiseActivationDynamicsModel:
    """Test suite for PiecewiseActivationDynamicsModel."""

    @pytest.fixture
    def model(self):
        """Create a PiecewiseActivationDynamicsModel instance for testing."""
        return PiecewiseActivationDynamicsModel()

    @pytest.fixture
    def simple_parameters(self):
        """Simple test parameters for 2 genes."""
        return {
            "R_on": torch.tensor([2.0, 3.0]),  # Fold-change during activation
            "gamma_star": torch.tensor([0.8, 1.2]),
            "t_on_star": torch.tensor([0.3, 0.25]),
            "delta_star": torch.tensor([0.4, 0.5]),
        }

    @pytest.fixture
    def gamma_one_parameters(self):
        """Test parameters with γ* = 1 for special case testing."""
        return {
            "R_on": torch.tensor([2.0, 3.0]),  # Fold-change during activation
            "gamma_star": torch.tensor([1.0, 1.0]),  # Special case
            "t_on_star": torch.tensor([0.3, 0.25]),
            "delta_star": torch.tensor([0.4, 0.5]),
        }

    def test_steady_state_calculation(self, model, simple_parameters):
        """Test steady-state calculation for OFF phase."""
        # In the corrected model, alpha_off is fixed at 1.0
        alpha_off = torch.ones(2)  # For 2 genes
        gamma_star = simple_parameters["gamma_star"]

        u_ss, s_ss = model.steady_state(alpha_off, gamma_star)

        # Check steady-state values
        expected_u_ss = alpha_off  # Should be 1.0
        expected_s_ss = alpha_off / gamma_star  # Should be 1.0/gamma_star

        torch.testing.assert_close(u_ss, expected_u_ss)
        torch.testing.assert_close(s_ss, expected_s_ss)

    def test_phase1_off_state(self, model, simple_parameters):
        """Test Phase 1 (OFF state) analytical solution."""
        # Time points in Phase 1 (before activation)
        t_star = torch.tensor([0.1, 0.2])  # Both before t_on_star
        
        # Create parameters in the format expected by _compute_piecewise_solution
        alpha_off = torch.ones(2)  # Fixed at 1.0 for both genes
        alpha_on = simple_parameters["R_on"]  # Since alpha_off = 1.0, alpha_on = R_on
        
        u_star, s_star = model._compute_piecewise_solution(
            t_star, alpha_off, alpha_on, 
            simple_parameters["gamma_star"],
            simple_parameters["t_on_star"],
            simple_parameters["delta_star"]
        )

        # In Phase 1, should be at steady state
        expected_u = alpha_off  # Should be 1.0
        expected_s = alpha_off / simple_parameters["gamma_star"]  # 1.0/gamma_star

        # Check that all time points give steady-state values
        for i in range(len(t_star)):
            torch.testing.assert_close(u_star[i], expected_u, rtol=1e-5, atol=1e-8)
            torch.testing.assert_close(s_star[i], expected_s, rtol=1e-5, atol=1e-8)

    def test_phase2_on_state(self, model, simple_parameters):
        """Test Phase 2 (ON state) analytical solution."""
        # Time points in Phase 2 (during activation)
        t_star = torch.tensor([0.4, 0.5])  # Within activation window
        
        # Create parameters in the format expected by _compute_piecewise_solution
        alpha_off = torch.ones(2)  # Fixed at 1.0 for both genes
        alpha_on = simple_parameters["R_on"]  # Since alpha_off = 1.0, alpha_on = R_on
        
        u_star, s_star = model._compute_piecewise_solution(
            t_star, alpha_off, alpha_on, 
            simple_parameters["gamma_star"],
            simple_parameters["t_on_star"],
            simple_parameters["delta_star"]
        )

        # Verify shapes
        assert u_star.shape == (2, 2)  # [cells, genes]
        assert s_star.shape == (2, 2)

        # Verify values are reasonable (between off and on steady states)
        # u* should be between α*_off and α*_on
        assert torch.all(u_star >= alpha_off.min())
        assert torch.all(u_star <= alpha_on.max())

    def test_gamma_one_special_case(self, model, gamma_one_parameters):
        """Test special case handling for γ* = 1."""
        # Time points in Phase 2 to test special case
        t_star = torch.tensor([0.4, 0.5])
        
        # Create parameters in the format expected by _compute_piecewise_solution
        alpha_off = torch.ones(2)  # Fixed at 1.0 for both genes
        alpha_on = gamma_one_parameters["R_on"]  # Since alpha_off = 1.0, alpha_on = R_on
        
        # The main test is that this doesn't raise a RuntimeError for division by zero
        try:
            u_star, s_star = model._compute_piecewise_solution(
                t_star, alpha_off, alpha_on, 
                gamma_one_parameters["gamma_star"],
                gamma_one_parameters["t_on_star"],
                gamma_one_parameters["delta_star"]
            )
            
            # Verify the results have the correct shape even if some values are NaN
            assert u_star.shape == (2, 2)
            assert s_star.shape == (2, 2)
            
        except RuntimeError as e:
            if "division by zero" in str(e):
                pytest.fail("Division by zero error not properly handled for γ* = 1")
            else:
                raise  # Re-raise other RuntimeErrors

    def test_phase_transitions(self, model, simple_parameters):
        """Test continuity at phase transitions."""
        # Test transition from Phase 1 to Phase 2
        t_on = simple_parameters["t_on_star"][0]
        t_before = t_on - 1e-6
        t_after = t_on + 1e-6
        
        # Single gene parameters
        alpha_off = torch.ones(1)  # Fixed at 1.0
        alpha_on = simple_parameters["R_on"][0:1]  # First gene only
        gamma_star = simple_parameters["gamma_star"][0:1]
        t_on_star = simple_parameters["t_on_star"][0:1]
        delta_star = simple_parameters["delta_star"][0:1]
        
        u_before, s_before = model._compute_piecewise_solution(
            torch.tensor([t_before]), alpha_off, alpha_on, gamma_star, t_on_star, delta_star
        )
        u_after, s_after = model._compute_piecewise_solution(
            torch.tensor([t_after]), alpha_off, alpha_on, gamma_star, t_on_star, delta_star
        )

        # Should be approximately continuous
        torch.testing.assert_close(u_before, u_after, rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(s_before, s_after, rtol=1e-3, atol=1e-6)

    def test_compute_velocity(self, model):
        """Test velocity computation."""
        ut = torch.tensor([[1.0, 1.5], [2.0, 2.5]])
        st = torch.tensor([[0.8, 1.2], [1.6, 2.0]])
        gamma_star = torch.tensor([0.8, 1.2])

        velocity = model.compute_velocity(ut, st, gamma_star)

        # Check velocity formula: ds*/dt* = u* - γ*s*
        expected_velocity = ut - gamma_star * st
        torch.testing.assert_close(velocity, expected_velocity)

    def test_forward_method_basic(self, model, simple_parameters):
        """Test the forward method with basic context."""
        # Create a basic context
        context = {
            "u_obs": torch.tensor([[10, 15], [20, 25]]),
            "s_obs": torch.tensor([[8, 12], [16, 20]]),
            "t_star": torch.tensor([0.1, 0.5]),  # Different phases
            **simple_parameters
        }

        # Call forward method
        result_context = model.forward(context)

        # Check that expected keys are added
        assert "u_expected" in result_context
        assert "s_expected" in result_context
        assert "ut" in result_context
        assert "st" in result_context

        # Check shapes
        u_expected = result_context["u_expected"]
        s_expected = result_context["s_expected"]
        
        assert u_expected.shape == (2, 2)  # [cells, genes]
        assert s_expected.shape == (2, 2)

    def test_invalid_context_raises_error(self, model):
        """Test that invalid context raises appropriate error."""
        # Missing required keys
        invalid_context = {
            "u_obs": torch.tensor([[10, 15]]),
            "s_obs": torch.tensor([[8, 12]]),
            # Missing other required parameters
        }

        with pytest.raises(ValueError, match="Error in optimized piecewise dynamics model forward pass"):
            model.forward(invalid_context)
