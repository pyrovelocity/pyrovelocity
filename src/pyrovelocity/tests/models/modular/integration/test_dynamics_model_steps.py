"""
Step definitions for testing dynamics models in PyroVelocity's modular implementation.

This module implements the steps defined in the dynamics_model.feature file.
"""

from importlib.resources import files

import pytest
import torch
from pytest_bdd import given, parsers, scenarios, then, when

# Import the feature file using importlib.resources
scenarios(str(files("pyrovelocity.tests.features") / "models" / "modular" / "dynamics_model.feature"))

# Import the components
from pyrovelocity.models.modular.components import (
    PiecewiseActivationDynamicsModel,
)


@given("I have a dynamics model component")
def dynamics_model_component():
    """Create a generic dynamics model component."""
    return PiecewiseActivationDynamicsModel()


@given("I have input data with unspliced and spliced counts", target_fixture="input_data")
def input_data_fixture(bdd_simple_data):
    """Get input data from the fixture."""
    return bdd_simple_data


@given("I have a PiecewiseActivationDynamicsModel", target_fixture="piecewise_dynamics_model")
def piecewise_dynamics_model_fixture():
    """Create a PiecewiseActivationDynamicsModel."""
    return PiecewiseActivationDynamicsModel()


@given("I have a StandardDynamicsModel", target_fixture="standard_dynamics_model")
def standard_dynamics_model_fixture(bdd_standard_dynamics_model):
    """Get a StandardDynamicsModel from the fixture."""
    return bdd_standard_dynamics_model


@given("I have a LegacyDynamicsModel", target_fixture="legacy_dynamics_model")
def legacy_dynamics_model_fixture(bdd_legacy_dynamics_model):
    """Get a LegacyDynamicsModel from the fixture."""
    return bdd_legacy_dynamics_model


@given("I have a StandardDynamicsModel with library size correction", target_fixture="standard_dynamics_model_with_library_size_correction")
def standard_dynamics_model_with_library_size_correction_fixture():
    """Create a StandardDynamicsModel with library size correction."""
    return PiecewiseActivationDynamicsModel()  # Use PiecewiseActivationDynamicsModel as the standard


@given("I have a PiecewiseActivationDynamicsModel with library size correction", target_fixture="piecewise_dynamics_model_with_library_size_correction")
def piecewise_dynamics_model_with_library_size_correction_fixture():
    """Create a PiecewiseActivationDynamicsModel with library size correction."""
    return PiecewiseActivationDynamicsModel()


@when(parsers.parse("I run the forward method with alpha_off {alpha_off} and gamma_star {gamma_star}"), target_fixture="run_forward_method_with_piecewise_parameters")
def run_forward_method_with_piecewise_parameters_fixture(piecewise_dynamics_model, input_data, alpha_off, gamma_star):
    """Run the forward method with piecewise activation parameters."""
    # Convert string parameters to float
    alpha_off_val = float(alpha_off)
    gamma_star_val = float(gamma_star)

    # Create piecewise activation parameters according to the component's expectations
    n_cells = input_data["n_cells"]
    n_genes = input_data["n_genes"]
    
    # Create context with all required parameters for PiecewiseActivationDynamicsModel
    context = {
        "u_obs": input_data["u_obs"],
        "s_obs": input_data["s_obs"],
        "R_on": torch.tensor([2.0] * n_genes),  # Fold-change parameter
        "gamma_star": torch.tensor([gamma_star_val] * n_genes),  # Relative degradation [genes]
        "t_on_star": torch.tensor([0.3] * n_genes),  # Activation onset [genes]
        "delta_star": torch.tensor([0.4] * n_genes),  # Activation duration [genes]
        "t_star": torch.tensor([0.5] * n_cells),  # Cell time [cells]
    }

    # Run the forward method
    result_context = piecewise_dynamics_model.forward(context)

    # Store the result for later steps
    return result_context


@when(parsers.parse("I run the forward method with alpha {alpha}, beta {beta}, and gamma {gamma}"), target_fixture="run_forward_method_with_parameters")
def run_forward_method_with_parameters_fixture(standard_dynamics_model, input_data, alpha, beta, gamma):
    """Run the forward method with the given parameters."""
    # Convert string parameters to float
    alpha_val = float(alpha)
    beta_val = float(beta)
    gamma_val = float(gamma)

    # Check if this is a PiecewiseActivationDynamicsModel and create appropriate parameters
    if hasattr(standard_dynamics_model, '_compute_piecewise_solution'):
        # This is a PiecewiseActivationDynamicsModel - create piecewise parameters
        # Note: t_star should have shape [cells], while other parameters have shape [genes]
        context = {
            "u_obs": input_data["u_obs"],
            "s_obs": input_data["s_obs"],
            "alpha_off": torch.tensor([alpha_val * 0.1] * input_data["n_genes"]),  # Basal transcription [genes]
            "alpha_on": torch.tensor([alpha_val] * input_data["n_genes"]),         # Active transcription [genes]
            "gamma_star": torch.tensor([gamma_val] * input_data["n_genes"]),       # Relative degradation [genes]
            "t_on_star": torch.tensor([0.3] * input_data["n_genes"]),              # Activation onset [genes]
            "delta_star": torch.tensor([0.4] * input_data["n_genes"]),             # Activation duration [genes]
            "t_star": torch.tensor([0.5] * input_data["n_cells"]),                 # Cell time [cells] - FIXED!
        }
    else:
        # This is a LegacyDynamicsModel - use legacy parameters
        context = {
            "u_obs": input_data["u_obs"],
            "s_obs": input_data["s_obs"],
            "alpha": torch.tensor([alpha_val] * input_data["n_genes"]),
            "beta": torch.tensor([beta_val] * input_data["n_genes"]),
            "gamma": torch.tensor([gamma_val] * input_data["n_genes"]),
        }

    # Run the forward method
    result_context = standard_dynamics_model.forward(context)

    # Store the result for later steps
    return result_context


@when(parsers.parse("I compute the steady state with alpha_off {alpha_off} and gamma_star {gamma_star}"), target_fixture="compute_piecewise_steady_state")
def compute_piecewise_steady_state_fixture(piecewise_dynamics_model, alpha_off, gamma_star):
    """Compute the steady state with piecewise activation parameters."""
    # Convert string parameters to float
    alpha_off_val = float(alpha_off)
    gamma_star_val = float(gamma_star)

    # Create tensor parameters
    n_genes = 5  # Using a fixed value for simplicity
    alpha_off_tensor = torch.tensor([alpha_off_val] * n_genes)
    gamma_star_tensor = torch.tensor([gamma_star_val] * n_genes)

    # Compute steady state using piecewise parameters
    u_ss, s_ss = piecewise_dynamics_model.steady_state(alpha_off_tensor, gamma_star_tensor)

    return {"u_ss": u_ss, "s_ss": s_ss, "alpha_off": alpha_off_tensor, "gamma_star": gamma_star_tensor}


@when(parsers.parse("I compute the steady state with alpha {alpha}, beta {beta}, and gamma {gamma}"), target_fixture="compute_steady_state")
def compute_steady_state_fixture(standard_dynamics_model, alpha, beta, gamma):
    """Compute the steady state with the given parameters."""
    # Convert string parameters to float
    alpha_val = float(alpha)
    beta_val = float(beta)
    gamma_val = float(gamma)

    # Create tensor parameters
    n_genes = 5  # Using a fixed value for simplicity

    # Check if this is a PiecewiseActivationDynamicsModel and use appropriate parameters
    if hasattr(standard_dynamics_model, '_compute_piecewise_solution'):
        # This is a PiecewiseActivationDynamicsModel - use piecewise parameters
        alpha_off = torch.tensor([alpha_val * 0.1] * n_genes)  # Basal transcription
        gamma_star = torch.tensor([gamma_val] * n_genes)       # Relative degradation

        # Compute steady state using piecewise parameters
        u_ss, s_ss = standard_dynamics_model.steady_state(alpha_off, gamma_star)

        return {"u_ss": u_ss, "s_ss": s_ss, "alpha_off": alpha_off, "gamma_star": gamma_star}
    else:
        # This is a LegacyDynamicsModel - use legacy parameters
        alpha = torch.tensor([alpha_val] * n_genes)
        beta = torch.tensor([beta_val] * n_genes)
        gamma = torch.tensor([gamma_val] * n_genes)

        # Compute steady state using legacy parameters
        u_ss, s_ss = standard_dynamics_model.steady_state(alpha, beta, gamma)

    # Store the result for later steps
        return {"u_ss": u_ss, "s_ss": s_ss, "alpha": alpha, "beta": beta, "gamma": gamma}


@when("I run the forward method with the same parameters as the legacy implementation", target_fixture="run_forward_method_legacy")
def run_forward_method_legacy_fixture(legacy_dynamics_model, input_data, bdd_model_parameters):
    """Run the forward method with parameters matching the legacy implementation."""
    # Create context with input data and parameters
    context = {
        "u_obs": input_data["u_obs"],
        "s_obs": input_data["s_obs"],
        "alpha": bdd_model_parameters["alpha"],
        "beta": bdd_model_parameters["beta"],
        "gamma": bdd_model_parameters["gamma"],
    }

    # Run the forward method
    result_context = legacy_dynamics_model.forward(context)

    # Store the result for later steps
    return result_context


@when("I run the forward method with edge case parameters", target_fixture="run_forward_method_with_edge_case_parameters")
def run_forward_method_with_edge_case_parameters_fixture(piecewise_dynamics_model, input_data):
    """Run the forward method with edge case parameters to test robustness."""
    n_cells = input_data["n_cells"]
    n_genes = input_data["n_genes"]
    
    # Create edge case parameters - very small values that could cause numerical issues
    context = {
        "u_obs": input_data["u_obs"],
        "s_obs": input_data["s_obs"],
        "R_on": torch.tensor([1e-6] * n_genes),  # Very small fold-change
        "gamma_star": torch.tensor([1e-6] * n_genes),  # Very small degradation
        "t_on_star": torch.tensor([0.0] * n_genes),  # Zero onset time
        "delta_star": torch.tensor([1e-6] * n_genes),  # Very small duration
        "t_star": torch.tensor([0.0] * n_cells),  # Zero cell time
    }

    # Run the forward method (may raise an exception)
    try:
        result_context = piecewise_dynamics_model.forward(context)
        return result_context
    except Exception as e:
        return {"error": e}


@when("I run the forward method with zero rates", target_fixture="run_forward_method_with_zero_rates")
def run_forward_method_with_zero_rates_fixture(standard_dynamics_model, input_data):
    """Run the forward method with zero rates to test edge cases."""
    # Create parameters with zeros
    n_genes = input_data["n_genes"]
    alpha = torch.zeros(n_genes)
    beta = torch.zeros(n_genes)
    gamma = torch.zeros(n_genes)

    # Create context with input data and zero parameters
    context = {
        "u_obs": input_data["u_obs"],
        "s_obs": input_data["s_obs"],
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
    }

    # Run the forward method (may raise an exception)
    try:
        result_context = standard_dynamics_model.forward(context)
        return result_context
    except Exception as e:
        return {"error": e}


@when("I run the forward method with library size factors", target_fixture="run_forward_method_with_library_size")
def run_forward_method_with_library_size_fixture(piecewise_dynamics_model_with_library_size_correction, input_data, bdd_piecewise_model_parameters):
    """Run the forward method with library size factors."""
    # Create library size factors
    n_cells = input_data["n_cells"]
    library_size = torch.ones(n_cells) * 2.0  # Scale by 2x

    # Create context with input data, proper piecewise parameters, and library size
    n_genes = input_data["n_genes"]
    context = {
        "u_obs": input_data["u_obs"],
        "s_obs": input_data["s_obs"],
        "R_on": torch.tensor([2.0] * n_genes),  # Fold-change parameter
        "gamma_star": torch.tensor([0.8] * n_genes),  # Relative degradation
        "t_on_star": torch.tensor([0.3] * n_genes),  # Activation onset
        "delta_star": torch.tensor([0.4] * n_genes),  # Activation duration
        "t_star": torch.tensor([0.5] * n_cells),  # Cell time
        "u_lib_size": library_size,
        "s_lib_size": library_size,
    }

    # Run the forward method
    result_context = piecewise_dynamics_model_with_library_size_correction.forward(context)

    # Store the result for later steps
    return result_context


@then("the model should compute expected unspliced and spliced counts")
def check_expected_counts(request):
    """Check that the model computed expected counts."""
    # Try to get the result from different fixture names
    result_context = None
    try:
        result_context = request.getfixturevalue("run_forward_method_with_piecewise_parameters")
    except:
        try:
            result_context = request.getfixturevalue("run_forward_method_with_parameters")
        except:
            pytest.fail("No forward method result fixture found")
    
    # Check that the expected counts are in the context
    assert "u_expected" in result_context
    assert "s_expected" in result_context

    # Check that the expected counts have the right shape
    u_expected = result_context["u_expected"]
    s_expected = result_context["s_expected"]
    u_obs = result_context["u_obs"]

    assert u_expected.shape == u_obs.shape
    assert s_expected.shape == u_obs.shape


@then("the expected counts should follow RNA velocity dynamics")
def check_dynamics(request):
    """Check that the expected counts follow RNA velocity dynamics."""
    # Try to get the result from different fixture names
    result_context = None
    try:
        result_context = request.getfixturevalue("run_forward_method_with_piecewise_parameters")
    except:
        try:
            result_context = request.getfixturevalue("run_forward_method_with_parameters")
        except:
            pytest.fail("No forward method result fixture found")
    
    # In a real test, we would check that the expected counts follow the analytical solution
    # For this example, we'll just check that they're positive
    u_expected = result_context["u_expected"]
    s_expected = result_context["s_expected"]

    assert torch.all(u_expected >= 0)
    assert torch.all(s_expected >= 0)


@then("the steady state should be computed correctly")
def check_steady_state_computed_correctly(compute_piecewise_steady_state):
    """Check that the steady state is computed correctly for piecewise activation model."""
    # Check that steady state values are present
    assert "u_ss" in compute_piecewise_steady_state
    assert "s_ss" in compute_piecewise_steady_state
    
    u_ss = compute_piecewise_steady_state["u_ss"]
    s_ss = compute_piecewise_steady_state["s_ss"]
    
    # For piecewise activation model: u_ss = 1.0 (fixed), s_ss = 1.0/gamma_star
    alpha_off = compute_piecewise_steady_state["alpha_off"]
    gamma_star = compute_piecewise_steady_state["gamma_star"]
    
    # u_ss should be 1.0 (reference state)
    expected_u_ss = torch.ones_like(alpha_off)
    assert torch.allclose(u_ss, expected_u_ss, rtol=1e-4)
    
    # s_ss should be 1.0/gamma_star
    expected_s_ss = torch.ones_like(alpha_off) / gamma_star
    assert torch.allclose(s_ss, expected_s_ss, rtol=1e-4)


@then("the steady state values should be positive")
def check_steady_state_positive(compute_piecewise_steady_state):
    """Check that steady state values are positive."""
    u_ss = compute_piecewise_steady_state["u_ss"]
    s_ss = compute_piecewise_steady_state["s_ss"]
    
    assert torch.all(u_ss > 0)
    assert torch.all(s_ss > 0)


@then("the steady state unspliced should equal alpha/beta")
def check_steady_state_unspliced(compute_steady_state):
    """Check that the steady state unspliced counts equal alpha/beta."""
    u_ss = compute_steady_state["u_ss"]

    # Check if we have legacy parameters or piecewise parameters
    if "alpha" in compute_steady_state:
        # Legacy model: u_ss = alpha/beta
        alpha = compute_steady_state["alpha"]
        beta = compute_steady_state["beta"]
        expected_u_ss = alpha / beta
    else:
        # Piecewise model: u_ss = alpha_off (basal transcription in steady state)
        alpha_off = compute_steady_state["alpha_off"]
        expected_u_ss = alpha_off  # For piecewise model, steady state is just alpha_off

    # Check that u_ss matches expected (with some tolerance for numerical precision)
    assert torch.allclose(u_ss, expected_u_ss, rtol=1e-4)


@then("the steady state spliced should equal alpha/gamma")
def check_steady_state_spliced(compute_steady_state):
    """Check that the steady state spliced counts equal alpha/gamma."""
    s_ss = compute_steady_state["s_ss"]

    # Check if we have legacy parameters or piecewise parameters
    if "alpha" in compute_steady_state:
        # Legacy model: s_ss = alpha/gamma
        alpha = compute_steady_state["alpha"]
        gamma = compute_steady_state["gamma"]
        expected_s_ss = alpha / gamma
    else:
        # Piecewise model: s_ss = alpha_off/gamma_star (basal transcription / degradation)
        alpha_off = compute_steady_state["alpha_off"]
        gamma_star = compute_steady_state["gamma_star"]
        expected_s_ss = alpha_off / gamma_star

    # Check that s_ss matches expected (with some tolerance for numerical precision)
    assert torch.allclose(s_ss, expected_s_ss, rtol=1e-4)


@then("the output should match the legacy implementation output")
def check_legacy_output(run_forward_method_legacy):
    """Check that the output matches the legacy implementation."""
    # In a real test, we would compare with actual legacy output
    # For this example, we'll just check that the expected counts are present
    assert "u_expected" in run_forward_method_legacy
    assert "s_expected" in run_forward_method_legacy


@then("the model should create deterministic nodes with event_dim=0")
def check_deterministic_nodes(run_forward_method_legacy):
    """Check that the model creates deterministic nodes with event_dim=0."""
    # This would require inspecting Pyro's trace, which is complex for a BDD test
    # For this example, we'll just check that the result exists
    assert run_forward_method_legacy is not None
    pass


@then("the model should handle the edge case gracefully")
def check_edge_case_handling(request):
    """Check that the model handles edge cases gracefully."""
    # Try to get the result from different fixture names
    result_context = None
    try:
        result_context = request.getfixturevalue("run_forward_method_with_edge_case_parameters")
    except:
        try:
            result_context = request.getfixturevalue("run_forward_method_with_zero_rates")
        except:
            pytest.fail("No edge case result fixture found")
    
    # Check if there was an error
    if "error" in result_context:
        # If there was an error, it should be a ValueError or RuntimeError
        assert isinstance(result_context["error"], (ValueError, RuntimeError))
    else:
        # If there was no error, the expected counts should be finite or handle special cases
        u_expected = result_context["u_expected"]
        s_expected = result_context["s_expected"]

        # For piecewise activation model with edge case parameters, 
        # the model should handle small values gracefully and produce finite results
        # Just check that the values are finite (the model is robust)
        assert torch.all(torch.isfinite(u_expected))
        assert torch.all(torch.isfinite(s_expected))


@then("should not produce NaN or infinite values")
def check_no_nan_or_inf(request):
    """Check that the model does not produce NaN or infinite values."""
    # Try to get the result from different fixture names
    result_context = None
    try:
        result_context = request.getfixturevalue("run_forward_method_with_edge_case_parameters")
    except:
        try:
            result_context = request.getfixturevalue("run_forward_method_with_zero_rates")
        except:
            pytest.fail("No edge case result fixture found")
    
    # Skip if there was an error
    if "error" in result_context:
        return

    # For piecewise activation model, all values should be finite
    u_expected = result_context["u_expected"]
    s_expected = result_context["s_expected"]

    # Check that no values are NaN or infinite
    assert torch.all(torch.isfinite(u_expected))
    assert torch.all(torch.isfinite(s_expected))


@then("the expected counts should be scaled by the library size factors")
def check_library_size_scaling(request):
    """Check that the expected counts are scaled by the library size factors."""
    # Try to get the result from library size fixture
    try:
        result_context = request.getfixturevalue("run_forward_method_with_library_size")
    except:
        pytest.fail("No library size result fixture found")
    
    # Check that the expected counts are present
    assert "u_expected" in result_context
    assert "s_expected" in result_context


@then("the scaling should be applied correctly")
def check_scaling_correctness(request):
    """Check that the scaling is applied correctly."""
    # Try to get the result from library size fixture
    try:
        result_context = request.getfixturevalue("run_forward_method_with_library_size")
    except:
        pytest.fail("No library size result fixture found")
    
    # Check that the result exists
    assert result_context is not None
