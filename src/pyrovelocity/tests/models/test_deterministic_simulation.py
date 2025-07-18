import diffrax
import jax
from diffrax import Solution
from jax import numpy as jnp

from pyrovelocity.models.experimental.deterministic_simulation import (
    analytical_solution_dstate_dt_dimless,
    calculate_xi,
    dstate_dt_dimensioned,
    dstate_dt_dimless,
    solve_transcription_splicing_model,
    solve_transcription_splicing_model_analytical,
    solve_transcription_splicing_model_impl,
)

import pytest


@pytest.fixture(autouse=True)
def jax_debug_mode():
    """Enable JAX debug mode for deterministic simulation tests with proper cleanup."""
    # Store original state
    original_debug_nans = jax.config.jax_debug_nans
    
    # Enable debug mode
    jax.config.update("jax_debug_nans", True)
    
    yield
    
    # Restore original state
    jax.config.update("jax_debug_nans", original_debug_nans)


def test_dstate_dt_dimless_basic():
    """Test the basic functionality of dstate_dt_dimless."""
    t = jnp.array([0.0])
    state = jnp.array([1.0, 0.5])
    params = (0.5,)
    expected_output = jnp.array([0, 0.75])
    output = dstate_dt_dimless(t, state, params)
    assert jnp.allclose(
        output, expected_output
    ), f"Expected {expected_output}, got {output}"


def test_dstate_dt_dimensioned_basic():
    """Test the basic functionality of dstate_dt_dimensioned."""
    t = jnp.array([0.0])
    state = jnp.array([1.0, 0.5])
    params = (1.0, 0.5, 0.2)
    expected_output = jnp.array([0.5, 0.4])
    output = dstate_dt_dimensioned(t, state, params)
    assert jnp.allclose(
        output, expected_output
    ), f"Expected {expected_output}, got {output}"


def test_solve_transcription_splicing_model_basic():
    """Test solving the model with a simple set of inputs."""
    ts = jnp.linspace(0, 10, 100)
    initial_state = jnp.array([0.5, 0.5])
    params = (0.5,)

    solution = solve_transcription_splicing_model_impl(
        ts, initial_state, params
    )

    assert isinstance(
        solution, Solution
    ), "Returned object should be a Solution instance"

    assert solution.ts.shape == (
        100,
    ), f"Expected ts shape to be (100,), got {solution.ts.shape}"
    assert solution.ys.shape == (
        100,
        2,
    ), f"Expected ys shape to be (100, 2), got {solution.ys.shape}"

    assert (
        solution.result == diffrax.RESULTS.successful
    ), f"Solver did not succeed; result was {solution.result}"


def test_solve_transcription_splicing_model_dimensioned():
    """Test solving the dimensioned model with a simple set of inputs."""
    ts = jnp.linspace(0, 10, 100)
    initial_state = jnp.array([1.0, 0.5])
    params = (1.0, 0.5, 0.2)

    solution = solve_transcription_splicing_model_impl(
        ts, initial_state, params, model=dstate_dt_dimensioned
    )

    assert isinstance(
        solution, Solution
    ), "Returned object should be a Solution instance"
    assert solution.ts.shape == (
        100,
    ), "Expected ts shape to match input time shape"
    assert solution.ys.shape == (
        100,
        2,
    ), "Expected ys shape to match input state shape"
    assert (
        solution.result == diffrax.RESULTS.successful
    ), "Solver did not succeed; check the result attribute"


def test_solve_transcription_splicing_model_impl_jit_equivalent():
    ts = jnp.linspace(0, 10, 100)
    initial_state = jnp.array([0.5, 0.5])
    params = (0.5,)

    solution_impl = solve_transcription_splicing_model_impl(
        ts,
        initial_state,
        params,
    )
    solution = solve_transcription_splicing_model(
        ts,
        initial_state,
        params,
    )
    assert jnp.allclose(
        solution_impl.ts, solution.ts
    ), "ts values should be equal"
    assert jnp.allclose(
        solution_impl.ys, solution.ys
    ), "ys values should be equal"


def test_solve_transcription_splicing_model_impl_dimensioned_jit_equivalent():
    ts = jnp.linspace(0, 10, 100)
    initial_state = jnp.array([1.0, 0.5])
    params = (1.0, 0.5, 0.2)

    solution_impl = solve_transcription_splicing_model_impl(
        ts,
        initial_state,
        params,
        model=dstate_dt_dimensioned,
    )
    solution = solve_transcription_splicing_model(
        ts,
        initial_state,
        params,
        model=dstate_dt_dimensioned,
    )
    assert jnp.allclose(
        solution_impl.ts, solution.ts
    ), "ts values should be equal"
    assert jnp.allclose(
        solution_impl.ys, solution.ys
    ), "ys values should be equal"


def test_calculate_xi_basic():
    """Test the calculation of the Xi parameter for the analytical solution."""
    initial_u_star = 0.5
    gamma_star = 0.2
    expected_xi = (initial_u_star - 1) / (gamma_star - 1)
    xi = calculate_xi(initial_u_star, gamma_star)
    assert jnp.allclose(
        xi, expected_xi
    ), f"Expected Xi to be {expected_xi}, got {xi}"


def test_analytical_solution_dstate_dt_dimless_basic():
    """Test the analytical solution for the dimensionless transcription-splicing model."""
    ts = jnp.linspace(0, 10, 100)
    initial_state = jnp.array([0.5, 0.5])
    params = (0.2,)

    analytical_solution = analytical_solution_dstate_dt_dimless(
        ts, initial_state, params
    )

    assert analytical_solution.shape == (
        100,
        2,
    ), "Unexpected shape of the analytical solution."


def test_solve_transcription_splicing_model_analytical_basic():
    """Test solving the model analytically with a simple set of inputs."""
    ts = jnp.linspace(0, 100, 100)
    initial_state = jnp.array([0.5, 0.5])
    params = (0.5,)

    solution = solve_transcription_splicing_model_analytical(
        ts, initial_state, params
    )

    expected_output = initial_state
    output = solution.ys[0, :]
    assert jnp.allclose(
        output,
        expected_output,
    ), f"Expected {expected_output}, got {output}"

    expected_output = jnp.array([1.0, 2.0])
    output = solution.ys[-1, :]
    assert jnp.allclose(
        output,
        expected_output,
        atol=1e-6,
    ), f"Expected {expected_output}, got {output}"

    assert isinstance(
        solution, Solution
    ), "Returned object should be a Solution instance"
    assert solution.ts.shape == (
        100,
    ), f"Expected ts shape to be (100,), got {solution.ts.shape}"
    assert solution.ys.shape == (
        100,
        2,
    ), f"Expected ys shape to be (2, 100), got {solution.ys.shape}"
    assert (
        solution.result == diffrax.RESULTS.successful
    ), "Solver did not succeed; check the result attribute"
