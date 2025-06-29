"""
Tests for PyroVelocity JAX/NumPyro prior components.

This module contains tests for the prior components, including:

- test_register_standard_priors: Test registration of standard prior functions
"""

import jax
import jax.numpy as jnp

from pyrovelocity.models.jax.registry import get_prior




def test_register_standard_priors():
    """Test registration of standard prior functions."""
    # Check that piecewise activation prior function is registered
    piecewise_fn = get_prior("piecewise_activation")
    assert piecewise_fn is not None
