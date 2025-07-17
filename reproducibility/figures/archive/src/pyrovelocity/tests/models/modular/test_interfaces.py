"""
Tests for the interface definitions in PyroVelocity's modular architecture.

This module contains tests for the Protocol interfaces defined in interfaces.py,
verifying that they correctly define the contract that component implementations
must follow.

The tests in this module verify:
1. That each Protocol is runtime-checkable
2. That classes implementing the Protocol interfaces are correctly recognized
3. That the context-based interface pattern works as expected
4. That edge cases and error handling are properly handled

The context-based interface pattern is a key design pattern in the PyroVelocity
modular architecture. Components communicate through a shared context dictionary
that is passed between components during model execution. Each component updates
the context with its outputs, which are then used by subsequent components.
"""

from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import pyro
import pytest
import torch

from pyrovelocity.models.modular.interfaces import (
    DynamicsModel,
    InferenceGuide,
    LikelihoodModel,
    ObservationModel,
    ParamTensor,
    PriorModel,
)


class TestDynamicsModelInterface:
    """Tests for the DynamicsModel interface."""

    def test_dynamics_model_protocol(self):
        """Test that DynamicsModel is a runtime checkable Protocol."""
        assert isinstance(DynamicsModel, type)
        assert issubclass(DynamicsModel, Protocol)

    def test_dynamics_model_implementation(self):
        """Test that a class implementing the DynamicsModel interface is recognized."""

        class TestDynamicsModelImpl:
            def forward(
                self,
                context: Dict[str, Any],
            ) -> Dict[str, Any]:
                # Extract parameters from context
                u_obs = context.get("u_obs")
                s_obs = context.get("s_obs")
                alpha = context.get("alpha")
                beta = context.get("beta")
                gamma = context.get("gamma")

                # Compute expected counts
                u_expected = u_obs * 0.5
                s_expected = s_obs * 0.5

                # Update context with expected counts
                context["u_expected"] = u_expected
                context["s_expected"] = s_expected

                return context

            def steady_state(
                self,
                alpha: ParamTensor,
                beta: ParamTensor,
                gamma: ParamTensor,
                **kwargs: Any,
            ) -> Tuple[ParamTensor, ParamTensor]:
                # Simple implementation for testing
                return alpha / beta, alpha / gamma

            def compute_velocity(
                self,
                ut: ParamTensor,
                st: ParamTensor,
                alpha: ParamTensor,
                beta: ParamTensor,
                gamma: ParamTensor,
                **kwargs: Any,
            ) -> ParamTensor:
                # Simple implementation for testing: ds/dt = β * u - γ * s
                return beta * ut - gamma * st

        # Create an instance of the implementation
        impl = TestDynamicsModelImpl()

        # Check that it's recognized as implementing the Protocol
        assert isinstance(impl, DynamicsModel)

    def test_dynamics_model_context_interface(self):
        """Test the context-based interface for DynamicsModel."""

        class TestDynamicsModelImpl:
            def forward(
                self,
                context: Dict[str, Any],
            ) -> Dict[str, Any]:
                # Extract parameters from context
                u_obs = context.get("u_obs")
                s_obs = context.get("s_obs")
                alpha = context.get("alpha")
                beta = context.get("beta")
                gamma = context.get("gamma")

                # Compute expected counts
                u_expected = u_obs * 0.5
                s_expected = s_obs * 0.5

                # Update context with expected counts
                context["u_expected"] = u_expected
                context["s_expected"] = s_expected

                return context

            def steady_state(
                self,
                alpha: ParamTensor,
                beta: ParamTensor,
                gamma: ParamTensor,
                **kwargs: Any,
            ) -> Tuple[ParamTensor, ParamTensor]:
                # Simple implementation for testing
                return alpha / beta, alpha / gamma

        # Create an instance of the implementation
        impl = TestDynamicsModelImpl()

        # Create a context dictionary
        context = {
            "u_obs": torch.ones(10, 5),
            "s_obs": torch.ones(10, 5),
            "alpha": torch.tensor(1.0),
            "beta": torch.tensor(0.5),
            "gamma": torch.tensor(0.2),
        }

        # Call the forward method
        result = impl.forward(context)

        # Check that the context was updated correctly
        assert "u_expected" in result
        assert "s_expected" in result
        assert torch.allclose(result["u_expected"], context["u_obs"] * 0.5)
        assert torch.allclose(result["s_expected"], context["s_obs"] * 0.5)

        # Check steady state
        u_ss, s_ss = impl.steady_state(
            context["alpha"], context["beta"], context["gamma"]
        )
        assert torch.allclose(u_ss, context["alpha"] / context["beta"])
        assert torch.allclose(s_ss, context["alpha"] / context["gamma"])


class TestPriorModelInterface:
    """Tests for the PriorModel interface."""

    def test_prior_model_protocol(self):
        """Test that PriorModel is a runtime checkable Protocol."""
        assert isinstance(PriorModel, type)
        assert issubclass(PriorModel, Protocol)

    def test_prior_model_implementation(self):
        """Test that a class implementing the PriorModel interface is recognized."""

        class TestPriorModelImpl:
            def forward(
                self,
                context: Dict[str, Any],
            ) -> Dict[str, Any]:
                # Extract data from context
                u_obs = context.get("u_obs")
                s_obs = context.get("s_obs")

                # Sample parameters (mock implementation)
                n_genes = u_obs.shape[1]
                alpha = torch.ones(n_genes)
                beta = torch.ones(n_genes) * 0.5
                gamma = torch.ones(n_genes) * 0.2

                # Update context with sampled parameters
                context["alpha"] = alpha
                context["beta"] = beta
                context["gamma"] = gamma

                return context

        # Create an instance of the implementation
        impl = TestPriorModelImpl()

        # Check that it's recognized as implementing the Protocol
        assert isinstance(impl, PriorModel)

    def test_prior_model_context_interface(self):
        """Test the context-based interface for PriorModel."""

        class TestPriorModelImpl:
            def forward(
                self,
                context: Dict[str, Any],
            ) -> Dict[str, Any]:
                # Extract data from context
                u_obs = context.get("u_obs")
                s_obs = context.get("s_obs")

                # Sample parameters (mock implementation)
                n_genes = u_obs.shape[1]
                alpha = torch.ones(n_genes)
                beta = torch.ones(n_genes) * 0.5
                gamma = torch.ones(n_genes) * 0.2

                # Update context with sampled parameters
                context["alpha"] = alpha
                context["beta"] = beta
                context["gamma"] = gamma

                return context

        # Create an instance of the implementation
        impl = TestPriorModelImpl()

        # Create a context dictionary
        context = {
            "u_obs": torch.ones(10, 5),
            "s_obs": torch.ones(10, 5),
        }

        # Call the forward method
        result = impl.forward(context)

        # Check that the context was updated correctly
        assert "alpha" in result
        assert "beta" in result
        assert "gamma" in result
        assert result["alpha"].shape == (5,)
        assert result["beta"].shape == (5,)
        assert result["gamma"].shape == (5,)
        assert torch.allclose(result["alpha"], torch.ones(5))
        assert torch.allclose(result["beta"], torch.ones(5) * 0.5)
        assert torch.allclose(result["gamma"], torch.ones(5) * 0.2)


class TestLikelihoodModelInterface:
    """Tests for the LikelihoodModel interface."""

    def test_likelihood_model_protocol(self):
        """Test that LikelihoodModel is a runtime checkable Protocol."""
        assert isinstance(LikelihoodModel, type)
        assert issubclass(LikelihoodModel, Protocol)

    def test_likelihood_model_implementation(self):
        """Test that a class implementing the LikelihoodModel interface is recognized."""

        class TestLikelihoodModelImpl:
            def forward(
                self,
                context: Dict[str, Any],
            ) -> Dict[str, Any]:
                # Extract data from context
                u_obs = context.get("u_obs")
                s_obs = context.get("s_obs")
                u_expected = context.get("u_expected")
                s_expected = context.get("s_expected")

                # Create mock distributions
                u_dist = "mock_u_distribution"
                s_dist = "mock_s_distribution"

                # Update context with distributions
                context["u_dist"] = u_dist
                context["s_dist"] = s_dist

                return context

        # Create an instance of the implementation
        impl = TestLikelihoodModelImpl()

        # Check that it's recognized as implementing the Protocol
        assert isinstance(impl, LikelihoodModel)

    def test_likelihood_model_context_interface(self):
        """Test the context-based interface for LikelihoodModel."""

        class TestLikelihoodModelImpl:
            def forward(
                self,
                context: Dict[str, Any],
            ) -> Dict[str, Any]:
                # Extract data from context
                u_obs = context.get("u_obs")
                s_obs = context.get("s_obs")
                u_expected = context.get("u_expected")
                s_expected = context.get("s_expected")

                # Create mock distributions
                u_dist = "mock_u_distribution"
                s_dist = "mock_s_distribution"

                # Update context with distributions
                context["u_dist"] = u_dist
                context["s_dist"] = s_dist

                return context

        # Create an instance of the implementation
        impl = TestLikelihoodModelImpl()

        # Create a context dictionary
        context = {
            "u_obs": torch.ones(10, 5),
            "s_obs": torch.ones(10, 5),
            "u_expected": torch.ones(10, 5) * 0.5,
            "s_expected": torch.ones(10, 5) * 0.5,
        }

        # Call the forward method
        result = impl.forward(context)

        # Check that the context was updated correctly
        assert "u_dist" in result
        assert "s_dist" in result
        assert result["u_dist"] == "mock_u_distribution"
        assert result["s_dist"] == "mock_s_distribution"


class TestObservationModelInterface:
    """Tests for the ObservationModel interface."""

    def test_observation_model_protocol(self):
        """Test that ObservationModel is a runtime checkable Protocol."""
        assert isinstance(ObservationModel, type)
        assert issubclass(ObservationModel, Protocol)

    def test_observation_model_implementation(self):
        """Test that a class implementing the ObservationModel interface is recognized."""

        class TestObservationModelImpl:
            def forward(
                self,
                context: Dict[str, Any],
            ) -> Dict[str, Any]:
                # Extract data from context
                u_obs = context.get("u_obs")
                s_obs = context.get("s_obs")

                # Transform data (mock implementation)
                u_transformed = u_obs
                s_transformed = s_obs
                scaling = torch.ones(u_obs.shape[0], 1)

                # Update context with transformed data
                context["u_transformed"] = u_transformed
                context["s_transformed"] = s_transformed
                context["scaling"] = scaling

                return context

        # Create an instance of the implementation
        impl = TestObservationModelImpl()

        # Check that it's recognized as implementing the Protocol
        assert isinstance(impl, ObservationModel)

    def test_observation_model_context_interface(self):
        """Test the context-based interface for ObservationModel."""

        class TestObservationModelImpl:
            def forward(
                self,
                context: Dict[str, Any],
            ) -> Dict[str, Any]:
                # Extract data from context
                u_obs = context.get("u_obs")
                s_obs = context.get("s_obs")

                # Transform data (mock implementation)
                u_transformed = u_obs
                s_transformed = s_obs
                scaling = torch.ones(u_obs.shape[0], 1)

                # Update context with transformed data
                context["u_transformed"] = u_transformed
                context["s_transformed"] = s_transformed
                context["scaling"] = scaling

                return context

        # Create an instance of the implementation
        impl = TestObservationModelImpl()

        # Create a context dictionary
        context = {
            "u_obs": torch.ones(10, 5),
            "s_obs": torch.ones(10, 5),
        }

        # Call the forward method
        result = impl.forward(context)

        # Check that the context was updated correctly
        assert "u_transformed" in result
        assert "s_transformed" in result
        assert "scaling" in result
        assert torch.allclose(result["u_transformed"], context["u_obs"])
        assert torch.allclose(result["s_transformed"], context["s_obs"])
        assert torch.allclose(result["scaling"], torch.ones(10, 1))


class TestInferenceGuideInterface:
    """Tests for the InferenceGuide interface."""

    def test_inference_guide_protocol(self):
        """Test that InferenceGuide is a runtime checkable Protocol."""
        assert isinstance(InferenceGuide, type)
        assert issubclass(InferenceGuide, Protocol)

    def test_inference_guide_implementation(self):
        """Test that a class implementing the InferenceGuide interface is recognized."""

        class TestInferenceGuideImpl:
            def __call__(
                self,
                model: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                # Simple implementation for testing
                return lambda *args, **kwargs: None

        # Create an instance of the implementation
        impl = TestInferenceGuideImpl()

        # Check that it's recognized as implementing the Protocol
        assert isinstance(impl, InferenceGuide)

    def test_inference_guide_functionality(self):
        """Test the functionality of the InferenceGuide interface."""

        class TestInferenceGuideImpl:
            def __call__(
                self,
                model: Any,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                # Create a simple guide function
                def guide_function(*guide_args, **guide_kwargs):
                    return "guide_result"
                return guide_function

        # Create an instance of the implementation
        impl = TestInferenceGuideImpl()

        # Create a mock model function
        def mock_model(*model_args, **model_kwargs):
            return "model_result"

        # Call the guide factory
        guide = impl(mock_model)

        # Check that the guide is callable
        assert callable(guide)

        # Call the guide
        result = guide()

        # Check the result
        assert result == "guide_result"


class TestContextBasedErrorHandling:
    """Tests for error handling in the context-based interface."""

    def test_missing_context_keys(self):
        """Test error handling when required context keys are missing."""

        class TestDynamicsModelImpl:
            def forward(
                self,
                context: Dict[str, Any],
            ) -> Dict[str, Any]:
                # Extract parameters from context
                u_obs = context["u_obs"]  # This will raise KeyError if missing
                s_obs = context["s_obs"]  # This will raise KeyError if missing

                # Compute expected counts
                u_expected = u_obs * 0.5
                s_expected = s_obs * 0.5

                # Update context with expected counts
                context["u_expected"] = u_expected
                context["s_expected"] = s_expected

                return context

        # Create an instance of the implementation
        impl = TestDynamicsModelImpl()

        # Create a context dictionary with missing keys
        context = {}

        # Call the forward method should raise KeyError
        with pytest.raises(KeyError):
            impl.forward(context)

    def test_context_validation(self):
        """Test context validation in the interface."""

        class TestDynamicsModelWithValidation:
            def forward(
                self,
                context: Dict[str, Any],
            ) -> Dict[str, Any]:
                # Validate context
                self._validate_context(context)

                # Extract parameters from context
                u_obs = context["u_obs"]
                s_obs = context["s_obs"]

                # Compute expected counts
                u_expected = u_obs * 0.5
                s_expected = s_obs * 0.5

                # Update context with expected counts
                context["u_expected"] = u_expected
                context["s_expected"] = s_expected

                return context

            def _validate_context(self, context: Dict[str, Any]) -> None:
                """Validate the context dictionary."""
                required_keys = ["u_obs", "s_obs"]
                for key in required_keys:
                    if key not in context:
                        raise ValueError(f"Missing required key: {key}")

        # Create an instance of the implementation
        impl = TestDynamicsModelWithValidation()

        # Create a context dictionary with missing keys
        context = {}

        # Call the forward method should raise ValueError
        with pytest.raises(ValueError, match="Missing required key"):
            impl.forward(context)

        # Create a valid context dictionary
        context = {
            "u_obs": torch.ones(10, 5),
            "s_obs": torch.ones(10, 5),
        }

        # Call the forward method should not raise
        result = impl.forward(context)

        # Check that the context was updated correctly
        assert "u_expected" in result
        assert "s_expected" in result
