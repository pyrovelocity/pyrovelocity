"""Tests for the registry system in PyroVelocity's modular architecture."""

from typing import Any, Dict, Optional, Tuple

import pyro
import pyro.distributions as dist
import pytest
import torch
from beartype.typing import Callable

from pyrovelocity.models.modular.interfaces import (
    BatchTensor,
    DynamicsModel,
    InferenceGuide,
    LikelihoodModel,
    ModelState,
    ObservationModel,
    ParamTensor,
    PriorModel,
)
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    InferenceGuideRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    PriorModelRegistry,
    Registry,
)


# Test implementations of the interfaces
class TestDynamicsModel(DynamicsModel):
    def forward(
        self,
        u: BatchTensor,
        s: BatchTensor,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        scaling: Optional[ParamTensor] = None,
        t: Optional[BatchTensor] = None,
        **kwargs: Any,
    ) -> Tuple[BatchTensor, BatchTensor]:
        # Simple implementation for testing
        return u * alpha / beta, s * gamma

    def steady_state(
        self,
        alpha: ParamTensor,
        beta: ParamTensor,
        gamma: ParamTensor,
        **kwargs: Any,
    ) -> Tuple[ParamTensor, ParamTensor]:
        # Simple implementation for testing
        return alpha / beta, alpha / gamma


class TestPriorModel(PriorModel):
    def forward(
        self,
        u_obs: BatchTensor,
        s_obs: BatchTensor,
        plate: pyro.plate,
        **kwargs: Any,
    ) -> ModelState:
        # Simple implementation for testing
        with plate:
            alpha = pyro.sample(
                "alpha", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0))
            )
            beta = pyro.sample(
                "beta", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0))
            )
            gamma = pyro.sample(
                "gamma", dist.LogNormal(torch.tensor(0.0), torch.tensor(1.0))
            )

        return {"alpha": alpha, "beta": beta, "gamma": gamma}


class TestLikelihoodModel(LikelihoodModel):
    def forward(
        self,
        u_obs: BatchTensor,
        s_obs: BatchTensor,
        u_logits: BatchTensor,
        s_logits: BatchTensor,
        plate: pyro.plate,
        **kwargs: Any,
    ) -> None:
        # Simple implementation for testing
        with plate:
            pyro.sample("u", dist.Poisson(u_logits), obs=u_obs)
            pyro.sample("s", dist.Poisson(s_logits), obs=s_obs)


class TestObservationModel(ObservationModel):
    def forward(
        self,
        u_obs: BatchTensor,
        s_obs: BatchTensor,
        **kwargs: Any,
    ) -> Tuple[BatchTensor, BatchTensor]:
        # Simple implementation for testing
        return u_obs, s_obs


class TestInferenceGuide(InferenceGuide):
    def __call__(
        self,
        model: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Callable:
        # Simple implementation for testing
        def guide(*args, **kwargs):
            return None

        return guide


class TestGenericClass:
    """A generic class for testing the Registry."""

    pass


class TestRegistry:
    """Tests for the generic Registry class."""

    def test_registry_initialization(self):
        """Test that the registry is initialized correctly."""
        registry = Registry[TestGenericClass]
        assert registry._registry == {}

    def test_register_decorator(self):
        """Test that the register decorator works correctly."""
        registry = Registry[TestGenericClass]
        registry.clear()  # Start with a clean registry

        @registry.register("test")
        class TestClass(TestGenericClass):
            pass

        assert "test" in registry._registry
        assert registry._registry["test"] == TestClass

    def test_register_duplicate(self):
        """Test that registering a duplicate name raises an error."""
        registry = Registry[TestGenericClass]
        registry.clear()  # Start with a clean registry

        @registry.register("test")
        class TestClass1(TestGenericClass):
            pass

        with pytest.raises(ValueError):

            @registry.register("test")
            class TestClass2(TestGenericClass):
                pass

    def test_get(self):
        """Test that get returns the correct class."""
        registry = Registry[TestGenericClass]
        registry.clear()  # Start with a clean registry

        @registry.register("test")
        class TestClass(TestGenericClass):
            pass

        assert registry.get("test") == TestClass

    def test_get_unknown(self):
        """Test that get raises an error for unknown components."""
        registry = Registry[TestGenericClass]
        registry.clear()  # Start with a clean registry

        with pytest.raises(ValueError):
            registry.get("unknown")

    def test_create(self):
        """Test that create returns an instance of the correct class."""
        registry = Registry[TestGenericClass]
        registry.clear()  # Start with a clean registry

        @registry.register("test")
        class TestClass(TestGenericClass):
            def __init__(self, value=None):
                self.value = value

        instance = registry.create("test", value=42)
        assert isinstance(instance, TestClass)
        assert instance.value == 42

    def test_list_available(self):
        """Test that list_available returns the correct list of components."""
        registry = Registry[TestGenericClass]
        registry.clear()  # Start with a clean registry

        @registry.register("test1")
        class TestClass1(TestGenericClass):
            pass

        @registry.register("test2")
        class TestClass2(TestGenericClass):
            pass

        available = registry.list_available()
        assert "test1" in available
        assert "test2" in available
        assert len(available) == 2

    def test_clear(self):
        """Test that clear removes all registered components."""
        registry = Registry[TestGenericClass]
        registry.clear()  # Start with a clean registry

        @registry.register("test")
        class TestClass(TestGenericClass):
            pass

        assert "test" in registry._registry
        registry.clear()
        assert "test" not in registry._registry
        assert registry._registry == {}


class TestSpecializedRegistries:
    """Tests for the specialized registry classes."""

    def test_dynamics_model_registry(self):
        """Test the DynamicsModelRegistry."""
        DynamicsModelRegistry.clear()  # Start with a clean registry

        @DynamicsModelRegistry.register("test")
        class TestDynamicsModelImpl(TestDynamicsModel):
            pass

        assert "test" in DynamicsModelRegistry._registry
        assert DynamicsModelRegistry._registry["test"] == TestDynamicsModelImpl

        # Test validate_compatibility
        model = TestDynamicsModelImpl()
        assert DynamicsModelRegistry.validate_compatibility(model)

    def test_prior_model_registry(self):
        """Test the PriorModelRegistry."""
        PriorModelRegistry.clear()  # Start with a clean registry

        @PriorModelRegistry.register("test")
        class TestPriorModelImpl(TestPriorModel):
            pass

        assert "test" in PriorModelRegistry._registry
        assert PriorModelRegistry._registry["test"] == TestPriorModelImpl

    def test_likelihood_model_registry(self):
        """Test the LikelihoodModelRegistry."""
        LikelihoodModelRegistry.clear()  # Start with a clean registry

        @LikelihoodModelRegistry.register("test")
        class TestLikelihoodModelImpl(TestLikelihoodModel):
            pass

        assert "test" in LikelihoodModelRegistry._registry
        assert (
            LikelihoodModelRegistry._registry["test"] == TestLikelihoodModelImpl
        )

        # Test validate_compatibility
        model = TestLikelihoodModelImpl()
        assert LikelihoodModelRegistry.validate_compatibility(model)

    def test_observation_model_registry(self):
        """Test the ObservationModelRegistry."""
        ObservationModelRegistry.clear()  # Start with a clean registry

        @ObservationModelRegistry.register("test")
        class TestObservationModelImpl(TestObservationModel):
            pass

        assert "test" in ObservationModelRegistry._registry
        assert (
            ObservationModelRegistry._registry["test"]
            == TestObservationModelImpl
        )

    def test_inference_guide_registry(self):
        """Test the InferenceGuideRegistry."""
        InferenceGuideRegistry.clear()  # Start with a clean registry

        @InferenceGuideRegistry.register("test")
        class TestInferenceGuideImpl(TestInferenceGuide):
            pass

        assert "test" in InferenceGuideRegistry._registry
        assert (
            InferenceGuideRegistry._registry["test"] == TestInferenceGuideImpl
        )


class TestRegistryIntegration:
    """Tests for the integration of the registry system."""

    def setup_method(self):
        """Set up the test by registering components."""
        # Save original registry state
        self.original_dynamics_registry = dict(DynamicsModelRegistry._registry)
        self.original_prior_registry = dict(PriorModelRegistry._registry)
        self.original_likelihood_registry = dict(
            LikelihoodModelRegistry._registry
        )
        self.original_observation_registry = dict(
            ObservationModelRegistry._registry
        )
        self.original_inference_guide_registry = dict(
            InferenceGuideRegistry._registry
        )

        # Clear all registries for this test
        DynamicsModelRegistry.clear()
        PriorModelRegistry.clear()
        LikelihoodModelRegistry.clear()
        ObservationModelRegistry.clear()
        InferenceGuideRegistry.clear()

        # Register test components
        @DynamicsModelRegistry.register("standard")
        class StandardDynamicsModel(TestDynamicsModel):
            pass

        @PriorModelRegistry.register("lognormal")
        class LogNormalPriorModel(TestPriorModel):
            pass

        @LikelihoodModelRegistry.register("poisson")
        class PoissonLikelihoodModel(TestLikelihoodModel):
            pass

        @ObservationModelRegistry.register("standard")
        class StandardObservationModel(TestObservationModel):
            pass

        @InferenceGuideRegistry.register("auto")
        class AutoGuide(TestInferenceGuide):
            pass

    def test_component_registration_and_retrieval(self):
        """Test that components can be registered and retrieved correctly."""
        # Check that components are registered
        assert "standard" in DynamicsModelRegistry.list_available()
        assert "lognormal" in PriorModelRegistry.list_available()
        assert "poisson" in LikelihoodModelRegistry.list_available()
        assert "standard" in ObservationModelRegistry.list_available()
        assert "auto" in InferenceGuideRegistry.list_available()

        # Retrieve components
        dynamics_model_class = DynamicsModelRegistry.get("standard")
        prior_model_class = PriorModelRegistry.get("lognormal")
        likelihood_model_class = LikelihoodModelRegistry.get("poisson")
        observation_model_class = ObservationModelRegistry.get("standard")
        inference_guide_class = InferenceGuideRegistry.get("auto")

        # Create instances
        dynamics_model = DynamicsModelRegistry.create("standard")
        prior_model = PriorModelRegistry.create("lognormal")
        likelihood_model = LikelihoodModelRegistry.create("poisson")
        observation_model = ObservationModelRegistry.create("standard")
        inference_guide = InferenceGuideRegistry.create("auto")

        # Check that instances are of the correct type
        assert isinstance(dynamics_model, dynamics_model_class)
        assert isinstance(prior_model, prior_model_class)
        assert isinstance(likelihood_model, likelihood_model_class)
        assert isinstance(observation_model, observation_model_class)
        assert isinstance(inference_guide, inference_guide_class)

    def test_component_compatibility(self):
        """Test that component compatibility can be validated."""
        dynamics_model = DynamicsModelRegistry.create("standard")
        likelihood_model = LikelihoodModelRegistry.create("poisson")

        # Test compatibility validation
        assert DynamicsModelRegistry.validate_compatibility(dynamics_model)
        assert LikelihoodModelRegistry.validate_compatibility(likelihood_model)

    def teardown_method(self):
        """Restore the original registry state after the test."""
        # Restore original registry state
        DynamicsModelRegistry._registry = self.original_dynamics_registry
        PriorModelRegistry._registry = self.original_prior_registry
        LikelihoodModelRegistry._registry = self.original_likelihood_registry
        ObservationModelRegistry._registry = self.original_observation_registry
        InferenceGuideRegistry._registry = (
            self.original_inference_guide_registry
        )
