"""
Tests for the unified configuration system in PyroVelocity's modular architecture.

This module contains tests for the unified configuration system, including the
generic ComponentConfig class and the ModelConfig class.
"""

from typing import Any, Dict, Optional, Tuple, cast

import pyro
import pytest
import torch
from hydra_zen import instantiate
from omegaconf import DictConfig, OmegaConf

from pyrovelocity.models.modular.config import (
    ComponentConfig,
    ComponentType,
    ModelConfig,
)
from pyrovelocity.models.modular.factory import (
    ComponentFactory,
    create_model_from_config,
)
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
from pyrovelocity.models.modular.model import PyroVelocityModel
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    InferenceGuideRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    PriorModelRegistry,
)


# Mock implementations for testing
class MockDynamicsModel:
    """Mock dynamics model for testing."""

    def __init__(self, param1=None, param2=None, shared_time=None, cell_specific_kinetics=None,
                 correct_library_size=None, kinetics_num=None, **kwargs):
        self.param1 = param1
        self.param2 = param2
        self.shared_time = shared_time
        self.cell_specific_kinetics = cell_specific_kinetics
        self.correct_library_size = correct_library_size
        self.kinetics_num = kinetics_num

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Implement the DynamicsModel protocol."""
        return context


class MockPriorModel:
    """Mock prior model for testing."""

    def __init__(self, param1=None, param2=None, scale_alpha=None, scale_beta=None,
                 scale_gamma=None, scale_u=None, scale_s=None, scale_dt=None, **kwargs):
        self.param1 = param1
        self.param2 = param2
        self.scale_alpha = scale_alpha
        self.scale_beta = scale_beta
        self.scale_gamma = scale_gamma
        self.scale_u = scale_u
        self.scale_s = scale_s
        self.scale_dt = scale_dt

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Implement the PriorModel protocol."""
        return context


class MockLikelihoodModel:
    """Mock likelihood model for testing."""

    def __init__(self, param1=None, param2=None, use_observed_lib_size=True, **kwargs):
        self.param1 = param1
        self.param2 = param2
        self.use_observed_lib_size = use_observed_lib_size

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Implement the LikelihoodModel protocol."""
        return context


class MockObservationModel:
    """Mock observation model for testing."""

    def __init__(self, param1=None, param2=None, log_transform=None, normalize=None,
                 use_raw=None, **kwargs):
        self.param1 = param1
        self.param2 = param2
        self.log_transform = log_transform
        self.normalize = normalize
        self.use_raw = use_raw

    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Implement the ObservationModel protocol."""
        return context


class MockInferenceGuide:
    """Mock inference guide for testing."""

    def __init__(self, param1=None, param2=None, guide_type=None, init_loc_fn=None,
                 init_scale=None, **kwargs):
        self.param1 = param1
        self.param2 = param2
        self.guide_type = guide_type
        self.init_loc_fn = init_loc_fn
        self.init_scale = init_scale

    def __call__(
        self,
        model,
        *args: Any,
        **kwargs: Any,
    ):
        """Implement the InferenceGuide protocol."""

        def guide(*args, **kwargs):
            return None

        return guide


@pytest.fixture
def setup_registries():
    """Set up the registries with mock components for testing."""
    # Clear all registries
    DynamicsModelRegistry.clear()
    PriorModelRegistry.clear()
    LikelihoodModelRegistry.clear()
    ObservationModelRegistry.clear()
    InferenceGuideRegistry.clear()

    # Make the mock classes implement the required protocols
    # This is a hack to make beartype happy with our mock classes
    DynamicsModel.register(MockDynamicsModel)
    PriorModel.register(MockPriorModel)
    LikelihoodModel.register(MockLikelihoodModel)
    ObservationModel.register(MockObservationModel)
    InferenceGuide.register(MockInferenceGuide)

    # Register mock components
    @DynamicsModelRegistry.register("mock")
    class RegisteredMockDynamicsModel(MockDynamicsModel):
        pass

    @PriorModelRegistry.register("mock")
    class RegisteredMockPriorModel(MockPriorModel):
        pass

    @LikelihoodModelRegistry.register("mock")
    class RegisteredMockLikelihoodModel(MockLikelihoodModel):
        pass

    @ObservationModelRegistry.register("mock")
    class RegisteredMockObservationModel(MockObservationModel):
        pass

    @InferenceGuideRegistry.register("mock")
    class RegisteredMockInferenceGuide(MockInferenceGuide):
        pass

    # Register standard components (assuming these exist in the real codebase)
    @DynamicsModelRegistry.register("standard")
    class StandardDynamicsModel(MockDynamicsModel):
        pass

    @PriorModelRegistry.register("lognormal")
    class LogNormalPriorModel(MockPriorModel):
        pass

    @LikelihoodModelRegistry.register("poisson")
    class PoissonLikelihoodModel(MockLikelihoodModel):
        pass

    @ObservationModelRegistry.register("standard")
    class StandardObservationModel(MockObservationModel):
        pass

    @InferenceGuideRegistry.register("auto")
    class AutoGuide(MockInferenceGuide):
        pass


class TestComponentConfig:
    """Tests for the ComponentConfig class."""

    def test_component_config_creation(self):
        """Test creating a ComponentConfig."""
        # Create a ComponentConfig
        config = ComponentConfig(name="mock", params={"param1": 42, "param2": "test"})

        # Check that the attributes are set correctly
        assert config.name == "mock"
        assert config.params == {"param1": 42, "param2": "test"}

    def test_component_config_from_dict(self):
        """Test creating a ComponentConfig from a dictionary."""
        # Create a dictionary
        config_dict = {"name": "mock", "params": {"param1": 42, "param2": "test"}}

        # Create a ComponentConfig from the dictionary
        config = ComponentConfig.from_dict(config_dict)

        # Check that the attributes are set correctly
        assert config.name == "mock"
        assert config.params == {"param1": 42, "param2": "test"}

    def test_component_config_to_dict(self):
        """Test converting a ComponentConfig to a dictionary."""
        # Create a ComponentConfig
        config = ComponentConfig(name="mock", params={"param1": 42, "param2": "test"})

        # Convert to a dictionary
        config_dict = config.to_dict()

        # Check that the dictionary is correct
        assert config_dict == {"name": "mock", "params": {"param1": 42, "param2": "test"}}


class TestModelConfig:
    """Tests for the ModelConfig class."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        # Create component configs
        dynamics_config = ComponentConfig(name="mock", params={"param1": 1})
        prior_config = ComponentConfig(name="mock", params={"param1": 2})
        likelihood_config = ComponentConfig(name="mock", params={"param1": 3})
        inference_config = ComponentConfig(name="mock", params={"param1": 5})

        # Create a ModelConfig
        config = ModelConfig(
            dynamics_model=dynamics_config,
            prior_model=prior_config,
            likelihood_model=likelihood_config,
            inference_guide=inference_config,
            metadata={"test": "value"},
        )

        # Check that the attributes are set correctly
        assert config.dynamics_model == dynamics_config
        assert config.prior_model == prior_config
        assert config.likelihood_model == likelihood_config
        assert config.inference_guide == inference_config
        assert config.metadata == {"test": "value"}

    def test_model_config_from_dict(self):
        """Test creating a ModelConfig from a dictionary."""
        # Create a dictionary
        config_dict = {
            "dynamics_model": {"name": "mock", "params": {"param1": 1}},
            "prior_model": {"name": "mock", "params": {"param1": 2}},
            "likelihood_model": {"name": "mock", "params": {"param1": 3}},
            "observation_model": {"name": "mock", "params": {"param1": 4}},  # This should be ignored
            "inference_guide": {"name": "mock", "params": {"param1": 5}},
            "metadata": {"test": "value"},
        }

        # Create a ModelConfig from the dictionary
        config = ModelConfig.from_dict(config_dict)

        # Check that the attributes are set correctly
        assert config.dynamics_model.name == "mock"
        assert config.dynamics_model.params == {"param1": 1}
        assert config.prior_model.name == "mock"
        assert config.prior_model.params == {"param1": 2}
        assert config.likelihood_model.name == "mock"
        assert config.likelihood_model.params == {"param1": 3}
        assert config.inference_guide.name == "mock"
        assert config.inference_guide.params == {"param1": 5}
        assert config.metadata == {"test": "value"}

    def test_model_config_to_dict(self):
        """Test converting a ModelConfig to a dictionary."""
        # Create component configs
        dynamics_config = ComponentConfig(name="mock", params={"param1": 1})
        prior_config = ComponentConfig(name="mock", params={"param1": 2})
        likelihood_config = ComponentConfig(name="mock", params={"param1": 3})
        inference_config = ComponentConfig(name="mock", params={"param1": 5})

        # Create a ModelConfig
        config = ModelConfig(
            dynamics_model=dynamics_config,
            prior_model=prior_config,
            likelihood_model=likelihood_config,
            inference_guide=inference_config,
            metadata={"test": "value"},
        )

        # Convert to a dictionary
        config_dict = config.to_dict()

        # Check that the dictionary is correct
        expected_dict = {
            "dynamics_model": {"name": "mock", "params": {"param1": 1}},
            "prior_model": {"name": "mock", "params": {"param1": 2}},
            "likelihood_model": {"name": "mock", "params": {"param1": 3}},
            "inference_guide": {"name": "mock", "params": {"param1": 5}},
            "metadata": {"test": "value"},
        }
        assert config_dict == expected_dict

    def test_standard_model_config(self):
        """Test the standard model configuration."""
        # Create a standard model configuration
        config = ModelConfig.standard()

        # Check that it has the expected component configurations
        assert config.dynamics_model.name == "standard"
        assert config.prior_model.name == "lognormal"
        assert config.likelihood_model.name == "poisson"
        assert config.inference_guide.name == "auto"


class TestComponentFactory:
    """Tests for the ComponentFactory class."""

    def test_create_component(self, setup_registries):
        """Test creating a component using the ComponentFactory."""
        # Create a ComponentConfig
        config = ComponentConfig(name="mock", params={"param1": 42, "param2": "test"})

        # Create a component using the factory
        component = ComponentFactory.create_component(
            config, ComponentType.DYNAMICS_MODEL
        )

        # Check that the component is of the correct type
        assert isinstance(component, MockDynamicsModel)
        assert component.param1 == 42
        assert component.param2 == "test"

    def test_create_component_from_dict(self, setup_registries):
        """Test creating a component from a dictionary using the ComponentFactory."""
        # Create a dictionary
        config_dict = {"name": "mock", "params": {"param1": 42, "param2": "test"}}

        # Create a component using the factory
        component = ComponentFactory.create_component_from_dict(
            config_dict, ComponentType.DYNAMICS_MODEL
        )

        # Check that the component is of the correct type
        assert isinstance(component, MockDynamicsModel)
        assert component.param1 == 42
        assert component.param2 == "test"

    def test_create_dynamics_model(self, setup_registries):
        """Test creating a dynamics model using the ComponentFactory."""
        # Create a ComponentConfig
        config = ComponentConfig(name="mock", params={"param1": 42, "param2": "test"})

        # Create a dynamics model using the factory
        model = ComponentFactory.create_dynamics_model(config)

        # Check that the model is of the correct type
        assert isinstance(model, MockDynamicsModel)
        assert model.param1 == 42
        assert model.param2 == "test"

    def test_create_prior_model(self, setup_registries):
        """Test creating a prior model using the ComponentFactory."""
        # Create a ComponentConfig
        config = ComponentConfig(name="mock", params={"param1": 42, "param2": "test"})

        # Create a prior model using the factory
        model = ComponentFactory.create_prior_model(config)

        # Check that the model is of the correct type
        assert isinstance(model, MockPriorModel)
        assert model.param1 == 42
        assert model.param2 == "test"

    def test_create_likelihood_model(self, setup_registries):
        """Test creating a likelihood model using the ComponentFactory."""
        # Create a ComponentConfig
        config = ComponentConfig(name="mock", params={"param1": 42, "param2": "test"})

        # Create a likelihood model using the factory
        model = ComponentFactory.create_likelihood_model(config)

        # Check that the model is of the correct type
        assert isinstance(model, MockLikelihoodModel)
        assert model.param1 == 42
        assert model.param2 == "test"

    def test_create_observation_model(self, setup_registries):
        """Test creating an observation model using the ComponentFactory."""
        # Create a ComponentConfig
        config = ComponentConfig(name="mock", params={"param1": 42, "param2": "test"})

        # Create an observation model using the factory
        model = ComponentFactory.create_observation_model(config)

        # Check that the model is of the correct type
        assert isinstance(model, MockObservationModel)
        assert model.param1 == 42
        assert model.param2 == "test"

    def test_create_inference_guide(self, setup_registries):
        """Test creating an inference guide using the ComponentFactory."""
        # Create a ComponentConfig
        config = ComponentConfig(name="mock", params={"param1": 42, "param2": "test"})

        # Create an inference guide using the factory
        guide = ComponentFactory.create_inference_guide(config)

        # Check that the guide is of the correct type
        assert isinstance(guide, MockInferenceGuide)
        assert guide.param1 == 42
        assert guide.param2 == "test"


class TestModelCreation:
    """Tests for creating models from configurations."""

    def test_create_model_from_config(self, setup_registries):
        """Test creating a PyroVelocityModel from a ModelConfig."""
        # Create component configs
        dynamics_config = ComponentConfig(name="mock", params={"param1": 1})
        prior_config = ComponentConfig(name="mock", params={"param1": 2})
        likelihood_config = ComponentConfig(name="mock", params={"param1": 3})
        inference_config = ComponentConfig(name="mock", params={"param1": 5})

        # Create a ModelConfig
        config = ModelConfig(
            dynamics_model=dynamics_config,
            prior_model=prior_config,
            likelihood_model=likelihood_config,
            inference_guide=inference_config,
        )

        # Create a model from the config
        model = create_model_from_config(config)

        # Check that the model is a PyroVelocityModel
        assert isinstance(model, PyroVelocityModel)

        # Check that the components are of the correct types
        assert isinstance(model.dynamics_model, MockDynamicsModel)
        assert isinstance(model.prior_model, MockPriorModel)
        assert isinstance(model.likelihood_model, MockLikelihoodModel)
        assert isinstance(model.guide_model, MockInferenceGuide)

        # Check that the parameters were passed correctly
        assert model.dynamics_model.param1 == 1
        assert model.prior_model.param1 == 2
        assert model.likelihood_model.param1 == 3
        assert model.guide_model.param1 == 5

    def test_create_model_from_dict(self, setup_registries):
        """Test creating a PyroVelocityModel from a dictionary."""
        # Create a dictionary
        config_dict = {
            "dynamics_model": {"name": "mock", "params": {"param1": 1}},
            "prior_model": {"name": "mock", "params": {"param1": 2}},
            "likelihood_model": {"name": "mock", "params": {"param1": 3}},
            "observation_model": {"name": "mock", "params": {"param1": 4}},  # This should be ignored
            "inference_guide": {"name": "mock", "params": {"param1": 5}},
        }

        # Create a model from the dictionary
        model = create_model_from_config(config_dict)

        # Check that the model is a PyroVelocityModel
        assert isinstance(model, PyroVelocityModel)

        # Check that the components are of the correct types
        assert isinstance(model.dynamics_model, MockDynamicsModel)
        assert isinstance(model.prior_model, MockPriorModel)
        assert isinstance(model.likelihood_model, MockLikelihoodModel)
        assert isinstance(model.guide_model, MockInferenceGuide)

        # Check that the parameters were passed correctly
        assert model.dynamics_model.param1 == 1
        assert model.prior_model.param1 == 2
        assert model.likelihood_model.param1 == 3
        assert model.guide_model.param1 == 5

    def test_create_standard_model(self, setup_registries):
        """Test creating a standard model."""
        # Create a standard model
        model = create_model_from_config(ModelConfig.standard())

        # Check that the model is a PyroVelocityModel
        assert isinstance(model, PyroVelocityModel)

        # Check that the components are of the correct types
        assert isinstance(model.dynamics_model, MockDynamicsModel)
        assert isinstance(model.prior_model, MockPriorModel)
        assert isinstance(model.likelihood_model, MockLikelihoodModel)
        assert isinstance(model.guide_model, MockInferenceGuide)
