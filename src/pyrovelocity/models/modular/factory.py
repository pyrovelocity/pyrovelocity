"""
Factory methods and configuration management for PyroVelocity's modular architecture.

This module provides factory functions for creating PyroVelocity models from
configuration objects, leveraging the registry system and hydra-zen for type-safe
configuration management.

This module has been simplified to include only the essential components needed for
validation against the legacy implementation.
"""

from typing import Any, Dict, Optional, Union, cast

from beartype import beartype
from omegaconf import DictConfig, OmegaConf

from pyrovelocity.models.modular.config import (
    ComponentConfig,
    ComponentType,
    ModelConfig,
)
from pyrovelocity.models.modular.inference.config import InferenceConfig
from pyrovelocity.models.modular.interfaces import (
    DynamicsModel,
    InferenceGuide,
    LikelihoodModel,
    ObservationModel,
    PriorModel,
)
from pyrovelocity.models.modular.model import PyroVelocityModel
from pyrovelocity.models.modular.registry import (
    DynamicsModelRegistry,
    InferenceGuideRegistry,
    LikelihoodModelRegistry,
    ObservationModelRegistry,
    PriorModelRegistry,
    register_standard_components,
)


class ComponentFactory:
    """
    Factory for creating components in the PyroVelocity modular architecture.

    This class provides methods for creating components from configurations,
    replacing the multiple component-specific factory functions in the original
    implementation.
    """

    @staticmethod
    @beartype
    def create_component(
        config: ComponentConfig, component_type: ComponentType
    ) -> Union[DynamicsModel, PriorModel, LikelihoodModel, ObservationModel, InferenceGuide]:
        """
        Create a component from a configuration.

        Args:
            config: Configuration for the component
            component_type: Type of component to create

        Returns:
            An instance of the specified component

        Raises:
            ValueError: If the specified component is not registered or the component type is invalid
        """
        # Select the appropriate registry based on the component type
        if component_type == ComponentType.DYNAMICS_MODEL:
            return DynamicsModelRegistry.create(config.name, **config.params)
        elif component_type == ComponentType.PRIOR_MODEL:
            return PriorModelRegistry.create(config.name, **config.params)
        elif component_type == ComponentType.LIKELIHOOD_MODEL:
            return LikelihoodModelRegistry.create(config.name, **config.params)
        elif component_type == ComponentType.OBSERVATION_MODEL:
            return ObservationModelRegistry.create(config.name, **config.params)
        elif component_type == ComponentType.INFERENCE_GUIDE:
            return InferenceGuideRegistry.create(config.name, **config.params)
        else:
            raise ValueError(f"Invalid component type: {component_type}")

    @staticmethod
    @beartype
    def create_component_from_dict(
        config_dict: Dict[str, Any], component_type: ComponentType
    ) -> Union[DynamicsModel, PriorModel, LikelihoodModel, ObservationModel, InferenceGuide]:
        """
        Create a component from a dictionary configuration.

        Args:
            config_dict: Dictionary containing the configuration
            component_type: Type of component to create

        Returns:
            An instance of the specified component

        Raises:
            ValueError: If the specified component is not registered or the component type is invalid
        """
        config = ComponentConfig.from_dict(config_dict)
        return ComponentFactory.create_component(config, component_type)

    @staticmethod
    @beartype
    def create_dynamics_model(
        config: Union[ComponentConfig, Dict[str, Any]]
    ) -> DynamicsModel:
        """
        Create a dynamics model from a configuration.

        Args:
            config: Configuration for the dynamics model, either as a ComponentConfig
                   object or a dictionary.

        Returns:
            An instance of the specified dynamics model.

        Raises:
            ValueError: If the specified model is not registered.
        """
        if isinstance(config, dict):
            config = ComponentConfig.from_dict(config)
        return cast(
            DynamicsModel,
            ComponentFactory.create_component(config, ComponentType.DYNAMICS_MODEL),
        )

    @staticmethod
    @beartype
    def create_prior_model(
        config: Union[ComponentConfig, Dict[str, Any]]
    ) -> PriorModel:
        """
        Create a prior model from a configuration.

        Args:
            config: Configuration for the prior model, either as a ComponentConfig
                   object or a dictionary.

        Returns:
            An instance of the specified prior model.

        Raises:
            ValueError: If the specified model is not registered.
        """
        if isinstance(config, dict):
            config = ComponentConfig.from_dict(config)
        return cast(
            PriorModel,
            ComponentFactory.create_component(config, ComponentType.PRIOR_MODEL),
        )

    @staticmethod
    @beartype
    def create_likelihood_model(
        config: Union[ComponentConfig, Dict[str, Any]]
    ) -> LikelihoodModel:
        """
        Create a likelihood model from a configuration.

        Args:
            config: Configuration for the likelihood model, either as a ComponentConfig
                   object or a dictionary.

        Returns:
            An instance of the specified likelihood model.

        Raises:
            ValueError: If the specified model is not registered.
        """
        if isinstance(config, dict):
            config = ComponentConfig.from_dict(config)
        return cast(
            LikelihoodModel,
            ComponentFactory.create_component(config, ComponentType.LIKELIHOOD_MODEL),
        )

    @staticmethod
    @beartype
    def create_observation_model(
        config: Union[ComponentConfig, Dict[str, Any]]
    ) -> ObservationModel:
        """
        Create an observation model from a configuration.

        Args:
            config: Configuration for the observation model, either as a ComponentConfig
                   object or a dictionary.

        Returns:
            An instance of the specified observation model.

        Raises:
            ValueError: If the specified model is not registered.
        """
        if isinstance(config, dict):
            config = ComponentConfig.from_dict(config)
        return cast(
            ObservationModel,
            ComponentFactory.create_component(config, ComponentType.OBSERVATION_MODEL),
        )

    @staticmethod
    @beartype
    def create_inference_guide(
        config: Union[ComponentConfig, Dict[str, Any]]
    ) -> InferenceGuide:
        """
        Create an inference guide from a configuration.

        Args:
            config: Configuration for the inference guide, either as a ComponentConfig
                   object or a dictionary.

        Returns:
            An instance of the specified inference guide.

        Raises:
            ValueError: If the specified guide is not registered.
        """
        if isinstance(config, dict):
            config = ComponentConfig.from_dict(config)
        return cast(
            InferenceGuide,
            ComponentFactory.create_component(config, ComponentType.INFERENCE_GUIDE),
        )


@beartype
def create_model_from_config(
    config: Union[ModelConfig, Dict[str, Any], DictConfig]
) -> PyroVelocityModel:
    """
    Create a PyroVelocityModel from a configuration.

    This function creates a PyroVelocityModel by instantiating each component
    from the provided configuration and composing them together.

    Args:
        config: Configuration for the PyroVelocityModel, either as a ModelConfig
               object, a dictionary, or a DictConfig.

    Returns:
        An instance of PyroVelocityModel with the specified components.

    Raises:
        ValueError: If any of the specified components are not registered.
    """
    # Convert config to a ModelConfig if it's a dictionary or DictConfig
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
        config = ModelConfig.from_dict(config_dict)
    elif isinstance(config, dict):
        config = ModelConfig.from_dict(config)

    # Create each component using the ComponentFactory
    dynamics_model = ComponentFactory.create_dynamics_model(config.dynamics_model)
    prior_model = ComponentFactory.create_prior_model(config.prior_model)
    likelihood_model = ComponentFactory.create_likelihood_model(config.likelihood_model)
    inference_guide = ComponentFactory.create_inference_guide(config.inference_guide)

    # Create and return the model (observation model functionality is now in likelihood model)
    return PyroVelocityModel(
        dynamics_model=dynamics_model,
        prior_model=prior_model,
        likelihood_model=likelihood_model,
        guide_model=inference_guide,
    )


def create_legacy_model1() -> PyroVelocityModel:
    """
    Create a PyroVelocityModel that replicates the legacy model with add_offset=False.

    This function creates a PyroVelocityModel with components specifically configured
    to match the behavior of the legacy PyroVelocity implementation with add_offset=False.
    It uses the LegacyDynamicsModel to create a model that exactly matches the
    behavior of the legacy implementation.

    Returns:
        A PyroVelocityModel instance that replicates the legacy model.
    """
    # Create a configuration with the legacy dynamics model
    config = ModelConfig(
        dynamics_model=ComponentConfig(
            name="legacy",
            params={
                "shared_time": True,
                "t_scale_on": False,
                "correct_library_size": True,
            },
        ),
        prior_model=ComponentConfig(
            name="lognormal",
            params={},
        ),
        likelihood_model=ComponentConfig(
            name="legacy",
            params={},
        ),
        inference_guide=ComponentConfig(
            name="auto",
            params={
                "init_scale": 0.1,
            },
        ),
    )

    # Create and return the model
    return create_model_from_config(config)


def create_legacy_model2() -> PyroVelocityModel:
    """
    Create a PyroVelocityModel that replicates the legacy model with add_offset=True.

    This function creates a PyroVelocityModel with components specifically configured
    to match the behavior of the legacy PyroVelocity implementation with add_offset=True.
    It uses the LegacyDynamicsModel to create a model that exactly matches the
    behavior of the legacy implementation.

    Returns:
        A PyroVelocityModel instance that replicates the legacy model.
    """
    # Create a configuration with the legacy dynamics model
    config = ModelConfig(
        dynamics_model=ComponentConfig(
            name="legacy",
            params={
                "shared_time": True,
                "t_scale_on": False,
                "correct_library_size": True,
                "add_offset": True,
            },
        ),
        prior_model=ComponentConfig(
            name="lognormal",
            params={},
        ),
        likelihood_model=ComponentConfig(
            name="legacy",
            params={},
        ),
        inference_guide=ComponentConfig(
            name="legacy_auto",
            params={
                "init_scale": 0.1,
                "add_offset": True,
            },
        ),
    )

    return create_model_from_config(config)


def create_piecewise_activation_model(
    inference_method: str = "svi",
    guide_type: str = "AutoLowRankMultivariateNormal",
    default_inference_config: Optional[InferenceConfig] = None,
) -> PyroVelocityModel:
    """
    Create a PyroVelocityModel with piecewise activation components and inference configuration.

    This function creates a PyroVelocityModel specifically configured for
    piecewise activation parameter recovery validation. It uses:
    - PiecewiseActivationDynamicsModel for dimensionless analytical dynamics
    - PiecewiseActivationPriorModel for hierarchical priors
    - PoissonLikelihoodModel for count data likelihood (includes data preprocessing)
    - Configurable inference guide for SVI or MCMC

    Args:
        inference_method: Default inference method ("svi" or "mcmc")
        guide_type: Guide type for SVI ("AutoNormal", "AutoDiagonalNormal",
                   "AutoMultivariateNormal", "AutoLowRankMultivariateNormal")
        default_inference_config: Optional default inference configuration to store on model

    Returns:
        A PyroVelocityModel instance configured for piecewise activation validation
        with optional default inference configuration.

    Examples:
        >>> # SVI with AutoLowRankMultivariateNormal guide
        >>> model = create_piecewise_activation_model()

        >>> # SVI with different guide
        >>> model = create_piecewise_activation_model(guide_type="AutoNormal")

        >>> # With default inference config
        >>> from pyrovelocity.models.modular.inference.config import InferenceConfig
        >>> config = InferenceConfig(method="svi", num_epochs=1000, learning_rate=0.01)
        >>> model = create_piecewise_activation_model(default_inference_config=config)
    """
    # Create configuration for piecewise activation model
    config = ModelConfig(
        dynamics_model=ComponentConfig(
            name="piecewise_activation",
            params={},
        ),
        prior_model=ComponentConfig(
            name="piecewise_activation",
            params={},
        ),
        likelihood_model=ComponentConfig(
            name="piecewise_activation_poisson",
            params={},
        ),
        inference_guide=ComponentConfig(
            name="auto",
            params={
                "guide_type": guide_type,
                "init_scale": 0.1,
            },
        ),
    )

    # Create the model
    model = create_model_from_config(config)

    # Store default inference configuration if provided
    if default_inference_config is not None:
        from pyrovelocity.models.modular.model import ModelState
        model.state = ModelState(
            dynamics_state=model.state.dynamics_state,
            prior_state=model.state.prior_state,
            likelihood_state=model.state.likelihood_state,
            guide_state=model.state.guide_state,
            inference_config=default_inference_config,
        )

    return model


__all__ = [
    # Factory class
    "ComponentFactory",
    # New factory functions
    "create_model_from_config",
    # Legacy model replication functions
    "create_legacy_model1",
    "create_legacy_model2",
    # Piecewise activation model
    "create_piecewise_activation_model",
]

register_standard_components()
