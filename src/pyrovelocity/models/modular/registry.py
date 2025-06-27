"""
Registry system for PyroVelocity's modular architecture.

This module implements the registry pattern for component registration and retrieval.
It provides a generic Registry class and specialized registries for each component type.

This module has been simplified to include only the essential components needed for
validation against the legacy implementation.
"""

from typing import Any, Callable, Dict, Generic, Type, TypeVar

from beartype import beartype
from beartype.typing import Optional

from pyrovelocity.models.modular.interfaces import (
    DynamicsModel,
    InferenceGuide,
    LikelihoodModel,
    ObservationModel,
    PriorModel,
)

# Generic type variable for the Registry class
T = TypeVar("T")


class Registry(Generic[T]):
    """
    Generic registry for component implementations.

    This class implements the registry pattern, allowing component implementations
    to be registered by name and retrieved later. It uses a class-level dictionary
    to store the registered components.

    Type Parameters:
        T: The type of components stored in this registry.
    """

    # Class-level registry dictionary
    _registry: Dict[str, Type[T]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Register a component implementation.

        This method returns a decorator that registers a component class
        in the registry under the given name.

        Args:
            name: The name to register the component under.

        Returns:
            A decorator function that registers the decorated class.

        Examples:
            @DynamicsModelRegistry.register("standard")
            class StandardDynamicsModel:
                ...
        """

        def decorator(component_class: Type[T]) -> Type[T]:
            if name in cls._registry:
                raise ValueError(
                    f"Component '{name}' is already registered in {cls.__name__}"
                )
            cls._registry[name] = component_class
            return component_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[T]:
        """
        Get a component implementation class.

        Args:
            name: The name of the registered component.

        Returns:
            The registered component class.

        Raises:
            ValueError: If no component is registered under the given name.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown component: '{name}'. Available components: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> T:
        """
        Create an instance of a registered component.

        Args:
            name: The name of the registered component.
            **kwargs: Additional arguments to pass to the component constructor.

        Returns:
            An instance of the registered component.

        Raises:
            ValueError: If no component is registered under the given name.
        """
        component_class = cls.get(name)
        return component_class(**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all available registered components.

        Returns:
            A list of names of all registered components.
        """
        return list(cls._registry.keys())

    @classmethod
    def available_models(cls) -> list[str]:
        """
        Alias for list_available for backward compatibility.

        Returns:
            A list of names of all registered components.
        """
        return cls.list_available()

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered components.

        This method is primarily intended for testing purposes.
        """
        cls._registry.clear()


class DynamicsModelRegistry(Registry[DynamicsModel]):
    """Registry for dynamics model implementations."""

    _registry: Dict[str, Type[DynamicsModel]] = {}

    @classmethod
    @beartype
    def validate_compatibility(
        cls,
        dynamics_model: DynamicsModel,
        prior_model: Optional[PriorModel] = None,
    ) -> bool:
        """
        Validate that a dynamics model is compatible with a prior model.

        Args:
            dynamics_model: The dynamics model to validate.
            prior_model: The prior model to check compatibility with.

        Returns:
            True if the models are compatible, False otherwise.
        """
        # In a real implementation, this would check specific compatibility requirements
        # For now, we just return True as a placeholder
        return True


class PriorModelRegistry(Registry[PriorModel]):
    """Registry for prior model implementations."""

    _registry: Dict[str, Type[PriorModel]] = {}


class LikelihoodModelRegistry(Registry[LikelihoodModel]):
    """Registry for likelihood model implementations."""

    _registry: Dict[str, Type[LikelihoodModel]] = {}

    @classmethod
    @beartype
    def validate_compatibility(
        cls,
        likelihood_model: LikelihoodModel,
        dynamics_model: Optional[DynamicsModel] = None,
    ) -> bool:
        """
        Validate that a likelihood model is compatible with a dynamics model.

        Args:
            likelihood_model: The likelihood model to validate.
            dynamics_model: The dynamics model to check compatibility with.

        Returns:
            True if the models are compatible, False otherwise.
        """
        # In a real implementation, this would check specific compatibility requirements
        # For now, we just return True as a placeholder
        return True


class ObservationModelRegistry(Registry[ObservationModel]):
    """Registry for observation model implementations."""

    _registry: Dict[str, Type[ObservationModel]] = {}


class InferenceGuideRegistry(Registry[InferenceGuide]):
    """Registry for inference guide implementations."""

    _registry: Dict[str, Type[InferenceGuide]] = {}


# Create global registry instances
dynamics_registry = DynamicsModelRegistry()
prior_registry = PriorModelRegistry()
likelihood_registry = LikelihoodModelRegistry()
observation_registry = ObservationModelRegistry()
observation_model_registry = (
    observation_registry  # Alias for backward compatibility
)
inference_guide_registry = InferenceGuideRegistry()


# Force explicit registration of component implementations
def register_standard_components():
    """
    Register only piecewise activation component implementations with their respective registries.

    This function imports and registers only the piecewise activation component implementations
    needed for validation workflows, ensuring they are available for factory methods and other
    parts of the system.

    All legacy components have been removed as part of the code cleanup process.
    """
    # Import only remaining component implementations
    from pyrovelocity.models.modular.components.dynamics import PiecewiseActivationDynamicsModel
    from pyrovelocity.models.modular.components.guides import AutoGuideFactory
    from pyrovelocity.models.modular.components.likelihoods import PiecewiseActivationPoissonLikelihoodModel
    from pyrovelocity.models.modular.components.priors import PiecewiseActivationPriorModel

    # The import itself should trigger the registrations through decorators
    # But we can also explicitly register them if needed
    if "piecewise_activation" not in DynamicsModelRegistry._registry:
        DynamicsModelRegistry._registry["piecewise_activation"] = PiecewiseActivationDynamicsModel

    if "piecewise_activation" not in PriorModelRegistry._registry:
        PriorModelRegistry._registry["piecewise_activation"] = PiecewiseActivationPriorModel

    if "piecewise_activation_poisson" not in LikelihoodModelRegistry._registry:
        LikelihoodModelRegistry._registry["piecewise_activation_poisson"] = PiecewiseActivationPoissonLikelihoodModel

    if "auto" not in InferenceGuideRegistry._registry:
        InferenceGuideRegistry._registry["auto"] = AutoGuideFactory


# Export all registry classes and instances
__all__ = [
    # Registry classes
    "Registry",
    "DynamicsModelRegistry",
    "PriorModelRegistry",
    "LikelihoodModelRegistry",
    "ObservationModelRegistry",
    "InferenceGuideRegistry",
    # Registry instances
    "dynamics_registry",
    "prior_registry",
    "likelihood_registry",
    "observation_registry",
    "observation_model_registry",  # Added alias to exports
    "inference_guide_registry",
    # Functions
    "register_standard_components",
]
