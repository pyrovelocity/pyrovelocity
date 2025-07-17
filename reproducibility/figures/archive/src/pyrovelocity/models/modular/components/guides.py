"""
Protocol-First inference guide implementations for PyroVelocity's modular architecture.

This module contains inference guide implementations that directly implement the
InferenceGuide Protocol. These implementations follow the Protocol-First approach,
which embraces composition over inheritance and allows for more flexible component composition.

This module has been simplified to include only the essential components needed for
validation against the legacy implementation:
- AutoGuideFactory: Factory for creating AutoGuides for PyroVelocity models
- LegacyAutoGuideFactory: Factory for creating AutoGuideList guides that replicate the legacy PyroVelocity guide
"""

from typing import Any, Callable, Dict, List, Optional, Union

import pyro
import torch
from beartype import beartype
from pyro import poutine
from pyro.infer import Predictive, autoguide
from pyro.infer.autoguide import (
    AutoLowRankMultivariateNormal,
    AutoNormal,
    init_to_median,
)
from pyro.infer.autoguide.guides import AutoGuideList

from pyrovelocity.models.modular.interfaces import InferenceGuide
from pyrovelocity.models.modular.registry import inference_guide_registry


@inference_guide_registry.register("auto")
class AutoGuideFactory:
    """Factory for creating AutoGuides for PyroVelocity models.

    This class provides a factory for creating different types of AutoGuides
    based on the specified guide type. It supports various AutoGuide types
    provided by Pyro, such as AutoNormal, AutoDiagonalNormal, etc.

    This implementation directly implements the InferenceGuide Protocol.

    Attributes:
        guide_type: The type of AutoGuide to create.
        init_loc_fn: Function to initialize the location parameters of the guide.
        init_scale: Initial scale for the guide parameters.
        name: A unique name for this component instance.
    """

    @beartype
    def __init__(
        self,
        guide_type: str = "AutoLowRankMultivariateNormal",
        init_loc_fn: Optional[Callable] = None,
        init_scale: float = 0.1,  # Match legacy model's default scale
        name: str = "inference_guide",
        **kwargs: Any,
    ) -> None:
        """Initialize the AutoGuideFactory.

        The default parameters are set to match the legacy implementation
        to ensure consistent behavior.

        Args:
            guide_type: The type of AutoGuide to create.
            init_loc_fn: Function to initialize the location parameters of the guide.
            init_scale: Initial scale for the guide parameters.
            name: A unique name for this component instance.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.guide_type = guide_type
        self.init_loc_fn = init_loc_fn or init_to_median
        self.init_scale = init_scale
        self._guide = None
        self._model = None

    @beartype
    def create_guide(
        self, model: Union[Callable, Any], **kwargs: Any
    ) -> autoguide.AutoGuide:
        """Create an AutoGuide for the given model.

        Args:
            model: The model function for which to create a guide.
            **kwargs: Additional keyword arguments.

        Returns:
            An AutoGuide instance.

        Raises:
            ValueError: If the specified guide_type is not supported.
        """
        # Store the model for later use
        self._model = model

        # Map guide_type to the corresponding AutoGuide class
        guide_classes = {
            "AutoNormal": autoguide.AutoNormal,
            "AutoDiagonalNormal": autoguide.AutoDiagonalNormal,
            "AutoMultivariateNormal": autoguide.AutoMultivariateNormal,
            "AutoLowRankMultivariateNormal": autoguide.AutoLowRankMultivariateNormal,
            "AutoIAFNormal": autoguide.AutoIAFNormal,
            "AutoLaplaceApproximation": autoguide.AutoLaplaceApproximation,
            "AutoDelta": autoguide.AutoDelta,
        }

        if self.guide_type not in guide_classes:
            raise ValueError(
                f"Unsupported guide type: {self.guide_type}. "
                f"Supported types are: {list(guide_classes.keys())}"
            )

        # Create the guide
        guide_cls = guide_classes[self.guide_type]

        # Different guide types accept different parameters
        if self.guide_type == "AutoDelta":
            self._guide = guide_cls(
                model,
                init_loc_fn=self.init_loc_fn,
            )
        else:
            self._guide = guide_cls(
                model,
                init_loc_fn=self.init_loc_fn,
                init_scale=self.init_scale,
            )
        return self._guide

    @beartype
    def get_guide(self) -> autoguide.AutoGuide:
        """Get the created guide.

        Returns:
            The created AutoGuide instance.

        Raises:
            RuntimeError: If the guide has not been created yet.
        """
        if self._guide is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )
        return self._guide

    @beartype
    def get_posterior(self) -> Dict[str, torch.Tensor]:
        """Get the posterior distribution.

        Returns:
            Dictionary of posterior samples.

        Raises:
            RuntimeError: If the guide has not been created yet.
        """
        if self._guide is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )
        return self.sample_posterior()

    @beartype
    def sample_posterior(
        self,
        model: Optional[Callable] = None,
        guide: Optional[Callable] = None,
        num_samples: int = 1000,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Sample from the posterior distribution.

        Args:
            model: Optional model function (uses stored model if None)
            guide: Optional guide function (uses stored guide if None)
            num_samples: Number of samples to draw
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of posterior samples

        Raises:
            RuntimeError: If guide has not been created yet
        """
        if self._guide is None and guide is None:
            raise RuntimeError(
                "Guide has not been created yet. Call create_guide first."
            )

        # Use stored model/guide if not provided
        if model is None:
            model = self._model
        if guide is None:
            guide = self._guide

        # Create a predictive object
        predictive = Predictive(
            model=model,
            guide=guide,
            num_samples=num_samples,
            return_sites=kwargs.get("return_sites", None),
        )

        # Sample from the posterior
        posterior_samples = predictive()

        # Filter out non-sample sites
        posterior_samples = {
            k: v
            for k, v in posterior_samples.items()
            if not k.startswith("_") and k != "obs"
        }

        return posterior_samples

    def __call__(self, *args: Any, **kwargs: Any) -> Callable:
        """
        Create a guide function for the given model.

        This method is called by Pyro's SVI when the guide is used directly
        in svi.step() as the guide parameter. It should delegate to the guide
        object created by create_guide.

        Args:
            *args: Additional positional arguments (first is model when used with SVI)
            **kwargs: Additional keyword arguments

        Returns:
            A guide function compatible with the model
        """
        # If being used directly in SVI, the first argument will be the model
        if len(args) > 0 and callable(args[0]) and self._guide is None:
            model = args[0]
            self.create_guide(model)
            self._model = model

        # If we have a guide, delegate to it
        if self._guide is not None:
            return self._guide(*args, **kwargs)

        # Otherwise, raise an error
        raise RuntimeError(
            "Guide has not been created yet. Call create_guide first."
        )
