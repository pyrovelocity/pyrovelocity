"""
Configuration classes for PyroVelocity JAX/NumPyro factory system.

This module provides configuration classes for the factory system, including:

- DynamicsFunctionConfig: Configuration for dynamics functions
- PriorFunctionConfig: Configuration for prior functions
- LikelihoodFunctionConfig: Configuration for likelihood functions
- GuideFunctionConfig: Configuration for guide factory functions
- ModelConfig: Configuration for models
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class DynamicsFunctionConfig:
    """Configuration for dynamics functions.

    Attributes:
        name: Name of the dynamics function to use
        params: Parameters to pass to the dynamics function
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PriorFunctionConfig:
    """Configuration for prior functions.

    Attributes:
        name: Name of the prior function to use
        params: Parameters to pass to the prior function
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LikelihoodFunctionConfig:
    """Configuration for likelihood functions.

    Attributes:
        name: Name of the likelihood function to use
        params: Parameters to pass to the likelihood function
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)




@dataclass
class GuideFunctionConfig:
    """Configuration for guide factory functions.

    Attributes:
        name: Name of the guide factory function to use
        params: Parameters to pass to the guide factory function
    """

    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for models.

    Attributes:
        dynamics_function: Configuration for the dynamics function
        prior_function: Configuration for the prior function
        likelihood_function: Configuration for the likelihood function
        guide_function: Configuration for the guide factory function
        metadata: Additional metadata for the model
    """

    dynamics_function: DynamicsFunctionConfig
    prior_function: PriorFunctionConfig
    likelihood_function: LikelihoodFunctionConfig
    guide_function: GuideFunctionConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
