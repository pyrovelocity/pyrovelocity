"""
Context validation utilities for PyroVelocity's modular architecture.

This module provides utilities for validating and manipulating the context
dictionary that is passed between components during model execution. These
utilities were previously part of the BaseComponent class and are now
available as standalone functions.

The utilities in this module are designed to be used by components that
directly implement Protocol interfaces without inheriting from base classes.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from expression import Result


def create_error(
    component_name: str,
    operation: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Result:
    """Create a Result.Error with component context."""
    error_msg = f"{component_name}.{operation}: {message}"
    if details:
        error_msg += f" (details: {details})"
    return Result.Error(error_msg)


def validate_context(
    component_name: str,
    context: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
    tensor_keys: Optional[List[str]] = None,
) -> Union[Result, Dict[str, Any]]:
    """
    Validate the context dictionary for required keys and tensor types.

    This function checks that the context dictionary contains all required keys
    and that the values for tensor keys are torch.Tensor instances. It returns
    a Result.Error if validation fails, or the original context if validation
    succeeds.

    Args:
        component_name: The name of the component performing the validation.
        context: The context dictionary to validate.
        required_keys: Optional list of keys that must be present in the context.
        tensor_keys: Optional list of keys whose values must be torch.Tensor instances.

    Returns:
        A Result.Error if validation fails, or the original context if validation succeeds.

    Examples:
        ```python
        def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
            # Validate context
            validation_result = validate_context(
                self.__class__.__name__,
                context,
                required_keys=["u_obs", "s_obs"],
                tensor_keys=["u_obs", "s_obs"]
            )
            if isinstance(validation_result, Result) and validation_result.is_error():
                return validation_result

            # Process data
            # ...
            return context
        ```
    """
    # Check required keys
    if required_keys:
        for key in required_keys:
            if key not in context:
                return create_error(
                    component_name,
                    "validate_context",
                    f"Missing required key: {key}",
                    {"provided_keys": list(context.keys())},
                )

    # Check tensor keys
    if tensor_keys:
        for key in tensor_keys:
            if key in context and not isinstance(context[key], torch.Tensor):
                return create_error(
                    component_name,
                    "validate_context",
                    f"Key {key} must be a torch.Tensor, got {type(context[key])}",
                    {"key_type": str(type(context[key]))},
                )

    return context
