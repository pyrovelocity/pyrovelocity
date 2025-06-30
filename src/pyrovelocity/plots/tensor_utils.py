"""
Framework-agnostic tensor utilities for PyroVelocity plotting functions.

This module provides utilities to convert tensors from different frameworks
(PyTorch, JAX, NumPy) to a common NumPy format for plotting.
"""

from typing import Dict, Union
import numpy as np
from beartype import beartype
from numpy.typing import ArrayLike


@beartype
def convert_to_numpy(data: ArrayLike) -> np.ndarray:
    """
    Convert tensor to numpy array, handling PyTorch, JAX, and NumPy tensors.
    
    Args:
        data: Input tensor or array from any framework
        
    Returns:
        NumPy array representation of the data
        
    Examples:
        >>> import torch
        >>> torch_tensor = torch.tensor([1, 2, 3])
        >>> numpy_array = convert_to_numpy(torch_tensor)
        >>> isinstance(numpy_array, np.ndarray)
        True
        
        >>> import jax.numpy as jnp
        >>> jax_array = jnp.array([1, 2, 3])
        >>> numpy_array = convert_to_numpy(jax_array)
        >>> isinstance(numpy_array, np.ndarray)
        True
    """
    # Already NumPy array
    if isinstance(data, np.ndarray):
        return data
    
    # PyTorch tensor
    elif hasattr(data, 'detach'):
        return data.detach().cpu().numpy()
    
    # JAX array
    elif hasattr(data, '__array__') and hasattr(data, 'shape'):
        return np.array(data)
    
    # Fallback: try to convert to numpy
    else:
        return np.array(data)


@beartype
def convert_parameters_to_numpy(parameters: Dict[str, ArrayLike]) -> Dict[str, np.ndarray]:
    """
    Convert all parameters in dictionary to numpy arrays.
    
    Args:
        parameters: Dictionary of parameters from any framework
        
    Returns:
        Dictionary with all values converted to NumPy arrays
        
    Examples:
        >>> import torch
        >>> params = {'a': torch.tensor([1, 2]), 'b': torch.tensor([3, 4])}
        >>> numpy_params = convert_parameters_to_numpy(params)
        >>> all(isinstance(v, np.ndarray) for v in numpy_params.values())
        True
    """
    return {key: convert_to_numpy(value) for key, value in parameters.items()}


@beartype
def ensure_numpy_parameters(
    parameters: Union[Dict[str, ArrayLike], None]
) -> Dict[str, np.ndarray]:
    """
    Ensure parameters are in numpy format for framework-agnostic plotting.
    
    This function automatically detects the tensor framework and converts
    to NumPy arrays as needed.
    
    Args:
        parameters: Dictionary of parameters from any framework, or None
        
    Returns:
        Dictionary with all values as NumPy arrays
    """
    if not parameters:
        return {}
    
    return convert_parameters_to_numpy(parameters)


@beartype
def framework_agnostic_sigmoid(x: ArrayLike) -> Union[np.ndarray, np.number]:
    """
    Framework-agnostic sigmoid function using NumPy.
    
    Args:
        x: Input value or array
        
    Returns:
        Sigmoid output using NumPy operations
    """
    x_np = convert_to_numpy(x)
    return 1 / (1 + np.exp(-np.clip(x_np, -500, 500)))  # Clip to prevent overflow


@beartype
def framework_agnostic_log2(x: ArrayLike) -> Union[np.ndarray, np.number]:
    """
    Framework-agnostic log2 function with numerical stability.
    
    Args:
        x: Input array
        
    Returns:
        Log2 of input with small epsilon added for numerical stability
    """
    x_np = convert_to_numpy(x)
    return np.log2(np.maximum(x_np, 1e-10))


@beartype
def framework_agnostic_exp(x: ArrayLike) -> Union[np.ndarray, np.number]:
    """
    Framework-agnostic exp function with numerical stability.
    
    Args:
        x: Input array
        
    Returns:
        Exponential of input with clipping to prevent overflow
    """
    x_np = convert_to_numpy(x)
    return np.exp(np.clip(x_np, -500, 500))