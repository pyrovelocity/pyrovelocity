"""
Utility modules for PyroVelocity's modular architecture.

This package contains utility functions and classes that are used by the
Protocol-First component implementations in the modular architecture.

The utility modules are organized by functionality:
- core_utils: Error handling and general utilities
- context_utils: Context validation and manipulation utilities
- pyro_utils: Utilities for working with Pyro

These utilities are designed to be used by Protocol-First component implementations
that directly implement the Protocol interfaces without inheriting from base classes.
"""

from pyrovelocity.models.modular.utils.context_utils import create_error, validate_context

__all__ = [
    "create_error",
    "validate_context",
]
