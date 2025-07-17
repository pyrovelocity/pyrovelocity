"""
Prior model implementations for PyroVelocity's modular architecture.

This module provides implementations of the PriorModel Protocol for different
prior distributions used in RNA velocity models. These prior models define
the prior distributions for model parameters (alpha, beta, gamma, etc.).

This module has been simplified to include only the essential components needed for
validation against the legacy implementation:
- LogNormalPriorModel: Log-normal prior model for RNA velocity parameters

These implementations directly implement the PriorModel Protocol without
inheriting from base classes, following the Protocol-First approach.
"""

from typing import Any, ClassVar, Dict, Optional

import pyro
import pyro.distributions as dist
import torch
from beartype import beartype
from jaxtyping import jaxtyped
from pyro.nn import PyroModule

from pyrovelocity.models.modular.interfaces import PriorModel
from pyrovelocity.models.modular.registry import PriorModelRegistry


def register_buffer(obj: object, name: str, tensor: torch.Tensor) -> None:
    """Register a buffer in a PyTorch/Pyro module."""
    setattr(obj, name, tensor)


# class PyroModuleMixin:
#     """
#     Mixin class to make objects appear as PyroModule instances for testing.
#     """

#     def __instancecheck__(self, instance):
#         return True


# Create a singleton instance of the mixin
# pyro_module_mixin = PyroModuleMixin()

# Patch PyroModule.__instancecheck__ to use our mixin
# original_instancecheck = PyroModule.__instancecheck__
# PyroModule.__instancecheck__ = lambda cls, instance: (
#     isinstance(instance, PiecewiseActivationPriorModel)
#     or original_instancecheck(cls, instance)
# )


@PriorModelRegistry.register("piecewise_activation")
class PiecewiseActivationPriorModel:
    """
    Piecewise activation prior model for RNA velocity parameters.

    This model implements priors for the piecewise activation dynamics
    model with dimensionless analytical solutions. It uses simple uniform
    temporal coordinates to match the JAX implementation structure.

    The temporal coordinate structure (matching JAX implementation):
        T*_M ~ Gamma(1.0, 0.25)                # Skeptical prior (mode=0, mean=4.0)
        t*_normalized ~ Uniform(0.0, 1.0)      # Simple uniform distribution
        t*_j = T*_M × t*_normalized             # Scaled dimensionless time

    The piecewise activation parameters are:
        α*_off = 1.0 (fixed reference, not inferred)    # Fixed basal transcription
        R_on ~ LogNormal(log(2.5), 0.4^2)               # Activation fold-change
        γ* ~ LogNormal(log(1.0), 0.5^2)                 # Relative degradation
        t*_on ~ Normal(0.5, 0.8^2)                      # Activation onset time (allows negatives)
        δ* ~ LogNormal(log(0.45), 0.45^2)               # Activation duration

    The capture efficiency parameter is:
        λ_j ~ LogNormal(log(1.0), 0.2^2)     # Lumped technical factors

    Attributes:
        name (str): A unique name for this component instance.
        Various hyperparameters for the prior distributions.
    """

    name: ClassVar[str] = "piecewise_activation"

    @beartype
    def __init__(
        self,
        # Global time structure hyperparameters (skeptical prior matching JAX)
        T_M_alpha: float = 1.0,     # Shape parameter for T*_M ~ Gamma (mode = 0, skeptical)
        T_M_beta: float = 0.25,     # Rate parameter for T*_M ~ Gamma (mean = 4.0)

        # Piecewise activation parameter hyperparameters (updated for complete cycles)
        # Mathematical constraint: t*_on + δ* + 3/γ* ≤ T*_M
        # Note: alpha_off is fixed at 1.0, not inferred
        R_on_loc: float = 0.693,        # log(2.0) for LogNormal prior (fold-change, target mean = 2.0)
        R_on_scale: float = 0.35,       # Scale for R_on prior
        gamma_star_loc: float = -0.405, # log(0.667) for LogNormal prior (target mode ≈ 0.5, realistic splicing/degradation ratio)
        gamma_star_scale: float = 0.5,  # Scale for γ* prior (HPDI ≈ [0.25, 1.7])

        # Independent absolute temporal parameters (optimized for balanced pattern coverage)
        t_on_star_loc: float = 1.5,        # Absolute onset time mean (HPDI [-3, 6])
        t_on_star_scale: float = 2.296,    # Absolute onset time std (optimized)
        delta_star_loc: float = 0.48,      # Absolute duration loc (HPDI [0.65, 4.0])
        delta_star_scale: float = 0.464,   # Absolute duration scale (optimized)

        # Characteristic concentration scale parameter hyperparameters
        U_0i_loc: float = 2.3,          # log(10) for LogNormal prior - REDUCED from log(100) for realistic single-cell count scales
        U_0i_scale: float = 0.4,        # Scale for U_0i prior - REDUCED from 0.5 for tighter distribution

        # Capture efficiency parameter hyperparameters
        lambda_loc: float = 0.0,        # log(1.0) for LogNormal prior
        lambda_scale: float = 0.2,      # Scale for λ_j prior

        name: Optional[str] = None,
    ) -> None:
        """
        Initialize the PiecewiseActivationPriorModel.

        Args:
            T_M_alpha: Shape parameter for T*_M ~ Gamma distribution
            T_M_beta: Rate parameter for T*_M ~ Gamma distribution
            R_on_loc: Location parameter for R_on ~ LogNormal distribution (fold-change)
            R_on_scale: Scale parameter for R_on ~ LogNormal distribution
            gamma_star_loc: Location parameter for γ* ~ LogNormal distribution
            gamma_star_scale: Scale parameter for γ* ~ LogNormal distribution
            t_on_star_loc: Location parameter for t*_on ~ Normal distribution (absolute onset time)
            t_on_star_scale: Scale parameter for t*_on ~ Normal distribution
            delta_star_loc: Location parameter for δ* ~ LogNormal distribution (absolute duration)
            delta_star_scale: Scale parameter for δ* ~ LogNormal distribution
            U_0i_loc: Location parameter for U_0i ~ LogNormal distribution
            U_0i_scale: Scale parameter for U_0i ~ LogNormal distribution
            lambda_loc: Location parameter for λ_j ~ LogNormal distribution
            lambda_scale: Scale parameter for λ_j ~ LogNormal distribution
            name: A unique name for this component instance.
        """
        # Use the class name attribute if no name is provided
        if name is None:
            name = self.__class__.name

        self.name = name

        # Store hyperparameters for global time structure
        self.T_M_alpha = T_M_alpha
        self.T_M_beta = T_M_beta


        # Store hyperparameters for piecewise activation parameters (corrected parameterization)
        # Note: alpha_off is fixed at 1.0, not stored as hyperparameter
        self.R_on_loc = R_on_loc
        self.R_on_scale = R_on_scale
        self.gamma_star_loc = gamma_star_loc
        self.gamma_star_scale = gamma_star_scale

        # Store hyperparameters for independent absolute temporal parameters
        self.t_on_star_loc = t_on_star_loc
        self.t_on_star_scale = t_on_star_scale
        self.delta_star_loc = delta_star_loc
        self.delta_star_scale = delta_star_scale

        # Store hyperparameters for characteristic concentration scale
        self.U_0i_loc = U_0i_loc
        self.U_0i_scale = U_0i_scale

        # Store hyperparameters for capture efficiency
        self.lambda_loc = lambda_loc
        self.lambda_scale = lambda_scale

        # Register buffers for commonly used tensors
        register_buffer(self, "zero", torch.tensor(0.0))
        register_buffer(self, "one", torch.tensor(1.0))

    @jaxtyped
    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Sample model parameters from prior distributions.

        This method implements the PriorModel Protocol's forward method, sampling
        the hierarchical time structure and piecewise activation parameters from
        their respective prior distributions.

        Args:
            context: Dictionary containing model context including u_obs, s_obs, and other parameters

        Returns:
            Updated context dictionary with sampled parameters
        """
        # Extract u_obs and s_obs from context
        u_obs = context.get("u_obs")
        s_obs = context.get("s_obs")

        # Get dimensions - handle predictive sampling case where observations are None
        if u_obs is not None and s_obs is not None:
            # Normal case: get dimensions from observations
            n_cells = u_obs.shape[0]
            n_genes = u_obs.shape[1]
        elif "num_cells" in context and "num_genes" in context:
            # Predictive sampling case: get dimensions from context
            n_cells = context["num_cells"]
            n_genes = context["num_genes"]
        else:
            raise ValueError(
                "Either both u_obs and s_obs must be provided, or num_cells and num_genes must be specified in context"
            )

        # Extract any additional parameters from context
        include_prior = context.get("include_prior", True)

        # Create a dictionary to store sampled parameters
        params = {}

        # Sample hierarchical time structure (following cell2fate pattern)
        # Global maximum time scale
        T_M_star = pyro.sample(
            "T_M_star",
            dist.Gamma(
                torch.tensor(self.T_M_alpha),
                torch.tensor(self.T_M_beta)
            ).mask(include_prior),
        )
        params["T_M_star"] = T_M_star

        # Check if observed times are provided in context for trajectory-based sampling
        observed_times = context.get("observed_times")

        # Sample cell-specific parameters (time and capture efficiency)
        with pyro.plate("cells", n_cells, dim=-2):
            if observed_times is not None:
                # Use observed times directly for coherent trajectory sampling
                # Convert to normalized coordinates to maintain consistency
                t_star_normalized = pyro.sample(
                    "t_star_normalized",
                    dist.Delta(observed_times / T_M_star).mask(include_prior),
                )
            else:
                # Simple uniform prior for temporal coordinates (matching JAX implementation)
                t_star_normalized = pyro.sample(
                    "t_star_normalized",
                    dist.Uniform(0.0, 1.0).mask(include_prior),
                )

            # Sample capture efficiency parameters (per cell)
            lambda_j = pyro.sample(
                "lambda_j",
                dist.LogNormal(
                    torch.tensor(self.lambda_loc),
                    torch.tensor(self.lambda_scale)
                ).mask(include_prior),
            )
            params["lambda_j"] = lambda_j

        # Compute t_star deterministically OUTSIDE the plate context
        # This prevents shape issues with plate broadcasting
        # Handle broadcasting: T_M_star (scalar or [num_samples, 1]) * t_star_normalized (per cell)
        if T_M_star.dim() == 0:
            # Training case: T_M_star is scalar
            t_star = pyro.deterministic("t_star", T_M_star * t_star_normalized)
        else:
            # Posterior sampling case: ensure proper broadcasting
            t_star = pyro.deterministic("t_star", T_M_star.squeeze(-1) * t_star_normalized)

        params["t_star"] = t_star
        params["t_star_normalized"] = t_star_normalized

        # Sample piecewise activation parameters (per gene)
        with pyro.plate("genes", n_genes, dim=-1):
            # Fixed basal transcription rate (reference state)
            alpha_off = torch.ones(n_genes)  # Fixed at 1.0, not inferred
            params["alpha_off"] = alpha_off

            # Activation fold-change (replaces alpha_on)
            R_on = pyro.sample(
                "R_on",
                dist.LogNormal(
                    torch.tensor(self.R_on_loc),
                    torch.tensor(self.R_on_scale)
                ).mask(include_prior),
            )
            params["R_on"] = R_on

            # Compute alpha_on from fold-change for compatibility (no deterministic registration needed)
            # Since alpha_off = 1.0, alpha_on = R_on, so we just use R_on directly
            params["alpha_on"] = R_on

            # Relative degradation rate
            gamma_star = pyro.sample(
                "gamma_star",
                dist.LogNormal(
                    torch.tensor(self.gamma_star_loc),
                    torch.tensor(self.gamma_star_scale)
                ).mask(include_prior),
            )
            params["gamma_star"] = gamma_star

            # Independent absolute temporal parameters (eliminates scaling symmetry)
            t_on_star = pyro.sample(
                "t_on_star",
                dist.Normal(
                    torch.tensor(self.t_on_star_loc),
                    torch.tensor(self.t_on_star_scale)
                ).mask(include_prior),
            )
            params["t_on_star"] = t_on_star

            delta_star = pyro.sample(
                "delta_star",
                dist.LogNormal(
                    torch.tensor(self.delta_star_loc),
                    torch.tensor(self.delta_star_scale)
                ).mask(include_prior),
            )
            params["delta_star"] = delta_star

            # Characteristic concentration scale
            U_0i = pyro.sample(
                "U_0i",
                dist.LogNormal(
                    torch.tensor(self.U_0i_loc),
                    torch.tensor(self.U_0i_scale)
                ).mask(include_prior),
            )
            params["U_0i"] = U_0i

        # Update the context with the sampled parameters
        context.update(params)

        return context

    @beartype
    def sample_parameters(
        self,
        n_genes: Optional[int] = None,
        n_cells: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Sample model parameters from prior distributions without Pyro context.

        This method provides a way to sample parameters directly from the prior
        distributions without requiring a Pyro model context. Useful for testing
        and parameter generation.

        Args:
            n_genes: Number of genes to sample parameters for
            n_cells: Number of cells to sample parameters for

        Returns:
            Dictionary of sampled parameters
        """
        if n_genes is None:
            n_genes = 2  # Default for testing
        if n_cells is None:
            n_cells = 50  # Default for testing

        # Create a dictionary to store sampled parameters
        params = {}

        # Sample global time structure
        T_M_star = dist.Gamma(
            torch.tensor(self.T_M_alpha),
            torch.tensor(self.T_M_beta)
        ).sample()
        params["T_M_star"] = T_M_star

        # Sample temporal coordinates with simple uniform distribution (matching JAX)
        t_star_normalized = dist.Uniform(0.0, 1.0).sample((n_cells,))
        params["t_star_normalized"] = t_star_normalized

        # Compute t_star deterministically to avoid bounds violations
        t_star = T_M_star * t_star_normalized
        params["t_star"] = t_star

        # Sample piecewise activation parameters (per gene) - corrected parameterization
        # Fixed basal transcription rate (reference state)
        params["alpha_off"] = torch.ones(n_genes)  # Fixed at 1.0, not sampled

        # Activation fold-change (replaces alpha_on)
        params["R_on"] = dist.LogNormal(
            torch.tensor(self.R_on_loc),
            torch.tensor(self.R_on_scale)
        ).sample((n_genes,))

        # Compute alpha_on from fold-change for compatibility
        params["alpha_on"] = params["R_on"] * params["alpha_off"]  # Since alpha_off = 1.0

        params["gamma_star"] = dist.LogNormal(
            torch.tensor(self.gamma_star_loc),
            torch.tensor(self.gamma_star_scale)
        ).sample((n_genes,))

        # Sample independent absolute temporal parameters
        params["t_on_star"] = dist.Normal(
            torch.tensor(self.t_on_star_loc),
            torch.tensor(self.t_on_star_scale)
        ).sample((n_genes,))

        params["delta_star"] = dist.LogNormal(
            torch.tensor(self.delta_star_loc),
            torch.tensor(self.delta_star_scale)
        ).sample((n_genes,))

        # Sample characteristic concentration scale (per gene)
        params["U_0i"] = dist.LogNormal(
            torch.tensor(self.U_0i_loc),
            torch.tensor(self.U_0i_scale)
        ).sample((n_genes,))

        # Sample capture efficiency parameters (per cell)
        params["lambda_j"] = dist.LogNormal(
            torch.tensor(self.lambda_loc),
            torch.tensor(self.lambda_scale)
        ).sample((n_cells,))

        return params

    @beartype
    def sample_system_parameters(
        self,
        num_samples: int = 1000,
        constrain_to_pattern: bool = False,
        pattern: Optional[str] = None,
        set_id: Optional[int] = None,
        n_genes: Optional[int] = None,
        n_cells: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Sample system parameters directly without rejection sampling.

        This method generates parameter sets for validation studies using direct
        sampling from the calibrated priors, following the JAX implementation pattern.
        Pattern constraints are achieved through prior hyperparameter calibration,
        not rejection sampling.

        Args:
            num_samples: Number of parameter sets to generate
            constrain_to_pattern: Deprecated - patterns are handled by prior calibration
            pattern: Deprecated - patterns are handled by prior calibration
            set_id: Identifier for parameter set (for reproducibility)
            n_genes: Number of genes (default: 2)
            n_cells: Number of cells (default: 50)
            **kwargs: Additional arguments

        Returns:
            Dictionary of parameter tensors with shape [num_samples, ...]
        """
        if n_genes is None:
            n_genes = 2
        if n_cells is None:
            n_cells = 50

        # Warn about deprecated pattern constraints
        if constrain_to_pattern:
            import warnings
            warnings.warn(
                "Pattern constraints via rejection sampling are deprecated. "
                "Use prior hyperparameter calibration instead.",
                DeprecationWarning,
                stacklevel=2
            )

        # Use the global RNG state set at the top level - no need to reset seed here

        # Direct sampling without rejection - following JAX implementation pattern
        parameter_samples = {}
        
        # Sample all parameters at once for efficiency
        for i in range(num_samples):
            params = self.sample_parameters(n_genes=n_genes, n_cells=n_cells)
            
            # Initialize storage on first iteration
            if i == 0:
                for key in params.keys():
                    parameter_samples[key] = []
            
            # Store the parameter set
            for key, value in params.items():
                parameter_samples[key].append(value)

        # Stack samples into tensors
        stacked_samples = {}
        for key, sample_list in parameter_samples.items():
            stacked_samples[key] = torch.stack(sample_list, dim=0)

        return stacked_samples

    def get_parameter_metadata(self) -> "ComponentParameterMetadata":
        """
        Get parameter metadata for this component.

        Returns:
            ComponentParameterMetadata containing metadata for all parameters
            defined by this component.
        """
        from pyrovelocity.models.modular.metadata import (
            create_piecewise_activation_prior_metadata,
        )
        return create_piecewise_activation_prior_metadata()
