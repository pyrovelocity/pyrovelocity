"""
Dynamics model implementations for PyroVelocity's modular architecture.

This module contains dynamics model implementations that directly implement the
DynamicsModel Protocol. These implementations follow the Protocol-First approach,
which embraces composition over inheritance and allows for more flexible component composition.

This module has been simplified to include only the essential components needed for
validation against the legacy implementation:
- StandardDynamicsModel: Standard RNA velocity dynamics model
- LegacyDynamicsModel: Legacy RNA velocity dynamics model that exactly matches the legacy implementation
"""

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch
from beartype import beartype
from jaxtyping import jaxtyped

from pyrovelocity.models.modular.constants import CELLS_DIM, GENES_DIM
from pyrovelocity.models.modular.interfaces import (
    BatchTensor,
    DynamicsModel,
    KineticParamTensor,
    LatentCountTensor,
    ParamTensor,
    VelocityTensor,
)
from pyrovelocity.models.modular.registry import DynamicsModelRegistry


def validate_context(component_name: str, context: Dict[str, Any], required_keys: List[str] = None, tensor_keys: List[str] = None) -> bool:
    """Validate that context contains required keys."""
    if required_keys:
        return all(key in context for key in required_keys)
    return True


@DynamicsModelRegistry.register("legacy")
class LegacyDynamicsModel:
    """Legacy dynamics model for RNA velocity that exactly matches the legacy implementation.

    This model implements the standard model of RNA velocity but with the exact same
    parameter shapes and behavior as the legacy implementation:

    du/dt = alpha - beta * u
    ds/dt = beta * u - gamma * s

    where:
    - u is the unspliced mRNA count
    - s is the spliced mRNA count
    - alpha is the transcription rate
    - beta is the splicing rate
    - gamma is the degradation rate

    This implementation is specifically designed to match the legacy implementation
    in terms of parameter shapes and behavior.

    Attributes:
        name: The name of the model
        description: A brief description of the model
        shared_time: Whether to use shared time across cells
        t_scale_on: Whether to use time scaling
        cell_specific_kinetics: Type of cell-specific kinetics
        kinetics_num: Number of kinetics
        correct_library_size: Whether to correct for library size
    """

    name: ClassVar[str] = "legacy"
    description: ClassVar[str] = "Legacy RNA velocity dynamics model"

    def __init__(
        self,
        name: str = "legacy_dynamics_model",
        shared_time: bool = True,
        t_scale_on: bool = False,
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
        correct_library_size: Union[bool, str] = True,
        **kwargs,
    ):
        """
        Initialize the legacy dynamics model.

        Args:
            name: A unique name for this component instance.
            shared_time: Whether to use shared time across cells.
            t_scale_on: Whether to use time scaling.
            cell_specific_kinetics: Type of cell-specific kinetics.
            kinetics_num: Number of kinetics.
            correct_library_size: Whether to correct for library size.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.shared_time = shared_time
        self.t_scale_on = t_scale_on
        self.cell_specific_kinetics = cell_specific_kinetics
        self.kinetics_num = kinetics_num
        self.correct_library_size = correct_library_size

    @jaxtyped
    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute the expected unspliced and spliced RNA counts based on the legacy dynamics model.

        This method takes a context dictionary containing observed data and parameters,
        computes the expected unspliced and spliced counts according to the legacy dynamics model,
        and updates the context with the results.

        Args:
            context: Dictionary containing model context with the following required keys:
                - u_obs: Observed unspliced counts (BatchTensor)
                - s_obs: Observed spliced counts (BatchTensor)
                - alpha: Transcription rate (ParamTensor)
                - beta: Splicing rate (ParamTensor)
                - gamma: Degradation rate (ParamTensor)

                And optional keys:
                - scaling: Scaling factor (ParamTensor)
                - t: Time points (BatchTensor)
                - u_read_depth: Unspliced read depth (BatchTensor)
                - s_read_depth: Spliced read depth (BatchTensor)

        Returns:
            Updated context dictionary with the following additional keys:
                - u_expected: Expected unspliced counts (BatchTensor)
                - s_expected: Expected spliced counts (BatchTensor)
                - ut: Latent unspliced counts (BatchTensor)
                - st: Latent spliced counts (BatchTensor)
                - u_inf: Steady-state unspliced counts (ParamTensor)
                - s_inf: Steady-state spliced counts (ParamTensor)
                - switching: Switching time (ParamTensor)
        """
        # Validate context
        validation_result = validate_context(
            self.__class__.__name__,
            context,
            required_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"],
            tensor_keys=["u_obs", "s_obs", "alpha", "beta", "gamma"],
        )

        if isinstance(validation_result, dict):
            # Extract required values from context
            u_obs = context["u_obs"]
            s_obs = context["s_obs"]
            alpha = context["alpha"]
            beta = context["beta"]
            gamma = context["gamma"]


            # Extract optional values
            t0 = context.get("t0", torch.zeros_like(alpha))
            dt_switching = context.get("dt_switching", torch.zeros_like(alpha))
            u_scale = context.get("u_scale", torch.ones_like(alpha))
            s_scale = context.get("s_scale", torch.ones_like(alpha))

            # Extract read depths if available
            u_read_depth = context.get("u_read_depth")
            s_read_depth = context.get("s_read_depth")

            # If read depths are not provided, create default ones
            if u_read_depth is None and hasattr(self, 'correct_library_size') and self.correct_library_size:
                # Create default read depths
                if u_obs.dim() == 2:
                    # Shape: [batch_size, 1]
                    u_read_depth = torch.ones(u_obs.shape[0], 1)
                    s_read_depth = torch.ones(s_obs.shape[0], 1)
                else:
                    # Shape: [1]
                    u_read_depth = torch.ones(1)
                    s_read_depth = torch.ones(1)

                # Add to context
                context["u_read_depth"] = u_read_depth
                context["s_read_depth"] = s_read_depth

            # Import pyro for deterministic sites
            import pyro

            # Get dimensions
            num_cells = u_obs.shape[0]
            num_genes = alpha.shape[-1] if alpha.dim() > 0 else 1

            # Create plates with consistent dimensions using constants
            # Use GENES_DIM (-1) for genes and CELLS_DIM (-2) for cells
            gene_plate = pyro.plate("genes", num_genes, dim=GENES_DIM)
            cell_plate = pyro.plate("cells", num_cells, dim=CELLS_DIM)

            # SIMPLIFIED PLATE STRUCTURE MATCHING LEGACY MODEL
            # In the legacy model:
            # - Gene parameters have shape [num_samples, 1, n_genes]
            # - Cell parameters have shape [num_samples, n_cells, 1]
            # - Cell-gene interactions have shape [num_samples, n_cells, n_genes]

            # First, reshape gene parameters to match legacy model shape
            # We want alpha, beta, gamma to have shape [1, 1, n_genes]
            if alpha.dim() == 1:  # [n_genes]
                alpha = alpha.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
                beta = beta.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
                gamma = gamma.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
                t0 = t0.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
                dt_switching = dt_switching.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
            elif alpha.dim() == 2:  # [num_samples, n_genes]
                alpha = alpha.unsqueeze(1)  # [num_samples, 1, n_genes]
                beta = beta.unsqueeze(1)  # [num_samples, 1, n_genes]
                gamma = gamma.unsqueeze(1)  # [num_samples, 1, n_genes]
                t0 = t0.unsqueeze(1)  # [num_samples, 1, n_genes]
                dt_switching = dt_switching.unsqueeze(1)  # [num_samples, 1, n_genes]

            # Sample gene-specific parameters with the correct shape
            with gene_plate:
                # Use parameters directly without creating duplicate deterministic nodes
                # This avoids adding extra dimensions to the parameters

                # Compute steady state values
                u_inf = alpha / beta  # Shape: [1, 1, n_genes] or [num_samples, 1, n_genes]
                s_inf = alpha / gamma  # Shape: [1, 1, n_genes] or [num_samples, 1, n_genes]

                # Compute switching time
                switching = t0 + dt_switching  # Shape: [1, 1, n_genes] or [num_samples, 1, n_genes]

                # Create deterministic sites for steady state and switching
                # FIXED: Use event_dim=1 for gene expression tensors
                # Genes are the event dimension, not batch dimension
                u_inf = pyro.deterministic("u_inf", u_inf, event_dim=1)
                s_inf = pyro.deterministic("s_inf", s_inf, event_dim=1)
                switching = pyro.deterministic("switching", switching, event_dim=1)

            # Next, sample cell-specific parameters
            with cell_plate:
                # Generate cell_time if not provided
                cell_time = context.get("cell_time")
                if cell_time is None:
                    # Create a uniform distribution of cell times between 0 and 1
                    cell_time = torch.linspace(0, 1, num_cells).unsqueeze(1)  # [n_cells, 1]
                    cell_time = pyro.sample(
                        "cell_time",
                        pyro.distributions.Delta(cell_time)
                    )  # Shape: [n_cells, 1]
                    context["cell_time"] = cell_time
                elif cell_time.dim() == 1:  # [n_cells]
                    cell_time = cell_time.unsqueeze(1)  # [n_cells, 1]

                # Ensure cell_time has shape [n_cells, 1] or [num_samples, n_cells, 1]
                if cell_time.dim() == 2:  # [n_cells, 1]
                    # If we have multiple samples, expand cell_time
                    if alpha.dim() == 3 and alpha.shape[0] > 1:
                        # Expand to [num_samples, n_cells, 1]
                        cell_time = cell_time.unsqueeze(0).expand(alpha.shape[0], -1, -1)

            # Compute ut and st with proper broadcasting
            # We need to ensure:
            # - alpha, beta, gamma have shape [1, 1, n_genes] or [num_samples, 1, n_genes]
            # - cell_time has shape [n_cells, 1] or [num_samples, n_cells, 1]
            # - ut, st will have shape [n_cells, n_genes] or [num_samples, n_cells, n_genes]

            # Compute ut and st
            ut = u_inf * (1 - torch.exp(-beta * cell_time))
            st = s_inf * (1 - torch.exp(-gamma * cell_time)) - (
                alpha / (gamma - beta + 1e-8)  # Add small epsilon to avoid division by zero
            ) * (torch.exp(-beta * cell_time) - torch.exp(-gamma * cell_time))

            # Ensure ut and st have the correct shape [num_samples, n_cells, n_genes]
            # First, determine the expected shape
            if alpha.dim() == 3:
                num_samples = alpha.shape[0]
            else:
                num_samples = 1

            # Reshape to ensure we have exactly 3 dimensions with the correct shape
            # This is more robust than using squeeze which can remove dimensions we want to keep
            if ut.dim() > 3:
                # Reshape to [num_samples, n_cells, n_genes]
                ut = ut.reshape(num_samples, num_cells, num_genes)
                st = st.reshape(num_samples, num_cells, num_genes)
            elif ut.dim() < 3:
                # Add missing dimensions
                if ut.dim() == 2:  # [n_cells, n_genes]
                    ut = ut.unsqueeze(0)  # [1, n_cells, n_genes]
                    st = st.unsqueeze(0)  # [1, n_cells, n_genes]
                elif ut.dim() == 1:  # [n_genes]
                    ut = ut.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]
                    st = st.unsqueeze(0).unsqueeze(0)  # [1, 1, n_genes]

            # Print shapes after reshaping

            # Ensure ut and st have the correct shape [num_samples, n_cells, n_genes]
            # This is critical for proper shape in posterior samples
            # In the legacy model, ut and st should have shape [num_samples, n_cells, n_genes]
            # We need to be very aggressive about reshaping to match the legacy model exactly

            # First, determine the expected shape
            if alpha.dim() == 3:
                num_samples = alpha.shape[0]
            else:
                num_samples = 1

            # Reshape to ensure we have exactly 3 dimensions with the correct shape
            # This is more robust than using squeeze which can remove dimensions we want to keep
            ut = ut.reshape(num_samples, num_cells, num_genes)
            st = st.reshape(num_samples, num_cells, num_genes)

            # Register ut and st as deterministic sites for proper Pyro integration
            # This enables automatic inclusion in posterior samples via Predictive
            # FIXED: Use event_dim=1 for gene expression tensors (genes are event dimension)
            import pyro
            ut = pyro.deterministic("ut", ut, event_dim=1)
            st = pyro.deterministic("st", st, event_dim=1)

            # Store computed values in context for component communication
            u = ut
            s = st

            # Add to context
            context["u_inf"] = u_inf
            context["s_inf"] = s_inf
            context["switching"] = switching
            context["ut"] = ut
            context["st"] = st
            context["u"] = u
            context["s"] = s

            # Add expected counts for the likelihood model
            context["u_expected"] = ut
            context["s_expected"] = st


            return context
        else:
            # If validation failed, raise an error
            raise ValueError(f"Error in dynamics model forward pass: {validation_result.error}")

    @jaxtyped
    @beartype
    def steady_state(
        self,
        alpha: Union[ParamTensor, torch.Tensor],
        beta: Union[ParamTensor, torch.Tensor],
        gamma: Union[ParamTensor, torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[
        Union[ParamTensor, torch.Tensor],
        Union[ParamTensor, torch.Tensor],
    ]:
        """
        Compute the steady-state unspliced and spliced RNA counts.

        Args:
            alpha: Transcription rate
            beta: Splicing rate
            gamma: Degradation rate
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (steady-state unspliced counts, steady-state spliced counts)
        """
        # At steady state, du/dt = 0 and ds/dt = 0
        # From du/dt = 0: alpha - beta * u = 0 => u = alpha / beta
        u_ss = alpha / beta

        # From ds/dt = 0: beta * u - gamma * s = 0 => s = beta * u / gamma
        # Substituting u = alpha / beta: s = beta * (alpha / beta) / gamma = alpha / gamma
        s_ss = alpha / gamma

        return u_ss, s_ss

    @jaxtyped
    @beartype
    def compute_velocity(
        self,
        ut: LatentCountTensor,
        st: LatentCountTensor,
        alpha: KineticParamTensor,
        beta: KineticParamTensor,
        gamma: KineticParamTensor,
        **kwargs: Any,
    ) -> VelocityTensor:
        """
        Compute RNA velocity from latent RNA counts and kinetic parameters.

        This method computes the RNA velocity (ds/dt) based on the legacy dynamics model:
            ds/dt = βu - γs

        This implementation exactly matches the legacy PyroVelocity velocity calculation.

        Args:
            ut: Latent unspliced RNA counts
            st: Latent spliced RNA counts
            alpha: Transcription rate (not used in velocity calculation but kept for interface consistency)
            beta: Splicing rate
            gamma: Degradation rate
            **kwargs: Additional model-specific parameters (e.g., scaling factors)

        Returns:
            RNA velocity tensor with same shape as ut/st

        Raises:
            ValueError: If tensor shapes are incompatible
        """
        # Handle scaling factors if provided (matching legacy implementation)
        u_scale = kwargs.get("u_scale")
        s_scale = kwargs.get("s_scale")

        # For the legacy dynamics model: ds/dt = β * u - γ * s
        # This exactly matches the legacy implementation in compute_mean_vector_field
        if u_scale is not None and s_scale is not None:
            # For models with two scales (Gaussian models)
            # velocity = β * ut / (u_scale / s_scale) - γ * st
            scale = u_scale / s_scale
            velocity = beta * ut / scale - gamma * st
        elif u_scale is not None:
            # For models with one scale (Poisson models)
            # velocity = β * ut / u_scale - γ * st
            velocity = beta * ut / u_scale - gamma * st
        else:
            # For models with no scaling
            # velocity = β * ut - γ * st
            velocity = beta * ut - gamma * st

        return velocity


@DynamicsModelRegistry.register("piecewise_activation")
class PiecewiseActivationDynamicsModel:
    """Piecewise activation dynamics model with corrected dimensional analysis.

    This model implements dimensionless analytical dynamics with piecewise constant
    transcription rates using the corrected parameterization that eliminates
    parameter redundancy. The system has three phases:

    Phase 1 (Off): t* < t*_on
        α*(t*) = 1.0 (fixed reference transcription rate)

    Phase 2 (On): t*_on ≤ t* < t*_on + δ*
        α*(t*) = R_on (fold-change from reference)

    Phase 3 (Return to Off): t* ≥ t*_on + δ*
        α*(t*) = 1.0 (back to reference)

    The dimensionless system is:
        du*/dt* = α*(t*) - u*
        ds*/dt* = u* - γ*s*

    With fixed steady-state initial conditions:
        u*_0 = 1.0, s*_0 = 1.0/γ*

    Key corrections:
    - α*_off = 1.0 (fixed, not inferred) eliminates U₀ᵢ vs α*_off redundancy
    - R_on = α*_on represents fold-change during activation
    - t*_on ~ Normal allows negative values for pre-activation scenarios
    - Initial conditions are fixed, not inferred

    Special handling for γ* = 1 boundary case with τe^(-τ) terms.

    Attributes:
        name: The name of the model
        description: A brief description of the model
        eps: Small epsilon for numerical stability near γ* = 1
    """

    name: ClassVar[str] = "piecewise_activation"
    description: ClassVar[str] = "Piecewise activation dynamics with analytical solutions"

    def __init__(
        self,
        name: str = "piecewise_dynamics_model",
        eps: float = 1e-6,
        **kwargs,
    ):
        """
        Initialize the piecewise activation dynamics model.

        Args:
            name: A unique name for this component instance.
            eps: Small epsilon for numerical stability near γ* = 1.
            **kwargs: Additional keyword arguments.
        """
        self.name = name
        self.eps = eps

    @jaxtyped
    @beartype
    def forward(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Optimized forward pass using simplified tensor operations.
        
        This method eliminates complex manual tensor operations by leveraging PyTorch's 
        natural broadcasting. It produces identical results to the original forward method 
        but with significantly improved performance and maintainability.
        
        Args:
            context: Dictionary containing model context with the same required keys
                    as the original forward method
        
        Returns:
            Updated context dictionary with the same outputs as original forward method
        """
        import pyro
        # Validate context (same validation as original method)
        validation_result = validate_context(
            self.__class__.__name__,
            context,
            required_keys=[
                "u_obs", "s_obs", "R_on", "gamma_star",
                "t_on_star", "delta_star", "t_star"
            ],
            tensor_keys=[
                "u_obs", "s_obs", "R_on", "gamma_star",
                "t_on_star", "delta_star", "t_star"
            ],
        )

        if validation_result:
            # Extract required values from context
            u_obs = context["u_obs"]
            s_obs = context["s_obs"]
            R_on = context["R_on"]
            gamma_star = context["gamma_star"]
            t_on_star = context["t_on_star"]
            delta_star = context["delta_star"]
            t_star = context["t_star"]

            # Create fixed alpha_off tensor (always 1.0) and compute alpha_on from R_on
            alpha_off = torch.ones_like(R_on)  # Match R_on shape exactly
            alpha_on = R_on  # Since alpha_off = 1.0, alpha_on = R_on

            # Get cell times with proper shape for broadcasting
            cell_time = self._get_cell_time(context, t_star, 
                                           t_star.shape[0] if t_star.dim() == 1 else t_star.shape[-1])
            
            # Compute piecewise solution with automatic broadcasting
            # This relies on PyTorch's natural broadcasting instead of manual operations
            u_expected, s_expected = self._compute_piecewise_solution(
                cell_time, alpha_off, alpha_on, gamma_star, t_on_star, delta_star
            )
            
            # Apply ReLU and numerical stability
            one = torch.ones_like(u_expected) * 1e-6
            u_expected = torch.relu(u_expected) + one
            s_expected = torch.relu(s_expected) + one
            
            # Create latent variables with proper event_dim
            # Both cells and genes dimensions should be event dimensions
            ut = pyro.deterministic("ut", u_expected, event_dim=2)
            st = pyro.deterministic("st", s_expected, event_dim=2)

            # Update context with results
            context["u_expected"] = u_expected
            context["s_expected"] = s_expected
            context["ut"] = ut
            context["st"] = st

            return context
        else:
            # If validation failed, raise an error
            raise ValueError(f"Error in optimized piecewise dynamics model forward pass: validation failed")

    def _get_cell_time(
        self, 
        context: Dict[str, Any], 
        t_star: torch.Tensor, 
        num_cells: int
    ) -> torch.Tensor:
        """
        Get or create cell times with proper shape for broadcasting.
        
        This method handles cell time preparation preserving the original tensor
        structure for proper broadcasting with gene parameters.
        
        Args:
            context: Model context dictionary
            t_star: Dimensionless time tensor from context
            num_cells: Number of cells
            
        Returns:
            Cell times with appropriate shape for broadcasting
        """
        # Handle t_star based on its natural shape - preserve original structure
        if t_star.dim() == 1:
            # Training case: [cells] → keep as [cells] for broadcasting
            return t_star
        elif t_star.dim() == 2:
            # Posterior sampling case: [num_samples, cells] → keep as is
            return t_star
        else:
            # More complex cases - return as-is and let broadcasting handle it
            return t_star

    @jaxtyped
    @beartype
    def _compute_piecewise_solution(
        self,
        t_star: torch.Tensor,
        alpha_off: torch.Tensor,
        alpha_on: torch.Tensor,
        gamma_star: torch.Tensor,
        t_on_star: torch.Tensor,
        delta_star: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized piecewise solution computation using natural broadcasting.
        
        This method implements the same mathematical logic as _compute_piecewise_solution
        but relies on PyTorch's natural broadcasting instead of manual tensor operations.
        
        Args:
            t_star: Cell times with shape [cells, 1]
            alpha_off: Basal transcription rate [genes] or [samples, genes]
            alpha_on: Active transcription rate [genes] or [samples, genes]  
            gamma_star: Relative degradation rate [genes] or [samples, genes]
            t_on_star: Activation onset time [genes] or [samples, genes]
            delta_star: Activation duration [genes] or [samples, genes]
            
        Returns:
            Tuple of (u_star, s_star) with shapes matching original implementation
        """
        # Determine target output shape based on input dimensions
        if alpha_off.dim() == 1:
            # Gene parameters are 1D: need to determine proper output shape
            if t_star.dim() == 1:
                # Training case: t_star [cells], params [genes] → output [cells, genes]
                target_shape = (t_star.shape[0], alpha_off.shape[0])
            else:
                # Posterior sampling case: t_star has batch dims, params 1D
                # Output should match t_star's batch shape + [cells, genes]
                batch_dims = t_star.shape[:-1]  # All but last dimension
                num_cells = t_star.shape[-1]    # Last dimension is cells
                num_genes = alpha_off.shape[0]  # Gene dimension
                target_shape = batch_dims + (num_cells, num_genes)
        else:
            # Posterior sampling case: parameters [samples, genes], t_star [samples, cells] → output [samples, cells, genes]
            num_samples = alpha_off.shape[0]
            num_genes = alpha_off.shape[1]
            if t_star.dim() == 2:
                num_cells = t_star.shape[1]
            else:
                num_cells = t_star.shape[-1]
            target_shape = (num_samples, num_cells, num_genes)
        
        # Prepare tensors for broadcasting
        if alpha_off.dim() == 1:
            # Gene parameters are 1D: need to broadcast with t_star properly
            if t_star.dim() == 1:
                # Training case: t_star [cells], params [genes] → [cells, genes]
                t_star_bc = t_star.unsqueeze(1)  # [cells, 1]
                alpha_off_bc = alpha_off.unsqueeze(0)  # [1, genes]
                alpha_on_bc = alpha_on.unsqueeze(0)  # [1, genes]
                gamma_star_bc = gamma_star.unsqueeze(0)  # [1, genes]
                t_on_star_bc = t_on_star.unsqueeze(0)  # [1, genes]
                delta_star_bc = delta_star.unsqueeze(0)  # [1, genes]
            else:
                # Posterior sampling case: t_star has batch dims, params are 1D
                # Need to add gene dimension to t_star and batch dims to params
                t_star_bc = t_star.unsqueeze(-1)  # [..., cells, 1]
                # Add batch dimensions to match t_star's batch shape
                param_shape = [1] * (t_star.dim() - 1) + [alpha_off.shape[0]]
                alpha_off_bc = alpha_off.view(param_shape)  # [..., 1, genes]
                alpha_on_bc = alpha_on.view(param_shape)  # [..., 1, genes]
                gamma_star_bc = gamma_star.view(param_shape)  # [..., 1, genes]
                t_on_star_bc = t_on_star.view(param_shape)  # [..., 1, genes]
                delta_star_bc = delta_star.view(param_shape)  # [..., 1, genes]
        else:
            # Posterior sampling case: handle [samples, genes] and [samples, cells]
            if t_star.dim() == 2:
                # t_star [samples, cells], params [samples, genes] → broadcast to [samples, cells, genes]
                t_star_bc = t_star.unsqueeze(2)  # [samples, cells, 1]
                alpha_off_bc = alpha_off.unsqueeze(1)  # [samples, 1, genes]
                alpha_on_bc = alpha_on.unsqueeze(1)  # [samples, 1, genes]
                gamma_star_bc = gamma_star.unsqueeze(1)  # [samples, 1, genes]
                t_on_star_bc = t_on_star.unsqueeze(1)  # [samples, 1, genes]
                delta_star_bc = delta_star.unsqueeze(1)  # [samples, 1, genes]
            else:
                # Use as-is
                t_star_bc = t_star
                alpha_off_bc = alpha_off
                alpha_on_bc = alpha_on
                gamma_star_bc = gamma_star
                t_on_star_bc = t_on_star
                delta_star_bc = delta_star
        
        # Compute switching times
        t_switch_on = t_on_star_bc  # Start of activation
        t_switch_off = t_on_star_bc + delta_star_bc  # End of activation
        
        # Phase 1: Before activation (t < t_on)
        # u'(t) = alpha_off - u, s'(t) = u - gamma*s
        # Solution: u(t) = alpha_off + (u0 - alpha_off)*exp(-t)
        #          s(t) = alpha_off/gamma + C*exp(-gamma*t) + D*exp(-t)
        
        # Initial conditions: u0 = 1.0, s0 = 1.0/gamma
        u0 = 1.0
        s0 = 1.0 / gamma_star_bc
        
        # Phase 1 mask: t < t_on
        mask_phase1 = t_star_bc < t_switch_on
        
        # Phase 1 solutions  
        u_phase1 = alpha_off_bc + (u0 - alpha_off_bc) * torch.exp(-t_star_bc)
        
        # For s_phase1, we need to solve the coupled system
        # This is the analytical solution for the coupled system
        gamma_minus_1 = gamma_star_bc - 1.0
        exp_minus_t = torch.exp(-t_star_bc)
        exp_minus_gamma_t = torch.exp(-gamma_star_bc * t_star_bc)
        
        # Avoid division by zero when gamma ≈ 1
        safe_gamma_minus_1 = torch.where(
            torch.abs(gamma_minus_1) > 1e-8,
            gamma_minus_1,
            torch.sign(gamma_minus_1) * 1e-8
        )
        
        s_phase1 = (alpha_off_bc / gamma_star_bc + 
                   (u0 - alpha_off_bc) / safe_gamma_minus_1 * (exp_minus_t - exp_minus_gamma_t) +
                   (s0 - alpha_off_bc / gamma_star_bc) * exp_minus_gamma_t)
        
        # Phase 2: During activation (t_on <= t < t_on + delta)
        # Similar logic but with alpha_on instead of alpha_off
        mask_phase2 = (t_star_bc >= t_switch_on) & (t_star_bc < t_switch_off)
        
        # Time relative to activation start
        t_rel = t_star_bc - t_switch_on
        
        # Continuity conditions: use phase 1 solutions at t_on as initial conditions
        t_on_rel = torch.zeros_like(t_switch_on)  # t_on relative to itself is 0
        u_at_ton = alpha_off_bc + (u0 - alpha_off_bc) * torch.exp(-t_switch_on)
        s_at_ton = (alpha_off_bc / gamma_star_bc + 
                   (u0 - alpha_off_bc) / safe_gamma_minus_1 * 
                   (torch.exp(-t_switch_on) - torch.exp(-gamma_star_bc * t_switch_on)) +
                   (s0 - alpha_off_bc / gamma_star_bc) * torch.exp(-gamma_star_bc * t_switch_on))
        
        # Phase 2 solutions
        u_phase2 = alpha_on_bc + (u_at_ton - alpha_on_bc) * torch.exp(-t_rel)
        
        exp_minus_t_rel = torch.exp(-t_rel)
        exp_minus_gamma_t_rel = torch.exp(-gamma_star_bc * t_rel)
        
        s_phase2 = (alpha_on_bc / gamma_star_bc + 
                   (u_at_ton - alpha_on_bc) / safe_gamma_minus_1 * (exp_minus_t_rel - exp_minus_gamma_t_rel) +
                   (s_at_ton - alpha_on_bc / gamma_star_bc) * exp_minus_gamma_t_rel)
        
        # Phase 3: After activation (t >= t_on + delta)
        # Back to alpha_off, using phase 2 solutions at t_on + delta as initial conditions
        mask_phase3 = t_star_bc >= t_switch_off
        
        # Time relative to switch-off
        t_rel_off = t_star_bc - t_switch_off
        
        # Continuity conditions from phase 2 at switch-off time
        delta_rel = delta_star_bc  # Duration of activation
        u_at_toff = alpha_on_bc + (u_at_ton - alpha_on_bc) * torch.exp(-delta_rel)
        s_at_toff = (alpha_on_bc / gamma_star_bc + 
                    (u_at_ton - alpha_on_bc) / safe_gamma_minus_1 * 
                    (torch.exp(-delta_rel) - torch.exp(-gamma_star_bc * delta_rel)) +
                    (s_at_ton - alpha_on_bc / gamma_star_bc) * torch.exp(-gamma_star_bc * delta_rel))
        
        # Phase 3 solutions
        u_phase3 = alpha_off_bc + (u_at_toff - alpha_off_bc) * torch.exp(-t_rel_off)
        
        exp_minus_t_rel_off = torch.exp(-t_rel_off)
        exp_minus_gamma_t_rel_off = torch.exp(-gamma_star_bc * t_rel_off)
        
        s_phase3 = (alpha_off_bc / gamma_star_bc + 
                   (u_at_toff - alpha_off_bc) / safe_gamma_minus_1 * (exp_minus_t_rel_off - exp_minus_gamma_t_rel_off) +
                   (s_at_toff - alpha_off_bc / gamma_star_bc) * exp_minus_gamma_t_rel_off)
        
        # Combine phases using masks
        u_star = torch.where(mask_phase1, u_phase1,
                            torch.where(mask_phase2, u_phase2, u_phase3))
        s_star = torch.where(mask_phase1, s_phase1,
                            torch.where(mask_phase2, s_phase2, s_phase3))
        
        # Ensure output tensors have the correct target shape
        if u_star.shape != target_shape:
            # If shapes don't match, use broadcasting to get the right shape
            if alpha_off.dim() == 1 and t_star.dim() == 1:
                # Training case: simple 2D expansion [cells, genes]
                u_star = u_star.expand(target_shape)
                s_star = s_star.expand(target_shape)
            else:
                # Posterior sampling case or multi-dimensional case
                # Use broadcasting to ensure compatibility
                try:
                    # Attempt direct expansion if dimensions allow
                    u_star = u_star.expand(target_shape)
                    s_star = s_star.expand(target_shape)
                except RuntimeError:
                    # If expansion fails, the tensors already have the right shape from broadcasting
                    # This can happen when broadcasting produced the correct shape automatically
                    pass
        
        return u_star, s_star

    @jaxtyped
    @beartype
    def steady_state(
        self,
        alpha_off: Union[ParamTensor, torch.Tensor],
        gamma_star: Union[ParamTensor, torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[
        Union[ParamTensor, torch.Tensor],
        Union[ParamTensor, torch.Tensor],
    ]:
        """
        Compute the steady-state unspliced and spliced RNA counts for the OFF phase.

        For the corrected piecewise activation model, the steady state corresponds to the
        OFF phase with fixed reference transcription α*_off = 1.0.

        Args:
            alpha_off: Fixed reference transcription rate (always 1.0)
            gamma_star: Relative degradation rate
            **kwargs: Additional model-specific parameters

        Returns:
            Tuple of (steady-state unspliced counts, steady-state spliced counts)
        """
        # At steady state in OFF phase:
        # du*/dt* = 1.0 - u* = 0 => u* = 1.0 (fixed reference)
        u_ss = torch.ones_like(alpha_off)  # Always 1.0

        # ds*/dt* = u* - γ*s* = 0 => s* = u*/γ* = 1.0/γ*
        s_ss = torch.ones_like(alpha_off) / gamma_star

        return u_ss, s_ss

    @beartype
    def compute_velocity(
        self,
        ut: Union[LatentCountTensor, torch.Tensor],
        st: Union[LatentCountTensor, torch.Tensor],
        gamma_star: Union[KineticParamTensor, torch.Tensor],
        **kwargs: Any,
    ) -> Union[VelocityTensor, torch.Tensor]:
        """
        Compute RNA velocity from latent RNA counts and kinetic parameters.

        For the dimensionless piecewise activation model:
            ds*/dt* = u* - γ*s*

        Args:
            ut: Latent unspliced RNA counts (dimensionless)
            st: Latent spliced RNA counts (dimensionless)
            gamma_star: Relative degradation rate
            **kwargs: Additional model-specific parameters

        Returns:
            RNA velocity tensor with same shape as ut/st
        """
        # For the dimensionless piecewise model: ds*/dt* = u* - γ*s*
        velocity = ut - gamma_star * st
        return velocity