"""
Protocol-First likelihood model implementations for PyroVelocity's modular architecture.

This module contains likelihood model implementations that directly implement the
LikelihoodModel Protocol. These implementations follow the Protocol-First approach,
which embraces composition over inheritance and allows for more flexible component composition.

This module has been simplified to include only the essential components needed for
validation against the legacy implementation:
- PoissonLikelihoodModel: Poisson likelihood model for observed counts
- LegacyLikelihoodModel: Legacy likelihood model for observed counts that exactly matches the legacy implementation
"""

from typing import Any, Dict, List, Optional

import pyro
import torch
from anndata import AnnData
from beartype import beartype

from pyrovelocity.models.modular.interfaces import LikelihoodModel
from pyrovelocity.models.modular.registry import LikelihoodModelRegistry


def validate_context(component_name: str, context: Dict[str, Any], required_keys: List[str] = None, tensor_keys: List[str] = None) -> bool:
    """Validate that context contains required keys."""
    if required_keys:
        return all(key in context for key in required_keys)
    return True


class PiecewiseActivationPoissonLikelihoodModel:
    """Pure Poisson likelihood model for piecewise activation parameter recovery validation.

    This model implements the pure mathematical specification from the parameter recovery
    validation documentation without legacy preprocessing steps. It follows the mathematical
    description:

    u_{ij} ∼ Poisson(λ_j · U_{0i} · u*_{ij})
    s_{ij} ∼ Poisson(λ_j · U_{0i} · s*_{ij})

    where:
    - λ_j is the cell-specific effective capture efficiency
    - U_{0i} is the gene-specific characteristic concentration scale
    - u*_{ij}, s*_{ij} are the latent dimensionless RNA concentrations

    This implementation directly implements the LikelihoodModel Protocol.
    """

    def __init__(self, name: str = "piecewise_activation_poisson_likelihood"):
        """
        Initialize the PiecewiseActivationPoissonLikelihoodModel.

        Args:
            name: A unique name for this component instance.
        """
        self.name = name

    @beartype
    def forward(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Define likelihood distributions for observed data given expected values.

        This method implements the pure mathematical specification without preprocessing.
        It expects the context to contain:
        - u_obs, s_obs: observed counts
        - ut, st: latent dimensionless RNA concentrations (u*_{ij}, s*_{ij})
        - lambda_j: cell-specific effective capture efficiency
        - U_0i: gene-specific characteristic concentration scale

        Args:
            context: Dictionary containing model context

        Returns:
            Updated context dictionary with likelihood information
        """
        # Validate context
        validation_result = validate_context(
            self.__class__.__name__,
            context,
            required_keys=["u_obs", "s_obs", "ut", "st"],
            tensor_keys=["u_obs", "s_obs", "ut", "st"],
        )

        if validation_result:
            # Extract required values from context
            u_obs = context["u_obs"]
            s_obs = context["s_obs"]
            ut = context["ut"]  # u*_{ij} - latent dimensionless unspliced
            st = context["st"]  # s*_{ij} - latent dimensionless spliced

            # Extract scaling parameters
            lambda_j = context.get("lambda_j")  # Cell-specific capture efficiency
            U_0i = context.get("U_0i")  # Gene-specific concentration scale

            # Calculate rate parameters according to mathematical specification
            # u_{ij} ∼ Poisson(λ_j · U_{0i} · u*_{ij})
            # s_{ij} ∼ Poisson(λ_j · U_{0i} · s*_{ij})

            if lambda_j is not None and U_0i is not None:
                # Handle the case where ut/st have shape [N, 1, G] from dynamics model
                if ut.dim() == 3 and ut.shape[1] == 1:
                    # Remove the middle dimension: [N, 1, G] -> [N, G]
                    ut = ut.squeeze(1)
                    st = st.squeeze(1)

                # Now ut, st should be [N, G]
                # lambda_j should be [N], U_0i should be [G]

                # Reshape for broadcasting: lambda_j [N] -> [N, 1], U_0i [G] -> [1, G]
                lambda_j_expanded = lambda_j.unsqueeze(-1)  # [N, 1]
                U_0i_expanded = U_0i.unsqueeze(0)  # [1, G]

                # Compute rates with proper broadcasting: [N, 1] * [1, G] * [N, G] = [N, G]
                u_rate = lambda_j_expanded * U_0i_expanded * ut
                s_rate = lambda_j_expanded * U_0i_expanded * st
            else:
                # Fallback: use latent concentrations directly as rates
                # Handle the case where ut/st have shape [N, 1, G] from dynamics model
                if ut.dim() == 3 and ut.shape[1] == 1:
                    # Remove the middle dimension: [N, 1, G] -> [N, G]
                    ut = ut.squeeze(1)
                    st = st.squeeze(1)

                u_rate = ut
                s_rate = st

            # Ensure all rate values are positive (required for Poisson distribution)
            epsilon = 1e-6
            u_rate = torch.maximum(u_rate, torch.tensor(epsilon))
            s_rate = torch.maximum(s_rate, torch.tensor(epsilon))

            # Ensure observations are integers for Poisson distribution
            u_obs_int = u_obs.round().long()
            s_obs_int = s_obs.round().long()

            # Handle different tensor dimensions for training vs posterior sampling
            if u_rate.dim() == 2:  # [N, G] - standard training case
                n_cells, n_genes = u_rate.shape
            elif u_rate.dim() == 4:  # [1, 1, N, G] - SVI posterior sampling case
                # Extract the actual dimensions and reshape
                n_cells, n_genes = u_rate.shape[-2], u_rate.shape[-1]
                u_rate = u_rate.squeeze(0).squeeze(0)  # [N, G]
                s_rate = s_rate.squeeze(0).squeeze(0)  # [N, G]
                u_obs_int = u_obs_int.squeeze(0).squeeze(0) if u_obs_int.dim() > 2 else u_obs_int
                s_obs_int = s_obs_int.squeeze(0).squeeze(0) if s_obs_int.dim() > 2 else s_obs_int
            elif u_rate.dim() == 3:  # [1, N, G] or [B, N, G] - other posterior sampling cases
                if u_rate.shape[0] == 1:
                    # [1, N, G] case - squeeze the batch dimension
                    n_cells, n_genes = u_rate.shape[1], u_rate.shape[2]
                    u_rate = u_rate.squeeze(0)  # [N, G]
                    s_rate = s_rate.squeeze(0)  # [N, G]
                    u_obs_int = u_obs_int.squeeze(0) if u_obs_int.dim() > 2 else u_obs_int
                    s_obs_int = s_obs_int.squeeze(0) if s_obs_int.dim() > 2 else s_obs_int
                else:
                    raise ValueError(f"PiecewiseActivationPoissonLikelihoodModel cannot handle batch dimension > 1, got {u_rate.shape}")
            else:
                raise ValueError(f"PiecewiseActivationPoissonLikelihoodModel expects 2D, 3D, or 4D tensors, got {u_rate.shape}")

            # Create Poisson distributions
            u_dist = pyro.distributions.Poisson(rate=u_rate)
            s_dist = pyro.distributions.Poisson(rate=s_rate)

            # Use standard plates without batch dimension
            with pyro.plate("cells_likelihood", n_cells, dim=-2):
                with pyro.plate("genes_likelihood", n_genes, dim=-1):
                    # Observe data
                    pyro.sample("u_obs", u_dist, obs=u_obs_int)
                    pyro.sample("s_obs", s_dist, obs=s_obs_int)

            # Compute and store log probabilities for debugging and validation
            log_prob_u = u_dist.log_prob(u_obs_int)
            log_prob_s = s_dist.log_prob(s_obs_int)

            # Add distributions and log probabilities to context
            context["u_dist"] = u_dist
            context["s_dist"] = s_dist
            context["log_prob_u"] = log_prob_u
            context["log_prob_s"] = log_prob_s

            return context
        else:
            # If validation failed, raise an error
            raise ValueError(f"Error in piecewise activation likelihood model forward pass: validation failed")

    def __call__(
        self,
        adata: Optional[AnnData] = None,
        cell_state: Optional[torch.Tensor] = None,
        gene_offset: Optional[torch.Tensor] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
        """Generate likelihood distributions for observed data.

        Args:
            adata: AnnData object containing gene expression data
            cell_state: Latent cell state vectors
            gene_offset: Optional gene-specific offset factors
            time_info: Optional time-related information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of likelihood distributions
        """
        if adata is None and cell_state is None:
            raise ValueError("Either adata or cell_state must be provided")

        if adata is not None:
            return self._generate_distributions_from_adata(
                adata, gene_offset, time_info, **kwargs
            )
        else:
            return self._generate_distributions_from_cell_state(
                cell_state, gene_offset, time_info, **kwargs
            )

    def _generate_distributions_from_adata(
        self,
        adata: AnnData,
        gene_offset: Optional[torch.Tensor] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
        """Generate Poisson likelihood distributions from AnnData.

        Args:
            adata: AnnData object containing gene expression data
            gene_offset: Optional gene-specific offset factors
            time_info: Optional time-related information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of likelihood distributions
        """
        # Extract counts from AnnData
        if "X" not in adata.layers:
            raise ValueError("AnnData object must have 'X' in layers")

        # Convert to torch tensor
        layer_data = adata.layers["X"]
        if hasattr(layer_data, 'toarray'):
            # For sparse matrices
            counts = torch.tensor(layer_data.toarray(), dtype=torch.float32)
        else:
            # For numpy arrays
            counts = torch.tensor(layer_data, dtype=torch.float32)

        # Use mean expression as rate parameter
        rate = torch.mean(counts, dim=0)

        # Apply gene offset if provided
        if gene_offset is not None:
            rate = rate * gene_offset

        # Create Poisson likelihood distribution
        return {"obs_counts": pyro.distributions.Poisson(rate=rate)}

    def _generate_distributions_from_cell_state(
        self,
        cell_state: torch.Tensor,
        gene_offset: Optional[torch.Tensor] = None,
        time_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, pyro.distributions.Distribution]:
        """Generate Poisson likelihood distributions from cell state.

        Args:
            cell_state: Latent cell state vectors
            gene_offset: Optional gene-specific offset factors
            time_info: Optional time-related information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary of likelihood distributions
        """
        # Get projection matrix from kwargs or create a random one
        projection = kwargs.get("projection")
        if projection is None:
            # Create a random projection matrix
            n_latent = cell_state.shape[1]
            n_genes = kwargs.get("n_genes", 100)
            projection = torch.randn(n_latent, n_genes)

        # Project cell_state to gene space
        projected_state = torch.matmul(cell_state, projection)

        # Apply gene offset if provided
        if gene_offset is not None:
            projected_state = projected_state * gene_offset

        # Calculate rate parameter (ensure positive)
        rate = torch.exp(projected_state)

        # Create Poisson likelihood distribution
        return {"obs_counts": pyro.distributions.Poisson(rate=rate)}

    def log_prob(
        self,
        observations: torch.Tensor,
        predictions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate log probability of observations under Poisson distribution.

        Args:
            observations: Observed gene expression counts
            predictions: Predicted mean expression levels
            scale_factors: Optional scaling factors for observations

        Returns:
            Log probability of observations for each cell
        """
        # Apply scale factors if provided
        if scale_factors is not None:
            rate = predictions * scale_factors
        else:
            rate = predictions

        # Create Poisson distribution and calculate log probability
        distribution = torch.distributions.Poisson(rate=rate)
        log_probs = distribution.log_prob(observations)

        # Sum log probabilities across genes for each cell
        return torch.sum(log_probs, dim=-1)

    def sample(
        self,
        predictions: torch.Tensor,
        scale_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample observations from Poisson distribution.

        Args:
            predictions: Predicted mean expression levels
            scale_factors: Optional scaling factors for observations

        Returns:
            Sampled observations
        """
        # Apply scale factors if provided
        if scale_factors is not None:
            rate = predictions * scale_factors
        else:
            rate = predictions

        # Create Poisson distribution and sample
        distribution = torch.distributions.Poisson(rate=rate)
        # Set a fixed seed for reproducibility
        torch.manual_seed(0)
        return distribution.sample()
