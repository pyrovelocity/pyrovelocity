"""
PyroVelocity model implementation that composes component models.

This module provides the core model class for PyroVelocity's modular architecture,
implementing a composable approach where the full model is built from specialized
component models (dynamics, priors, likelihoods, observations, guides).

The modular architecture follows a component-based design pattern, where each
component is responsible for a specific aspect of the model. This approach enables:

1. Flexibility: Components can be swapped out independently
2. Extensibility: New components can be added without modifying existing code
3. Testability: Components can be tested in isolation
4. Reusability: Components can be reused across different models

The main class in this module is `PyroVelocityModel`, which composes the different
components into a cohesive probabilistic model. The model uses functional composition
for the forward method, enabling railway-oriented programming patterns.

Examples:
    >>> import torch
    >>> import pyro
    >>> from pyrovelocity.models.modular.factory import create_legacy_model1
    >>> from pyrovelocity.models.modular.model import PyroVelocityModel
    >>>
    >>> # Create a standard model with default components
    >>> model = create_legacy_model1()
    >>>
    >>> # Generate synthetic data
    >>> u_obs = torch.randn(10, 5)  # 10 cells, 5 genes
    >>> s_obs = torch.randn(10, 5)  # 10 cells, 5 genes
    >>>
    >>> # Run the model forward
    >>> results = model.forward(u_obs=u_obs, s_obs=s_obs)
    >>>
    >>> # Access model parameters
    >>> alpha = results.get("alpha")
    >>> beta = results.get("beta")
    >>> gamma = results.get("gamma")
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import anndata
import numpy as np
import pyro
import torch
from anndata import AnnData
from beartype import beartype
from jaxtyping import Float

from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
    LegacyAutoGuideFactory,
)
from pyrovelocity.models.modular.components.likelihoods import (
    LegacyLikelihoodModel,
    PiecewiseActivationPoissonLikelihoodModel,
)
from pyrovelocity.models.modular.data.anndata import (
    extract_layers,
    get_library_size,
    prepare_anndata,
    store_results,
)
from pyrovelocity.models.modular.inference.config import InferenceConfig
from pyrovelocity.models.modular.inference.unified import InferenceState
from pyrovelocity.models.modular.interfaces import (
    DynamicsModel as DynamicsModelProtocol,
)
from pyrovelocity.models.modular.interfaces import (
    InferenceGuide as GuideModelProtocol,
)
from pyrovelocity.models.modular.interfaces import (
    LikelihoodModel as LikelihoodModelProtocol,
)
from pyrovelocity.models.modular.interfaces import (
    PriorModel as PriorModelProtocol,
)


@dataclass(frozen=True)
class ModelState:
    """
    Immutable state container for the PyroVelocityModel with type-safe inference state.

    This dataclass holds the state of all component models and ensures immutability
    through the frozen=True parameter. The inference_state field provides direct
    type-safe access to inference results.

    Attributes:
        dynamics_state: State of the dynamics model component
        prior_state: State of the prior model component
        likelihood_state: State of the likelihood model component (includes data preprocessing)
        guide_state: State of the inference guide component
        inference_state: Type-safe inference state from training
        inference_config: Type-safe inference configuration used for training
        metadata: Optional dictionary for additional metadata (deprecated, use specific fields)
    """

    dynamics_state: Dict[str, Any] = field(default_factory=dict)
    prior_state: Dict[str, Any] = field(default_factory=dict)
    likelihood_state: Dict[str, Any] = field(default_factory=dict)
    guide_state: Dict[str, Any] = field(default_factory=dict)
    inference_state: Optional["InferenceState"] = field(default=None)
    inference_config: Optional["InferenceConfig"] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate state after initialization."""
        # Since the class is frozen, we can't modify attributes directly
        # This method is for validation only
        pass


class PyroVelocityModel:
    """
    Composable PyroVelocity model that integrates specialized component models.

    This class implements the core model for PyroVelocity's modular architecture,
    composing specialized component models (dynamics, priors, likelihoods,
    observations, guides) into a cohesive probabilistic model. It uses functional
    composition for the forward method, enabling railway-oriented programming
    patterns.

    The model follows a clear separation of concerns:

    - Dynamics model: Defines the velocity vector field
    - Prior model: Specifies priors for model parameters
    - Likelihood model: Defines the observation likelihood
    - Observation model: Handles data preprocessing and transformation
    - Guide model: Implements the inference guide for posterior approximation

    The PyroVelocityModel provides methods for:

    1. Running the model forward to compute expected RNA counts
    2. Training the model using SVI (Stochastic Variational Inference)
    3. Generating posterior samples for uncertainty quantification
    4. Computing RNA velocity from posterior samples
    5. Storing results in AnnData objects for visualization and analysis

    Attributes:
        dynamics_model: Component handling velocity vector field modeling
        prior_model: Component handling prior distributions
        likelihood_model: Component handling likelihood distributions and data preprocessing
        guide_model: Component handling inference guide
        state: Immutable state container for all model components

    Examples:
        >>> # Create a model with standard components
        >>> from pyrovelocity.models.modular.factory import create_legacy_model1
        >>> model = create_legacy_model1()
        >>>
        >>> # Create synthetic AnnData for testing
        >>> import anndata as ad
        >>> import numpy as np
        >>> import torch
        >>> import pandas as pd
        >>> import os
        >>> # Use pytest tmp_path fixture for temporary directory
        >>> tmp = getfixture("tmp_path")
        >>> tmp_dir = str(tmp)
        >>>
        >>> # Create synthetic data
        >>> n_cells, n_genes = 10, 5
        >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
        >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
        >>>
        >>> # Create AnnData object
        >>> adata = ad.AnnData(X=s_data)
        >>> adata.layers["spliced"] = s_data
        >>> adata.layers["unspliced"] = u_data
        >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        >>>
        >>> # Set up AnnData for PyroVelocity
        >>> adata = PyroVelocityModel.setup_anndata(adata)
        >>>
        >>> # Train the model with minimal epochs for testing
        >>> model.train(adata=adata, max_epochs=2)
        >>>
        >>> # Generate posterior samples
        >>> posterior_samples = model.generate_posterior_samples(
        ...     adata=adata, num_samples=2
        ... )
        >>>
        >>> # Store results in AnnData
        >>> adata = model.store_results_in_anndata(
        ...     adata=adata, posterior_samples=posterior_samples
        ... )
        >>>
        >>> # Clean up temporary directory
        >>> import shutil
        >>> if os.path.exists(tmp_dir):
        ...     shutil.rmtree(tmp_dir)
    """

    @beartype
    def __init__(
        self,
        dynamics_model: DynamicsModelProtocol,
        prior_model: PriorModelProtocol,
        likelihood_model: LikelihoodModelProtocol,
        guide_model: GuideModelProtocol,
        state: Optional[ModelState] = None,
    ):
        """
        Initialize the PyroVelocityModel with component models.

        Args:
            dynamics_model: Model component for velocity vector field
            prior_model: Model component for prior distributions
            likelihood_model: Model component for likelihood distributions and data preprocessing
            guide_model: Model component for inference guide
            state: Optional pre-initialized model state
        """
        # Store the component models
        self.dynamics_model = dynamics_model
        self.prior_model = prior_model
        self.likelihood_model = likelihood_model
        self.guide_model = guide_model

        # Initialize state if not provided
        if state is None:
            self.state = ModelState(
                dynamics_state=getattr(self.dynamics_model, "state", {}),
                prior_state=getattr(self.prior_model, "state", {}),
                likelihood_state=getattr(self.likelihood_model, "state", {}),
                guide_state=getattr(self.guide_model, "state", {}),
            )
        else:
            self.state = state

    def __call__(self, *args, **kwargs):
        """Make the model callable for Pyro's autoguide.

        This method delegates to the forward method, making the model compatible
        with Pyro's autoguide system.

        Args:
            *args: Positional arguments passed to forward
            **kwargs: Keyword arguments passed to forward

        Returns:
            Result from the forward method
        """
        return self.forward(*args, **kwargs)

    def _build_context(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        x: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
        time_points: Optional[torch.Tensor] = None,
        cell_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build context dictionary for model components.

        This helper method consolidates the context building logic shared between
        forward() and guide() methods, reducing code duplication and ensuring
        consistent context construction.

        Args:
            u_obs: Observed unspliced RNA counts
            s_obs: Observed spliced RNA counts  
            u_log_library: Log library size for unspliced counts
            s_log_library: Log library size for spliced counts
            x: Optional input data tensor
            time_points: Optional time points for dynamics model
            cell_state: Optional dictionary with cell state information
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing all context information for component models
        """
        # Initialize the context dictionary
        context = {
            "cell_state": cell_state or {},
            **kwargs,  # Include remaining kwargs
        }

        # Add optional parameters to context if provided
        if u_obs is not None:
            context["u_obs"] = u_obs
        if s_obs is not None:
            context["s_obs"] = s_obs
        if u_log_library is not None:
            context["u_log_library"] = u_log_library
        if s_log_library is not None:
            context["s_log_library"] = s_log_library
        if x is not None:
            context["x"] = x
        if time_points is not None:
            context["time_points"] = time_points

        return context

    @beartype
    def forward(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        x: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
        time_points: Optional[torch.Tensor] = None,
        cell_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model using functional composition.

        This method implements the forward pass through all model components
        using functional composition, following a railway-oriented programming
        approach where each component processes the data and passes it to the
        next component.

        The forward pass follows this sequence:
        1. Observation model: Processes input data and prepares it for the dynamics model
        2. Dynamics model: Computes expected RNA counts based on dynamics equations
        3. Prior model: Applies prior distributions to model parameters
        4. Likelihood model: Computes likelihood of observed data given expected counts

        The method can be called in two ways:
        - With x and time_points: For general input data and time points
        - With u_obs and s_obs: For RNA velocity-specific input data

        Args:
            x: Optional input data tensor of shape [batch_size, n_features]
            time_points: Optional time points for the dynamics model of shape [n_time_points]
            u_obs: Observed unspliced RNA counts of shape [batch_size, n_genes]
            s_obs: Observed spliced RNA counts of shape [batch_size, n_genes]
            cell_state: Optional dictionary with cell state information
            **kwargs: Additional keyword arguments passed to component models

        Returns:
            Dictionary containing model outputs and intermediate results, including:
            - alpha: Transcription rate parameter
            - beta: Splicing rate parameter
            - gamma: Degradation rate parameter
            - u_expected: Expected unspliced RNA counts
            - s_expected: Expected spliced RNA counts
            - tau: Latent time (if enabled)
            - Other model-specific outputs

        Examples:
            >>> # Run forward pass with RNA count data
            >>> import torch
            >>> import os
            >>> from pyrovelocity.models.modular.factory import create_legacy_model1
            >>>
            >>> # Use pytest tmp_path fixture for temporary directory
            >>> tmp = getfixture("tmp_path")
            >>> tmp_dir = str(tmp)
            >>>
            >>> # Create model and synthetic data
            >>> model = create_legacy_model1()
            >>> u_obs = torch.randn(10, 5)  # 10 cells, 5 genes
            >>> s_obs = torch.randn(10, 5)  # 10 cells, 5 genes
            >>>
            >>> # Run forward pass
            >>> results = model.forward(u_obs=u_obs, s_obs=s_obs)
            >>>
            >>> # Access results
            >>> alpha = results["alpha"]
            >>> beta = results["beta"]
            >>> gamma = results["gamma"]
            >>> u_expected = results["u_expected"]
            >>> s_expected = results["s_expected"]
            >>>
            >>> # Clean up temporary directory
            >>> import shutil
            >>> if os.path.exists(tmp_dir):
            ...     shutil.rmtree(tmp_dir)
        """


        # Build context dictionary for component models
        context = self._build_context(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
            x=x,
            time_points=time_points,
            cell_state=cell_state,
            **kwargs
        )



        # Apply prior distributions first to provide parameters for the dynamics model
        prior_context = self.prior_model.forward(context)

        # Process through the dynamics model with the parameters from the prior model
        dynamics_context = self.dynamics_model.forward(prior_context)

        # Apply likelihood model (which now handles data preprocessing)
        likelihood_context = self.likelihood_model.forward(dynamics_context)

        # Return the final context with all model outputs
        return likelihood_context

    @beartype
    def guide(
        self,
        u_obs: Optional[torch.Tensor] = None,
        s_obs: Optional[torch.Tensor] = None,
        u_log_library: Optional[torch.Tensor] = None,
        s_log_library: Optional[torch.Tensor] = None,
        x: Optional[Union[torch.Tensor, Dict[str, Any]]] = None,
        time_points: Optional[torch.Tensor] = None,
        cell_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Inference guide for the model used in variational inference.

        This method delegates to the guide model component to implement
        the inference guide for posterior approximation. The guide defines
        the variational distribution that approximates the true posterior
        distribution over model parameters.

        During training, this guide is used by Pyro's SVI (Stochastic Variational
        Inference) to optimize the variational parameters. After training, the guide
        can be used to sample from the approximate posterior distribution.

        The method handles two cases:
        1. When data is provided (x, u_obs, or s_obs): Processes data through the
           observation model before passing to the guide
        2. When no data is provided: Assumes posterior sampling mode and passes
           directly to the guide

        Args:
            x: Optional input data tensor of shape [batch_size, n_features]
            time_points: Optional time points for the dynamics model of shape [n_time_points]
            u_obs: Observed unspliced RNA counts of shape [batch_size, n_genes]
            s_obs: Observed spliced RNA counts of shape [batch_size, n_genes]
            cell_state: Optional dictionary with cell state information
            **kwargs: Additional keyword arguments passed to the guide model

        Returns:
            Dictionary containing guide outputs, typically including variational parameters

        Examples:
            >>> # Use guide for posterior sampling
            >>> import torch
            >>> from pyrovelocity.models.modular.factory import create_legacy_model1
            >>>
            >>> # Create model and synthetic data
            >>> model = create_legacy_model1()
            >>> u_obs = torch.randn(10, 5)  # 10 cells, 5 genes
            >>> s_obs = torch.randn(10, 5)  # 10 cells, 5 genes
            >>>
            >>> # Train model (simplified)
            >>> model.guide_model.create_guide(model.forward)
            >>>
            >>> # Use guide for inference
            >>> guide_results = model.guide(u_obs=u_obs, s_obs=s_obs)
        """
        # Build context dictionary for guide model
        context = self._build_context(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
            x=x,
            time_points=time_points,
            cell_state=cell_state,
            **kwargs
        )

        # Get the guide function
        guide_fn = self.guide_model.get_guide()

        # The AutoGuide expects the same signature as the model
        # So we need to call it with the same arguments
        return guide_fn(
            u_obs=u_obs,
            s_obs=s_obs,
            u_log_library=u_log_library,
            s_log_library=s_log_library,
            x=x,
            time_points=time_points,
            cell_state=cell_state,
            **kwargs
        )

    @property
    def name(self) -> str:
        """Return the model name."""
        return "PyroVelocityModel"

    def get_state(self) -> ModelState:
        """Return the current model state."""
        return self.state

    def __repr__(self) -> str:
        """
        Return a string representation of the PyroVelocityModel.

        This method provides a nicely formatted representation of the model,
        including information about each component and its configuration.

        Returns:
            Formatted string representation of the model
        """
        # Get component names and descriptions
        dynamics_name = getattr(self.dynamics_model, "name", self.dynamics_model.__class__.__name__)
        dynamics_desc = getattr(self.dynamics_model, "description", "")

        prior_name = getattr(self.prior_model, "name", self.prior_model.__class__.__name__)
        prior_desc = getattr(self.prior_model, "description", "")

        likelihood_name = getattr(self.likelihood_model, "name", self.likelihood_model.__class__.__name__)
        likelihood_desc = getattr(self.likelihood_model, "description", "")

        guide_name = getattr(self.guide_model, "name", self.guide_model.__class__.__name__)
        guide_desc = getattr(self.guide_model, "description", "")

        # Get component configurations
        dynamics_config = getattr(self.dynamics_model, "config", {})
        prior_config = getattr(self.prior_model, "config", {})
        likelihood_config = getattr(self.likelihood_model, "config", {})
        guide_config = getattr(self.guide_model, "config", {})

        # Format component configurations as strings
        def format_config(config):
            if not config:
                return ""
            return ", ".join(f"{k}={v}" for k, v in config.items())

        dynamics_config_str = format_config(dynamics_config)
        prior_config_str = format_config(prior_config)
        likelihood_config_str = format_config(likelihood_config)
        guide_config_str = format_config(guide_config)

        # Check if model has been trained using the new type-safe inference state field
        trained_status = "Trained" if self.state.inference_state is not None else "Untrained"

        # Build the representation string
        repr_str = [
            f"PyroVelocityModel ({trained_status})",
            "─" * 50,
            "Components:",
            f"  • Dynamics:    {dynamics_name} {f'({dynamics_config_str})' if dynamics_config_str else ''}",
            f"                 {dynamics_desc}",
            f"  • Prior:       {prior_name} {f'({prior_config_str})' if prior_config_str else ''}",
            f"                 {prior_desc}",
            f"  • Likelihood:  {likelihood_name} {f'({likelihood_config_str})' if likelihood_config_str else ''}",
            f"                 {likelihood_desc} (includes data preprocessing)",
            f"  • Guide:       {guide_name} {f'({guide_config_str})' if guide_config_str else ''}",
            f"                 {guide_desc}",
        ]

        # Add training information if available using new type-safe fields
        if self.state.inference_state is not None:
            inference_state = self.state.inference_state
            training_config = self.state.inference_config

            # Extract training information
            loss = getattr(inference_state, "loss", None)
            final_loss = loss[-1] if loss and isinstance(loss, list) else None
            num_epochs = len(loss) if loss and isinstance(loss, list) else None

            # Add training information to representation
            repr_str.extend([
                "─" * 50,
                "Training:",
                f"  • Epochs:      {num_epochs if num_epochs else 'Unknown'}",
                f"  • Final Loss:  {final_loss:.4f}" if final_loss is not None else "  • Final Loss:  Unknown",
            ])

            # Add optimizer information if available
            if hasattr(inference_state, "optimizer"):
                optimizer = inference_state.optimizer
                optimizer_name = optimizer.__class__.__name__ if optimizer else "Unknown"
                learning_rate = getattr(training_config, "learning_rate", "Unknown") if training_config else "Unknown"

                repr_str.extend([
                    f"  • Optimizer:   {optimizer_name}",
                    f"  • Learning Rate: {learning_rate}",
                ])

        return "\n".join(repr_str)

    def __str__(self) -> str:
        """
        Return a string representation of the PyroVelocityModel.

        This method delegates to __repr__ to ensure consistent string representation.

        Returns:
            String representation of the model
        """
        return self.__repr__()




    @beartype
    def train(
        self,
        adata: AnnData,
        config: InferenceConfig,
        seed: Optional[int] = None,
    ) -> "PyroVelocityModel":
        """
        Run inference on the model using type-safe configuration.

        This method trains the model using the data in the AnnData object and the provided
        inference configuration. It performs the following steps:
        1. Prepares data from the AnnData object
        2. Validates the inference configuration
        3. Runs unified inference (SVI or MCMC based on config)
        4. Stores the inference state in the model state

        Args:
            adata: AnnData object containing the data
            config: Inference configuration specifying method and parameters
            seed: Random seed for reproducibility

        Returns:
            The model instance with updated state (for method chaining)

        Examples:
            >>> # SVI inference
            >>> from pyrovelocity.models.modular.inference.config import InferenceConfig
            >>> config = InferenceConfig(
            ...     method="svi",
            ...     num_epochs=1000,
            ...     learning_rate=0.01,
            ...     guide="auto_normal"
            ... )
            >>> model.train(adata, config)

            >>> # MCMC inference
            >>> config = InferenceConfig(
            ...     method="mcmc",
            ...     num_samples=500,
            ...     kernel="nuts",
            ...     num_warmup=250
            ... )
            >>> model.train(adata, config)
        """
        # Enable Pyro validation
        pyro.enable_validation(True)

        # Prepare data from AnnData
        data_dict = prepare_anndata(adata)

        # Extract unspliced and spliced data
        u_obs = data_dict["X_unspliced"]
        s_obs = data_dict["X_spliced"]

        # Extract library sizes
        u_lib_size = data_dict.get("u_lib_size")
        s_lib_size = data_dict.get("s_lib_size")

        # Create training data dictionary
        train_data = {
            "u_obs": u_obs,
            "s_obs": s_obs,
        }

        if u_lib_size is not None:
            train_data["u_log_library"] = u_lib_size
        if s_lib_size is not None:
            train_data["s_log_library"] = s_lib_size

        # Import inference utilities
        from pyrovelocity.models.modular.inference.config import validate_config
        from pyrovelocity.models.modular.inference.unified import run_inference

        # Validate configuration
        config = validate_config(config)

        # Create guide for SVI (existing logic)
        guide = None
        if config.method == "svi":
            if hasattr(self.guide_model, 'create_guide'):
                guide = self.guide_model.create_guide(self.forward)
            else:
                guide = self.guide_model

        # Run unified inference
        inference_state = run_inference(
            model=self.forward,
            guide=guide,
            kwargs=train_data,
            config=config,
            seed=seed,
        )

        # Update model state with type-safe fields
        self.state = ModelState(
            dynamics_state=self.state.dynamics_state,
            prior_state=self.state.prior_state,
            likelihood_state=self.state.likelihood_state,
            guide_state=inference_state.params,
            inference_state=inference_state,
            inference_config=config,
        )

        return self

    @beartype
    def generate_posterior_samples(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        num_samples: int = 100,
        return_tensors: bool = True,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        Generate posterior samples using the trained model.

        This method generates posterior samples for model parameters using the trained model.
        The samples represent the posterior distribution over parameters like alpha, beta, and gamma,
        which can be used for uncertainty quantification and downstream analysis.

        The method requires that the model has been trained first. It uses the guide
        from the inference state to sample from the approximate posterior distribution.

        Args:
            adata: Optional AnnData object (if None, uses the one from training)
            indices: Optional sequence of indices to generate samples for (for batch processing)
            batch_size: Batch size for generating samples (for memory efficiency)
            num_samples: Number of posterior samples to generate
            return_tensors: If True, return torch tensors; if False, return numpy arrays (default: True)
            **kwargs: Additional keyword arguments including 'seed' for reproducibility

        Returns:
            Dictionary of posterior samples with keys for model parameters (alpha, beta, gamma, etc.)
            and values as torch tensors or numpy arrays of shape [num_samples, num_genes]

        Examples:
            >>> # Generate posterior samples after training
            >>> from pyrovelocity.models.modular.factory import create_legacy_model1
            >>> import anndata as ad
            >>> import numpy as np
            >>> import os
            >>>
            >>> # Create a temporary directory for any file operations
            >>> tmp_dir = os.path.join(os.getcwd(), "tmp_test_dir")
            >>> os.makedirs(tmp_dir, exist_ok=True)
            >>>
            >>> # Create synthetic data
            >>> n_cells, n_genes = 10, 5
            >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>>
            >>> # Create AnnData object
            >>> adata = ad.AnnData(X=s_data)
            >>> adata.layers["spliced"] = s_data
            >>> adata.layers["unspliced"] = u_data
            >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
            >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
            >>>
            >>> # Prepare AnnData
            >>> adata = PyroVelocityModel.setup_anndata(adata)
            >>>
            >>> # Create, train the model, and generate samples
            >>> model = create_legacy_model1()
            >>> model.train(adata=adata, max_epochs=2)  # Use small number for testing
            >>> posterior_samples = model.generate_posterior_samples(
            ...     adata=adata, num_samples=2, seed=42  # Use small number for testing
            ... )
            >>>
            >>> # Access parameter samples
            >>> alpha_samples = posterior_samples["alpha"]  # Shape: [2, num_genes]
            >>> beta_samples = posterior_samples["beta"]    # Shape: [2, num_genes]
            >>> gamma_samples = posterior_samples["gamma"]  # Shape: [2, num_genes]
            >>>
            >>> # Clean up temporary directory
            >>> import shutil
            >>> if os.path.exists(tmp_dir):
            ...     shutil.rmtree(tmp_dir)
        """
        # Check if model has been trained
        if self.state.inference_state is None:
            raise ValueError("Model must be trained before generating posterior samples")

        # 1. Inference method-specific: Extract samples via unified interface
        from pyrovelocity.models.modular.inference.unified import (
            extract_posterior_samples,
        )

        posterior_samples = extract_posterior_samples(
            self.state.inference_state,
            num_samples=num_samples
        )

        # 2. Extract observations for Predictive
        observations = self._extract_observations(adata)

        # 3. Use Pyro's Predictive to get all sites including deterministic ones
        import pyro

        predictive = pyro.infer.Predictive(
            self.forward,
            posterior_samples=posterior_samples,
            return_sites=None,  # Return all sites including deterministic
        )

        # Run predictive to get deterministic sites (ut, st) automatically
        model_samples = predictive(**observations)

        # 4. Combine posterior samples with deterministic sites
        # Deterministic sites from model take precedence
        posterior_samples = {**posterior_samples, **model_samples}

        # 5. Return in requested format
        if return_tensors:
            return posterior_samples
        else:
            # Convert to numpy arrays for legacy compatibility
            return {
                k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in posterior_samples.items()
            }

    def _extract_observations(self, adata: Optional[AnnData]) -> Dict[str, torch.Tensor]:
        """
        Extract observations from AnnData for Predictive.

        Args:
            adata: AnnData object containing observations

        Returns:
            Dictionary with u_obs and s_obs tensors
        """
        if adata is None:
            raise ValueError("AnnData object is required for generating posterior samples")

        import scipy.sparse

        # Extract unspliced and spliced observations
        u_obs = adata.layers["unspliced"]
        s_obs = adata.layers["spliced"]

        # Convert sparse matrices to dense if needed
        if isinstance(u_obs, scipy.sparse.spmatrix):
            u_obs = u_obs.toarray()
        if isinstance(s_obs, scipy.sparse.spmatrix):
            s_obs = s_obs.toarray()

        # Convert to tensors
        u_obs = torch.tensor(u_obs, dtype=torch.float32)
        s_obs = torch.tensor(s_obs, dtype=torch.float32)

        return {"u_obs": u_obs, "s_obs": s_obs}




    @beartype
    def generate_predictive_samples(
        self,
        num_cells: int,
        num_genes: int,
        samples: Optional[Dict[str, torch.Tensor]] = None,
        num_samples: Optional[int] = None,
        return_format: str = "dict",
        observed_times: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Dict[str, torch.Tensor], AnnData]:
        """
        Generate predictive samples using the model.

        This method provides a unified interface for generating both prior and posterior
        predictive samples. When samples=None, it generates prior predictive samples by
        sampling parameters from the prior. When samples are provided, it generates
        posterior predictive samples using those parameter values.

        Args:
            num_cells: Number of cells to generate
            num_genes: Number of genes to generate
            samples: Optional parameter samples to use (if None, samples from prior)
            num_samples: Number of predictive samples (only used if samples=None)
            return_format: Format for returned data ("dict", "anndata")
            observed_times: Optional tensor of observed times for coherent trajectory sampling
            **kwargs: Additional arguments passed to the model

        Returns:
            Generated predictive samples in the specified format

        Raises:
            ValueError: If return_format is invalid or model components are missing
        """
        import pyro
        from pyro.infer import Predictive

        # Validate return format
        valid_formats = ["dict", "anndata"]
        if return_format not in valid_formats:
            raise ValueError(f"return_format must be one of {valid_formats}, got {return_format}")

        # Create an unconditioned version of the model for predictive sampling
        def create_predictive_model():
            """Create a model for predictive sampling without conditioning on observations."""
            # Create dummy observations with the right shape
            dummy_u_obs = torch.zeros(num_cells, num_genes)
            dummy_s_obs = torch.zeros(num_cells, num_genes)

            # Create context with observations
            context = {
                "u_obs": dummy_u_obs,
                "s_obs": dummy_s_obs,
            }

            # Add observed times to context if provided
            if observed_times is not None:
                context["observed_times"] = observed_times

            # Run the full model pipeline with context
            prior_context = self.prior_model.forward(context)
            dynamics_context = self.dynamics_model.forward(prior_context)
            likelihood_context = self.likelihood_model.forward(dynamics_context)

            return likelihood_context

        # Use pyro.poutine.uncondition to remove observation conditioning
        unconditioned_model = pyro.poutine.uncondition(create_predictive_model)

        # Handle prior vs posterior predictive sampling
        if samples is None:
            # Prior predictive sampling: sample parameters from prior
            if num_samples is None:
                num_samples = 1

            # Generate prior predictive samples
            predictive = Predictive(
                unconditioned_model,
                num_samples=num_samples,
                return_sites=None,
            )

            predictive_samples = predictive()

        else:
            # Posterior predictive sampling: use provided parameter samples
            if isinstance(samples, dict):
                # Check if samples contain multiple posterior samples or single averaged values
                sample_sizes = [v.shape[0] if hasattr(v, 'shape') and len(v.shape) > 0 else 1
                              for v in samples.values() if isinstance(v, torch.Tensor)]

                if sample_sizes and max(sample_sizes) > 1:
                    # Multiple posterior samples - CORRECTED APPROACH: Sample-wise generation
                    num_posterior_samples = max(sample_sizes)
                    print(f"  🔄 Generating {num_posterior_samples} posterior predictive samples (sample-wise generation)...")

                    # Limit samples for computational efficiency
                    max_samples_to_use = min(num_posterior_samples, 30)

                    # Generate one observation per posterior sample (NO AVERAGING)
                    all_predictive_samples = []

                    for sample_idx in range(max_samples_to_use):
                        # Extract single parameter vector for this sample
                        single_sample = {}
                        for key, value in samples.items():
                            if isinstance(value, torch.Tensor) and value.shape[0] > 1:
                                single_sample[key] = value[sample_idx:sample_idx+1]  # Keep batch dimension
                            else:
                                single_sample[key] = value

                        # Generate ONE observation from this parameter vector
                        def single_posterior_predictive_model():
                            """Model with fixed parameters for single posterior sample."""
                            # Create dummy observations with the right shape
                            dummy_u_obs = torch.zeros(num_cells, num_genes)
                            dummy_s_obs = torch.zeros(num_cells, num_genes)

                            # Create context with observations
                            context = {
                                "u_obs": dummy_u_obs,
                                "s_obs": dummy_s_obs,
                            }

                            # Add observed times to context if provided
                            if observed_times is not None:
                                context["observed_times"] = observed_times

                            # Use single parameter vector (NO AVERAGING)
                            context.update(self._process_single_parameter_sample(single_sample, num_cells, num_genes))

                            # Run dynamics and likelihood model components
                            dynamics_context = self.dynamics_model.forward(context)
                            likelihood_context = self.likelihood_model.forward(dynamics_context)

                            return likelihood_context

                        # Generate single observation from this parameter sample
                        unconditioned_model = pyro.poutine.uncondition(single_posterior_predictive_model)
                        predictive = Predictive(unconditioned_model, num_samples=1, return_sites=None)
                        single_predictive_sample = predictive()

                        all_predictive_samples.append(single_predictive_sample)

                    # Combine all samples into proper batch structure
                    predictive_samples = self._combine_predictive_samples(all_predictive_samples)

                else:
                    # Single parameter set (backward compatibility): use existing logic
                    def posterior_predictive_model():
                        """Model with fixed parameters for posterior predictive sampling."""
                        # Create dummy observations with the right shape
                        dummy_u_obs = torch.zeros(num_cells, num_genes)
                        dummy_s_obs = torch.zeros(num_cells, num_genes)

                        # Create context with observations
                        context = {
                            "u_obs": dummy_u_obs,
                            "s_obs": dummy_s_obs,
                        }

                        # Add observed times to context if provided
                        if observed_times is not None:
                            context["observed_times"] = observed_times

                        # Add fixed parameter values to context with simplified processing
                        context.update(self._process_parameter_samples(samples, num_cells, num_genes))

                        # Skip prior sampling and go directly to dynamics and likelihood
                        dynamics_context = self.dynamics_model.forward(context)
                        likelihood_context = self.likelihood_model.forward(dynamics_context)

                        return likelihood_context

                    # Use unconditioned version to generate observations
                    unconditioned_posterior_model = pyro.poutine.uncondition(posterior_predictive_model)

                    # Generate samples
                    predictive = Predictive(
                        unconditioned_posterior_model,
                        num_samples=1,  # Generate one sample with fixed parameters
                        return_sites=None,
                    )

                    predictive_samples = predictive()
            else:
                raise ValueError("samples must be a dictionary of parameter values")

        # Convert to requested format
        if return_format == "dict":
            return predictive_samples
        elif return_format == "anndata":
            return self._convert_to_anndata(
                predictive_samples,
                num_cells,
                num_genes,
                samples=samples,
                **kwargs
            )
        elif return_format == "inference_data":
            # TODO: Implement conversion to ArviZ InferenceData format
            raise NotImplementedError("inference_data format not yet implemented")

        return predictive_samples

    @beartype
    def _process_parameter_samples(
        self,
        samples: Dict[str, torch.Tensor],
        num_cells: int,
        num_genes: int,
        single_sample: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Unified parameter processing for both single and multiple samples.

        This method consolidates parameter processing logic, handling tensor reshaping
        for both single parameter samples and parameter sets. It preserves the exact
        behavior of both previous methods while eliminating code duplication.

        Args:
            samples: Dictionary of parameter samples
            num_cells: Target number of cells
            num_genes: Target number of genes
            single_sample: If True, treat as single sample (precise extraction).
                          If False, treat as parameter set (legacy backward compatibility).

        Returns:
            Dictionary of processed parameters ready for model context
        """
        processed = {}

        for key, value in samples.items():
            if isinstance(value, torch.Tensor):
                # Extract value based on sample type
                if single_sample:
                    # Single sample: precise extraction with squeeze
                    if value.shape[0] == 1:
                        processed_value = value.squeeze()
                    else:
                        processed_value = value[0].squeeze()
                else:
                    # Parameter set: backward compatibility handling
                    if value.ndim == 3:
                        processed_value = value[0].squeeze(0)
                    elif value.ndim == 2:
                        if value.shape[0] == 1:
                            processed_value = value.squeeze(0)
                        else:
                            processed_value = value[0]
                    elif value.ndim >= 1:
                        processed_value = value.squeeze()
                    else:
                        processed_value = value

                # Apply parameter-specific reshaping
                processed[key] = self._reshape_parameter(
                    key, processed_value, num_cells, num_genes, single_sample
                )
            else:
                processed[key] = value

        return processed

    def _reshape_parameter(
        self,
        key: str,
        value: torch.Tensor,
        num_cells: int,
        num_genes: int,
        single_sample: bool
    ) -> torch.Tensor:
        """
        Reshape a single parameter tensor based on its type and target dimensions.

        Args:
            key: Parameter name
            value: Parameter tensor value
            num_cells: Target number of cells
            num_genes: Target number of genes  
            single_sample: Whether this is from a single sample

        Returns:
            Reshaped parameter tensor
        """
        # Cell-specific parameters
        if key in ["t_star", "cell_time"]:
            return self._reshape_cell_parameter(value, num_cells)

        # Gene-specific parameters
        gene_params = ["R_on", "alpha_on", "alpha_off", "gamma_star", "t_on_star", "delta_star", "U_0i"]
        if key in gene_params:
            return self._reshape_gene_parameter(value, num_genes, single_sample)

        # Cell-specific parameters (capture efficiency)
        if key in ["lambda_j"]:
            return self._reshape_cell_parameter(value, num_cells)

        # Other parameters: return as-is
        return value

    def _reshape_cell_parameter(self, value: torch.Tensor, num_cells: int) -> torch.Tensor:
        """Reshape parameter to match number of cells."""
        if value.numel() == 1:
            return value.expand(num_cells)
        elif value.numel() == num_cells:
            return value.reshape(num_cells)
        elif value.numel() > num_cells:
            return value.flatten()[:num_cells]
        else:
            repeat_factor = (num_cells + value.numel() - 1) // value.numel()
            repeated = value.repeat(repeat_factor)
            return repeated.flatten()[:num_cells]

    def _reshape_gene_parameter(self, value: torch.Tensor, num_genes: int, single_sample: bool) -> torch.Tensor:
        """Reshape parameter to match number of genes."""
        if single_sample:
            # Enhanced logic for single samples
            if value.numel() == num_genes:
                return value.reshape(num_genes)
            elif value.numel() == 1:
                return value.expand(num_genes)
            else:
                flattened = value.flatten()
                if flattened.numel() >= num_genes:
                    return flattened[:num_genes]
                else:
                    repeat_factor = (num_genes + flattened.numel() - 1) // flattened.numel()
                    repeated = flattened.repeat(repeat_factor)
                    return repeated[:num_genes]
        else:
            # Legacy logic for parameter sets
            if value.numel() == num_genes:
                return value.reshape(num_genes)
            else:
                return value

    @beartype
    def _process_single_parameter_sample(
        self,
        sample: Dict[str, torch.Tensor],
        num_cells: int,
        num_genes: int
    ) -> Dict[str, torch.Tensor]:
        """
        Process a SINGLE parameter sample for posterior predictive sampling.

        This method is now a wrapper around the unified processing function.
        Preserved for backward compatibility.

        Args:
            sample: Dictionary containing a single parameter sample
            num_cells: Target number of cells
            num_genes: Target number of genes

        Returns:
            Dictionary of processed parameters ready for model context
        """
        return self._process_parameter_samples(sample, num_cells, num_genes, single_sample=True)

    @beartype
    def _combine_predictive_samples(
        self,
        all_samples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Combine multiple predictive samples into a single dictionary with batch dimension.

        This method takes a list of individual predictive samples and combines them
        into the format expected by _convert_to_anndata.

        Args:
            all_samples: List of dictionaries, each containing one predictive sample

        Returns:
            Combined dictionary with batch dimension for all tensors
        """
        if not all_samples:
            return {}

        combined = {}

        # Get all keys from the first sample
        sample_keys = all_samples[0].keys()

        for key in sample_keys:
            # Collect all values for this key
            values = []
            for sample in all_samples:
                if key in sample:
                    values.append(sample[key])

            if values:
                # Stack along new batch dimension
                if isinstance(values[0], torch.Tensor):
                    combined[key] = torch.stack(values, dim=0)
                else:
                    # For non-tensor values, just take the first one
                    combined[key] = values[0]

        return combined

    @beartype
    def _convert_to_anndata(
        self,
        predictive_samples: Dict[str, torch.Tensor],
        num_cells: int,
        num_genes: int,
        samples: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> AnnData:
        """
        Convert predictive samples to AnnData format.

        This method creates a properly structured AnnData object from predictive samples,
        including count data in layers and metadata in uns. It follows PyroVelocity
        conventions for AnnData structure and naming.

        Args:
            predictive_samples: Dictionary of predictive samples from Pyro
            num_cells: Number of cells
            num_genes: Number of genes
            samples: Optional parameter samples used for generation (stored as metadata)
            **kwargs: Additional metadata to store

        Returns:
            AnnData object with proper structure for PyroVelocity

        Raises:
            ValueError: If required observation sites are missing from predictive samples
        """
        # Extract count data from predictive samples
        # Look for standard observation sites
        u_obs_key = None
        s_obs_key = None

        # Try different possible keys for observations
        possible_u_keys = ["u", "u_obs", "unspliced", "unspliced_obs"]
        possible_s_keys = ["s", "s_obs", "spliced", "spliced_obs"]

        for key in possible_u_keys:
            if key in predictive_samples:
                u_obs_key = key
                break

        for key in possible_s_keys:
            if key in predictive_samples:
                s_obs_key = key
                break

        if u_obs_key is None or s_obs_key is None:
            available_keys = list(predictive_samples.keys())
            raise ValueError(
                f"Could not find required observation sites in predictive samples. "
                f"Expected unspliced and spliced count data. Available keys: {available_keys}"
            )

        # Extract count matrices
        u_counts = predictive_samples[u_obs_key]
        s_counts = predictive_samples[s_obs_key]

        # Handle tensor conversion and shape
        if isinstance(u_counts, torch.Tensor):
            u_counts = u_counts.detach().cpu().numpy()
        if isinstance(s_counts, torch.Tensor):
            s_counts = s_counts.detach().cpu().numpy()



        # Handle batch dimension for multiple predictive samples
        if u_counts.ndim >= 3:  # Multiple dimensions - need to find the right ones
            # The tensor might be [num_samples, batch_dim, num_cells, num_genes] or similar
            # We need to find the dimensions that correspond to [num_cells, num_genes]

            # Look for dimensions that match our expected cell and gene counts
            shape = u_counts.shape
            cell_dim_idx = None
            gene_dim_idx = None

            for i, dim_size in enumerate(shape):
                if dim_size == num_cells and cell_dim_idx is None:
                    cell_dim_idx = i
                elif dim_size == num_genes and gene_dim_idx is None:
                    gene_dim_idx = i

            if cell_dim_idx is not None and gene_dim_idx is not None:
                # We found the cell and gene dimensions
                # Reshape to [num_samples, num_cells, num_genes] by moving and squeezing

                # First, move the cell and gene dimensions to the end
                u_counts = np.moveaxis(u_counts, [cell_dim_idx, gene_dim_idx], [-2, -1])
                s_counts = np.moveaxis(s_counts, [cell_dim_idx, gene_dim_idx], [-2, -1])

                # Squeeze out any singleton dimensions except the last two (cells, genes)
                while u_counts.ndim > 3:
                    # Find singleton dimensions (excluding last two)
                    singleton_dims = [i for i in range(u_counts.ndim - 2) if u_counts.shape[i] == 1]
                    if singleton_dims:
                        u_counts = np.squeeze(u_counts, axis=singleton_dims[0])
                        s_counts = np.squeeze(s_counts, axis=singleton_dims[0])
                    else:
                        # If no singleton dimensions, flatten the first dimensions
                        new_shape = (-1,) + u_counts.shape[-2:]
                        u_counts = u_counts.reshape(new_shape)
                        s_counts = s_counts.reshape(new_shape)
                        break

                if u_counts.ndim == 3:  # [num_samples, num_cells, num_genes]
                    # For posterior predictive checks, we want to store summary statistics
                    # Store mean as primary data and std/quantiles as additional layers
                    u_counts_mean = u_counts.mean(axis=0)
                    s_counts_mean = s_counts.mean(axis=0)

                    # Store full samples for uncertainty computation
                    u_counts_std = u_counts.std(axis=0)
                    s_counts_std = s_counts.std(axis=0)

                    # Store quantiles for credible intervals
                    u_counts_q025 = np.quantile(u_counts, 0.025, axis=0)
                    u_counts_q975 = np.quantile(u_counts, 0.975, axis=0)
                    s_counts_q025 = np.quantile(s_counts, 0.025, axis=0)
                    s_counts_q975 = np.quantile(s_counts, 0.975, axis=0)

                    # Use mean for primary data
                    u_counts = u_counts_mean
                    s_counts = s_counts_mean

                    # Flag that we have multiple samples for later storage
                    has_multiple_samples = True
                else:
                    # Fallback: take first sample
                    u_counts = u_counts[0] if u_counts.ndim > 2 else u_counts
                    s_counts = s_counts[0] if s_counts.ndim > 2 else s_counts
                    has_multiple_samples = False
            else:
                # Fallback: take first sample along first dimension
                u_counts = u_counts[0] if u_counts.ndim > 2 else u_counts
                s_counts = s_counts[0] if s_counts.ndim > 2 else s_counts
                has_multiple_samples = False
        else:
            has_multiple_samples = False

        # Ensure correct shape [num_cells, num_genes]
        if u_counts.shape != (num_cells, num_genes):
            raise ValueError(
                f"Unspliced counts shape {u_counts.shape} does not match expected "
                f"({num_cells}, {num_genes})"
            )
        if s_counts.shape != (num_cells, num_genes):
            raise ValueError(
                f"Spliced counts shape {s_counts.shape} does not match expected "
                f"({num_cells}, {num_genes})"
            )

        # Create AnnData object with spliced counts as main matrix
        adata = AnnData(X=s_counts.copy())

        # Add layers for both count types
        adata.layers["spliced"] = s_counts
        adata.layers["unspliced"] = u_counts

        # Add uncertainty layers if we have multiple samples
        if has_multiple_samples:
            adata.layers["spliced_std"] = s_counts_std
            adata.layers["unspliced_std"] = u_counts_std
            adata.layers["spliced_q025"] = s_counts_q025
            adata.layers["spliced_q975"] = s_counts_q975
            adata.layers["unspliced_q025"] = u_counts_q025
            adata.layers["unspliced_q975"] = u_counts_q975

        # Add cell and gene names
        adata.obs_names = [f"cell_{i}" for i in range(num_cells)]
        adata.var_names = [f"gene_{i}" for i in range(num_genes)]

        # Add basic metadata
        adata.uns["pyrovelocity"] = {
            "model_name": self.name,
            "generation_method": "predictive_sampling",
            "num_cells": num_cells,
            "num_genes": num_genes,
            "has_posterior_uncertainty": has_multiple_samples,
            "predictive_type": "posterior" if samples is not None else "prior",
        }

        # Store all parameters from predictive samples in clean dictionaries
        # This ensures we capture ALL parameters that were sampled during generation
        true_params = {}

        # First, extract parameters from predictive_samples (prior predictive case)
        for key, value in predictive_samples.items():
            # Skip observation sites (u_obs, s_obs) and latent variables (ut, st)
            if key not in [u_obs_key, s_obs_key, "ut", "st"]:
                if isinstance(value, torch.Tensor):
                    # Convert to numpy and handle batch dimension
                    param_array = value.detach().cpu().numpy()
                    if param_array.ndim > 1 and param_array.shape[0] == 1:
                        param_array = param_array[0]  # Remove batch dimension
                    true_params[key] = param_array
                else:
                    true_params[key] = value

        # Also store parameters from samples argument if provided (posterior predictive case)
        if samples is not None:
            for key, value in samples.items():
                if isinstance(value, torch.Tensor):
                    # Convert to numpy and handle batch dimension
                    param_array = value.detach().cpu().numpy()
                    if param_array.ndim > 1 and param_array.shape[0] == 1:
                        param_array = param_array[0]  # Remove batch dimension
                    true_params[key] = param_array
                else:
                    true_params[key] = value

        # Store parameters in AnnData using clean dictionary names
        if true_params:
            # For prior predictive: store as "true_parameters"
            # For posterior predictive: store as "fit_parameters" (will be handled later)
            adata.uns["true_parameters"] = true_params

            # Try to determine pattern type from parameters if available
            pattern_type = self._infer_pattern_type(true_params)
            if pattern_type:
                adata.uns["pattern"] = pattern_type  # Store as 'pattern' for compatibility

        # Store additional metadata
        for key, value in kwargs.items():
            if key not in adata.uns:
                adata.uns[key] = value

        # Add library size information
        adata.obs["total_unspliced"] = u_counts.sum(axis=1)
        adata.obs["total_spliced"] = s_counts.sum(axis=1)
        adata.obs["total_counts"] = adata.obs["total_unspliced"] + adata.obs["total_spliced"]

        # Extract and store temporal coordinates for UMAP visualization
        self._store_temporal_coordinates(adata, predictive_samples, samples, num_cells)

        # Add gene-level statistics
        adata.var["mean_unspliced"] = u_counts.mean(axis=0)
        adata.var["mean_spliced"] = s_counts.mean(axis=0)
        adata.var["total_unspliced"] = u_counts.sum(axis=0)
        adata.var["total_spliced"] = s_counts.sum(axis=0)

        # Add dimensionality reduction for UMAP visualization
        # This ensures posterior predictive check plots can display UMAP embeddings
        #     import scanpy as sc
        #     sc.pp.pca(adata, random_state=42)
        #     sc.pp.neighbors(adata, n_neighbors=10, random_state=42)
        #     sc.tl.umap(adata, random_state=42)
        #     sc.tl.leiden(adata, random_state=42)

        return adata

    @beartype
    def _store_temporal_coordinates(
        self,
        adata: AnnData,
        predictive_samples: Dict[str, torch.Tensor],
        samples: Optional[Dict[str, torch.Tensor]],
        num_cells: int
    ) -> None:
        """
        Extract and store temporal coordinates in AnnData for UMAP visualization.

        This method looks for temporal coordinates in both the predictive samples
        and the true parameters, and stores them in adata.obs with names that
        the UMAP plotting functions can detect.

        Args:
            adata: AnnData object to store coordinates in
            predictive_samples: Dictionary of predictive samples from Pyro
            samples: Optional parameter samples used for generation
            num_cells: Number of cells
        """
        # Priority order for temporal coordinate names (canonical parameter name first)
        target_time_names = ['t_star', 'latent_time', 'shared_time', 'pseudotime', 'time']

        # Look for temporal coordinates in predictive samples first
        temporal_coord = None
        source_name = None

        # Check predictive samples for temporal variables, prioritizing canonical names
        canonical_temporal_keys = ['t_star', 'cell_time']
        fallback_temporal_patterns = ['time', 'tau', 'latent']

        # First check for canonical parameter names
        for key in canonical_temporal_keys:
            if key in predictive_samples:
                value = predictive_samples[key]
                if isinstance(value, torch.Tensor):
                    # Convert to numpy
                    coord_array = value.detach().cpu().numpy()

                    # Handle different tensor shapes - AVERAGE ACROSS SAMPLES FIRST
                    if coord_array.ndim >= 2:
                        # Average across sample dimension (first dimension) if multiple samples
                        if coord_array.shape[0] > 1:
                            coord_array = coord_array.mean(axis=0)

                        # Try to find a dimension that matches num_cells
                        for dim_idx in range(coord_array.ndim):
                            if coord_array.shape[dim_idx] == num_cells:
                                # Extract the cell dimension
                                if coord_array.ndim == 1:
                                    temporal_coord = coord_array
                                elif coord_array.ndim == 2:
                                    if dim_idx == 0:
                                        temporal_coord = coord_array[:, 0] if coord_array.shape[1] == 1 else coord_array.flatten()
                                    else:
                                        temporal_coord = coord_array[0, :] if coord_array.shape[0] == 1 else coord_array.flatten()
                                elif coord_array.ndim > 2:
                                    # For higher dimensions, flatten and take first num_cells elements
                                    coord_flat = coord_array.flatten()
                                    if len(coord_flat) >= num_cells:
                                        temporal_coord = coord_flat[:num_cells]
                                break
                    elif coord_array.ndim == 1 and len(coord_array) == num_cells:
                        temporal_coord = coord_array

                    if temporal_coord is not None:
                        source_name = key
                        break

        # If not found, search by pattern matching
        if temporal_coord is None:
            for key, value in predictive_samples.items():
                if any(t_word in key.lower() for t_word in fallback_temporal_patterns):
                    if isinstance(value, torch.Tensor):
                        # Convert to numpy
                        coord_array = value.detach().cpu().numpy()

                    # Handle different tensor shapes
                    if coord_array.ndim >= 2:
                        # Try to find a dimension that matches num_cells
                        for dim_idx in range(coord_array.ndim):
                            if coord_array.shape[dim_idx] == num_cells:
                                # Extract the cell dimension
                                if coord_array.ndim == 2:
                                    if dim_idx == 0:
                                        temporal_coord = coord_array[:, 0]  # Take first column
                                    else:
                                        temporal_coord = coord_array[0, :]  # Take first row
                                elif coord_array.ndim > 2:
                                    # For higher dimensions, take first slice along other dimensions
                                    if dim_idx == 0:
                                        temporal_coord = coord_array[:, 0, 0] if coord_array.shape[1] > 0 else coord_array[:, 0]
                                    else:
                                        # Reshape to get cell dimension
                                        coord_flat = coord_array.flatten()
                                        if len(coord_flat) >= num_cells:
                                            temporal_coord = coord_flat[:num_cells]
                                break
                    elif coord_array.ndim == 1 and len(coord_array) == num_cells:
                        temporal_coord = coord_array

                    if temporal_coord is not None:
                        source_name = key
                        break

        # If not found in predictive samples, look for t_star directly in other sources
            if temporal_coord is None:
                if 'true_parameters' in adata.uns:
                    true_params = adata.uns['true_parameters']
                elif samples is not None:
                    # Convert samples to true_params format
                    true_params = {}
                    for key, value in samples.items():
                        if isinstance(value, torch.Tensor):
                            param_array = value.detach().cpu().numpy()
                            if param_array.ndim > 1 and param_array.shape[0] == 1:
                                param_array = param_array[0]
                            true_params[key] = param_array
                        else:
                            true_params[key] = value
                else:
                    true_params = {}

                # Look for t_star (cell-specific times) in true parameters
                for key in ['t_star', 't_cell', 'cell_time', 'latent_time']:
                    if key in true_params:
                        param_value = true_params[key]

                        # Convert to numpy array if needed
                        if isinstance(param_value, torch.Tensor):
                            param_array = param_value.detach().cpu().numpy()
                        else:
                            param_array = np.array(param_value)

                        # Handle multi-dimensional arrays (e.g., shape (10, 1, 1, 20))
                        if param_array.ndim > 1:
                            # Average across sample dimension (first dimension)
                            if param_array.shape[0] > 1:
                                param_array = param_array.mean(axis=0)

                            # Find the dimension that matches num_cells
                            for dim_idx in range(param_array.ndim):
                                if param_array.shape[dim_idx] == num_cells:
                                    # Extract the cell dimension
                                    if param_array.ndim == 1:
                                        temporal_coord = param_array
                                    elif param_array.ndim == 2:
                                        if dim_idx == 0:
                                            temporal_coord = param_array[:, 0] if param_array.shape[1] == 1 else param_array.flatten()
                                        else:
                                            temporal_coord = param_array[0, :] if param_array.shape[0] == 1 else param_array.flatten()
                                    else:
                                        # For higher dimensions, flatten and take first num_cells elements
                                        param_flat = param_array.flatten()
                                        if len(param_flat) >= num_cells:
                                            temporal_coord = param_flat[:num_cells]
                                    break

                            # If we found a matching dimension, use it
                            if temporal_coord is not None:
                                source_name = key
                                break

                        # Handle 1D arrays
                        elif param_array.ndim == 1 and len(param_array) == num_cells:
                            temporal_coord = param_array
                            source_name = key
                            break

        # Store temporal coordinate in adata.obs if found
        if temporal_coord is not None:
            # Ensure it's a 1D numpy array
            if not isinstance(temporal_coord, np.ndarray):
                temporal_coord = np.array(temporal_coord)

            if temporal_coord.ndim > 1:
                temporal_coord = temporal_coord.flatten()

            # Ensure correct length
            if len(temporal_coord) == num_cells:
                # Store with the first available target name for UMAP compatibility
                target_name = target_time_names[0]  # 't_star' (canonical parameter name)
                adata.obs[target_name] = temporal_coord

                # Also store with original name if different
                if source_name and source_name != target_name:
                    adata.obs[source_name] = temporal_coord

                # For backward compatibility, also store as 'latent_time' if not already stored
                if 'latent_time' not in adata.obs and target_name != 'latent_time':
                    adata.obs['latent_time'] = temporal_coord

                # Add metadata about the temporal coordinate
                if 'pyrovelocity' not in adata.uns:
                    adata.uns['pyrovelocity'] = {}
                adata.uns['pyrovelocity']['temporal_coordinate'] = {
                    'source': source_name,
                    'stored_as': target_name,
                    'range': [float(temporal_coord.min()), float(temporal_coord.max())],
                    'description': f'Temporal coordinate extracted from {source_name}'
                }

    @beartype
    def _infer_pattern_type(self, true_params: Dict[str, Any]) -> Optional[str]:
        """
        Infer the gene expression pattern type from true parameters.

        This method attempts to classify the expression pattern based on the
        parameter values, which is useful for validation studies.

        Args:
            true_params: Dictionary of true parameter values

        Returns:
            Inferred pattern type or None if cannot be determined
        """
        # Check if we have piecewise activation parameters
        required_keys = ["alpha_off", "alpha_on", "t_on_star", "delta_star"]
        if not all(key in true_params for key in required_keys):
            return None

        try:
            # Extract parameters (handle both tensor and array formats)
            alpha_off = true_params["alpha_off"]
            alpha_on = true_params["alpha_on"]
            t_on_star = true_params["t_on_star"]
            delta_star = true_params["delta_star"]

            # Convert to scalars if needed (take first gene/cell)
            if hasattr(alpha_off, '__len__') and len(alpha_off) > 0:
                alpha_off = float(alpha_off.flat[0])
            if hasattr(alpha_on, '__len__') and len(alpha_on) > 0:
                alpha_on = float(alpha_on.flat[0])
            if hasattr(t_on_star, '__len__') and len(t_on_star) > 0:
                t_on_star = float(t_on_star.flat[0])
            if hasattr(delta_star, '__len__') and len(delta_star) > 0:
                delta_star = float(delta_star.flat[0])

            # Apply pattern classification logic (same as in prior model)
            fold_change = alpha_on / alpha_off if alpha_off > 0 else float('inf')

            # Check for activation pattern first (most stringent requirements)
            if alpha_off < 0.15 and alpha_on > 1.5 and t_on_star < 0.4 and delta_star > 0.4 and fold_change > 7.5:
                return "activation"

            # Check for decay pattern (most specific constraints)
            if alpha_off > 0.08 and t_on_star > 0.35:
                return "decay"

            # Check for transient pattern (specific delta_star range)
            if alpha_off < 0.3 and alpha_on > 1.0 and t_on_star < 0.5 and delta_star < 0.35 and fold_change > 3.3:
                return "transient"

            # Check for sustained pattern (less stringent than activation)
            if alpha_off < 0.3 and alpha_on > 1.0 and t_on_star < 0.3 and delta_star > 0.35 and fold_change > 3.3:
                return "sustained"

            # If none of the patterns match, return unknown
            return "unknown"

        except (KeyError, TypeError, ValueError):
            # If we can't extract or convert parameters, return None
            return None

    @beartype
    def _prepare_training_data(self, adata: AnnData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data from AnnData object.

        Args:
            adata: AnnData object containing count data

        Returns:
            Tuple of (unspliced_counts, spliced_counts) as torch tensors
        """
        # Extract count matrices
        u_obs = adata.layers["unspliced"].copy()
        s_obs = adata.layers["spliced"].copy()

        # Convert to torch tensors
        u_obs = torch.tensor(u_obs, dtype=torch.float32)
        s_obs = torch.tensor(s_obs, dtype=torch.float32)

        return u_obs, s_obs

    @beartype
    def _compute_recovery_metrics(
        self,
        true_parameters: Dict[str, torch.Tensor],
        posterior_samples: Dict[str, torch.Tensor],
        success_threshold: float
    ) -> Dict[str, float]:
        """
        Compute parameter recovery metrics comparing true and inferred parameters.

        Args:
            true_parameters: Dictionary of true parameter values
            posterior_samples: Dictionary of posterior parameter samples
            success_threshold: Correlation threshold for success

        Returns:
            Dictionary containing recovery metrics
        """
        import numpy as np
        from scipy.stats import pearsonr

        correlations = []
        mae_values = []
        mape_values = []
        successful_params = 0
        total_params = 0

        for param_name, true_value in true_parameters.items():
            if param_name in posterior_samples:
                # Convert to numpy arrays
                if isinstance(true_value, torch.Tensor):
                    true_array = true_value.detach().cpu().numpy().flatten()
                else:
                    true_array = np.array(true_value).flatten()

                posterior_value = posterior_samples[param_name]
                if isinstance(posterior_value, torch.Tensor):
                    # Take mean across samples if multiple samples
                    if posterior_value.ndim > 1:
                        posterior_array = posterior_value.mean(dim=0).detach().cpu().numpy().flatten()
                    else:
                        posterior_array = posterior_value.detach().cpu().numpy().flatten()
                else:
                    posterior_array = np.array(posterior_value).flatten()

                # Ensure same length
                min_len = min(len(true_array), len(posterior_array))
                true_array = true_array[:min_len]
                posterior_array = posterior_array[:min_len]

                if len(true_array) > 0 and len(posterior_array) > 0:
                    # Compute correlation
                    if np.std(true_array) > 1e-8 and np.std(posterior_array) > 1e-8:
                        corr, _ = pearsonr(true_array, posterior_array)
                        if not np.isnan(corr):
                            correlations.append(corr)

                            # Compute MAE
                            mae = np.mean(np.abs(true_array - posterior_array))
                            mae_values.append(mae)

                            # Compute MAPE (avoid division by zero)
                            true_nonzero = true_array[np.abs(true_array) > 1e-8]
                            posterior_nonzero = posterior_array[np.abs(true_array) > 1e-8]
                            if len(true_nonzero) > 0:
                                mape = np.mean(np.abs((true_nonzero - posterior_nonzero) / true_nonzero)) * 100
                                mape_values.append(mape)

                            # Check if parameter recovery is successful
                            if corr >= success_threshold:
                                successful_params += 1
                            total_params += 1

        # Compute overall metrics
        overall_correlation = np.mean(correlations) if correlations else 0.0
        overall_mae = np.mean(mae_values) if mae_values else float('inf')
        overall_mape = np.mean(mape_values) if mape_values else float('inf')
        success_rate = successful_params / total_params if total_params > 0 else 0.0

        return {
            'overall_correlation': overall_correlation,
            'mean_absolute_error': overall_mae,
            'mean_absolute_percentage_error': overall_mape,
            'success_rate': success_rate,
            'successful_parameters': successful_params,
            'total_parameters': total_params,
            'parameter_correlations': dict(zip(
                [name for name in true_parameters.keys() if name in posterior_samples],
                correlations
            ))
        }
