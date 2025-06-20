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
from jaxtyping import Array, Float, Int

from pyrovelocity.models.modular.components.guides import (
    AutoGuideFactory,
    LegacyAutoGuideFactory,
)
from pyrovelocity.models.modular.components.likelihoods import (
    LegacyLikelihoodModel,
    PiecewiseActivationPoissonLikelihoodModel,
)
from pyrovelocity.models.modular.components.priors import LogNormalPriorModel
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


        # Initialize the context dictionary to pass between components
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
        # Initialize the context dictionary to pass to the guide
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

        # Check if model has been trained
        trained_status = "Trained" if "inference_state" in self.state.metadata else "Untrained"

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

        # Add training information if available
        if "inference_state" in self.state.metadata:
            inference_state = self.state.metadata["inference_state"]
            training_config = self.state.metadata.get("training_config", {})

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
                learning_rate = training_config.get("learning_rate", "Unknown")

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

    def with_state(self, state: ModelState) -> "PyroVelocityModel":
        """
        Create a new model instance with the given state.

        This method implements the immutable state pattern, returning a new
        model instance with the updated state rather than modifying the
        current instance.

        Args:
            state: New model state

        Returns:
            New PyroVelocityModel instance with the given state
        """
        return PyroVelocityModel(
            dynamics_model=self.dynamics_model,
            prior_model=self.prior_model,
            likelihood_model=self.likelihood_model,
            guide_model=self.guide_model,
            state=state,
        )

    @beartype
    def predict(
        self, x: Any, time_points: Optional[Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate predictions using the model without sampling from the posterior.

        This method processes the input data through the observation model and dynamics model
        to generate predictions. Unlike the forward method, this method is focused on
        prediction rather than inference, and doesn't apply prior distributions or
        likelihood models.

        The method is useful for:
        1. Making predictions with fixed parameter values
        2. Simulating RNA dynamics over time
        3. Testing the dynamics model in isolation

        Args:
            x: Input data tensor of shape [batch_size, n_features]
            time_points: Time points for the dynamics model of shape [n_time_points]
            **kwargs: Additional keyword arguments passed to component models

        Returns:
            Dictionary containing predictions, typically including:
            - u_predicted: Predicted unspliced RNA counts
            - s_predicted: Predicted spliced RNA counts
            - velocity: Predicted RNA velocity

        Examples:
            >>> # Make predictions with the model
            >>> import torch
            >>> from pyrovelocity.models.modular.factory import create_legacy_model1
            >>>
            >>> # Create model and synthetic data
            >>> model = create_legacy_model1()
            >>> x = torch.randn(10, 5)  # 10 cells, 5 genes
            >>> time_points = torch.linspace(0, 1, 10)  # 10 time points
            >>>
            >>> # Generate predictions
            >>> predictions = model.predict(x, time_points)
            >>>
            >>> # Access predictions
            >>> u_predicted = predictions.get("u_predicted")
            >>> s_predicted = predictions.get("s_predicted")
            >>> velocity = predictions.get("velocity")
        """
        # Initialize the context dictionary
        context = {
            "x": x,
            "time_points": time_points,
            **kwargs,
        }

        # Process through the dynamics model (data preprocessing is now handled by likelihood model)
        dynamics_context = self.dynamics_model.forward(context)

        # Extract predictions
        predictions = dynamics_context.get("predictions", {})

        return predictions

    @beartype
    def predict_future_states(
        self, current_state: Tuple[torch.Tensor, torch.Tensor], time_delta: Union[float, torch.Tensor], **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict future RNA states based on current state and time delta.

        This method delegates to the dynamics model to predict future states of
        unspliced and spliced RNA counts given the current state and a time delta.
        It's useful for simulating RNA dynamics over time and predicting cell
        trajectories.

        The method solves the RNA velocity differential equations to predict
        how the RNA counts will evolve over the specified time delta.

        Args:
            current_state: Current state as a tuple of (u, s) where:
                - u: Unspliced RNA counts of shape [batch_size, n_genes]
                - s: Spliced RNA counts of shape [batch_size, n_genes]
            time_delta: Time delta for prediction, either a scalar or tensor of shape [batch_size]
            **kwargs: Additional keyword arguments passed to the dynamics model,
                including parameters like alpha, beta, gamma

        Returns:
            Tuple of (u_future, s_future) representing the predicted future state:
                - u_future: Predicted unspliced RNA counts of shape [batch_size, n_genes]
                - s_future: Predicted spliced RNA counts of shape [batch_size, n_genes]

        Examples:
            >>> # Predict future states
            >>> import torch
            >>> from pyrovelocity.models.modular.factory import create_legacy_model1
            >>>
            >>> # Create model and synthetic data
            >>> model = create_legacy_model1()
            >>> u_current = torch.ones(10, 5)  # 10 cells, 5 genes
            >>> s_current = torch.ones(10, 5)  # 10 cells, 5 genes
            >>> current_state = (u_current, s_current)
            >>>
            >>> # Predict future state after time_delta
            >>> time_delta = 0.5  # Use float to match type annotation
            >>> # Create default parameters
            >>> alpha = torch.ones(5)  # One value per gene
            >>> beta = torch.ones(5)  # One value per gene
            >>> gamma = torch.ones(5)  # One value per gene
            >>> u_future, s_future = model.predict_future_states(
            ...     current_state, time_delta, alpha=alpha, beta=beta, gamma=gamma
            ... )
        """
        # Delegate to the dynamics model
        return self.dynamics_model.predict_future_states(
            current_state, time_delta, **kwargs
        )

    @classmethod
    @beartype
    def setup_anndata(
        cls,
        adata: AnnData,
        spliced_layer: str = "spliced",
        unspliced_layer: str = "unspliced",
        use_raw: bool = False,
        **kwargs
    ) -> AnnData:
        """
        Set up AnnData object for use with PyroVelocityModel.

        This method prepares the AnnData object for use with the PyroVelocityModel by:
        1. Verifying that required layers exist
        2. Computing library sizes for normalization
        3. Computing library size statistics
        4. Adding indices for batch processing

        The method modifies the AnnData object in-place by adding several new columns
        to the obs attribute:
        - u_lib_size_raw: Raw library size for unspliced counts
        - s_lib_size_raw: Raw library size for spliced counts
        - u_lib_size: Log-transformed library size for unspliced counts
        - s_lib_size: Log-transformed library size for spliced counts
        - u_lib_size_mean: Mean of log library size for unspliced counts
        - s_lib_size_mean: Mean of log library size for spliced counts
        - u_lib_size_scale: Standard deviation of log library size for unspliced counts
        - s_lib_size_scale: Standard deviation of log library size for spliced counts
        - ind_x: Indices for batch processing

        Args:
            adata: AnnData object to set up
            spliced_layer: Name of the spliced layer in adata.layers
            unspliced_layer: Name of the unspliced layer in adata.layers
            use_raw: Whether to use raw data from adata.raw
            **kwargs: Additional keyword arguments (unused)

        Returns:
            Modified AnnData object with additional columns

        Raises:
            ValueError: If required layers are not found in the AnnData object

        Examples:
            >>> # Set up AnnData for PyroVelocity
            >>> import anndata as ad
            >>> import numpy as np
            >>> import os
            >>> from pyrovelocity.models.modular.model import PyroVelocityModel
            >>>
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
            >>> # Set up AnnData
            >>> adata = PyroVelocityModel.setup_anndata(
            ...     adata,
            ...     spliced_layer="spliced",
            ...     unspliced_layer="unspliced"
            ... )
            >>>
            >>> # Check that library sizes were computed
            >>> print("u_lib_size in adata.obs:", "u_lib_size" in adata.obs)
            >>> print("s_lib_size in adata.obs:", "s_lib_size" in adata.obs)
            >>>
            >>> # Clean up temporary directory
            >>> import shutil
            >>> if os.path.exists(tmp_dir):
            ...     shutil.rmtree(tmp_dir)
        """
        # Make sure the required layers exist
        required_layers = [spliced_layer, unspliced_layer]
        for layer in required_layers:
            if layer not in adata.layers:
                raise ValueError(f"Layer '{layer}' not found in AnnData object")

        # Compute library sizes if they don't exist
        if "u_lib_size_raw" not in adata.obs:
            u, _ = extract_layers(adata, spliced_layer, unspliced_layer, use_raw)
            adata.obs["u_lib_size_raw"] = u.sum(dim=1).cpu().numpy()

        if "s_lib_size_raw" not in adata.obs:
            _, s = extract_layers(adata, spliced_layer, unspliced_layer, use_raw)
            adata.obs["s_lib_size_raw"] = s.sum(dim=1).cpu().numpy()

        # Compute log library sizes
        if "u_lib_size" not in adata.obs:
            adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"].astype(float) + 1e-6)

        if "s_lib_size" not in adata.obs:
            adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"].astype(float) + 1e-6)

        # Compute library size statistics
        if "u_lib_size_mean" not in adata.obs:
            adata.obs["u_lib_size_mean"] = adata.obs["u_lib_size"].mean()

        if "s_lib_size_mean" not in adata.obs:
            adata.obs["s_lib_size_mean"] = adata.obs["s_lib_size"].mean()

        if "u_lib_size_scale" not in adata.obs:
            adata.obs["u_lib_size_scale"] = adata.obs["u_lib_size"].std()

        if "s_lib_size_scale" not in adata.obs:
            adata.obs["s_lib_size_scale"] = adata.obs["s_lib_size"].std()

        # Add indices for batch processing
        if "ind_x" not in adata.obs:
            adata.obs["ind_x"] = np.arange(adata.n_obs)

        return adata

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
        # Import inference utilities
        from pyrovelocity.models.modular.inference.posterior import (
            sample_posterior,
        )

        # Get inference state from model state
        inference_state = self.state.metadata.get("inference_state")
        if inference_state is None:
            raise ValueError("Model has not been trained yet")

        # Generate posterior samples
        posterior_samples = sample_posterior(
            model=self,
            state=inference_state,
            num_samples=num_samples,
            seed=kwargs.get("seed", None),
        )

        # Now, we need to run the model with the posterior samples to get deterministic sites
        # This is critical for getting ut and st which are deterministic sites
        import pyro

        # Create a predictive object for the model, using the guide samples
        # Return all sites, including deterministic ones
        model_predictive = pyro.infer.Predictive(
            self,
            posterior_samples=posterior_samples,
            return_sites=None,  # Return all sites, including deterministic
            num_samples=num_samples
        )

        # Run the model predictive to get all sites, including deterministic ones
        # We need to pass the same arguments as during training
        # For the modular model, we can pass u_obs and s_obs
        if adata is not None:
            # Extract u_obs and s_obs from adata
            import scipy.sparse
            u_obs = torch.tensor(
                adata.layers["unspliced"].toarray()
                if isinstance(adata.layers["unspliced"], scipy.sparse.spmatrix)
                else adata.layers["unspliced"]
            )
            s_obs = torch.tensor(
                adata.layers["spliced"].toarray()
                if isinstance(adata.layers["spliced"], scipy.sparse.spmatrix)
                else adata.layers["spliced"]
            )


            # Run the model predictive with u_obs and s_obs
            # Don't use plate context managers here - they're already in the model
            model_samples = model_predictive(u_obs=u_obs, s_obs=s_obs)

            # Combine guide and model samples
            # Guide samples take precedence if there's a conflict
            posterior_samples = {**model_samples, **posterior_samples}

            # If ut and st are not in the posterior samples, we need to compute them
            # This is critical for the legacy model compatibility
            if "ut" not in posterior_samples or "st" not in posterior_samples:
                # Check if we have the necessary parameters to compute ut and st
                if all(k in posterior_samples for k in ["alpha", "beta", "gamma", "cell_time"]):
                    # Extract parameters
                    alpha = posterior_samples["alpha"]
                    beta = posterior_samples["beta"]
                    gamma = posterior_samples["gamma"]
                    cell_time = posterior_samples["cell_time"]

                    # Compute steady state values
                    u_inf = alpha / beta
                    s_inf = alpha / gamma

                    # Compute ut and st based on the transcription model
                    # For cells before switching time
                    t0 = posterior_samples.get("t0", torch.zeros_like(alpha))
                    dt_switching = posterior_samples.get("dt_switching", torch.zeros_like(alpha))
                    switching = t0 + dt_switching

                    # Compute ut and st
                    ut = u_inf * (1 - torch.exp(-beta * cell_time))
                    st = s_inf * (1 - torch.exp(-gamma * cell_time)) - (
                        alpha / (gamma - beta)
                    ) * (torch.exp(-beta * cell_time) - torch.exp(-gamma * cell_time))

                    # Add to posterior samples
                    posterior_samples["ut"] = ut
                    posterior_samples["st"] = st
                    posterior_samples["u_inf"] = u_inf
                    posterior_samples["s_inf"] = s_inf
                    posterior_samples["switching"] = switching

        # Return tensors or numpy arrays based on return_tensors parameter
        if return_tensors:
            # Return torch tensors directly (no conversion)
            return posterior_samples
        else:
            # Convert PyTorch tensors to NumPy arrays (legacy behavior)
            posterior_samples_np = {
                k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in posterior_samples.items()
            }
            return posterior_samples_np

    @beartype
    def store_results_in_anndata(
        self,
        adata: AnnData,
        posterior_samples: Dict[str, Union[np.ndarray, torch.Tensor]],
        model_name: str = "pyrovelocity",
    ) -> AnnData:
        """
        Store model results in AnnData for visualization and analysis.

        This method computes velocity from posterior samples and stores all results
        in the AnnData object. The results are stored in various slots of the AnnData object:
        - adata.obs: Cell-level information (e.g., latent time)
        - adata.var: Gene-level information (e.g., kinetic parameters)
        - adata.layers: Matrix data (e.g., velocity, uncertainty)
        - adata.uns: Unstructured data (e.g., model configuration)

        The stored results can be used with visualization tools like scVelo for
        creating velocity stream plots and other visualizations.

        Args:
            adata: AnnData object to store results in
            posterior_samples: Dictionary of posterior samples from generate_posterior_samples()
            model_name: Name prefix to use for storing results (to distinguish between models)

        Returns:
            Updated AnnData object with model results

        Examples:
            >>> # Store results in AnnData after generating posterior samples
            >>> from pyrovelocity.models.modular.factory import create_legacy_model1
            >>> import anndata as ad
            >>> import numpy as np
            >>> import torch
            >>> tmp = getfixture("tmp_path")
            >>> # Create synthetic data
            >>> n_cells, n_genes = 10, 5
            >>> u_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>> s_data = np.random.poisson(5, size=(n_cells, n_genes))
            >>> # Create AnnData object
            >>> adata = ad.AnnData(X=s_data)
            >>> adata.layers["spliced"] = s_data
            >>> adata.layers["unspliced"] = u_data
            >>> adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
            >>> adata.var_names = [f"gene_{i}" for i in range(n_genes)]
            >>> # Add UMAP coordinates for visualization
            >>> adata.obsm = {}
            >>> adata.obsm["X_umap"] = np.random.normal(0, 1, size=(n_cells, 2))
            >>> # Add cluster information
            >>> adata.obs["clusters"] = np.random.choice(["A", "B", "C"], size=n_cells)
            >>> # Prepare AnnData
            >>> adata = PyroVelocityModel.setup_anndata(adata)
            >>> # Create, train model, and generate samples
            >>> model = create_legacy_model1()
            >>> model.train(adata=adata, max_epochs=2)  # Use small number for testing
            >>> posterior_samples = model.generate_posterior_samples(
            ...     adata=adata, num_samples=2  # Use small number for testing
            ... )
            >>> # Convert torch tensors to numpy arrays if needed
            >>> numpy_posterior_samples = {}
            >>> for key, value in posterior_samples.items():
            ...     if isinstance(value, torch.Tensor):
            ...         numpy_posterior_samples[key] = value.detach().cpu().numpy()
            ...     else:
            ...         numpy_posterior_samples[key] = value
            >>> # Store results in AnnData
            >>> adata_out = model.store_results_in_anndata(
            ...     adata=adata,
            ...     posterior_samples=numpy_posterior_samples,
            ...     model_name="standard_model"
            ... )
            >>> # Check that results were stored
            >>> assert "standard_model_alpha" in adata_out.var
            >>> assert "standard_model_beta" in adata_out.var
            >>> assert "standard_model_gamma" in adata_out.var
        """
        # Import inference utilities
        # Import store_results function
        from pyrovelocity.models.modular.data.anndata import store_results
        from pyrovelocity.models.modular.inference.posterior import (
            compute_velocity,
        )

        # Convert torch tensors to numpy arrays if needed
        numpy_posterior_samples = {}
        for key, value in posterior_samples.items():
            if isinstance(value, torch.Tensor):
                numpy_posterior_samples[key] = value.detach().cpu().numpy()
            else:
                numpy_posterior_samples[key] = value

        # Compute velocity from posterior samples
        velocity_results = compute_velocity(
            model=self,
            posterior_samples=numpy_posterior_samples,
            adata=adata,
        )

        # Combine results
        results = {**numpy_posterior_samples, **velocity_results}

        # Store results in AnnData
        return store_results(adata, results, model_name)

    @beartype
    def get_velocity(
        self,
        adata: AnnData,
        indices: Optional[Sequence[int]] = None,
        num_samples: int = 100,
        basis: str = "umap",
        n_jobs: int = 1,
        random_seed: int = 99,
        compute_uncertainty: bool = True,
        uncertainty_method: str = "std",
        model_name: str = "pyrovelocity",
        **kwargs
    ) -> AnnData:
        """
        Compute RNA velocity and uncertainty from posterior samples and store results in AnnData.

        This method computes RNA velocity and optionally uncertainty from posterior samples
        generated by the model, following the same approach as the legacy implementation's
        compute_statistics_from_posterior_samples. It generates posterior samples, computes
        velocity and uncertainty, stores all results in the AnnData object, and computes
        velocity embeddings using scvelo.

        The method mutates the input AnnData object by adding the following fields:
        - adata.layers[f"velocity_{model_name}"] - Mean velocity values
        - adata.layers[f"spliced_{model_name}"] - Mean posterior spliced counts
        - adata.var["velocity_genes"] - Boolean indicating velocity genes
        - adata.obsm[f"velocity_{model_name}_{basis}"] - Embedded velocity vectors
        - adata.uns[f"velocity_{model_name}_graph"] - Velocity graph
        - adata.uns[f"velocity_{model_name}_graph_neg"] - Negative velocity graph
        - adata.uns[f"velocity_{model_name}_params"] - Velocity parameters
        - adata.obs[f"velocity_{model_name}_self_transition"] - Self-transition probabilities
        - adata.var[f"velocity_{model_name}_uncertainty"] - Velocity uncertainty (if computed)

        Args:
            adata: AnnData object containing the data (will be mutated)
            indices: Optional indices to subset the data
            num_samples: Number of posterior samples to generate
            basis: Basis for velocity embedding (default: "umap")
            n_jobs: Number of jobs for parallel processing
            random_seed: Random seed for reproducibility
            compute_uncertainty: Whether to compute velocity uncertainty
            uncertainty_method: Method for computing uncertainty ("std", "quantile", "entropy")
            model_name: Name prefix for stored results in AnnData
            **kwargs: Additional keyword arguments passed to generate_posterior_samples

        Returns:
            The mutated AnnData object with velocity results stored

        Examples:
            >>> # Compute velocity from a trained model
            >>> import anndata as ad
            >>> import numpy as np
            >>> from pyrovelocity.models.modular.factory import create_legacy_model1
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
            >>> # Create and train model
            >>> model = create_legacy_model1()
            >>> model.train(adata=adata, max_epochs=2)  # Use small number for testing
            >>>
            >>> # Compute velocity and store results in AnnData
            >>> adata_result = model.get_velocity(adata=adata)
            >>> assert f"velocity_{model.name}" in adata_result.layers
            >>> assert f"velocity_{model.name}_umap" in adata_result.obsm
        """
        import scvelo as scv

        from pyrovelocity.models.modular.inference.posterior import (
            compute_uncertainty,
            compute_velocity,
        )

        # Generate posterior samples once (return numpy arrays for compatibility with downstream functions)
        posterior_samples = self.generate_posterior_samples(
            adata=adata, indices=indices, num_samples=num_samples, return_tensors=False, **kwargs
        )

        # Compute velocity from posterior samples (both mean and samples)
        velocity_results_mean = compute_velocity(
            model=self,
            posterior_samples=posterior_samples,
            adata=adata,
            use_mean=True,
        )

        velocity_mean = velocity_results_mean["velocity"]
        if isinstance(velocity_mean, torch.Tensor):
            velocity_mean = velocity_mean.detach().cpu().numpy()

        # Store mean velocity in AnnData layers
        adata.layers[f"velocity_{model_name}"] = velocity_mean

        # Store mean spliced counts if available in posterior samples
        if "st" in posterior_samples:
            st_mean = posterior_samples["st"].mean(0).squeeze()
            if isinstance(st_mean, torch.Tensor):
                st_mean = st_mean.detach().cpu().numpy()
            adata.layers[f"spliced_{model_name}"] = st_mean

        # Mark all genes as velocity genes (following legacy behavior)
        adata.var["velocity_genes"] = True

        # Compute velocity uncertainty if requested
        if compute_uncertainty:
            velocity_results_samples = compute_velocity(
                model=self,
                posterior_samples=posterior_samples,
                adata=adata,
                use_mean=False,
            )

            velocity_samples = velocity_results_samples["velocity"]
            if isinstance(velocity_samples, np.ndarray):
                velocity_samples = torch.tensor(velocity_samples)

            uncertainty = compute_uncertainty(velocity_samples, method=uncertainty_method)

            if isinstance(uncertainty, torch.Tensor):
                uncertainty = uncertainty.detach().cpu().numpy()
            elif not isinstance(uncertainty, np.ndarray):
                uncertainty = np.array(uncertainty)

            # Handle uncertainty shape for storage in adata.var (gene-level information)
            if uncertainty.ndim == 2:
                # If uncertainty is [num_cells, num_genes], take mean across cells for gene-level storage
                uncertainty_gene_level = uncertainty.mean(axis=0)
                # Also store the full uncertainty in layers for cell-gene level information
                adata.layers[f"velocity_{model_name}_uncertainty"] = uncertainty
            elif uncertainty.ndim == 1 and uncertainty.shape[0] == adata.n_vars:
                # If uncertainty is already [num_genes], use directly
                uncertainty_gene_level = uncertainty
            elif uncertainty.ndim == 1 and uncertainty.shape[0] == adata.n_obs:
                # If uncertainty is [num_cells], broadcast to genes (unusual case)
                uncertainty_gene_level = np.full(adata.n_vars, uncertainty.mean())
                adata.layers[f"velocity_{model_name}_uncertainty"] = uncertainty.reshape(-1, 1)
            else:
                raise ValueError(
                    f"Uncertainty shape {uncertainty.shape} does not match AnnData dimensions "
                    f"({adata.n_obs}, {adata.n_vars})"
                )

            # Store gene-level uncertainty in var
            adata.var[f"velocity_{model_name}_uncertainty"] = uncertainty_gene_level

        # Compute velocity graph and embedding using scvelo
        scv.tl.velocity_graph(
            adata,
            vkey=f"velocity_{model_name}",
            xkey="spliced",
            n_jobs=n_jobs,
        )
        scv.tl.velocity_embedding(
            adata,
            vkey=f"velocity_{model_name}",
            basis=basis
        )

        # Store velocity parameters in uns (following legacy pattern)
        adata.uns[f"velocity_{model_name}_params"] = {
            "num_samples": num_samples,
            "basis": basis,
            "model_name": model_name,
            "uncertainty_method": uncertainty_method if compute_uncertainty else None,
        }

        return adata

    @beartype
    def generate_trajectory_times(
        self,
        n_cells: int,
        trajectory_type: str = "linear",
        time_range: Tuple[float, float] = (0.0, 50.0),
        n_trajectories: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate coherent time sequences for trajectory-based validation.

        This method creates time vectors that represent cells sampled along
        coherent developmental trajectories, addressing the issue where
        independent time sampling breaks parameter recovery validation.

        Args:
            n_cells: Total number of cells to generate times for
            trajectory_type: Type of trajectory ('linear', 'branching', 'cyclic')
            time_range: (min_time, max_time) for the trajectory
            n_trajectories: Number of separate trajectories (for branching)
            seed: Random seed for reproducibility

        Returns:
            Tensor of shape [n_cells] with coherent time assignments
        """
        if seed is not None:
            torch.manual_seed(seed)

        min_time, max_time = time_range

        if trajectory_type == "linear":
            # Single linear trajectory with uniform time sampling
            times = torch.linspace(min_time, max_time, n_cells)

        elif trajectory_type == "branching":
            # Multiple trajectories with cells distributed among them
            cells_per_trajectory = n_cells // n_trajectories
            times = []

            for i in range(n_trajectories):
                # Each trajectory has slightly different time range
                traj_min = min_time + i * (max_time - min_time) * 0.1
                traj_max = max_time - (n_trajectories - 1 - i) * (max_time - min_time) * 0.1

                traj_times = torch.linspace(traj_min, traj_max, cells_per_trajectory)
                times.append(traj_times)

            # Handle remaining cells
            remaining_cells = n_cells - cells_per_trajectory * n_trajectories
            if remaining_cells > 0:
                extra_times = torch.linspace(min_time, max_time, remaining_cells)
                times.append(extra_times)

            times = torch.cat(times)

        elif trajectory_type == "cyclic":
            # Cyclic trajectory with periodic time structure
            base_times = torch.linspace(0, 2 * torch.pi, n_cells)
            # Map to time range with cyclic structure
            normalized_times = (torch.sin(base_times) + 1) / 2  # [0, 1]
            times = min_time + normalized_times * (max_time - min_time)

        else:
            raise ValueError(f"Unknown trajectory_type: {trajectory_type}")

        # Add small amount of noise to avoid exact duplicates
        noise = torch.randn(n_cells) * 0.01 * (max_time - min_time)
        times = times + noise

        # Ensure times stay within bounds
        times = torch.clamp(times, min_time, max_time)

        return times

    @beartype
    def generate_validation_datasets(
        self,
        patterns: List[str] = ['activation', 'decay', 'transient', 'sustained'],
        n_genes: int = 5,
        n_cells: int = 200,
        n_parameter_sets: int = 2,
        seed: int = 42,
        compute_dimred: bool = True,
        use_coherent_trajectories: bool = True,
        trajectory_type: str = "linear"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate synthetic validation datasets for parameter recovery validation.

        This method creates synthetic datasets for each specified expression pattern
        using pattern-constrained parameter sampling and predictive data generation.
        Each dataset contains the synthetic data, true parameters, pattern information,
        and metadata needed for validation studies.

        Args:
            patterns: List of expression patterns to generate datasets for
                     ('activation', 'decay', 'transient', 'sustained')
            n_genes: Number of genes in each dataset
            n_cells: Number of cells in each dataset
            n_parameter_sets: Number of parameter sets per pattern
            seed: Base random seed for reproducibility
            compute_dimred: Whether to compute dimensionality reduction (PCA, UMAP, etc.)
            use_coherent_trajectories: Whether to use coherent time trajectories instead of random sampling
            trajectory_type: Type of trajectory ('linear', 'branching', 'cyclic') when using coherent trajectories

        Returns:
            Dictionary with keys like 'activation_1', 'decay_2' containing:
            - 'adata': AnnData object with synthetic count data
            - 'true_parameters': Dictionary of true parameter values
            - 'pattern': Expression pattern name
            - 'metadata': Additional metadata about the dataset

        Example:
            >>> from pyrovelocity.models.modular.factory import create_piecewise_activation_model
            >>> model = create_piecewise_activation_model()
            >>> datasets = model.generate_validation_datasets(
            ...     patterns=['activation', 'decay'],
            ...     n_genes=3,
            ...     n_cells=100,
            ...     n_parameter_sets=2,
            ...     seed=42
            ... )
            >>> print(f"Generated {len(datasets)} datasets")
            >>> for key, data in datasets.items():
            ...     print(f"{key}: {data['adata'].shape}, pattern: {data['pattern']}")
        """
        import torch

        validation_datasets = {}

        # Generate datasets for each pattern and parameter set
        for pattern in patterns:
            for set_id in range(n_parameter_sets):
                # Create unique seed for each dataset
                dataset_seed = seed + len(validation_datasets)
                torch.manual_seed(dataset_seed)

                # Generate pattern-constrained parameters
                true_parameters = self.sample_system_parameters(
                    num_samples=1,
                    constrain_to_pattern=True,
                    pattern=pattern,
                    set_id=set_id,
                    n_genes=n_genes,
                    n_cells=n_cells
                )

                # Generate coherent trajectory times if requested
                if use_coherent_trajectories:
                    # Generate coherent time sequence for this dataset
                    trajectory_times = self.generate_trajectory_times(
                        n_cells=n_cells,
                        trajectory_type=trajectory_type,
                        time_range=(0.0, 50.0),  # Use typical T_M_star range
                        seed=dataset_seed
                    )

                    # Generate synthetic data using the true parameters and coherent times
                    synthetic_adata = self.generate_predictive_samples(
                        num_cells=n_cells,
                        num_genes=n_genes,
                        samples=true_parameters,
                        return_format="anndata",
                        observed_times=trajectory_times
                    )
                else:
                    # Generate synthetic data using the true parameters (original method)
                    synthetic_adata = self.generate_predictive_samples(
                        num_cells=n_cells,
                        num_genes=n_genes,
                        samples=true_parameters,
                        return_format="anndata"
                    )

                if compute_dimred:
                    import scanpy as sc
                    # Add PCA for dimensionality reduction and as fallback for UMAP computation
                    sc.pp.pca(synthetic_adata, random_state=dataset_seed)
                    sc.pp.neighbors(synthetic_adata, n_neighbors=10, random_state=dataset_seed)
                    sc.tl.umap(synthetic_adata, random_state=dataset_seed)
                    sc.tl.leiden(synthetic_adata, random_state=dataset_seed)

                # Create dataset key
                dataset_key = f"{pattern}_{set_id + 1}"

                # Store dataset with metadata
                validation_datasets[dataset_key] = {
                    'adata': synthetic_adata,
                    'true_parameters': true_parameters,
                    'pattern': pattern,
                    'metadata': {
                        'n_genes': n_genes,
                        'n_cells': n_cells,
                        'set_id': set_id,
                        'seed': dataset_seed,
                        'pattern_constraints': True
                    }
                }

        return validation_datasets

    @beartype
    def validate_parameter_recovery(
        self,
        validation_datasets: Dict[str, Dict[str, Any]],
        max_epochs: int = 1000,
        learning_rate: float = 0.01,
        success_threshold: float = 0.9,
        num_posterior_samples: int = 1000,
        adaptive_guide_rank: bool = True,
        seed: int = 42,
        create_predictive_plots: bool = True,
        save_plots_path: Optional[str] = "reports/docs/validation"
    ) -> Dict[str, Any]:
        """
        Validate parameter recovery across multiple synthetic datasets.

        This method performs comprehensive parameter recovery validation by training
        the model on synthetic datasets with known ground truth parameters, then
        comparing the inferred parameters to the true values. It includes training,
        posterior sampling, predictive checks, and metric computation.

        Args:
            validation_datasets: Dictionary of validation datasets from generate_validation_datasets()
            max_epochs: Maximum number of training epochs per dataset
            learning_rate: Learning rate for SVI training
            success_threshold: Correlation threshold for successful parameter recovery
            num_posterior_samples: Number of posterior samples to generate
            adaptive_guide_rank: Whether to use adaptive guide rank based on parameter count
            seed: Random seed for reproducibility
            create_predictive_plots: Whether to create posterior predictive check plots
            save_plots_path: Directory path to save plots (created if doesn't exist)

        Returns:
            Dictionary containing validation results for each dataset with:
            - 'training_result': Training metrics and convergence info
            - 'posterior_samples': Extracted posterior parameter samples
            - 'recovery_metrics': Parameter recovery correlation and error metrics
            - 'success': Boolean indicating if validation passed success threshold
            - 'plot_paths': Paths to generated plots (if create_predictive_plots=True)

        Example:
            >>> from pyrovelocity.models.modular.factory import create_piecewise_activation_model
            >>> model = create_piecewise_activation_model()
            >>> datasets = model.generate_validation_datasets(n_genes=3, n_cells=100)
            >>> results = model.validate_parameter_recovery(
            ...     validation_datasets=datasets,
            ...     max_epochs=500,
            ...     success_threshold=0.85,
            ...     create_predictive_plots=True
            ... )
            >>> for key, result in results.items():
            ...     print(f"{key}: {result['recovery_metrics']['overall_correlation']:.3f}")
        """
        import os

        import torch

        from pyrovelocity.models.modular.inference.config import InferenceConfig
        from pyrovelocity.models.modular.inference.svi import run_svi_inference
        from pyrovelocity.plots.predictive_checks import (
            plot_posterior_predictive_checks,
        )

        # Set seed for reproducibility
        torch.manual_seed(seed)

        validation_results = {}

        # Create plots directory if needed
        if create_predictive_plots and save_plots_path:
            os.makedirs(save_plots_path, exist_ok=True)

        print(f"🚀 Starting parameter recovery validation on {len(validation_datasets)} datasets...")

        for dataset_key, dataset_info in validation_datasets.items():
            print(f"\n📊 Validating {dataset_key} ({dataset_info['pattern']} pattern)...")

            try:
                # Extract dataset components
                adata = dataset_info['adata']
                true_parameters = dataset_info['true_parameters']
                pattern = dataset_info['pattern']

                # Prepare data for training
                u_obs, s_obs = self._prepare_training_data(adata)

                # Calculate adaptive guide rank if requested and update guide
                if adaptive_guide_rank:
                    total_params = sum(
                        param.numel() for param in true_parameters.values()
                        if isinstance(param, torch.Tensor)
                    )
                    guide_rank = max(5, min(50, int(0.12 * total_params)))
                    print(f"  📐 Using adaptive guide rank: {guide_rank} (total params: {total_params})")

                    # Update guide rank if the guide supports it
                    if hasattr(self.guide_model, 'rank'):
                        self.guide_model.rank = guide_rank
                    elif hasattr(self.guide_model, '_rank'):
                        self.guide_model._rank = guide_rank

                # Create inference configuration
                config = InferenceConfig(
                    method="svi",
                    optimizer="adam",
                    learning_rate=learning_rate,
                    num_epochs=max_epochs,
                    seed=seed
                )

                # Ensure guide is created before training
                if not hasattr(self.guide_model, '_guide') or self.guide_model._guide is None:
                    print(f"  🔧 Creating guide for model...")
                    self.guide_model.create_guide(self.forward)

                # Run SVI training
                print(f"  🔄 Training model (max {max_epochs} epochs)...")
                training_state, posterior_samples = run_svi_inference(
                    model=self.forward,
                    guide=self.guide,
                    args=(u_obs, s_obs),
                    config=config,
                    seed=seed
                )

                print(f"  ✅ Training completed in {training_state.step} epochs")
                print(f"     Final ELBO: {training_state.best_loss:.2f}")



                # Compute parameter recovery metrics
                recovery_metrics = self._compute_recovery_metrics(
                    true_parameters, posterior_samples, success_threshold
                )

                # Generate posterior predictive data and plots
                plot_paths = {}
                if create_predictive_plots:
                    print(f"  📈 Creating posterior predictive check plots...")

                    # Generate posterior predictive data
                    posterior_adata = self.generate_predictive_samples(
                        num_cells=adata.n_obs,
                        num_genes=adata.n_vars,
                        samples=posterior_samples,
                        return_format="anndata"
                    )

                    # Process posterior samples for plotting (handle batch dimensions)
                    processed_posterior_samples = self._process_posterior_samples_for_plotting(posterior_samples)

                    # Create dataset-specific subdirectory for plots
                    dataset_plots_path = os.path.join(save_plots_path, dataset_key)
                    os.makedirs(dataset_plots_path, exist_ok=True)

                    # Create comprehensive posterior predictive check plots
                    plot_filename = f"{dataset_key}_posterior_predictive_checks"
                    try:
                        fig = plot_posterior_predictive_checks(
                            model=self,
                            posterior_adata=posterior_adata,
                            posterior_parameters=processed_posterior_samples,
                            save_path=dataset_plots_path,
                            figure_name=plot_filename,
                            combine_individual_pdfs=True,
                        )
                    except Exception as plot_error:
                        print(f"  ❌ Plotting error: {plot_error}")
                        # Continue without plotting
                        fig = None

                    plot_paths['posterior_predictive_checks'] = os.path.join(
                        dataset_plots_path, f"{plot_filename}.png"
                    )

                # Determine validation success
                success = recovery_metrics['overall_correlation'] >= success_threshold
                status_emoji = "✅" if success else "❌"

                print(f"  {status_emoji} Recovery: {recovery_metrics['overall_correlation']:.3f} correlation")
                print(f"     Success rate: {recovery_metrics['success_rate']:.1%}")

                # Store results
                validation_results[dataset_key] = {
                    'pattern': pattern,
                    'training_result': {
                        'num_epochs': training_state.step,
                        'final_elbo': training_state.best_loss,
                        'converged': training_state.step < max_epochs,
                        'loss_history': training_state.loss_history
                    },
                    'posterior_samples': posterior_samples,
                    'recovery_metrics': recovery_metrics,
                    'success': success,
                    'plot_paths': plot_paths
                }

            except Exception as e:
                print(f"  ❌ Validation failed for {dataset_key}: {str(e)}")
                validation_results[dataset_key] = {
                    'pattern': dataset_info['pattern'],
                    'error': str(e),
                    'success': False
                }

        # Compute overall validation summary
        successful_validations = sum(1 for result in validation_results.values() if result.get('success', False))
        total_validations = len(validation_results)
        overall_success_rate = successful_validations / total_validations if total_validations > 0 else 0.0

        print(f"\n🎯 Validation Summary:")
        print(f"   Successful: {successful_validations}/{total_validations} ({overall_success_rate:.1%})")

        # Add summary to results
        validation_results['_summary'] = {
            'total_datasets': total_validations,
            'successful_datasets': successful_validations,
            'overall_success_rate': overall_success_rate,
            'success_threshold': success_threshold
        }

        return validation_results

    @beartype
    def _process_posterior_samples_for_plotting(
        self,
        posterior_samples: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Process posterior samples to make them compatible with plotting functions.

        The plotting functions expect parameters to be flattened 1D arrays, but posterior
        samples from SVI have batch dimensions. This method handles the tensor reshaping
        to make the samples compatible with the plotting code.

        Args:
            posterior_samples: Raw posterior samples from SVI with batch dimensions

        Returns:
            Processed samples suitable for plotting functions
        """
        processed_samples = {}

        for key, value in posterior_samples.items():
            if isinstance(value, torch.Tensor):
                # Handle different tensor shapes
                if value.ndim == 1:
                    # Already 1D, use as-is
                    processed_samples[key] = value
                elif value.ndim == 2:
                    # 2D tensor: [num_samples, param_dim] or [batch_size, param_dim]
                    # Flatten to 1D for plotting
                    processed_samples[key] = value.flatten()
                elif value.ndim == 3:
                    # 3D tensor: [batch_size, num_samples, param_dim]
                    # Remove batch dimension and flatten
                    if value.shape[0] == 1:
                        # Remove batch dimension: [1, num_samples, param_dim] -> [num_samples, param_dim]
                        squeezed = value.squeeze(0)
                        processed_samples[key] = squeezed.flatten()
                    else:
                        # Multiple batches: flatten everything
                        processed_samples[key] = value.flatten()
                else:
                    # Higher dimensions: flatten everything
                    processed_samples[key] = value.flatten()
            else:
                # Non-tensor values: keep as-is
                processed_samples[key] = value

        return processed_samples

    @beartype
    def sample_system_parameters(
        self,
        num_samples: int = 1000,
        constrain_to_pattern: bool = False,
        pattern: Optional[str] = None,
        set_id: Optional[int] = None,
        n_genes: Optional[int] = None,
        n_cells: Optional[int] = None,
        max_attempts: int = 10000,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Sample system parameters with optional pattern constraints.

        This method provides a clean user interface for parameter sampling while
        delegating to the prior model component. It enables controlled parameter
        generation for validation studies and synthetic data creation.

        The method supports both unconstrained sampling (using full prior ranges)
        and pattern-constrained sampling (enforcing specific gene expression patterns).
        This is essential for parameter recovery validation studies where we need
        to generate parameters that produce known expression patterns.

        Args:
            num_samples: Number of parameter sets to generate
            constrain_to_pattern: Whether to apply pattern constraints
            pattern: Expression pattern to constrain to ('activation', 'decay', 'transient', 'sustained')
            set_id: Identifier for parameter set (for reproducibility)
            n_genes: Number of genes (default: 2)
            n_cells: Number of cells (default: 50)
            max_attempts: Maximum attempts to find valid parameters when constraining
            **kwargs: Additional arguments passed to the prior model

        Returns:
            Dictionary of parameter tensors with shape [num_samples, ...]

        Raises:
            ValueError: If pattern is invalid or constraints cannot be satisfied

        Examples:
            >>> # Unconstrained parameter sampling
            >>> from pyrovelocity.models.modular.factory import create_piecewise_activation_model
            >>> model = create_piecewise_activation_model()
            >>> prior_samples = model.sample_system_parameters(
            ...     num_samples=1000,
            ...     constrain_to_pattern=False,
            ...     n_genes=2,
            ...     n_cells=50
            ... )
            >>> print(f"Generated {len(prior_samples)} parameter types")
            >>> print(f"alpha_off shape: {prior_samples['alpha_off'].shape}")

            >>> # Pattern-constrained parameter sampling
            >>> activation_samples = model.sample_system_parameters(
            ...     num_samples=100,
            ...     constrain_to_pattern=True,
            ...     pattern='activation',
            ...     set_id=1,
            ...     n_genes=2,
            ...     n_cells=50
            ... )
            >>> # Verify activation pattern constraints
            >>> alpha_off = activation_samples['alpha_off']
            >>> alpha_on = activation_samples['alpha_on']
            >>> fold_change = alpha_on / alpha_off
            >>> assert torch.all(alpha_off < 0.2), "Activation: alpha_off should be < 0.2"
            >>> assert torch.all(alpha_on > 1.5), "Activation: alpha_on should be > 1.5"
            >>> assert torch.all(fold_change > 7.5), "Activation: fold_change should be > 7.5"
        """
        # Delegate to the prior model component
        return self.prior_model.sample_system_parameters(
            num_samples=num_samples,
            constrain_to_pattern=constrain_to_pattern,
            pattern=pattern,
            set_id=set_id,
            n_genes=n_genes,
            n_cells=n_cells,
            max_attempts=max_attempts,
            **kwargs
        )

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
        num_genes: int
    ) -> Dict[str, torch.Tensor]:
        """
        Process parameter samples for posterior predictive sampling.

        This method handles tensor reshaping for single parameter sets (backward compatibility).
        For multiple posterior samples, use _process_single_parameter_sample instead.

        Args:
            samples: Dictionary of parameter samples (should be single parameter set)
            num_cells: Target number of cells
            num_genes: Target number of genes

        Returns:
            Dictionary of processed parameters ready for model context
        """
        processed = {}

        for key, value in samples.items():
            if isinstance(value, torch.Tensor):
                # Handle tensor processing for single parameter sets
                if value.ndim == 3:
                    # 3D tensor: take first sample only (no averaging)
                    processed_value = value[0].squeeze(0)
                elif value.ndim == 2:
                    # 2D tensor: check if it's a single parameter set
                    if value.shape[0] == 1:
                        # Single parameter set: squeeze batch dimension
                        processed_value = value.squeeze(0)
                    else:
                        # Multiple samples: take first sample only (no averaging)
                        processed_value = value[0]
                elif value.ndim >= 1:
                    # 1D or 2D tensor: squeeze out batch dimensions
                    processed_value = value.squeeze()
                else:
                    processed_value = value

                # Handle cell-specific parameters
                if key in ["t_star", "cell_time"]:
                    # These parameters should have shape [num_cells] or be expandable to it
                    if processed_value.numel() == 1:
                        # Single value: expand to all cells
                        processed[key] = processed_value.expand(num_cells)
                    elif processed_value.shape == torch.Size([num_cells]):
                        # Already correct shape
                        processed[key] = processed_value
                    elif processed_value.numel() >= num_cells:
                        # Take first num_cells values
                        processed[key] = processed_value.flatten()[:num_cells]
                    else:
                        # Repeat to match num_cells
                        repeat_factor = (num_cells + processed_value.numel() - 1) // processed_value.numel()
                        repeated = processed_value.repeat(repeat_factor)
                        processed[key] = repeated.flatten()[:num_cells]
                elif key in ["R_on", "alpha_on", "alpha_off", "gamma_star"] and processed_value.numel() == num_genes:
                    # Gene-specific parameters should have shape [num_genes]
                    processed[key] = processed_value.reshape(num_genes)
                else:
                    # Other parameters: use as-is
                    processed[key] = processed_value
            else:
                processed[key] = value

        return processed

    @beartype
    def _process_single_parameter_sample(
        self,
        sample: Dict[str, torch.Tensor],
        num_cells: int,
        num_genes: int
    ) -> Dict[str, torch.Tensor]:
        """
        Process a SINGLE parameter sample for posterior predictive sampling.

        This method handles tensor reshaping for a single parameter vector
        WITHOUT any averaging operations. This is the mathematically correct
        approach for Bayesian posterior predictive checking.

        Args:
            sample: Dictionary containing a single parameter sample
            num_cells: Target number of cells
            num_genes: Target number of genes

        Returns:
            Dictionary of processed parameters ready for model context
        """
        processed = {}

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                # Handle complex tensor shapes from posterior samples
                # The tensor has shape [1, ...] where we need to extract the single sample

                # Remove the first dimension (batch dimension) and squeeze all singleton dimensions
                if value.shape[0] == 1:
                    # Remove batch dimension and squeeze singleton dimensions
                    processed_value = value.squeeze()
                else:
                    # This shouldn't happen for single samples, but handle gracefully
                    processed_value = value[0].squeeze()

                # Handle parameter-specific reshaping based on expected shapes
                if key in ["t_star", "cell_time"]:
                    # These parameters should have shape [num_cells] or be expandable to it
                    if processed_value.numel() == 1:
                        # Single value: expand to all cells
                        processed[key] = processed_value.expand(num_cells)
                    elif processed_value.numel() == num_cells:
                        # Already correct size: reshape to [num_cells]
                        processed[key] = processed_value.reshape(num_cells)
                    elif processed_value.numel() > num_cells:
                        # Take first num_cells values
                        processed[key] = processed_value.flatten()[:num_cells]
                    else:
                        # Repeat to match num_cells
                        repeat_factor = (num_cells + processed_value.numel() - 1) // processed_value.numel()
                        repeated = processed_value.repeat(repeat_factor)
                        processed[key] = repeated.flatten()[:num_cells]

                elif key in ["R_on", "alpha_on", "alpha_off", "gamma_star", "t_on_star", "delta_star", "U_0i"]:
                    # Gene-specific parameters should have shape [num_genes]
                    if processed_value.numel() == num_genes:
                        processed[key] = processed_value.reshape(num_genes)
                    elif processed_value.numel() == 1:
                        # Single value: expand to all genes
                        processed[key] = processed_value.expand(num_genes)
                    else:
                        # Use flattened version and take first num_genes elements
                        flattened = processed_value.flatten()
                        if flattened.numel() >= num_genes:
                            processed[key] = flattened[:num_genes]
                        else:
                            # Repeat to match num_genes
                            repeat_factor = (num_genes + flattened.numel() - 1) // flattened.numel()
                            repeated = flattened.repeat(repeat_factor)
                            processed[key] = repeated[:num_genes]

                elif key in ["lambda_j"]:
                    # Cell-specific parameters (like capture efficiency)
                    if processed_value.numel() == num_cells:
                        processed[key] = processed_value.reshape(num_cells)
                    elif processed_value.numel() == 1:
                        # Single value: expand to all cells
                        processed[key] = processed_value.expand(num_cells)
                    else:
                        # Use flattened version and take first num_cells elements
                        flattened = processed_value.flatten()
                        if flattened.numel() >= num_cells:
                            processed[key] = flattened[:num_cells]
                        else:
                            # Repeat to match num_cells
                            repeat_factor = (num_cells + flattened.numel() - 1) // flattened.numel()
                            repeated = flattened.repeat(repeat_factor)
                            processed[key] = repeated[:num_cells]

                else:
                    # Other parameters: use squeezed version
                    processed[key] = processed_value
            else:
                processed[key] = value

        return processed

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
