import anndata
import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import scanpy as sc
import os
from pathlib import Path

from pyrovelocity.models.jax.factory.factory import create_piecewise_activation_model
from pyrovelocity.models.jax.inference.config import create_inference_config
from pyrovelocity.models.jax.inference.unified import (
    run_inference, 
)
from pyrovelocity.plots.predictive_checks import (
    plot_prior_predictive_checks,
    plot_posterior_predictive_checks,
)
from pyrovelocity.utils import print_anndata
from numpyro.infer import Predictive
from pyrovelocity.models.metadata import (
    get_parameter_display_names,
    get_parameter_short_labels,
)


RANDOM_SEED = 42
REPORTS_SAVE_PATH = "reports/docs/posterior_predictive_jax"

# ============================================================================
# CONFIGURATION: Select which inference method to run
# ============================================================================
# Edit this section to choose which inference method to test.
# Available options:

AVAILABLE_METHODS = {
    "svi_auto_normal": {
        "config": create_inference_config(
            method="svi",
            num_epochs=1000,
            learning_rate=0.01,
            guide_type="auto_normal",
            early_stopping=True,
            early_stopping_patience=10,
        ),
        "guide_type": "auto_normal"
    },
    "svi_auto_diagonal_normal": {
        "config": create_inference_config(
            method="svi",
            num_epochs=1000,
            learning_rate=0.01,
            guide_type="auto_diagonal_normal",
            early_stopping=True,
            early_stopping_patience=10,
        ),
        "guide_type": "auto_diagonal_normal"
    },
    "svi_auto_multivariate_normal": {
        "config": create_inference_config(
            method="svi",
            num_epochs=1000,
            learning_rate=0.01,
            guide_type="auto_multivariate_normal",
            early_stopping=True,
            early_stopping_patience=10,
        ),
        "guide_type": "auto_multivariate_normal"
    },
    "svi_auto_lowrank_multivariate_normal": {
        "config": create_inference_config(
            method="svi",
            num_epochs=1000,
            learning_rate=0.01,
            guide_type="auto_lowrank_multivariate_normal",
            early_stopping=True,
            early_stopping_patience=10,
        ),
        "guide_type": "auto_lowrank_multivariate_normal"
    },
    "mcmc_nuts": {
        "config": create_inference_config(
            method="mcmc",
            num_samples=500,
            num_warmup=250,
            num_chains=1,
        ),
        "guide_type": None
    }
}

# ============================================================================
# EDIT THIS LINE to choose which method to run:
# ============================================================================
SELECTED_METHOD = "svi_auto_lowrank_multivariate_normal"

# Validate selection
if SELECTED_METHOD not in AVAILABLE_METHODS:
    raise ValueError(f"Invalid method '{SELECTED_METHOD}'. Available methods: {list(AVAILABLE_METHODS.keys())}")

METHOD_CONFIG = AVAILABLE_METHODS[SELECTED_METHOD]

# Set random seeds for reproducibility
numpyro.set_platform("cpu")  # Ensure CPU execution for consistency
jax.config.update("jax_enable_x64", True)  # Enable double precision

# Master seed - user controls this one value
MASTER_SEED = RANDOM_SEED
rng_key = jax.random.PRNGKey(MASTER_SEED)
np.random.seed(RANDOM_SEED)

# Split keys for major workflow components
rng_key, rng_key_data = jax.random.split(rng_key)
rng_key, rng_key_inference = jax.random.split(rng_key)
rng_key, rng_key_prediction = jax.random.split(rng_key)

print("=" * 70)
print("🚀 JAX Single-Method Posterior Predictive Check Workflow")
print("=" * 70)
print(f"🎯 Random seed: {RANDOM_SEED}")
print(f"🔬 Selected inference method: {SELECTED_METHOD}")
print(f"📊 Guide type: {METHOD_CONFIG['guide_type'] or 'N/A (MCMC)'}")
print("=" * 70)

# Step 1: Generate sample data
print(f"\n📊 Step 1: Generating prior predictive data (seed: {RANDOM_SEED})...")

# Create model for sample data generation
model = create_piecewise_activation_model()

# Create a metadata-aware wrapper for the JAX model
class MetadataAwareJAXModel:
    """Wrapper to provide metadata to JAX models for consistent plot labeling."""
    
    def __init__(self, jax_model):
        self._jax_model = jax_model
        # Store the component name for metadata lookup
        self._component_name = "piecewise_activation_prior"
    
    def __call__(self, *args, **kwargs):
        """Forward calls to the underlying JAX model."""
        return self._jax_model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Forward attribute access to the underlying JAX model."""
        return getattr(self._jax_model, name)
    
    @property
    def component_name(self):
        """Provide component name for metadata lookup."""
        return self._component_name

# Wrap the model with metadata support
model = MetadataAwareJAXModel(model)

# Generate synthetic data with known true parameters from prior
print("Generating synthetic data...")

# Create synthetic data dimensions
num_cells = 200
num_genes = 100

# Generate dummy observations for the model to get prior predictive samples
# u_obs = jnp.ones((1, num_cells, num_genes))
# s_obs = jnp.ones((1, num_cells, num_genes))

# Use NumPyro's Predictive class for deterministic sampling
prior_predictive = Predictive(model, num_samples=1)

# Generate single prior sample for "observed" data using proper key management  
prior_samples = prior_predictive(rng_key_data, u_obs=None, s_obs=None, num_cells=num_cells, num_genes=num_genes)

# Use the generated Poisson samples as our "observed" data (NOT expected values!)
# This is critical for realistic Poisson noise and proper scatter in phase space plots
u_obs_shape = prior_samples["u_obs"].shape
s_obs_shape = prior_samples["s_obs"].shape

if len(u_obs_shape) == 5:
    # Shape: [batch, sample, dim1, cells, genes] - extract the last 2 dimensions
    u_observed = prior_samples["u_obs"][0, 0, 0, :, :] 
    s_observed = prior_samples["s_obs"][0, 0, 0, :, :]
elif len(u_obs_shape) == 4:
    # Expected shape: [batch, sample, cells, genes]
    u_observed = prior_samples["u_obs"][0, 0, :, :] 
    s_observed = prior_samples["s_obs"][0, 0, :, :]
elif len(u_obs_shape) == 3:
    # Shape might be [sample, cells, genes] 
    u_observed = prior_samples["u_obs"][0, :, :] 
    s_observed = prior_samples["s_obs"][0, :, :]
else:
    # Shape might be [cells, genes]
    u_observed = prior_samples["u_obs"]
    s_observed = prior_samples["s_obs"]

# Create AnnData object for training
prior_predictive_adata = anndata.AnnData(
    X=np.array(s_observed),  # Use spliced as main expression
    layers={
        "unspliced": np.array(u_observed),
        "spliced": np.array(s_observed)
    }
)

# Store t_star as latent_time for proper time coordinate visualization
if "t_star" in prior_samples:
    # Adapt t_star extraction based on its shape
    t_star_shape = prior_samples["t_star"].shape
    if len(t_star_shape) == 3:
        # Shape is (1, 200, 1) - extract the middle dimension
        t_star_values = prior_samples["t_star"][0, :, 0]  # [cells]
    elif len(t_star_shape) == 2:
        t_star_values = prior_samples["t_star"][0, :]  # [cells]
    else:
        t_star_values = prior_samples["t_star"]  # Already 1D
        
    prior_predictive_adata.obs["latent_time"] = np.array(t_star_values)
    print(f"✅ Stored t_star as latent_time in prior data: shape {t_star_values.shape}, range [{t_star_values.min():.3f}, {t_star_values.max():.3f}]")
else:
    print("⚠️ No t_star found in prior samples")

# Store true parameters for validation
prior_predictive_adata.uns["true_parameters"] = {}
for key_name, value in prior_samples.items():
    if key_name not in ["u_expected", "s_expected"]:
        # Store single-sample parameters
        if hasattr(value, "shape") and len(value.shape) > 0:
            prior_predictive_adata.uns["true_parameters"][key_name] = np.array(value[0])
        else:
            prior_predictive_adata.uns["true_parameters"][key_name] = np.array(value)

print(f"\n🗺️ Computing UMAP and clustering for prior predictive data...")
sc.pp.pca(prior_predictive_adata, random_state=RANDOM_SEED)
sc.pp.neighbors(prior_predictive_adata, n_neighbors=10, random_state=RANDOM_SEED)
sc.tl.umap(prior_predictive_adata, random_state=RANDOM_SEED)
sc.tl.leiden(prior_predictive_adata, random_state=RANDOM_SEED)

print("✅ Prior predictive data generated:")
print_anndata(prior_predictive_adata)

# Extract parameters for plotting functions
prior_parameter_samples = {}
if "true_parameters" in prior_predictive_adata.uns:
    for param_key, value in prior_predictive_adata.uns['true_parameters'].items():
        # Normalize parameter names to match metadata registry
        normalized_key = param_key
        if param_key == "boundaryconcentration":
            normalized_key = "boundary_concentration"
        prior_parameter_samples[normalized_key] = value  # Let plotting functions handle conversion

# Check if prior predictive plots already exist (caching)
sample_data_path = Path(REPORTS_SAVE_PATH) / str(RANDOM_SEED) / "sample_data"
key_files = [
    f"01_posterior_predictive_check_sample_data_jax_{RANDOM_SEED}.pdf",
    f"combined_prior_predictive_checks_jax_{RANDOM_SEED}.pdf"
]
plots_exist = sample_data_path.exists() and all((sample_data_path / f).exists() for f in key_files)

if plots_exist:
    print(f"📋 Found cached prior predictive plots, skipping regeneration...")
    print(f"   Cache location: {sample_data_path}")
else:
    print(f"🎨 Generating prior predictive plots...")
    # Generate prior predictive plots (this is the expensive part)
    # Pass the wrapped model which provides metadata access
    plot_prior_predictive_checks(
        model=model,
        prior_adata=prior_predictive_adata,
        prior_parameters=prior_parameter_samples,
        figsize=(7.5, 5.0),
        save_path=f"{REPORTS_SAVE_PATH}/{RANDOM_SEED}/sample_data",
        figure_name=f"posterior_predictive_check_sample_data_jax_{RANDOM_SEED}",
        combine_individual_pdfs=True,
        default_fontsize=5,
        num_genes=10,
        true_parameters_adata=prior_predictive_adata,
    )
    print(f"✅ Prior predictive plots generated")


# Step 2: Create model with selected inference method and train
print(f"\n🔬 Step 2: Model Training with {SELECTED_METHOD}")

# Create method-specific output directory
method_save_path = f"{REPORTS_SAVE_PATH}/{RANDOM_SEED}/{SELECTED_METHOD}"
os.makedirs(method_save_path, exist_ok=True)

# Create model
base_model = create_piecewise_activation_model()
# Wrap with metadata support for consistent parameter labeling
model = MetadataAwareJAXModel(base_model)
print(f"✅ Created JAX model for {SELECTED_METHOD}")

print(f"🎯 Training with {SELECTED_METHOD}...")

# Prepare data for JAX model
u_obs_batch = jnp.expand_dims(jnp.array(prior_predictive_adata.layers["unspliced"]), 0)  # [1, cells, genes]
s_obs_batch = jnp.expand_dims(jnp.array(prior_predictive_adata.layers["spliced"]), 0)  # [1, cells, genes]

# Run inference using proper seed management
config = METHOD_CONFIG['config']

# Run inference using unified interface
inference_object, inference_state = run_inference(
    model=model,
    args=(u_obs_batch, s_obs_batch),
    kwargs={},
    config=config,
    key=rng_key_inference
)
    
print(f"✅ {SELECTED_METHOD} training completed")
    
# Extract posterior samples
posterior_samples = inference_state.posterior_samples
    
# Attach training state to model for compatibility with plotting functions
class ModelState:
    def __init__(self, inference_state):
        self.inference_state = inference_state
    
# Attach state to model
model.state = ModelState(inference_state)
    
print(f"✅ Generated {len(posterior_samples)} types of posterior parameters")
for key_name in sorted(posterior_samples.keys()):
    param_shape = posterior_samples[key_name].shape if hasattr(posterior_samples[key_name], 'shape') else len(posterior_samples[key_name])
    print(f"    {key_name}: {param_shape}")

# Step 3: Generate posterior predictive data using posterior samples
print(f"\n📊 Step 3: Generating posterior predictive data for {SELECTED_METHOD}...")
print("  Using full posterior samples to generate synthetic data with uncertainty")

# Generate posterior predictive data with uncertainty using NumPyro Predictive
# Create posterior predictive using NumPyro
predictive = Predictive(
    model, 
    posterior_samples=posterior_samples,
    num_samples=1  # Generate one sample for visualization
)
    
# Generate posterior predictive samples
posterior_predictive_samples = predictive(rng_key_prediction, u_obs=u_obs_batch, s_obs=s_obs_batch)
    
# Extract expected counts
u_posterior_expected = posterior_predictive_samples["u_expected"][0, 0, :, :]  # [cells, genes]
s_posterior_expected = posterior_predictive_samples["s_expected"][0, 0, :, :]  # [cells, genes]
    
# Create AnnData object for posterior predictive data
posterior_predictive_adata = anndata.AnnData(
    X=np.array(s_posterior_expected),  # Use spliced as main expression
    layers={
        "unspliced": np.array(u_posterior_expected),
        "spliced": np.array(s_posterior_expected)
    }
)
    
# Store t_star as latent_time for proper time coordinate visualization
if "t_star" in posterior_predictive_samples:
    t_star_values = posterior_predictive_samples["t_star"][0, :]  # [cells]
    posterior_predictive_adata.obs["latent_time"] = np.array(t_star_values)
    print(f"✅ Stored t_star as latent_time: shape {t_star_values.shape}, range [{t_star_values.min():.3f}, {t_star_values.max():.3f}]")
else:
    print("⚠️ No t_star found in posterior predictive samples")
    
print("✅ Posterior predictive data generated:")
print_anndata(posterior_predictive_adata)

# Copy UMAP coordinates for consistent visualization
print(f"\n🗺️ Copying UMAP coordinates from prior predictive data for consistent visualization...")
posterior_predictive_adata.obsm['X_umap'] = prior_predictive_adata.obsm['X_umap'].copy()
posterior_predictive_adata.uns['umap'] = prior_predictive_adata.uns['umap'].copy()

# Copy Leiden cluster labels from prior predictive data
print(f"📋 Copying Leiden cluster labels from prior predictive data...")
posterior_predictive_adata.obs['leiden'] = prior_predictive_adata.obs['leiden'].copy()
posterior_predictive_adata.uns['leiden'] = prior_predictive_adata.uns['leiden'].copy()

# Recompute PCA and neighbors for posterior predictive data
sc.pp.pca(posterior_predictive_adata, random_state=RANDOM_SEED)
sc.pp.neighbors(posterior_predictive_adata, n_neighbors=10, random_state=RANDOM_SEED)

# Step 4: Generate posterior predictive check plots
print(f"\n🎨 Step 4: Creating posterior predictive check plots for {SELECTED_METHOD}...")

# Extract posterior parameters for plotting (let plotting functions handle conversion)
posterior_parameter_samples = {}
for key_name, value in posterior_samples.items():
    if key_name not in ["u_expected", "s_expected"]:
        # Normalize parameter names to match metadata registry
        normalized_key = key_name
        if key_name == "boundaryconcentration":
            normalized_key = "boundary_concentration"
        
        if hasattr(value, "shape") and len(value.shape) > 1:
            # Take mean across samples for plotting
            posterior_parameter_samples[normalized_key] = np.mean(np.array(value), axis=0)
        else:
            posterior_parameter_samples[normalized_key] = np.array(value)

_ = plot_posterior_predictive_checks(
    model=model,  # Uses wrapped model with metadata support
    posterior_adata=posterior_predictive_adata,
    posterior_parameters=posterior_parameter_samples,
    figsize=(7.5, 5.0),
    save_path=method_save_path,
    figure_name=f"piecewise_activation_posterior_checks_jax_{SELECTED_METHOD}_{RANDOM_SEED}",
    combine_individual_pdfs=True,
    default_fontsize=5,
    observed_adata=prior_predictive_adata,
    num_genes=10,
    true_parameters_adata=prior_predictive_adata,  # Contains true parameters for validation
)

print(f"\n{'='*70}")
print(f"✅ JAX {SELECTED_METHOD} Posterior Predictive Check Completed!")
print(f"{'='*70}")
print(f"📁 Results saved to: {method_save_path}")
print(f"🎯 Random seed used: {RANDOM_SEED}")
print(f"🔬 Inference method: {SELECTED_METHOD}")
print(f"📊 Guide type: {METHOD_CONFIG['guide_type'] or 'N/A (MCMC)'}")
print(f"🧮 JAX/NumPyro backend: CPU with double precision")
