import torch
import pyro
import numpy as np
import scanpy as sc
import os
from pathlib import Path

from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.models.modular.inference.config import InferenceConfig
from pyrovelocity.plots.predictive_checks import (
    plot_prior_predictive_checks,
    plot_posterior_predictive_checks,
)
from pyrovelocity.utils import print_anndata


RANDOM_SEED = 42
REPORTS_SAVE_PATH = "reports/docs/posterior_predictive"

# ============================================================================
# CONFIGURATION: Select which inference method to run
# ============================================================================
# Edit this section to choose which inference method to test.
# Available options:

AVAILABLE_METHODS = {
    "svi_autonormal": {
        "config": InferenceConfig(
            method="svi",
            num_epochs=1000,
            learning_rate=0.01,
            guide="AutoNormal",
            early_stopping=True,
            early_stopping_patience=10,
        ),
        "guide_type": "AutoNormal"
    },
    "svi_autodiagonalnormal": {
        "config": InferenceConfig(
            method="svi",
            num_epochs=1000,
            learning_rate=0.01,
            guide="AutoDiagonalNormal",
            early_stopping=True,
            early_stopping_patience=10,
        ),
        "guide_type": "AutoDiagonalNormal"
    },
    "svi_automultivariatenormal": {
        "config": InferenceConfig(
            method="svi",
            num_epochs=1000,
            learning_rate=0.01,
            guide="AutoMultivariateNormal",
            early_stopping=True,
            early_stopping_patience=10,
        ),
        "guide_type": "AutoMultivariateNormal"
    },
    "svi_autolowrankmultivariatenormal": {
        "config": InferenceConfig(
            method="svi",
            num_epochs=1000,
            learning_rate=0.01,
            guide="AutoLowRankMultivariateNormal",
            early_stopping=True,
            early_stopping_patience=10,
        ),
        "guide_type": "AutoLowRankMultivariateNormal"
    },
    "mcmc_nuts": {
        "config": InferenceConfig(
            method="mcmc",
            num_samples=500,
            num_warmup=250,
            kernel="nuts",
            num_chains=1,
        ),
        "guide_type": None  # MCMC doesn't use guides
    }
}

# ============================================================================
# EDIT THIS LINE to choose which method to run:
# ============================================================================
SELECTED_METHOD = "svi_autonormal"  # Change this to test different methods

# Validate selection
if SELECTED_METHOD not in AVAILABLE_METHODS:
    raise ValueError(f"Invalid method '{SELECTED_METHOD}'. Available methods: {list(AVAILABLE_METHODS.keys())}")

METHOD_CONFIG = AVAILABLE_METHODS[SELECTED_METHOD]

torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("🚀 Single-Method Posterior Predictive Check Workflow")
print("=" * 70)
print(f"🎯 Random seed: {RANDOM_SEED}")
print(f"🔬 Selected inference method: {SELECTED_METHOD}")
print(f"📊 Guide type: {METHOD_CONFIG['guide_type'] or 'N/A (MCMC)'}")
print("=" * 70)

# Step 1: Generate sample data
print(f"\n📊 Step 1: Generating prior predictive data (seed: {RANDOM_SEED})...")

# Create model for sample data generation
model = create_piecewise_activation_model()

# Generate synthetic data with known true parameters from prior
prior_predictive_adata = model.generate_predictive_samples(
    num_cells=200,
    num_genes=100,
    num_samples=1,
    return_format="anndata"
)

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
    for key, value in prior_predictive_adata.uns['true_parameters'].items():
        prior_parameter_samples[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value

# Check if prior predictive plots already exist (caching)
sample_data_path = Path(REPORTS_SAVE_PATH) / str(RANDOM_SEED) / "sample_data"
key_files = [
    f"01_posterior_predictive_check_sample_data_{RANDOM_SEED}.pdf",
    f"combined_prior_predictive_checks_{RANDOM_SEED}.pdf"
]
plots_exist = sample_data_path.exists() and all((sample_data_path / f).exists() for f in key_files)

if plots_exist:
    print(f"📋 Found cached prior predictive plots, skipping regeneration...")
    print(f"   Cache location: {sample_data_path}")
else:
    print(f"🎨 Generating prior predictive plots...")
    # Generate prior predictive plots (this is the expensive part)
    plot_prior_predictive_checks(
        model=model,
        prior_adata=prior_predictive_adata,
        prior_parameters=prior_parameter_samples,
        figsize=(7.5, 5.0),
        save_path=f"{REPORTS_SAVE_PATH}/{RANDOM_SEED}/sample_data",
        figure_name=f"posterior_predictive_check_sample_data_{RANDOM_SEED}",
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

# Create model with appropriate guide type
guide_type = METHOD_CONFIG['guide_type']
if guide_type is not None:
    # SVI method
    model = create_piecewise_activation_model(guide_type=guide_type)
    print(f"✅ Created SVI model with {guide_type} guide")
else:
    # MCMC method
    model = create_piecewise_activation_model()
    print(f"✅ Created MCMC model")

print(f"🎯 Training with {SELECTED_METHOD}...")

# Set seed for this method
config = METHOD_CONFIG['config']
config_with_seed = InferenceConfig(
    method=config.method,
    num_epochs=getattr(config, 'num_epochs', None),
    learning_rate=getattr(config, 'learning_rate', None),
    guide=getattr(config, 'guide', None),
    early_stopping=getattr(config, 'early_stopping', None),
    early_stopping_patience=getattr(config, 'early_stopping_patience', None),
    num_samples=getattr(config, 'num_samples', None),
    num_warmup=getattr(config, 'num_warmup', None),
    kernel=getattr(config, 'kernel', None),
    num_chains=getattr(config, 'num_chains', None),
    seed=RANDOM_SEED
)

trained_model = model.train(
    adata=prior_predictive_adata,
    config=config_with_seed,
    seed=RANDOM_SEED
)
print(f"✅ {SELECTED_METHOD} training completed")


# Step 3: Generate posterior samples from trained model
print(f"\n🔬 Step 3: Generating posterior samples from {SELECTED_METHOD}...")

# Generate posterior samples using the trained model
posterior_parameter_samples = trained_model.generate_posterior_samples(
    adata=prior_predictive_adata,
    num_samples=30,
    return_tensors=True
)

print(f"✅ Generated {len(posterior_parameter_samples)} types of posterior parameters")
for key in sorted(posterior_parameter_samples.keys()):
    param_shape = posterior_parameter_samples[key].shape if hasattr(posterior_parameter_samples[key], 'shape') else len(posterior_parameter_samples[key])
    print(f"    {key}: {param_shape}")


# Step 4: Generate posterior predictive data using posterior samples
print(f"\n📊 Step 4: Generating posterior predictive data for {SELECTED_METHOD}...")
print("  Using full posterior samples to generate synthetic data with uncertainty")

# Generate posterior predictive data with uncertainty
posterior_predictive_adata = trained_model.generate_predictive_samples(
    num_cells=prior_predictive_adata.n_obs,
    num_genes=prior_predictive_adata.n_vars,
    samples=posterior_parameter_samples,  # Use torch tensors directly
    return_format="anndata"
)

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

# Step 5: Generate posterior predictive check plots
print(f"\n🎨 Step 5: Creating posterior predictive check plots for {SELECTED_METHOD}...")

_ = plot_posterior_predictive_checks(
    model=trained_model,
    posterior_adata=posterior_predictive_adata,
    posterior_parameters=posterior_parameter_samples,
    figsize=(7.5, 5.0),
    save_path=method_save_path,
    figure_name=f"piecewise_activation_posterior_checks_{SELECTED_METHOD}_{RANDOM_SEED}",
    combine_individual_pdfs=True,
    default_fontsize=5,
    observed_adata=prior_predictive_adata,
    num_genes=10,
    true_parameters_adata=prior_predictive_adata,  # Contains true parameters for validation
)

print(f"\n{'='*70}")
print(f"✅ {SELECTED_METHOD} Posterior Predictive Check Completed!")
print(f"{'='*70}")
print(f"📁 Results saved to: {method_save_path}")
print(f"🎯 Random seed used: {RANDOM_SEED}")
print(f"🔬 Inference method: {SELECTED_METHOD}")
print(f"📊 Guide type: {METHOD_CONFIG['guide_type'] or 'N/A (MCMC)'}")

# Print directory structure
print(f"\n📂 Output directory structure:")
print(f"   {REPORTS_SAVE_PATH}/{RANDOM_SEED}/")
print(f"   ├── sample_data/          # Cached prior predictive plots")
print(f"   └── {SELECTED_METHOD}/     # {SELECTED_METHOD} results")
