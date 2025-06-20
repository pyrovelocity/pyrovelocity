import torch
import pyro
import numpy as np
import scanpy as sc
import os
from pathlib import Path
from typing import Tuple

from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.models.modular.inference.config import InferenceConfig
from pyrovelocity.plots.predictive_checks import (
    plot_prior_predictive_checks,
    plot_posterior_predictive_checks,
)
from pyrovelocity.utils import print_anndata


RANDOM_SEED = 42
REPORTS_SAVE_PATH = "reports/docs/posterior_predictive"

# Define inference methods to test
INFERENCE_METHODS = [
    {
        "name": "svi_autonormal",
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
    {
        "name": "svi_autodiagonalnormal",
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
    {
        "name": "svi_automultivariatenormal",
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
    {
        "name": "svi_autolowrankmultivariatenormal",
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
    {
        "name": "mcmc_nuts",
        "config": InferenceConfig(
            method="mcmc",
            num_samples=500,
            num_warmup=250,
            kernel="nuts",
            num_chains=1,
        ),
        "guide_type": None  # MCMC doesn't use guides
    }
]

def check_sample_data_cache(seed: int, save_path: str) -> bool:
    """Check if sample data plots already exist for the given seed."""
    sample_data_path = Path(save_path) / str(seed) / "sample_data"
    if not sample_data_path.exists():
        return False

    # Check for key files that indicate complete sample data generation
    key_files = [
        "01_posterior_predictive_check_sample_data_42.pdf",
        "combined_prior_predictive_checks_42.pdf"
    ]

    for file_name in key_files:
        if not (sample_data_path / file_name).exists():
            return False

    return True


def generate_sample_data_if_needed(seed: int, save_path: str) -> Tuple:
    """Generate sample data and prior predictive plots if not cached."""
    cache_exists = check_sample_data_cache(seed, save_path)

    if cache_exists:
        print(f"📋 Found cached sample data for seed {seed}, skipping regeneration...")
        print(f"   Cache location: {Path(save_path) / str(seed) / 'sample_data'}")

        # Load the cached prior predictive data
        # For now, we'll regenerate it since loading AnnData from cache is complex
        # In a production system, you'd want to save/load the AnnData object
        print(f"   Note: Still regenerating AnnData object (caching AnnData objects requires additional implementation)")
        cache_exists = False

    if not cache_exists:
        print(f"📊 Generating prior predictive data (seed: {seed})...")

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
        sc.pp.pca(prior_predictive_adata, random_state=seed)
        sc.pp.neighbors(prior_predictive_adata, n_neighbors=10, random_state=seed)
        sc.tl.umap(prior_predictive_adata, random_state=seed)
        sc.tl.leiden(prior_predictive_adata, random_state=seed)

        print("✅ Prior predictive data generated:")
        print_anndata(prior_predictive_adata)

        # Extract parameters for plotting functions
        prior_parameter_samples = {}
        if "true_parameters" in prior_predictive_adata.uns:
            for key, value in prior_predictive_adata.uns['true_parameters'].items():
                prior_parameter_samples[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value

        # Generate prior predictive plots
        fig_prior = plot_prior_predictive_checks(
            model=model,
            prior_adata=prior_predictive_adata,
            prior_parameters=prior_parameter_samples,
            figsize=(7.5, 5.0),
            save_path=f"{save_path}/{seed}/sample_data",
            figure_name=f"posterior_predictive_check_sample_data_{seed}",
            combine_individual_pdfs=True,
            default_fontsize=5,
            num_genes=10,
            true_parameters_adata=prior_predictive_adata,
        )

        print(f"✅ Sample data and prior predictive plots generated")

    else:
        # This branch would load cached data in a full implementation
        prior_predictive_adata = None
        prior_parameter_samples = {}

    return prior_predictive_adata, prior_parameter_samples


torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("🚀 Multi-Method Posterior Predictive Check Workflow")
print("=" * 70)
print(f"🎯 Random seed: {RANDOM_SEED}")
print(f"📊 Testing {len(INFERENCE_METHODS)} inference methods:")
for method in INFERENCE_METHODS:
    print(f"   • {method['name']}")
print("=" * 70)


# Step 1: Generate or load cached sample data
print(f"\n� Step 1: Sample Data Generation")
prior_predictive_adata, prior_parameter_samples = generate_sample_data_if_needed(
    RANDOM_SEED, REPORTS_SAVE_PATH
)


# Step 2: Test multiple inference methods
print(f"\n🔬 Step 2: Multi-Method Inference Testing")

for i, method_info in enumerate(INFERENCE_METHODS, 1):
    method_name = method_info["name"]
    config = method_info["config"]
    guide_type = method_info["guide_type"]

    print(f"\n{'='*50}")
    print(f"🧪 Method {i}/{len(INFERENCE_METHODS)}: {method_name}")
    print(f"{'='*50}")

    # Create method-specific output directory
    method_save_path = f"{REPORTS_SAVE_PATH}/{RANDOM_SEED}/{method_name}"
    os.makedirs(method_save_path, exist_ok=True)

    # Create model with appropriate guide type
    if guide_type is not None:
        # SVI method
        model = create_piecewise_activation_model(guide_type=guide_type)
        print(f"✅ Created SVI model with {guide_type} guide")
    else:
        # MCMC method
        model = create_piecewise_activation_model()
        print(f"✅ Created MCMC model")

    print(f"🎯 Training with {method_name}...")

    # Set seed for this method
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
    print(f"✅ {method_name} training completed")


    # Step 3: Generate posterior samples from trained model
    print(f"\n🔬 Generating posterior samples from {method_name}...")

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
    print(f"\n📊 Generating posterior predictive data for {method_name}...")
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
    print(f"\n🎨 Creating posterior predictive check plots for {method_name}...")

    _ = plot_posterior_predictive_checks(
        model=trained_model,
        posterior_adata=posterior_predictive_adata,
        posterior_parameters=posterior_parameter_samples,
        figsize=(7.5, 5.0),
        save_path=method_save_path,
        figure_name=f"piecewise_activation_posterior_checks_{method_name}_{RANDOM_SEED}",
        combine_individual_pdfs=True,
        default_fontsize=5,
        observed_adata=prior_predictive_adata,
        num_genes=10,
        true_parameters_adata=prior_predictive_adata,  # Contains true parameters for validation
    )

    print(f"✅ {method_name} posterior predictive check completed!")
    print(f"📁 Plots saved to: {method_save_path}")

print(f"\n{'='*70}")
print(f"🎉 Multi-Method Posterior Predictive Check Workflow Completed!")
print(f"{'='*70}")
print(f"📁 All results saved to: {REPORTS_SAVE_PATH}/{RANDOM_SEED}/")
print(f"🎯 Random seed used: {RANDOM_SEED}")
print(f"📊 Methods tested: {', '.join([m['name'] for m in INFERENCE_METHODS])}")

# Print directory structure
print(f"\n📂 Output directory structure:")
print(f"   {REPORTS_SAVE_PATH}/{RANDOM_SEED}/")
print(f"   ├── sample_data/          # Cached prior predictive plots")
for method in INFERENCE_METHODS:
    print(f"   ├── {method['name']}/     # {method['name']} results")
print(f"   └── ...")
