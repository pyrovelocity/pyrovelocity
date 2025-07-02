"""JAX implementation validation script for PyroVelocity."""

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
from pyrovelocity.models.jax.inference.unified import run_inference
from pyrovelocity.plots.predictive_checks import (
    plot_prior_predictive_checks,
    plot_posterior_predictive_checks,
)
from pyrovelocity.utils import print_anndata
from numpyro.infer import Predictive


RANDOM_SEED = 42
REPORTS_SAVE_PATH = "reports/docs/posterior_predictive_jax"

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

SELECTED_METHOD = "svi_auto_lowrank_multivariate_normal"

if SELECTED_METHOD not in AVAILABLE_METHODS:
    raise ValueError(f"Invalid method '{SELECTED_METHOD}'. Available methods: {list(AVAILABLE_METHODS.keys())}")

METHOD_CONFIG = AVAILABLE_METHODS[SELECTED_METHOD]

numpyro.set_platform("cpu")
numpyro.set_host_device_count(1)

rng_key = jax.random.PRNGKey(RANDOM_SEED)
os.makedirs(REPORTS_SAVE_PATH, exist_ok=True)
os.makedirs(f"{REPORTS_SAVE_PATH}/{RANDOM_SEED}", exist_ok=True)
os.makedirs(f"{REPORTS_SAVE_PATH}/{RANDOM_SEED}/sample_data", exist_ok=True)




# Step 1: Generate prior predictive data
print(f"\n📊 Step 1: Generating prior predictive data (seed: {RANDOM_SEED})...")

model = create_piecewise_activation_model()

num_cells = 200
num_genes = 100

dummy_u_obs = jnp.zeros((1, num_cells, num_genes))
dummy_s_obs = jnp.zeros((1, num_cells, num_genes))

print("  Running prior predictive sampling...")
rng_key, rng_key_prior = jax.random.split(rng_key)

predictive = Predictive(model, num_samples=1)
prior_samples = predictive(rng_key_prior, u_obs=None, s_obs=None, 
                          num_cells=num_cells, num_genes=num_genes)

prior_predictive_adata = anndata.AnnData(
    X=np.array(prior_samples["s_obs"][0, 0, :, :]),
    layers={
        "unspliced": np.array(prior_samples["u_obs"][0, 0, :, :]),
        "spliced": np.array(prior_samples["s_obs"][0, 0, :, :])
    }
)

# Store parameters in consistent format
prior_parameter_samples = {}
for key in ["U_0i", "lambda_j", "T_M_star", "boundary_concentration", 
            "R_on", "gamma_star", "t_on_star", "delta_star", "t_star"]:
    if key in prior_samples:
        prior_parameter_samples[key] = prior_samples[key]

# Store t_star for temporal coordinate validation
if "t_star" in prior_samples:
    t_star_shape = prior_samples["t_star"].shape
    if len(t_star_shape) == 3:
        t_star_values = prior_samples["t_star"][0, :, 0]
    elif len(t_star_shape) == 2:
        t_star_values = prior_samples["t_star"][0, :]
    else:
        t_star_values = prior_samples["t_star"]
        
    prior_predictive_adata.obs["t_star"] = np.array(t_star_values)
    print(f"✅ Stored t_star in prior data: shape {t_star_values.shape}, range [{t_star_values.min():.3f}, {t_star_values.max():.3f}]")

# Store true parameters for validation
prior_predictive_adata.uns["true_parameters"] = {
    key: np.array(value) for key, value in prior_parameter_samples.items()
}
print(f"\n🗺️ Computing UMAP and clustering for prior predictive data...")
sc.pp.pca(prior_predictive_adata, random_state=RANDOM_SEED)
sc.pp.neighbors(prior_predictive_adata, n_neighbors=10, random_state=RANDOM_SEED)
sc.tl.umap(prior_predictive_adata, random_state=RANDOM_SEED)
sc.tl.leiden(prior_predictive_adata, random_state=RANDOM_SEED)

print("✅ Prior predictive data generated:")
print_anndata(prior_predictive_adata)

# Generate prior predictive plots
sample_data_path = Path(f"{REPORTS_SAVE_PATH}/{RANDOM_SEED}/sample_data")
key_files = [
    f"01_data_overview_posterior_predictive_check_sample_data_jax_{RANDOM_SEED}.pdf",
    f"combined_prior_predictive_checks_jax_{RANDOM_SEED}.pdf"
]
plots_exist = sample_data_path.exists() and all((sample_data_path / f).exists() for f in key_files)

if plots_exist:
    print(f"📋 Found cached prior predictive plots, skipping regeneration...")
else:
    print(f"🎨 Generating prior predictive plots...")
    plot_prior_predictive_checks(
        model=None,
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


# Step 2: Train model
print(f"\n🔬 Step 2: Model Training with {SELECTED_METHOD}")

method_save_path = f"{REPORTS_SAVE_PATH}/{RANDOM_SEED}/{SELECTED_METHOD}"
os.makedirs(method_save_path, exist_ok=True)

model = create_piecewise_activation_model()
print(f"✅ Created JAX model for {SELECTED_METHOD}")

print(f"🎯 Training with {SELECTED_METHOD}...")

u_obs_batch = jnp.expand_dims(jnp.array(prior_predictive_adata.layers["unspliced"]), 0)
s_obs_batch = jnp.expand_dims(jnp.array(prior_predictive_adata.layers["spliced"]), 0)

rng_key, rng_key_inference = jax.random.split(rng_key)
config = METHOD_CONFIG['config']

inference_object, inference_state = run_inference(
    model=model,
    args=(u_obs_batch, s_obs_batch),
    kwargs={},
    config=config,
    key=rng_key_inference
)
    
print(f"✅ {SELECTED_METHOD} training completed")

posterior_samples = inference_state.posterior_samples

# Store inference state for plotting functions that need it
class ModelWithState:
    def __init__(self, model, inference_state):
        self.model = model
        self.state = type('State', (), {'inference_state': inference_state})()
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

model_with_state = ModelWithState(model, inference_state)
    
print(f"✅ Generated {len(posterior_samples)} types of posterior parameters")
for key_name in sorted(posterior_samples.keys()):
    param_shape = posterior_samples[key_name].shape if hasattr(posterior_samples[key_name], 'shape') else len(posterior_samples[key_name])
    print(f"    {key_name}: {param_shape}")

# Step 3: Generate posterior predictive data
print(f"\n📊 Step 3: Generating posterior predictive data for {SELECTED_METHOD}...")

rng_key, rng_key_prediction = jax.random.split(rng_key)

predictive = Predictive(
    model, 
    posterior_samples=posterior_samples,
    num_samples=1
)
    
posterior_predictive_samples = predictive(rng_key_prediction, u_obs=u_obs_batch, s_obs=s_obs_batch)
    
u_posterior_expected = posterior_predictive_samples["u_expected"][0, 0, :, :]
s_posterior_expected = posterior_predictive_samples["s_expected"][0, 0, :, :]
    
posterior_predictive_adata = anndata.AnnData(
    X=np.array(s_posterior_expected),
    layers={
        "unspliced": np.array(u_posterior_expected),
        "spliced": np.array(s_posterior_expected)
    }
)
    
if "t_star" in posterior_predictive_samples:
    t_star_values = posterior_predictive_samples["t_star"][0, :, 0]
    posterior_predictive_adata.obs["t_star"] = np.array(t_star_values)
    print(f"✅ Stored t_star: shape {t_star_values.shape}, range [{t_star_values.min():.3f}, {t_star_values.max():.3f}]")
    
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
# Generate posterior predictive plots
print(f"\n🎨 Step 4: Generating {SELECTED_METHOD} posterior predictive plots...")

plot_posterior_predictive_checks(
    model=model_with_state,
    posterior_adata=posterior_predictive_adata,
    posterior_parameters=posterior_samples,
    true_parameters_adata=prior_predictive_adata,
    observed_adata=prior_predictive_adata,
    figsize=(7.5, 5.0),
    save_path=method_save_path,
    figure_name=f"posterior_predictive_check_{SELECTED_METHOD}_jax_{RANDOM_SEED}",
    combine_individual_pdfs=True,
    default_fontsize=6,
    num_genes=10,
)

print(f"\n✅ JAX validation complete!")
print(f"📊 Results saved to: {REPORTS_SAVE_PATH}/{RANDOM_SEED}")
print(f"   - Prior predictive plots: sample_data/")
print(f"   - Posterior predictive plots: {SELECTED_METHOD}/")