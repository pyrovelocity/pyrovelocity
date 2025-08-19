"""JAX implementation validation script for PyroVelocity."""

import anndata
import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import scanpy as sc
import os
from pathlib import Path

from pyrovelocity.models.jax.factory.factory import create_piecewise_activation_model, create_poisson_model
from pyrovelocity.models.jax.inference.config import create_inference_config
from pyrovelocity.models.jax.inference.unified import run_inference
from pyrovelocity.plots.predictive import (
    plot_prior_predictive_checks,
    plot_posterior_predictive_checks,
)
from pyrovelocity.utils import print_anndata
from numpyro.infer import Predictive
from numpyro.handlers import condition


RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 42))
MAX_TIME = float(os.environ.get("MAX_TIME", 7.0))
REPORTS_SAVE_PATH = "reports/docs/posterior_predictive_jax"

# Model selection configuration
MODEL_TYPES = {
    "piecewise_activation": create_piecewise_activation_model,
    "poisson_baseline": create_poisson_model,
}

MODEL_TYPE = os.environ.get("MODEL_TYPE", "piecewise_activation")
print(f"Selected model type: {MODEL_TYPE}")

if MODEL_TYPE not in MODEL_TYPES:
    raise ValueError(f"Invalid MODEL_TYPE '{MODEL_TYPE}'. Available types: {list(MODEL_TYPES.keys())}")

create_model_fn = MODEL_TYPES[MODEL_TYPE]
num_samples = 1000
num_cells = 200
num_genes = 100
num_epochs = 1000


AVAILABLE_METHODS = {
    "svi_auto_normal": {
        "config": create_inference_config(
            method="svi",
            num_samples=num_samples,
            num_epochs=num_epochs,
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
            num_samples=num_samples,
            num_epochs=num_epochs,
            learning_rate=0.01,
            guide_type="auto_lowrank_multivariate_normal",
            early_stopping=True,
            early_stopping_patience=10,
        ),
        "guide_type": "auto_lowrank_multivariate_normal"
    },
    "svi_auto_diagonal_normal": {
        "config": create_inference_config(
            method="svi",
            num_samples=num_samples,
            num_epochs=num_epochs,
            learning_rate=0.01,
            guide_type="auto_diagonal_normal",
            early_stopping=True,
            early_stopping_patience=10,
        ),
        "guide_type": "auto_diagonal_normal"
    },
    "svi_auto_iaf_normal": {
        "config": create_inference_config(
            method="svi",
            num_samples=num_samples,
            num_epochs=num_epochs,
            learning_rate=0.001,  # Reduced learning rate for stability
            guide_type="auto_iaf_normal",
            early_stopping=True,
            early_stopping_patience=10,
        ),
        "guide_type": "auto_iaf_normal"
    },
    "mcmc_nuts": {
        "config": create_inference_config(
            method="mcmc",
            num_samples=num_samples,
            num_warmup=500,
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

model = create_model_fn()

dummy_u_obs = jnp.zeros((1, num_cells, num_genes))
dummy_s_obs = jnp.zeros((1, num_cells, num_genes))

print("  Running prior predictive sampling...")
rng_key, rng_key_prior = jax.random.split(rng_key)

# Create synthetic data with model-specific conditioning
if MODEL_TYPE == "piecewise_activation":
    # Piecewise model: condition on T_M_star parameter
    conditioned_model = condition(model, {"T_M_star": jnp.array(MAX_TIME)})
    condition_param = "T_M_star"
    condition_value = MAX_TIME
else:
    # Poisson model: no conditioning needed (uses constant t_star=0.5)
    conditioned_model = model
    condition_param = "t_star"
    condition_value = 0.5

predictive = Predictive(conditioned_model, num_samples=1)
prior_samples = predictive(rng_key_prior, u_obs=None, s_obs=None, 
                          num_cells=num_cells, num_genes=num_genes)

print(f"  ✅ Using {condition_param} = {condition_value} for consistent synthetic data generation")

prior_predictive_adata = anndata.AnnData(
    X=np.array(prior_samples["s_obs"][0, 0, :, :]),
    layers={
        "unspliced": np.array(prior_samples["u_obs"][0, 0, :, :]),
        "spliced": np.array(prior_samples["s_obs"][0, 0, :, :])
    }
)

# Store parameters in consistent format
prior_parameter_samples = {}
for key in ["U_0i", "S_0i", "lambda_j", "T_M_star", 
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
    f"01_posterior_predictive_check_sample_data_jax_{RANDOM_SEED}.pdf",
    f"combined_prior_predictive_checks_{RANDOM_SEED}.pdf"
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
        check_type="prior",
        save_path=f"{REPORTS_SAVE_PATH}/{RANDOM_SEED}/sample_data",
        figure_name=f"posterior_predictive_check_sample_data_jax_{RANDOM_SEED}",
        create_individual_plots=True,
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

model = create_model_fn()
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

# COMPREHENSIVE FIX: Proper posterior predictive checking with multiple samples
# Generate multiple predictive datasets to capture posterior predictive uncertainty
n_predictive_samples = min(50, posterior_samples[list(posterior_samples.keys())[0]].shape[0])
print(f"  📊 Generating {n_predictive_samples} posterior predictive datasets...")

all_u_samples = []
all_s_samples = []
all_t_star_samples = []

for sample_idx in range(n_predictive_samples):
    # Extract single parameter set for this posterior sample
    single_posterior_sample = {}
    for key, value in posterior_samples.items():
        if hasattr(value, 'shape') and len(value.shape) > 0:
            single_posterior_sample[key] = value[sample_idx:sample_idx+1]
        else:
            single_posterior_sample[key] = value
    
    # CRITICAL FIX: Extract t_star from posterior samples (don't generate new ones!)
    # t_star is a latent variable we've already inferred, not something to predict
    if "t_star" in single_posterior_sample:
        t_star_from_posterior = single_posterior_sample["t_star"][0, :, 0]  # [n_cells,]
        all_t_star_samples.append(t_star_from_posterior)
    
    # Generate predictive data for this parameter set
    # The Predictive will use the t_star from posterior_samples to generate u_obs, s_obs
    predictive = Predictive(
        model, 
        posterior_samples=single_posterior_sample,
        num_samples=1
    )
    
    # Generate new random key for each sample
    rng_key_prediction, rng_key_sample = jax.random.split(rng_key_prediction)
    
    sample_predictive = predictive(rng_key_sample, u_obs=None, s_obs=None, 
                                  num_cells=num_cells, num_genes=num_genes)
    
    u_sample = sample_predictive["u_obs"][0, 0, :, :]
    s_sample = sample_predictive["s_obs"][0, 0, :, :]
    
    all_u_samples.append(u_sample)
    all_s_samples.append(s_sample)

# Stack all samples and compute statistics
all_u_samples = np.stack(all_u_samples, axis=0)  # [n_samples, n_cells, n_genes]
all_s_samples = np.stack(all_s_samples, axis=0)  # [n_samples, n_cells, n_genes]

# Compute posterior predictive summary statistics
# For count data, median preserves integer nature better than mean
u_median = np.median(all_u_samples, axis=0).astype(np.int32)
s_median = np.median(all_s_samples, axis=0).astype(np.int32)

# Still compute mean for comparison, but acknowledge it's not integer
u_mean = np.mean(all_u_samples, axis=0)
s_mean = np.mean(all_s_samples, axis=0)

# Standard deviation and CV for dispersion
u_std = np.std(all_u_samples, axis=0)
s_std = np.std(all_s_samples, axis=0)

# Coefficient of variation (CV) is more meaningful for count data
u_cv = np.where(u_mean > 0, u_std / u_mean, 0)
s_cv = np.where(s_mean > 0, s_std / s_mean, 0)

# Quantiles - round to nearest integer for count data
u_q025 = np.round(np.quantile(all_u_samples, 0.025, axis=0)).astype(np.int32)
u_q975 = np.round(np.quantile(all_u_samples, 0.975, axis=0)).astype(np.int32)
s_q025 = np.round(np.quantile(all_s_samples, 0.025, axis=0)).astype(np.int32)
s_q975 = np.round(np.quantile(all_s_samples, 0.975, axis=0)).astype(np.int32)

# Compute proportion of zeros (important for count data)
u_zero_prop = np.mean(all_u_samples == 0, axis=0)
s_zero_prop = np.mean(all_s_samples == 0, axis=0)

print(f"  📊 Computed statistics over {n_predictive_samples} posterior samples")
print(f"    - U counts: median={np.median(u_median):.0f}, mean={u_mean.mean():.2f}, CV={np.median(u_cv):.2f}")
print(f"    - S counts: median={np.median(s_median):.0f}, mean={s_mean.mean():.2f}, CV={np.median(s_cv):.2f}")
print(f"    - Zero proportions: U={u_zero_prop.mean():.2%}, S={s_zero_prop.mean():.2%}")

# Create AnnData with median (integer) values as primary data
# Store mean and other continuous statistics in layers for reference
posterior_predictive_adata = anndata.AnnData(
    X=s_median,  # Use median (integer) as primary data
    layers={
        "unspliced": u_median,  # Integer median
        "spliced": s_median,    # Integer median
        "unspliced_mean": u_mean,  # Continuous mean for reference
        "spliced_mean": s_mean,    # Continuous mean for reference
        "unspliced_std": u_std,
        "spliced_std": s_std,
        "unspliced_cv": u_cv,   # Coefficient of variation
        "spliced_cv": s_cv,     # Coefficient of variation
        "unspliced_q025": u_q025,  # Integer quantiles
        "unspliced_q975": u_q975,  # Integer quantiles
        "spliced_q025": s_q025,    # Integer quantiles
        "spliced_q975": s_q975,    # Integer quantiles
        "unspliced_zero_prop": u_zero_prop,  # Proportion of zeros
        "spliced_zero_prop": s_zero_prop,    # Proportion of zeros
    }
)
    
# Add t_star statistics if available
if len(all_t_star_samples) > 0:
    all_t_star_samples = np.stack(all_t_star_samples, axis=0)  # [n_samples, n_cells]
    t_star_mean = np.mean(all_t_star_samples, axis=0)
    t_star_std = np.std(all_t_star_samples, axis=0)
    posterior_predictive_adata.obs["t_star"] = t_star_mean
    posterior_predictive_adata.obs["t_star_std"] = t_star_std
    print(f"✅ Stored t_star: shape {t_star_mean.shape}, range [{t_star_mean.min():.3f}, {t_star_mean.max():.3f}], avg_std={t_star_std.mean():.3f}")
    
print("✅ Posterior predictive data generated:")
print_anndata(posterior_predictive_adata)

# Compute and store MAE for reuse across plotting functions
print(f"\n📊 Computing MAE for parameter recovery correlation coloring...")
from pyrovelocity.plots.predictive.core import compute_and_store_mae

mae_scores = compute_and_store_mae(
    predicted_adata=posterior_predictive_adata,
    observed_adata=prior_predictive_adata,
    store_in_var=True
)

print(f"✅ MAE computed and stored: Mean={mae_scores.mean():.4f}, Std={mae_scores.std():.4f}, Range=[{mae_scores.min():.4f}, {mae_scores.max():.4f}]")

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
    figsize=(7.5, 5.0),
    save_path=method_save_path,
    figure_name=f"posterior_predictive_check_{SELECTED_METHOD}_jax_{RANDOM_SEED}",
    create_individual_plots=True,
    combine_individual_pdfs=True,
    default_fontsize=6,
    observed_adata=prior_predictive_adata,
    num_genes=10,
    true_parameters_adata=prior_predictive_adata,
)

print(f"\n✅ JAX validation complete!")
print(f"📊 Results saved to: {REPORTS_SAVE_PATH}/{RANDOM_SEED}")
print(f"   - Prior predictive plots: sample_data/")
print(f"   - Posterior predictive plots: {SELECTED_METHOD}/")