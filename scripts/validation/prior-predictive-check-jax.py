import jax
import jax.numpy as jnp
import numpyro
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from pyrovelocity.models.jax.factory.factory import create_piecewise_activation_model
from pyrovelocity.plots.predictive_checks import (
    plot_prior_predictive_checks,
)
from pyrovelocity.utils import print_anndata


RANDOM_SEED = 42

# Set random seeds for reproducibility
numpyro.set_platform("cpu")  # Ensure CPU execution for consistency
jax.config.update("jax_enable_x64", True)  # Enable double precision
key = jax.random.PRNGKey(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create JAX model
model = create_piecewise_activation_model()

print(f"Created JAX model: {model}")
print(f"Model type: {type(model)}")

# Generate num_samples datasets for prior predictive check
# Optimized configuration for hierarchical model convergence:
# - More samples (400) for better global parameter convergence
# - Fewer cells/genes to maintain computational efficiency
# - Total parameters: 400 × (100 + 25) = 50,000 (vs 25,000 previously)

# Note: JAX implementation uses NumPyro's Predictive class for sampling
print("Generating prior predictive samples...")

# Create synthetic data dimensions
num_cells = 100      # Reduced from 200 for efficiency
num_genes = 25       # Reduced from 50 for efficiency
num_samples = 400    # Increased from 100 for convergence

# Use NumPyro's Predictive class to generate prior samples
from numpyro.infer import Predictive

# Sample from the prior
prior_predictive = Predictive(model, num_samples=num_samples)
key, subkey = jax.random.split(key)

# Generate prior samples - pass None for observations to generate from prior
prior_samples = prior_predictive(subkey, u_obs=None, s_obs=None, 
                                num_cells=num_cells, num_genes=num_genes)

print(f"Generated prior samples with keys: {list(prior_samples.keys())}")

# Convert JAX arrays to NumPy for compatibility with AnnData and plotting
prior_parameter_samples = {}
for key_name, value in prior_samples.items():
    if isinstance(value, jnp.ndarray):
        prior_parameter_samples[key_name] = np.array(value)
    else:
        prior_parameter_samples[key_name] = value

# Create AnnData object from generated data
# Use the generated observations from the first sample for visualization
u_obs_generated = prior_samples["u_obs"][0, 0, :, :]  # [cells, genes]
s_obs_generated = prior_samples["s_obs"][0, 0, :, :]  # [cells, genes]

# Create AnnData object
import anndata as adata_module
prior_predictive_adata = adata_module.AnnData(
    X=np.array(s_obs_generated),  # Use spliced as main expression
    layers={
        "unspliced": np.array(u_obs_generated),
        "spliced": np.array(s_obs_generated)
    }
)

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

# Store true parameters in AnnData uns
prior_predictive_adata.uns["true_parameters"] = {}
for key_name, value in prior_parameter_samples.items():
    if key_name not in ["u_obs", "s_obs"]:
        # For multi-sample parameters, take the mean or last sample
        if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == num_samples:
            # Take mean across samples for parameter storage
            prior_predictive_adata.uns["true_parameters"][key_name] = np.mean(value, axis=0)
        else:
            prior_predictive_adata.uns["true_parameters"][key_name] = value

print("AnnData object summary:")
print_anndata(prior_predictive_adata)

# Extract parameters from AnnData object (stored in clean true_parameters dictionary)
print("Parameters stored in AnnData:")
if "true_parameters" in prior_predictive_adata.uns:
    print(f"  Found {len(prior_predictive_adata.uns['true_parameters'])} parameters in true_parameters")
    for key in sorted(prior_predictive_adata.uns['true_parameters'].keys()):
        print(f"    {key}")
else:
    print("  No parameters found in adata.uns['true_parameters']")

# Extract parameters for plotting functions (no prefix needed)
plot_parameter_samples = {}
if "true_parameters" in prior_predictive_adata.uns:
    for key, value in prior_predictive_adata.uns['true_parameters'].items():
        # Keep as original type (JAX arrays/NumPy arrays) for framework-agnostic plotting
        plot_parameter_samples[key] = value

# Add UMAP and clustering though these will be uninformative in the context of prior predictive checks
sc.pp.pca(prior_predictive_adata, random_state=RANDOM_SEED)
sc.pp.neighbors(prior_predictive_adata, n_neighbors=10, random_state=RANDOM_SEED)
sc.tl.umap(prior_predictive_adata, random_state=RANDOM_SEED)
sc.tl.leiden(prior_predictive_adata, random_state=RANDOM_SEED)

# Plot randomly sampled parameters and data
fig_prior = plot_prior_predictive_checks(
    model=model,
    prior_adata=prior_predictive_adata,
    prior_parameters=plot_parameter_samples,
    figsize=(7.5, 5.0),
    save_path="reports/docs/prior_predictive_jax",
    figure_name=f"piecewise_activation_prior_checks_jax_{RANDOM_SEED}",
    combine_individual_pdfs=True,
    default_fontsize=5,
    num_genes=10,
    true_parameters_adata=prior_predictive_adata,  # For prior predictive, true parameters are in the same adata
)

# Validate prior parameter ranges (updated for hierarchical temporal parameterization)
print("\nPrior parameter range validation:")
print(f"Total parameters extracted: {len(plot_parameter_samples)}")

for param_name, samples in plot_parameter_samples.items():
    if param_name.startswith(('R_on', 't_on_star', 'delta_star', 'gamma_star', 'T_M_star')):
        # Convert to tensor if needed and handle different shapes
        if isinstance(samples, np.ndarray):
            samples_array = samples
        else:
            samples_array = np.array(samples)

        # Handle different parameter shapes (scalar vs vector)
        if samples_array.size == 1:
            # Scalar parameter
            print(f"{param_name}: {samples_array.item():.3f}")
        else:
            # Vector parameter (gene-specific or cell-specific)
            print(f"{param_name} (n={samples_array.size}):")
            print(f"  Range: [{samples_array.min():.3f}, {samples_array.max():.3f}]")
            print(f"  Mean ± Std: {samples_array.mean():.3f} ± {samples_array.std():.3f}")

# Print information about the dataset
print(f"\nDataset information:")
print(f"  Cells: {prior_predictive_adata.n_obs}")
print(f"  Genes: {prior_predictive_adata.n_vars}")
print(f"  Layers: {list(prior_predictive_adata.layers.keys())}")
print(f"  Unspliced counts range: [{prior_predictive_adata.layers['unspliced'].min():.0f}, {prior_predictive_adata.layers['unspliced'].max():.0f}]")
print(f"  Spliced counts range: [{prior_predictive_adata.layers['spliced'].min():.0f}, {prior_predictive_adata.layers['spliced'].max():.0f}]")