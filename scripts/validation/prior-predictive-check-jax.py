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

# Generate dummy observations for the model
u_obs = jnp.ones((1, num_cells, num_genes))
s_obs = jnp.ones((1, num_cells, num_genes))

# Use NumPyro's Predictive class to generate prior samples
from numpyro.infer import Predictive

# Sample from the prior
prior_predictive = Predictive(model, num_samples=num_samples)
key, subkey = jax.random.split(key)

# Generate prior samples
prior_samples = prior_predictive(subkey, u_obs=u_obs, s_obs=s_obs)

print(f"Generated prior samples with keys: {list(prior_samples.keys())}")

# Convert JAX arrays to NumPy for compatibility with AnnData and plotting
prior_parameter_samples = {}
for key_name, value in prior_samples.items():
    if isinstance(value, jnp.ndarray):
        prior_parameter_samples[key_name] = np.array(value)
    else:
        prior_parameter_samples[key_name] = value

# Create AnnData object from generated data
# Use the expected counts from the last sample for visualization
u_expected = prior_samples["u_expected"][-1, 0, :, :]  # [cells, genes]
s_expected = prior_samples["s_expected"][-1, 0, :, :]  # [cells, genes]

# Create AnnData object
import anndata as adata_module
prior_predictive_adata = adata_module.AnnData(
    X=np.array(s_expected),  # Use spliced as main expression
    layers={
        "unspliced": np.array(u_expected),
        "spliced": np.array(s_expected)
    }
)

# Store true parameters in AnnData uns
prior_predictive_adata.uns["true_parameters"] = {}
for key_name, value in prior_parameter_samples.items():
    if key_name not in ["u_expected", "s_expected"]:
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
            samples_tensor = torch.tensor(samples)
        else:
            samples_tensor = samples

        # Handle different parameter shapes (scalar vs vector)
        if samples_tensor.numel() == 1:
            # Scalar parameter
            print(f"{param_name}: {samples_tensor.item():.3f}")
        else:
            # Vector parameter (gene-specific or cell-specific)
            print(f"{param_name} (n={samples_tensor.numel()}):")
            print(f"  Range: [{samples_tensor.min():.3f}, {samples_tensor.max():.3f}]")
            print(f"  Mean ± Std: {samples_tensor.mean():.3f} ± {samples_tensor.std():.3f}")

# Print information about the dataset
print(f"\nDataset information:")
print(f"  Cells: {prior_predictive_adata.n_obs}")
print(f"  Genes: {prior_predictive_adata.n_vars}")
print(f"  Layers: {list(prior_predictive_adata.layers.keys())}")
print(f"  Unspliced counts range: [{prior_predictive_adata.layers['unspliced'].min():.0f}, {prior_predictive_adata.layers['unspliced'].max():.0f}]")
print(f"  Spliced counts range: [{prior_predictive_adata.layers['spliced'].min():.0f}, {prior_predictive_adata.layers['spliced'].max():.0f}]")