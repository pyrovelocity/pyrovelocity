import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.plots.predictive_checks import (
    plot_prior_predictive_checks,
)
from pyrovelocity.utils import print_anndata


RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

model = create_piecewise_activation_model()

print(f"Created model: {model}")
print(f"Model components:")
print(f"  Dynamics: {model.dynamics_model}")
print(f"  Prior: {model.prior_model}")
print(f"  Likelihood: {model.likelihood_model}")
print(f"  Guide: {model.guide_model}")

# Generate num_samples datasets for prior predictive check
# Optimized configuration for hierarchical model convergence:
# - More samples (400) for better global parameter convergence
# - Fewer cells/genes to maintain computational efficiency
# - Total parameters: 400 × (100 + 25) = 50,000 (vs 25,000 previously)
prior_predictive_adata = model.generate_predictive_samples(
    num_cells=100,      # Reduced from 200 for efficiency
    num_genes=25,       # Reduced from 50 for efficiency
    num_samples=400,    # Increased from 100 for convergence
    return_format="anndata"
)

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
prior_parameter_samples = {}
if "true_parameters" in prior_predictive_adata.uns:
    for key, value in prior_predictive_adata.uns['true_parameters'].items():
        prior_parameter_samples[key] = torch.tensor(value) if not isinstance(value, torch.Tensor) else value

# Add UMAP and clustering though these will be uninformative in the context of prior predictive checks
sc.pp.pca(prior_predictive_adata, random_state=RANDOM_SEED)
sc.pp.neighbors(prior_predictive_adata, n_neighbors=10, random_state=RANDOM_SEED)
sc.tl.umap(prior_predictive_adata, random_state=RANDOM_SEED)
sc.tl.leiden(prior_predictive_adata, random_state=RANDOM_SEED)

# Plot randomly sampled parameters and data
fig_prior = plot_prior_predictive_checks(
    model=model,
    prior_adata=prior_predictive_adata,
    prior_parameters=prior_parameter_samples,
    figsize=(7.5, 5.0),
    save_path="reports/docs/prior_predictive",
    figure_name=f"piecewise_activation_prior_checks_{RANDOM_SEED}",
    combine_individual_pdfs=True,
    default_fontsize=5,
    num_genes=10,
    true_parameters_adata=prior_predictive_adata,  # For prior predictive, true parameters are in the same adata
)

# Validate prior parameter ranges (updated for hierarchical temporal parameterization)
print("\nPrior parameter range validation:")
print(f"Total parameters extracted: {len(prior_parameter_samples)}")

for param_name, samples in prior_parameter_samples.items():
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
