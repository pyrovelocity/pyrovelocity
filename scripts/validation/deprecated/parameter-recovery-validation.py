import torch
import pyro
import numpy as np

from pyrovelocity.models.modular.factory import create_piecewise_activation_model
from pyrovelocity.models.modular.validation import (
    create_validation_summary,
    display_validation_table,
    plot_validation_summary
)
from pyrovelocity.plots.predictive_checks import (
    plot_posterior_predictive_checks,
)


# Configurable random seed for reproducibility and variability analysis
# Change this value to generate different datasets and see variability in parameter recovery
# The seed will be included in the output filename: validation_summary_{RANDOM_SEED}.png
RANDOM_SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create the piecewise activation model
model = create_piecewise_activation_model()

print(f"Created model: {model}")
print(f"Model components:")
print(f"  Dynamics: {model.dynamics_model}")
print(f"  Prior: {model.prior_model}")
print(f"  Likelihood: {model.likelihood_model}")
print(f"  Guide: {model.guide_model}")

# Generate validation datasets for all patterns using the updated 3-pattern classification
print("📊 Generating validation datasets...")
print(f"Using random seed: {RANDOM_SEED}")
print(f"Output will include seed in filenames: validation_summary_{RANDOM_SEED}.png")

validation_datasets = model.generate_validation_datasets(
    # Updated to use the simplified 3-pattern classification system:
    # - pre_activation: Genes activated before observation window (negative t_on)
    # - transient: Complete activation-decay cycles within observation window
    # - sustained: Net increase over observation window (includes late activation)
    patterns=['pre_activation', 'transient', 'sustained'],
    n_genes=5,
    n_cells=200,
    n_parameter_sets=2,  # 2 parameter sets per pattern = 6 total datasets
    seed=RANDOM_SEED,
    use_coherent_trajectories=True,  # Enable coherent trajectory sampling
    trajectory_type="linear"  # Use linear trajectories for validation
)

print(f"✅ Generated {len(validation_datasets)} validation datasets")

# Display dataset information
for dataset_key, dataset_info in validation_datasets.items():
    adata = dataset_info['adata']
    pattern = dataset_info['pattern']
    true_params = dataset_info['true_parameters']

    print(f"\n{dataset_key}:")
    print(f"  Data shape: {adata.shape}")
    print(f"  Pattern: {pattern}")
    print(f"  True parameters: {list(true_params.keys())}")
    print(f"  Metadata: {dataset_info.get('metadata', {})}")

print(f"\n🎯 Ready for parameter recovery validation!")

print("🚀 Starting Parameter Recovery Validation...")
print("=" * 60)

# Run comprehensive parameter recovery validation
validation_results = model.validate_parameter_recovery(
    validation_datasets=validation_datasets,
    max_epochs=1000,
    learning_rate=0.01,
    success_threshold=0.9,
    num_posterior_samples=1000,
    adaptive_guide_rank=True,
    create_predictive_plots=True,
    save_plots_path="reports/docs/validation",
    seed=RANDOM_SEED
)

print("✅ Parameter recovery validation complete!")
print(f"📊 Validated {len(validation_results)} datasets")

# Display training summary
for dataset_key, result in validation_results.items():
    if 'training_result' in result:
        training = result['training_result']
        metrics = result['recovery_metrics']
        print(f"\n{dataset_key} ({result['pattern']} pattern):")
        print(f"  Training: {training['num_epochs']} epochs, ELBO: {training['final_elbo']:.2f}")
        print(f"  Recovery: {metrics['overall_correlation']:.3f} correlation, {metrics['success_rate']:.1%} success")
        print(f"  Plots: {result.get('plot_paths', {})}")

# Display validation results summary
print("📊 PARAMETER RECOVERY VALIDATION RESULTS")
print("=" * 80)

# Get overall summary
summary = create_validation_summary(validation_results)
print(f"\n🎯 Overall Success Rate: {summary['overall_success_rate']:.1%}")
print(f"📊 Mean Correlation: {summary['mean_correlation']:.3f}")
print(f"📉 Mean Error: {summary['mean_error']:.3f}")

# Pattern-wise performance
print(f"\n📈 Pattern-wise Performance:")
for pattern, metrics in summary['by_pattern'].items():
    print(f"  {pattern.upper()}: {metrics['correlation']:.3f} correlation, {metrics['success_rate']:.1%} success")

# Detailed results table
print(f"\n📋 Detailed Results:")
display_validation_table(validation_results)

# Generate comprehensive validation summary plot with seed in filename
plot_validation_summary(
    validation_results,
    save_path=f'reports/docs/validation/validation_summary_{RANDOM_SEED}.png',
    figsize=(15, 12),
    show_success_threshold=True
)

# Validate parameter ranges for the new parameterization
print("\n🔍 Parameter Range Validation:")
print("Checking for updated parameter names and ranges...")

# Sample some parameters to validate the new parameterization
sample_params = model.sample_system_parameters(
    num_samples=100,
    constrain_to_pattern=False,  # Use full prior ranges
    n_genes=5,
    n_cells=200
)

print("\nUpdated parameter ranges (new parameterization):")
for param_name, samples in sample_params.items():
    if param_name.startswith(('R_on', 'tilde_t_on', 'tilde_delta', 't_on_star', 'delta_star', 'gamma_star', 'T_M_star')):
        print(f"{param_name}:")
        print(f"  Range: [{samples.min():.3f}, {samples.max():.3f}]")
        print(f"  Mean ± Std: {samples.mean():.3f} ± {samples.std():.3f}")

print(f"\n✅ Parameter recovery validation completed with seed {RANDOM_SEED}!")
print(f"📊 Results saved with seed identifier: validation_summary_{RANDOM_SEED}.png")

# Generate posterior predictive check plots for successful validations
print("\n🎨 Creating posterior predictive check plots for successful validations...")

successful_validations = {k: v for k, v in validation_results.items()
                         if 'error' not in v and v.get('success', False)}

if successful_validations:
    print(f"Found {len(successful_validations)} successful validations to plot")

    for dataset_key, result in list(successful_validations.items())[:2]:  # Limit to first 2 for demo
        if 'posterior_samples' in result and 'posterior_adata' in result:
            print(f"  📈 Creating plots for {dataset_key}...")

            try:
                fig_posterior = plot_posterior_predictive_checks(
                    model=model,
                    posterior_adata=result['posterior_adata'],
                    posterior_parameters=result['posterior_samples'],
                    figsize=(15, 10),
                    save_path="reports/docs/validation",
                    figure_name=f"posterior_checks_{dataset_key}_{RANDOM_SEED}",
                    create_individual_plots=True,
                    combine_individual_pdfs=False
                )
                print(f"    ✅ Saved posterior predictive plots for {dataset_key}")
            except Exception as e:
                print(f"    ⚠️  Could not create plots for {dataset_key}: {e}")
else:
    print("No successful validations found for posterior predictive plotting")
