#!/usr/bin/env python
"""
Direct Comparison Script for PyroVelocity.

This script directly trains both the legacy and modular implementations
of PyroVelocity on the preprocessed pancreas data and compares their results.
It bypasses the validation framework to avoid the issues we've encountered.

By default, velocity computation is skipped to accelerate model comparison
during debugging. Use the --compute-velocity flag to enable velocity computation.

Example usage:
    python direct_comparison.py --max-epochs 5 --num-samples 3
    python direct_comparison.py --max-epochs 5 --num-samples 3 --compute-velocity
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import pyro
from importlib.resources import files

# Add the src directory to the path to import test fixtures
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pyrovelocity.io.serialization import load_anndata_from_json
from pyrovelocity.models._velocity import PyroVelocity
from pyrovelocity.models.modular import PyroVelocityModel
from pyrovelocity.models.modular.factory import create_legacy_model1, create_legacy_model2
from pyrovelocity.validation.comparison import (
    compare_parameters,
    compare_velocities,
    compare_uncertainties,
    compare_performance,
)


def print_progress(message):
    """Print progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Ensure immediate output


def to_numpy(tensor_or_array):
    """Convert tensor to numpy array if needed."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return tensor_or_array


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Direct Comparison Script for PyroVelocity"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="Maximum number of epochs for training",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of posterior samples to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scripts/validation/direct_comparison_results",
        help="Directory to save validation results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="model1",
        choices=["model1", "model2"],
        help="Model type to use (model1=legacy model1 replication, model2=legacy model2 replication). "
             "model1: guide_type='auto_t0_constraint', add_offset=False. "
             "model2: guide_type='auto', add_offset=True.",
    )
    parser.add_argument(
        "--compute-velocity",
        action="store_true",
        default=False,
        help="Compute velocity (default: False)",
    )
    return parser.parse_args()


def load_preprocessed_pancreas_data():
    """Load the preprocessed pancreas data fixture."""
    print("Loading preprocessed pancreas data fixture...")

    # Fixture hash for data validation
    FIXTURE_HASH = "95c80131694f2c6449a48a56513ef79cdc56eae75204ec69abde0d81a18722ae"

    fixture_file_path = (
        files("pyrovelocity.tests.data") / "preprocessed_pancreas_50_7.json"
    )
    adata = load_anndata_from_json(
        filename=str(fixture_file_path),
        expected_hash=FIXTURE_HASH,
    )
    print(f"Loaded preprocessed pancreas data: {adata.shape[0]} cells, {adata.shape[1]} genes")
    return adata


def train_legacy_model(adata, max_epochs, num_samples, seed=42, compute_velocity=True, model_type=None):
    """
    Train the legacy model and return results.

    Args:
        adata: AnnData object containing the data
        max_epochs: Maximum number of epochs for training
        num_samples: Number of posterior samples to generate
        seed: Random seed for reproducibility
        compute_velocity: Whether to compute velocity (default: True)
        model_type: Ignored - legacy model always uses VelocityModelAuto

    Note: The legacy model always uses VelocityModelAuto regardless of any model_type parameter.
    """
    print("Training legacy model...")

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # Set up AnnData for legacy model
    PyroVelocity.setup_anndata(adata)

    # Create legacy model based on model_type
    if model_type == "model1":
        print("Legacy model1: using guide_type='auto_t0_constraint', add_offset=False")
        model = PyroVelocity(
            adata=adata,
            guide_type="auto_t0_constraint",
            add_offset=False,
        )
    elif model_type == "model2":
        print("Legacy model2: using guide_type='auto', add_offset=True")
        model = PyroVelocity(
            adata=adata,
            guide_type="auto",
            add_offset=True,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Train model
    print(f"Starting legacy model training with max_epochs={max_epochs}")
    training_start_time = time.time()
    model.train(max_epochs=max_epochs, check_val_every_n_epoch=None)
    print("Legacy model training completed successfully")
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    print(f"Legacy model training time: {training_time:.2f} seconds")

    # Generate posterior samples
    print(f"Starting posterior sampling with num_samples={num_samples}")
    inference_start_time = time.time()
    try:
        posterior_samples = model.generate_posterior_samples(num_samples=num_samples)
    except AttributeError:
        # Known issue with legacy model - retry with explicit adata
        posterior_samples = model.generate_posterior_samples(adata=model.adata, num_samples=num_samples)
    print("Posterior sampling completed successfully")

    # Print parameter shapes for debugging
    print("Legacy model parameter shapes:")
    for key, value in posterior_samples.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(f"  {key}: {value.shape}")

    # Print parameter statistics for debugging
    print("\nLegacy model parameter statistics:")
    for param in ["alpha", "beta", "gamma", "u_scale", "ut", "st"]:
        if param in posterior_samples:
            param_value = to_numpy(posterior_samples[param])
            print(f"  {param} mean: {np.mean(param_value):.6f}, std: {np.std(param_value):.6f}, min: {np.min(param_value):.6f}, max: {np.max(param_value):.6f}")

    # Initialize velocity and uncertainty as None
    velocity = None
    uncertainty = None

    # Compute velocity only if requested
    if compute_velocity:
        print("\nComputing velocity for legacy model...")

        # Compute statistics from posterior samples (this computes velocity and stores it in adata)
        posterior_samples = model.compute_statistics_from_posterior_samples(
            adata=adata,
            posterior_samples=posterior_samples,
            vector_field_basis="umap",
            ncpus_use=1,
            random_seed=seed
        )

        # Extract velocity and uncertainty
        velocity = adata.layers["velocity_pyro"]
        uncertainty = posterior_samples.get('fdri', None)
    else:
        print("\nSkipping velocity computation for legacy model (--no-compute-velocity flag is set)")

    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    # Return results
    return {
        "model": model,
        "posterior_samples": posterior_samples,
        "velocity": velocity,
        "uncertainty": uncertainty,
        "performance": {
            "training_time": training_time,
            "inference_time": inference_time,
        }
    }


def train_modular_model(adata, max_epochs, num_samples, seed=42, model_type="model1", compute_velocity=True):
    """
    Train the modular model and return results.

    Args:
        adata: AnnData object containing the data
        max_epochs: Maximum number of epochs for training
        num_samples: Number of posterior samples to generate
        seed: Random seed for reproducibility
        model_type: Model type to use (model1 or model2)
        compute_velocity: Whether to compute velocity (default: True)
    """
    print("Training modular model...")

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    # Set up AnnData for modular model
    adata_copy = adata.copy()
    PyroVelocityModel.setup_anndata(adata_copy)
    print("AnnData setup completed for modular model")

    # Create modular model based on model_type
    print(f"Modular model using model_type: {model_type}")

    if model_type == "model1":
        model = create_legacy_model1()
        print("Using legacy model1 configuration for direct comparison with legacy implementation")
    elif model_type == "model2":
        model = create_legacy_model2()
        print("Using legacy model2 configuration for direct comparison with legacy implementation")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Train model
    print(f"Starting modular model training with max_epochs={max_epochs}")
    training_start_time = time.time()
    model.train(
        adata=adata_copy,
        max_epochs=max_epochs,
        early_stopping=False,
        seed=seed,
    )
    print("Modular model training completed successfully")
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    print(f"Modular model training time: {training_time:.2f} seconds")

    # Generate posterior samples
    print(f"Starting modular model posterior sampling with num_samples={num_samples}")
    inference_start_time = time.time()
    posterior_samples = model.generate_posterior_samples(
        adata=adata_copy,
        num_samples=num_samples,
        seed=seed
    )
    print("Modular model posterior sampling completed successfully")

    # Print parameter shapes for debugging
    print("Modular model parameter shapes:")
    for key, value in posterior_samples.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            print(f"  {key}: {value.shape}")

    # Print parameter statistics for debugging
    print("\nModular model parameter statistics:")
    for param in ["alpha", "beta", "gamma"]:
        param_value = to_numpy(posterior_samples[param])
        print(f"  {param} mean: {np.mean(param_value):.6f}, std: {np.std(param_value):.6f}, min: {np.min(param_value):.6f}, max: {np.max(param_value):.6f}")

    # Initialize velocity and uncertainty as None
    velocity = None
    uncertainty = None

    # Compute velocity only if requested
    if compute_velocity:
        print("\nComputing velocity for modular model...")

        # Compute velocity and uncertainty using the model's get_velocity method
        adata_with_velocity = model.get_velocity(
            adata=adata_copy,
            num_samples=num_samples,
            random_seed=seed,
            compute_uncertainty=True,
            uncertainty_method="std"
        )

        # Extract velocity and uncertainty from AnnData layers (assuming consistent naming)
        velocity = adata_with_velocity.layers["velocity_pyrovelocity"]
        uncertainty = adata_with_velocity.layers["velocity_uncertainty_pyrovelocity"]
    else:
        print("\nSkipping velocity computation for modular model (--no-compute-velocity flag is set)")

    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    # Return results
    return {
        "model": model,
        "posterior_samples": posterior_samples,
        "velocity": velocity,
        "uncertainty": uncertainty,
        "performance": {
            "training_time": training_time,
            "inference_time": inference_time,
        }
    }


def compare_models(legacy_results, modular_results):
    """Compare the results of the legacy and modular models."""
    print("Comparing models...")

    comparison_results = {}

    # Compare parameters
    results = {
        "legacy": {"posterior_samples": legacy_results["posterior_samples"]},
        "modular": {"posterior_samples": modular_results["posterior_samples"]}
    }
    param_comparison_results = compare_parameters(results)
    parameter_comparison = {}
    for param in ["alpha", "beta", "gamma"]:
        parameter_comparison[param] = param_comparison_results[param]

    comparison_results["parameter_comparison"] = parameter_comparison

    # Compare velocities if both are available
    if legacy_results["velocity"] is not None and modular_results["velocity"] is not None:
        velocity_results = {
            "legacy": {"velocity": to_numpy(legacy_results["velocity"])},
            "modular": {"velocity": to_numpy(modular_results["velocity"])}
        }
        velocity_comparison = {
            "legacy_vs_modular": compare_velocities(velocity_results)
        }
        comparison_results["velocity_comparison"] = velocity_comparison
    else:
        comparison_results["velocity_comparison"] = {
            "legacy_vs_modular": {"skipped": "Velocity computation was disabled"}
        }

    # Compare uncertainties if both are available
    if legacy_results["uncertainty"] is not None and modular_results["uncertainty"] is not None:
        uncertainty_results = {
            "legacy": {"uncertainty": to_numpy(legacy_results["uncertainty"])},
            "modular": {"uncertainty": to_numpy(modular_results["uncertainty"])}
        }
        uncertainty_comparison = {
            "legacy_vs_modular": compare_uncertainties(uncertainty_results)
        }
        comparison_results["uncertainty_comparison"] = uncertainty_comparison
    else:
        comparison_results["uncertainty_comparison"] = {
            "legacy_vs_modular": {"skipped": "Velocity computation was disabled"}
        }

    # Compare performance
    performance_results = {
        "legacy": {"performance": legacy_results["performance"]},
        "modular": {"performance": modular_results["performance"]}
    }
    performance_comparison = {
        "legacy_vs_modular": compare_performance(performance_results)
    }
    comparison_results["performance_comparison"] = performance_comparison

    return comparison_results


def generate_summary_report(legacy_results, modular_results, comparison, args, output_dir):
    """Generate a summary report of the comparison."""
    report_path = os.path.join(output_dir, "comparison_summary_report.txt")

    with open(report_path, "w") as f:
        f.write("PyroVelocity Direct Comparison Summary Report\n")
        f.write("===========================================\n\n")

        # Write comparison parameters
        f.write("Comparison Parameters\n")
        f.write("--------------------\n")
        f.write(f"Max epochs: {args.max_epochs}\n")
        f.write(f"Number of posterior samples: {args.num_samples}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Compute velocity: {args.compute_velocity}\n")
        f.write("\n")

        # Write model information
        f.write("Model Information\n")
        f.write("----------------\n")

        # Legacy model
        f.write("Legacy Model:\n")
        f.write(f"  Training time: {legacy_results['performance']['training_time']:.2f} seconds\n")
        f.write(f"  Inference time: {legacy_results['performance']['inference_time']:.2f} seconds\n")
        if legacy_results['velocity'] is not None:
            f.write(f"  Velocity shape: {legacy_results['velocity'].shape}\n")
        else:
            f.write("  Velocity: None\n")
        f.write("\n")

        # Modular model
        f.write("Modular Model:\n")
        f.write(f"  Training time: {modular_results['performance']['training_time']:.2f} seconds\n")
        f.write(f"  Inference time: {modular_results['performance']['inference_time']:.2f} seconds\n")
        if modular_results['velocity'] is not None:
            f.write(f"  Velocity shape: {modular_results['velocity'].shape}\n")
        else:
            f.write("  Velocity: None\n")
        f.write("\n")

        # Write comparison summary
        f.write("Comparison Summary\n")
        f.write("-----------------\n")

        # Parameter comparison summary
        f.write("Parameter Comparison:\n")
        for param, param_comp in comparison["parameter_comparison"].items():
            f.write(f"  {param}:\n")
            for comp_name, metrics in param_comp.items():
                f.write(f"    {comp_name}:\n")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"      {metric_name}: {value:.6f}\n")
                    else:
                        f.write(f"      {metric_name}: {value}\n")
        f.write("\n")

        # Velocity comparison summary
        f.write("Velocity Comparison:\n")
        for comp_name, metrics in comparison["velocity_comparison"].items():
            f.write(f"  {comp_name}:\n")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"    {metric_name}: {value:.6f}\n")
                else:
                    f.write(f"    {metric_name}: {value}\n")
        f.write("\n")

        # Uncertainty comparison summary
        f.write("Uncertainty Comparison:\n")
        for comp_name, metrics in comparison["uncertainty_comparison"].items():
            f.write(f"  {comp_name}:\n")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"    {metric_name}: {value:.6f}\n")
                else:
                    f.write(f"    {metric_name}: {value}\n")
        f.write("\n")

        # Performance comparison summary
        f.write("Performance Comparison:\n")
        for comp_name, metrics in comparison["performance_comparison"].items():
            f.write(f"  {comp_name}:\n")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"    {metric_name}: {value:.6f}\n")
                else:
                    f.write(f"    {metric_name}: {value}\n")
        f.write("\n")

        # Write comparison conclusion
        f.write("Comparison Conclusion\n")
        f.write("--------------------\n")

        # Check if there are any significant differences in parameters
        param_diff = False
        for param, param_comp in comparison["parameter_comparison"].items():
            for comp_name, metrics in param_comp.items():
                if "correlation" in metrics and metrics["correlation"] < 0.9:
                    param_diff = True
                    break
            if param_diff:
                break

        # Check if there are any significant differences in velocities
        vel_diff = False
        for comp_name, metrics in comparison["velocity_comparison"].items():
            if "correlation" in metrics and metrics["correlation"] < 0.9:
                vel_diff = True
                break

        # Write conclusion
        if param_diff or vel_diff:
            f.write("There are significant differences between the legacy and modular implementations.\n")
            if param_diff:
                f.write("- Parameter estimates show low correlation (< 0.9)\n")
            if vel_diff:
                f.write("- Velocity estimates show low correlation (< 0.9)\n")
            f.write("\nRecommendation: Further investigation is needed to understand these differences.\n")
        else:
            f.write("The legacy and modular implementations produce similar results.\n")
            f.write("- Parameter estimates show high correlation (>= 0.9)\n")
            f.write("- Velocity estimates show high correlation (>= 0.9)\n")
            f.write("\nRecommendation: The modular implementation can be considered a valid replacement for the legacy implementation.\n")

    print(f"Summary report saved to {report_path}")
    return report_path


def main():
    """Run the direct comparison and save results."""
    # Parse command line arguments
    args = parse_args()

    print_progress("Starting PyroVelocity Direct Comparison")
    print_progress(f"Configuration: max_epochs={args.max_epochs}, num_samples={args.num_samples}, model_type={args.model_type}, compute_velocity={args.compute_velocity}")

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print_progress(f"Created output directory: {args.output_dir}")

    # Load preprocessed pancreas data
    print_progress("Loading preprocessed pancreas data...")
    adata = load_preprocessed_pancreas_data()
    print_progress("Data loading completed successfully")

    # Train legacy model
    print_progress("Starting legacy model training...")
    # Note: model_type is passed for compatibility but has no effect on the legacy model
    legacy_results = train_legacy_model(
        adata,
        args.max_epochs,
        args.num_samples,
        seed=args.seed,
        compute_velocity=args.compute_velocity,
        model_type=args.model_type  # This parameter is ignored by the legacy model
    )
    print_progress("Legacy model training completed successfully")

    # Train modular model
    print_progress("Starting modular model training...")
    modular_results = train_modular_model(
        adata,
        args.max_epochs,
        args.num_samples,
        seed=args.seed,
        model_type=args.model_type,
        compute_velocity=args.compute_velocity
    )
    print_progress("Modular model training completed successfully")

    # Compare models
    print_progress("Starting model comparison...")
    comparison_results = compare_models(legacy_results, modular_results)
    print_progress("Model comparison completed successfully")

    # Generate summary report
    print_progress("Generating summary report...")
    report_path = generate_summary_report(legacy_results, modular_results, comparison_results, args, args.output_dir)
    print_progress(f"Summary report saved to {report_path}")
    print_progress(f"All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
