#!/usr/bin/env python
"""
Hierarchical Spacing Stick-Breaking Model Demonstration Script.

This script implements and demonstrates the hierarchical spacing stick-breaking model
for temporal coordinate assignment in PyroVelocity. The model showcases how learnable
spacing heterogeneity can generate diverse temporal patterns through hierarchical
parameter control.

Mathematical Specification:
- Global heterogeneity parameter: κ ~ Gamma(2.0, 2.0)
- Position-specific spacing rates: α_j ~ Gamma(κ, κ) for j = 1, ..., N-1
- Stick-breaking variables: ξ_j ~ Beta(1.0, α_j)
- Temporal coordinates: t*_1 = 0, t*_N = T_max, with recursive construction

Key Features:
1. Clean implementation of hierarchical spacing model
2. Sample generation with different heterogeneity levels
3. Visualization of multiple sample paths
4. Comparison with standard uniform stick-breaking
5. Uncertainty quantification with credible intervals
6. Analysis of spacing patterns (gaps, clusters)

Usage:
    python hierarchical-stick-breaking.py

The script generates educational plots showing how the hierarchical model works
in practice and what kinds of temporal patterns it can generate.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.distributions import Beta, Gamma

from pyrovelocity.plots.predictive_checks import combine_pdfs
from pyrovelocity.styles import configure_matplotlib_style


configure_matplotlib_style()


# Set color palette
colors = sns.color_palette("husl", 8)
heterogeneity_colors = {
    'low': colors[0],       # Blue
    'moderate': colors[2],  # Green
    'high': colors[1],      # Orange
    'very_high': colors[3], # Red
    'uniform': colors[7]    # Gray
}


def hierarchical_stick_breaking_sample(
    n_cells: int,
    T_max: float,
    kappa: float,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a single sample from the hierarchical spacing stick-breaking model.

    - κ ~ Gamma(2.0, 2.0) [global heterogeneity parameter]
    - α_j ~ Gamma(κ·(N-j), κ) for j ∈ {1, ..., N-1} [position-dependent hierarchical rates]
    - ξ_j ~ Beta(1.0, α_j) for j ∈ {1, ..., N-1} [hierarchical stick-breaking variables]
    - t*_j = t*_{j-1} + ξ_{j-1} × (T_max - t*_{j-1}) [recursive construction]

    Key Properties:
    - E[α_j] = N-j (matches standard stick-breaking expectation)
    - Var[α_j] = (N-j)/κ (heterogeneity controlled by κ)
    - As κ→∞: α_j → N-j deterministically → ξ_j ~ Beta(1, N-j) → exact standard stick-breaking
    - As κ→0: High variability in α_j → meaningful position-specific heterogeneity

    Args:
        n_cells: Number of cells (temporal coordinates)
        T_max: Maximum time value
        kappa: Global heterogeneity parameter
        seed: Random seed for reproducibility

    Returns:
        Tuple of (temporal_coordinates, spacing_rates, stick_breaking_variables)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # CORRECTED: Position-dependent hierarchical rates
    # α_j ~ Gamma(κ·(N-j), κ) for j = 1, ..., N-1
    position_weights = torch.arange(n_cells - 1, 0, -1).float()  # [N-1, N-2, ..., 1]

    spacing_rates = torch.zeros(n_cells - 1)
    for j in range(n_cells - 1):
        concentration = kappa * position_weights[j]  # κ·(N-j)
        rate = kappa
        alpha_dist = Gamma(concentration, rate)
        spacing_rates[j] = alpha_dist.sample()

    # Sample stick-breaking variables
    # ξ_j ~ Beta(1.0, α_j)
    stick_breaking_vars = torch.zeros(n_cells - 1)
    for j in range(n_cells - 1):
        beta_dist = Beta(1.0, spacing_rates[j])
        stick_breaking_vars[j] = beta_dist.sample()

    # Construct temporal coordinates recursively
    # t*_1 = 0, t*_N = T_max
    # t*_j = t*_{j-1} + ξ_{j-1}(T_max - t*_{j-1})
    temporal_coords = torch.zeros(n_cells)
    temporal_coords[0] = 0.0
    temporal_coords[-1] = T_max

    for j in range(1, n_cells - 1):
        remaining_time = T_max - temporal_coords[j-1]
        temporal_coords[j] = temporal_coords[j-1] + stick_breaking_vars[j-1] * remaining_time

    return temporal_coords, spacing_rates, stick_breaking_vars


def uniform_stick_breaking_sample(
    n_cells: int,
    T_max: float,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate a sample from standard uniform stick-breaking for comparison.

    Args:
        n_cells: Number of cells
        T_max: Maximum time value
        seed: Random seed for reproducibility

    Returns:
        Temporal coordinates from uniform stick-breaking
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Standard stick-breaking: ξ_j ~ Beta(1, N-j)
    temporal_coords = torch.zeros(n_cells)
    temporal_coords[0] = 0.0
    temporal_coords[-1] = T_max

    for j in range(1, n_cells - 1):
        beta_dist = Beta(1.0, n_cells - j)
        xi = beta_dist.sample()
        remaining_time = T_max - temporal_coords[j-1]
        temporal_coords[j] = temporal_coords[j-1] + xi * remaining_time

    return temporal_coords


def validate_convergence_to_standard(
    n_cells: int = 20,
    T_max: float = 10.0,
    kappa_values: List[float] = [1.0, 10.0, 100.0, 1000.0],
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Verify hierarchical model converges to standard stick-breaking as κ→∞.

    This function computes the Wasserstein distance between hierarchical and standard
    stick-breaking distributions for increasing κ values to validate mathematical correctness.

    Args:
        n_cells: Number of cells
        T_max: Maximum time value
        kappa_values: List of κ values to test convergence
        n_samples: Number of samples for distance computation
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping κ values to Wasserstein distances from standard stick-breaking
    """
    if seed is not None:
        torch.manual_seed(seed)

    print("🔬 Validating Convergence to Standard Stick-Breaking...")
    print("-" * 50)

    # Generate standard stick-breaking samples for comparison
    standard_samples = []
    for _ in range(n_samples):
        coords = uniform_stick_breaking_sample(n_cells, T_max, seed=None)
        standard_samples.append(coords)
    standard_samples = torch.stack(standard_samples)

    # Compute distances for each κ value
    distances = {}

    for kappa in kappa_values:
        # Generate hierarchical samples
        hierarchical_samples = []
        for _ in range(n_samples):
            coords, _, _ = hierarchical_stick_breaking_sample(n_cells, T_max, kappa, seed=None)
            hierarchical_samples.append(coords)
        hierarchical_samples = torch.stack(hierarchical_samples)

        # Compute Wasserstein distance (approximated by mean absolute difference of sorted samples)
        # This is a simplified metric but sufficient for convergence validation
        standard_sorted = torch.sort(standard_samples.flatten())[0]
        hierarchical_sorted = torch.sort(hierarchical_samples.flatten())[0]

        wasserstein_distance = torch.mean(torch.abs(standard_sorted - hierarchical_sorted)).item()
        distances[kappa] = wasserstein_distance

        print(f"κ = {kappa:8.1f}: Wasserstein distance = {wasserstein_distance:.6f}")

    print("\n✅ Convergence Analysis:")
    print(f"   Distance reduction: {distances[kappa_values[0]]:.6f} → {distances[kappa_values[-1]]:.6f}")
    print(f"   Improvement factor: {distances[kappa_values[0]] / distances[kappa_values[-1]]:.1f}x")

    if distances[kappa_values[-1]] < 0.01:
        print("   ✅ PASS: High κ converges to standard stick-breaking")
    else:
        print("   ❌ FAIL: High κ does not sufficiently converge")

    return distances


def generate_multiple_samples(
    n_cells: int,
    T_max: float,
    kappa_values: Dict[str, float],
    n_samples: int = 20,
    seed: Optional[int] = None
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generate multiple samples for different heterogeneity levels.

    Args:
        n_cells: Number of cells
        T_max: Maximum time value
        kappa_values: Dictionary mapping heterogeneity levels to kappa values
        n_samples: Number of samples per heterogeneity level
        seed: Base random seed

    Returns:
        Dictionary containing samples for each heterogeneity level
    """
    results = {}

    for level, kappa in kappa_values.items():
        samples = {
            'temporal_coords': [],
            'spacing_rates': [],
            'stick_breaking_vars': [],
            'kappa': kappa
        }

        for i in range(n_samples):
            sample_seed = seed + i if seed is not None else None

            if level == 'uniform':
                # Standard uniform stick-breaking
                coords = uniform_stick_breaking_sample(n_cells, T_max, sample_seed)
                samples['temporal_coords'].append(coords)
                # For uniform, we don't have spacing rates or stick-breaking vars
                samples['spacing_rates'].append(torch.ones(n_cells - 1))
                samples['stick_breaking_vars'].append(torch.zeros(n_cells - 1))
            else:
                # Hierarchical stick-breaking
                coords, rates, vars = hierarchical_stick_breaking_sample(
                    n_cells, T_max, kappa, sample_seed
                )
                samples['temporal_coords'].append(coords)
                samples['spacing_rates'].append(rates)
                samples['stick_breaking_vars'].append(vars)

        # Convert lists to tensors
        samples['temporal_coords'] = torch.stack(samples['temporal_coords'])
        samples['spacing_rates'] = torch.stack(samples['spacing_rates'])
        samples['stick_breaking_vars'] = torch.stack(samples['stick_breaking_vars'])

        results[level] = samples

    return results


def compute_spacing_statistics(
    temporal_coords: torch.Tensor
) -> Dict[str, float]:
    """
    Compute spacing statistics for temporal coordinates.

    Args:
        temporal_coords: Tensor of shape (n_samples, n_cells) with temporal coordinates

    Returns:
        Dictionary with spacing statistics
    """
    # Compute spacing between consecutive time points
    spacings = torch.diff(temporal_coords, dim=1)  # Shape: (n_samples, n_cells-1)

    # Compute statistics
    stats = {
        'mean_spacing': spacings.mean().item(),
        'std_spacing': spacings.std().item(),
        'min_spacing': spacings.min().item(),
        'max_spacing': spacings.max().item(),
        'cv_spacing': (spacings.std() / spacings.mean()).item(),  # Coefficient of variation
        'spacing_range': (spacings.max() - spacings.min()).item(),
    }

    return stats


def plot_sample_trajectories(
    samples: Dict[str, Dict[str, torch.Tensor]],
    n_cells: int,
    T_max: float,
    figsize: Tuple[float, float] = (7.5, 6.0),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple sample trajectories for different heterogeneity levels.

    Args:
        samples: Dictionary containing samples for each heterogeneity level
        n_cells: Number of cells
        T_max: Maximum time value
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Adjust subplot layout for 5 levels
    fig, axes = plt.subplots(2, 3, figsize=(11.25, 6.0), sharex=True, sharey=True)
    axes = axes.flatten()

    levels = ['uniform', 'low', 'moderate', 'high', 'very_high']
    level_names = {
        'uniform': 'Standard Uniform\nStick-Breaking',
        'low': r'Low Heterogeneity' + '\n' + r'($\kappa = 0.5$)',
        'moderate': r'Moderate Heterogeneity' + '\n' + r'($\kappa = 2.0$)',
        'high': r'High Heterogeneity' + '\n' + r'($\kappa = 10.0$)',
        'very_high': r'Very High Heterogeneity' + '\n' + r'($\kappa = 100.0$)'
    }

    for idx, level in enumerate(levels):
        ax = axes[idx]

        if level not in samples:
            continue

        coords = samples[level]['temporal_coords']
        color = heterogeneity_colors[level]

        # Plot individual trajectories with transparency
        for i in range(min(15, coords.shape[0])):  # Plot up to 15 trajectories
            cell_indices = torch.arange(n_cells)
            ax.plot(coords[i], cell_indices,
                   color=color, alpha=0.3, linewidth=0.8)

        # Plot mean trajectory
        mean_coords = coords.mean(dim=0)
        ax.plot(mean_coords, torch.arange(n_cells),
               color=color, linewidth=2.5, label='Mean')

        # Plot credible intervals
        lower = torch.quantile(coords, 0.025, dim=0)
        upper = torch.quantile(coords, 0.975, dim=0)
        ax.fill_betweenx(torch.arange(n_cells), lower, upper,
                        color=color, alpha=0.2, label='95% CI')

        ax.set_title(level_names[level], fontsize=10, pad=10)
        ax.set_xlim(0, T_max)
        ax.set_ylim(0, n_cells - 1)
        ax.grid(True, alpha=0.3)

        if idx >= 3:  # Bottom row (adjusted for 2x3 layout)
            ax.set_xlabel(r'Temporal Coordinate $t^*$')
        if idx % 3 == 0:  # Left column (adjusted for 2x3 layout)
            ax.set_ylabel('Cell Index')

    # Hide the extra subplot (6th position)
    if len(levels) < 6:
        axes[5].set_visible(False)

    # Add overall title and legend
    fig.suptitle('Hierarchical Spacing Stick-Breaking: Sample Trajectories',
                fontsize=12, y=0.95)

    # Add legend to the very_high subplot
    if 'very_high' in samples:
        axes[4].legend(loc='lower right', fontsize=8)

    plt.tight_layout()

    if save_path:
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(output_dir / f'01_sample_trajectories.{ext}',
                       dpi=300, bbox_inches='tight')
        print(f"Saved sample trajectories plot to {save_path}")

    return fig


def plot_spacing_analysis(
    samples: Dict[str, Dict[str, torch.Tensor]],
    figsize: Tuple[float, float] = (7.5, 6.0),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot spacing analysis comparing different heterogeneity levels.

    Args:
        samples: Dictionary containing samples for each heterogeneity level
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    levels = ['uniform', 'low', 'moderate', 'high', 'very_high']
    level_names = {
        'uniform': 'Uniform',
        'low': r'Low ($\kappa=0.5$)',
        'moderate': r'Moderate ($\kappa=2.0$)',
        'high': r'High ($\kappa=10.0$)',
        'very_high': r'Very High ($\kappa=100.0$)'
    }

    # Collect spacing data
    all_spacings = {}
    spacing_stats = {}

    for level in levels:
        if level not in samples:
            continue

        coords = samples[level]['temporal_coords']
        spacings = torch.diff(coords, dim=1)  # Shape: (n_samples, n_cells-1)
        all_spacings[level] = spacings.flatten()
        spacing_stats[level] = compute_spacing_statistics(coords)

    # Plot 1: Spacing distributions
    ax1 = axes[0, 0]
    for level in levels:
        if level in all_spacings:
            spacings = all_spacings[level].numpy()
            ax1.hist(spacings, bins=30, alpha=0.6,
                    color=heterogeneity_colors[level],
                    label=level_names[level], density=True)

    ax1.set_xlabel('Spacing Between Consecutive Cells')
    ax1.set_ylabel('Density')
    ax1.set_title('Spacing Distributions')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coefficient of variation
    ax2 = axes[0, 1]
    cv_values = [spacing_stats[level]['cv_spacing'] for level in levels if level in spacing_stats]
    level_labels = [level_names[level] for level in levels if level in spacing_stats]
    colors_list = [heterogeneity_colors[level] for level in levels if level in spacing_stats]

    bars = ax2.bar(range(len(cv_values)), cv_values, color=colors_list, alpha=0.7)
    ax2.set_xlabel('Heterogeneity Level')
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('Spacing Variability')
    ax2.set_xticks(range(len(level_labels)))
    ax2.set_xticklabels(level_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, cv in zip(bars, cv_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{cv:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot 3: Spacing vs position
    ax3 = axes[1, 0]
    for level in levels:
        if level not in samples:
            continue

        coords = samples[level]['temporal_coords']
        spacings = torch.diff(coords, dim=1)
        mean_spacings = spacings.mean(dim=0)
        std_spacings = spacings.std(dim=0)

        positions = torch.arange(1, len(mean_spacings) + 1)
        color = heterogeneity_colors[level]

        ax3.plot(positions, mean_spacings, color=color,
                label=level_names[level], linewidth=2, marker='o', markersize=3)
        ax3.fill_between(positions,
                        mean_spacings - std_spacings,
                        mean_spacings + std_spacings,
                        color=color, alpha=0.2)

    ax3.set_xlabel('Position in Sequence')
    ax3.set_ylabel('Mean Spacing')
    ax3.set_title('Spacing vs Position')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create summary table
    table_data = []
    for level in levels:
        if level in spacing_stats:
            stats = spacing_stats[level]
            table_data.append([
                level_names[level],
                f"{stats['mean_spacing']:.3f}",
                f"{stats['std_spacing']:.3f}",
                f"{stats['cv_spacing']:.3f}"
            ])

    table = ax4.table(cellText=table_data,
                     colLabels=['Level', 'Mean', 'Std', 'CV'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.3, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax4.set_title('Summary Statistics', fontsize=10, pad=20)

    plt.tight_layout()

    if save_path:
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(output_dir / f'02_spacing_analysis.{ext}',
                       dpi=300, bbox_inches='tight')
        print(f"Saved spacing analysis plot to {save_path}")

    return fig


def plot_parameter_analysis(
    samples: Dict[str, Dict[str, torch.Tensor]],
    figsize: Tuple[float, float] = (7.5, 8.0),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot analysis of hierarchical parameters (spacing rates and stick-breaking variables).

    Args:
        samples: Dictionary containing samples for each heterogeneity level
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    levels = ['low', 'moderate', 'high', 'very_high']  # Exclude uniform for parameter analysis
    level_names = {
        'low': r'Low ($\kappa=0.5$)',
        'moderate': r'Moderate ($\kappa=2.0$)',
        'high': r'High ($\kappa=10.0$)',
        'very_high': r'Very High ($\kappa=100.0$)'
    }

    # Plot 1: Spacing rates distributions
    ax1 = axes[0, 0]
    for level in levels:
        if level not in samples:
            continue

        spacing_rates = samples[level]['spacing_rates'].flatten()
        color = heterogeneity_colors[level]

        ax1.hist(spacing_rates.numpy(), bins=30, alpha=0.6,
                color=color, label=level_names[level], density=True)

    ax1.set_xlabel(r'Spacing Rate $\alpha_j$')
    ax1.set_ylabel('Density')
    ax1.set_title('Spacing Rate Distributions')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Stick-breaking variables distributions
    ax2 = axes[0, 1]
    for level in levels:
        if level not in samples:
            continue

        stick_vars = samples[level]['stick_breaking_vars'].flatten()
        # Filter out zeros (from uniform case)
        stick_vars = stick_vars[stick_vars > 0]
        color = heterogeneity_colors[level]

        ax2.hist(stick_vars.numpy(), bins=30, alpha=0.6,
                color=color, label=level_names[level], density=True)

    ax2.set_xlabel(r'Stick-Breaking Variable $\xi_j$')
    ax2.set_ylabel('Density')
    ax2.set_title('Stick-Breaking Variable Distributions')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Spacing rates vs position
    ax3 = axes[1, 0]
    for level in levels:
        if level not in samples:
            continue

        spacing_rates = samples[level]['spacing_rates']  # Shape: (n_samples, n_cells-1)
        mean_rates = spacing_rates.mean(dim=0)
        std_rates = spacing_rates.std(dim=0)

        positions = torch.arange(1, len(mean_rates) + 1)
        color = heterogeneity_colors[level]

        ax3.plot(positions, mean_rates, color=color,
                label=level_names[level], linewidth=2, marker='o', markersize=3)
        ax3.fill_between(positions,
                        mean_rates - std_rates,
                        mean_rates + std_rates,
                        color=color, alpha=0.2)

    ax3.set_xlabel('Position in Sequence')
    ax3.set_ylabel(r'Mean Spacing Rate $\alpha_j$')
    ax3.set_title('Spacing Rates vs Position')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Plot 4: Stick-breaking variables vs position
    ax4 = axes[1, 1]
    for level in levels:
        if level not in samples:
            continue

        stick_vars = samples[level]['stick_breaking_vars']
        # Filter out samples where stick_vars are all zeros
        valid_samples = (stick_vars.sum(dim=1) > 0)
        if valid_samples.sum() == 0:
            continue

        stick_vars = stick_vars[valid_samples]
        mean_vars = stick_vars.mean(dim=0)
        std_vars = stick_vars.std(dim=0)

        positions = torch.arange(1, len(mean_vars) + 1)
        color = heterogeneity_colors[level]

        ax4.plot(positions, mean_vars, color=color,
                label=level_names[level], linewidth=2, marker='o', markersize=3)
        ax4.fill_between(positions,
                        mean_vars - std_vars,
                        mean_vars + std_vars,
                        color=color, alpha=0.2)

    ax4.set_xlabel('Position in Sequence')
    ax4.set_ylabel(r'Mean Stick-Breaking Variable $\xi_j$')
    ax4.set_title('Stick-Breaking Variables vs Position')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Correlation between spacing rates and resulting spacings
    ax5 = axes[2, 0]
    for level in levels:
        if level not in samples:
            continue

        spacing_rates = samples[level]['spacing_rates']
        coords = samples[level]['temporal_coords']
        spacings = torch.diff(coords, dim=1)

        # Flatten for correlation analysis
        rates_flat = spacing_rates.flatten()
        spacings_flat = spacings.flatten()

        color = heterogeneity_colors[level]
        ax5.scatter(rates_flat.numpy(), spacings_flat.numpy(),
                   alpha=0.3, s=1, color=color, label=level_names[level])

    ax5.set_xlabel(r'Spacing Rate $\alpha_j$')
    ax5.set_ylabel('Resulting Spacing')
    ax5.set_title('Spacing Rate vs Resulting Spacing')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    ax5.set_yscale('log')

    # Plot 6: Heterogeneity parameter effect
    ax6 = axes[2, 1]
    kappa_values = []
    mean_spacing_vars = []

    for level in levels:
        if level not in samples:
            continue

        kappa = samples[level]['kappa']
        coords = samples[level]['temporal_coords']
        spacings = torch.diff(coords, dim=1)
        spacing_var = spacings.var(dim=1).mean()  # Mean variance across samples

        kappa_values.append(kappa)
        mean_spacing_vars.append(spacing_var.item())

    colors_list = [heterogeneity_colors[level] for level in levels if level in samples]
    ax6.scatter(kappa_values, mean_spacing_vars,
               c=colors_list, s=100, alpha=0.8)

    # Add labels for each point
    for i, level in enumerate([l for l in levels if l in samples]):
        ax6.annotate(level_names[level],
                    (kappa_values[i], mean_spacing_vars[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left')

    ax6.set_xlabel(r'Heterogeneity Parameter $\kappa$')
    ax6.set_ylabel('Mean Spacing Variance')
    ax6.set_title('Heterogeneity Parameter Effect')
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')
    ax6.set_yscale('log')

    plt.tight_layout()

    if save_path:
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(output_dir / f'03_parameter_analysis.{ext}',
                       dpi=300, bbox_inches='tight')
        print(f"Saved parameter analysis plot to {save_path}")

    return fig


def plot_uncertainty_quantification(
    samples: Dict[str, Dict[str, torch.Tensor]],
    figsize: Tuple[float, float] = (7.5, 6.0),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot uncertainty quantification for different heterogeneity levels.

    Args:
        samples: Dictionary containing samples for each heterogeneity level
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    levels = ['uniform', 'low', 'moderate', 'high', 'very_high']
    level_names = {
        'uniform': 'Uniform',
        'low': r'Low ($\kappa=0.5$)',
        'moderate': r'Moderate ($\kappa=2.0$)',
        'high': r'High ($\kappa=10.0$)',
        'very_high': r'Very High ($\kappa=100.0$)'
    }

    # Plot 1: Credible interval widths
    ax1 = axes[0, 0]
    for level in levels:
        if level not in samples:
            continue

        coords = samples[level]['temporal_coords']
        lower = torch.quantile(coords, 0.025, dim=0)
        upper = torch.quantile(coords, 0.975, dim=0)
        ci_widths = upper - lower

        positions = torch.arange(len(ci_widths))
        color = heterogeneity_colors[level]

        ax1.plot(positions, ci_widths, color=color,
                label=level_names[level], linewidth=2, marker='o', markersize=3)

    ax1.set_xlabel('Cell Index')
    ax1.set_ylabel('95% Credible Interval Width')
    ax1.set_title('Uncertainty vs Position')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coefficient of variation across cells
    ax2 = axes[0, 1]
    cv_data = []
    level_labels = []

    for level in levels:
        if level not in samples:
            continue

        coords = samples[level]['temporal_coords']
        cv_per_cell = coords.std(dim=0) / coords.mean(dim=0)
        cv_data.append(cv_per_cell.numpy())
        level_labels.append(level_names[level])

    # Create box plot
    bp = ax2.boxplot(cv_data, labels=level_labels, patch_artist=True)

    # Color the boxes
    for patch, level in zip(bp['boxes'], levels[:len(cv_data)]):
        if level in heterogeneity_colors:
            patch.set_facecolor(heterogeneity_colors[level])
            patch.set_alpha(0.7)

    ax2.set_xlabel('Heterogeneity Level')
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('Uncertainty Distribution')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # Plot 3: Predictive intervals comparison
    ax3 = axes[1, 0]

    # Select a few representative cells for detailed comparison
    n_cells = samples[list(samples.keys())[0]]['temporal_coords'].shape[1]
    representative_cells = [n_cells//4, n_cells//2, 3*n_cells//4]

    x_offset = 0
    width = 0.8 / len(levels)

    for cell_idx in representative_cells:
        for j, level in enumerate(levels):
            if level not in samples:
                continue

            coords = samples[level]['temporal_coords'][:, cell_idx]

            # Compute percentiles
            percentiles = [5, 25, 50, 75, 95]
            values = [torch.quantile(coords, p/100).item() for p in percentiles]

            x_pos = x_offset + j * width
            color = heterogeneity_colors[level]

            # Plot box-and-whisker style
            ax3.plot([x_pos, x_pos], [values[0], values[4]],
                    color=color, linewidth=2)  # Whiskers
            ax3.plot([x_pos, x_pos], [values[1], values[3]],
                    color=color, linewidth=6, alpha=0.7)  # Box
            ax3.plot(x_pos, values[2], 'o', color=color, markersize=4)  # Median

        x_offset += 1

    ax3.set_xlabel('Representative Cells')
    ax3.set_ylabel('Temporal Coordinate')
    ax3.set_title('Predictive Intervals Comparison')
    ax3.set_xticks(range(len(representative_cells)))
    ax3.set_xticklabels([f'Cell {idx}' for idx in representative_cells])
    ax3.grid(True, alpha=0.3)

    # Create custom legend
    legend_elements = [plt.Line2D([0], [0], color=heterogeneity_colors[level],
                                 linewidth=3, label=level_names[level])
                      for level in levels if level in samples]
    ax3.legend(handles=legend_elements, fontsize=8)

    # Plot 4: Entropy analysis
    ax4 = axes[1, 1]

    # Compute empirical entropy for each level
    entropies = []
    level_list = []

    for level in levels:
        if level not in samples:
            continue

        coords = samples[level]['temporal_coords']

        # Compute entropy by discretizing the temporal coordinates
        # and computing the entropy of the resulting distribution
        n_bins = 20
        total_entropy = 0

        for cell_idx in range(coords.shape[1]):
            cell_coords = coords[:, cell_idx]
            hist, _ = torch.histogram(cell_coords, bins=n_bins)
            probs = hist.float() / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities
            entropy = -(probs * torch.log(probs)).sum()
            total_entropy += entropy

        mean_entropy = total_entropy / coords.shape[1]
        entropies.append(mean_entropy.item())
        level_list.append(level)

    colors_list = [heterogeneity_colors[level] for level in level_list]
    bars = ax4.bar(range(len(entropies)), entropies, color=colors_list, alpha=0.7)

    ax4.set_xlabel('Heterogeneity Level')
    ax4.set_ylabel('Mean Entropy (nats)')
    ax4.set_title('Uncertainty Entropy')
    ax4.set_xticks(range(len(level_list)))
    ax4.set_xticklabels([level_names[level] for level in level_list],
                       rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, entropy in zip(bars, entropies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{entropy:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(output_dir / f'04_uncertainty_quantification.{ext}',
                       dpi=300, bbox_inches='tight')
        print(f"Saved uncertainty quantification plot to {save_path}")

    return fig


def plot_comparison_summary(
    samples: Dict[str, Dict[str, torch.Tensor]],
    figsize: Tuple[float, float] = (7.5, 10.0),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive comparison summary plot.

    Args:
        samples: Dictionary containing samples for each heterogeneity level
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.3)

    levels = ['uniform', 'low', 'moderate', 'high', 'very_high']
    level_names = {
        'uniform': 'Standard Uniform',
        'low': r'Low Heterogeneity ($\kappa=0.5$)',
        'moderate': r'Moderate Heterogeneity ($\kappa=2.0$)',
        'high': r'High Heterogeneity ($\kappa=10.0$)',
        'very_high': r'Very High Heterogeneity ($\kappa=100.0$)'
    }

    # Plot 1: Representative trajectories
    ax1 = fig.add_subplot(gs[0, :])

    for level in levels:
        if level not in samples:
            continue

        coords = samples[level]['temporal_coords']
        color = heterogeneity_colors[level]

        # Plot a few representative trajectories
        for i in range(min(3, coords.shape[0])):
            cell_indices = torch.arange(coords.shape[1])
            alpha = 0.8 if i == 0 else 0.4
            linewidth = 2 if i == 0 else 1
            label = level_names[level] if i == 0 else None

            ax1.plot(coords[i], cell_indices, color=color,
                    alpha=alpha, linewidth=linewidth, label=label)

    ax1.set_xlabel('Temporal Coordinate $t^*$')
    ax1.set_ylabel('Cell Index')
    ax1.set_title('Representative Sample Trajectories')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spacing variability
    ax2 = fig.add_subplot(gs[1, 0])

    cv_values = []
    level_labels = []

    for level in levels:
        if level not in samples:
            continue

        coords = samples[level]['temporal_coords']
        spacings = torch.diff(coords, dim=1)
        cv = (spacings.std() / spacings.mean()).item()
        cv_values.append(cv)
        level_labels.append(level_names[level])

    colors_list = [heterogeneity_colors[level] for level in levels if level in samples]
    ax2.bar(range(len(cv_values)), cv_values, color=colors_list, alpha=0.7)

    ax2.set_xlabel('Model Type')
    ax2.set_ylabel('Spacing CV')
    ax2.set_title('Spacing Variability')
    ax2.set_xticks(range(len(level_labels)))
    ax2.set_xticklabels([name.split('(')[0].strip() for name in level_labels],
                       rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Uncertainty comparison
    ax3 = fig.add_subplot(gs[1, 1])

    mean_ci_widths = []

    for level in levels:
        if level not in samples:
            continue

        coords = samples[level]['temporal_coords']
        lower = torch.quantile(coords, 0.025, dim=0)
        upper = torch.quantile(coords, 0.975, dim=0)
        ci_widths = upper - lower
        mean_ci_widths.append(ci_widths.mean().item())

    ax3.bar(range(len(mean_ci_widths)), mean_ci_widths,
            color=colors_list, alpha=0.7)

    ax3.set_xlabel('Model Type')
    ax3.set_ylabel('Mean CI Width')
    ax3.set_title('Uncertainty Level')
    ax3.set_xticks(range(len(level_labels)))
    ax3.set_xticklabels([name.split('(')[0].strip() for name in level_labels],
                       rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Mathematical explanation
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Add mathematical explanation text
    explanation_text = r"""
Mathematical Framework: Hierarchical Spacing Stick-Breaking

Standard Stick-Breaking:
• $\xi_j \sim \text{Beta}(1, N-j)$  [uniform spacing rates]
• $t^*_j = t^*_{j-1} + \xi_{j-1}(T_{\max} - t^*_{j-1})$

Hierarchical Spacing Model:
• $\kappa \sim \text{Gamma}(2.0, 2.0)$  [global heterogeneity parameter]
• $\alpha_j \sim \text{Gamma}(\kappa, \kappa)$  [position-specific spacing rates]
• $\xi_j \sim \text{Beta}(1.0, \alpha_j)$  [adaptive stick-breaking variables]
• $t^*_j = t^*_{j-1} + \xi_{j-1}(T_{\max} - t^*_{j-1})$  [recursive construction]

Key Insights:
• Higher $\kappa$ → more uniform spacing (lower heterogeneity)
• Lower $\kappa$ → more variable spacing (higher heterogeneity)
• Position-specific rates $\alpha_j$ allow adaptive spacing control
• Maintains ordering constraints: $0 = t^*_1 < t^*_2 < \ldots < t^*_N = T_{\max}$
    """

    ax4.text(0.05, 0.95, explanation_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    # Plot 5: Summary statistics table
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')

    # Compute summary statistics
    table_data = []
    for level in levels:
        if level not in samples:
            continue

        coords = samples[level]['temporal_coords']
        spacings = torch.diff(coords, dim=1)

        stats = {
            'mean_spacing': spacings.mean().item(),
            'cv_spacing': (spacings.std() / spacings.mean()).item(),
            'mean_ci_width': (torch.quantile(coords, 0.975, dim=0) -
                             torch.quantile(coords, 0.025, dim=0)).mean().item(),
        }

        if level != 'uniform':
            kappa = samples[level]['kappa']
            table_data.append([
                level_names[level],
                f"{kappa:.1f}",
                f"{stats['mean_spacing']:.3f}",
                f"{stats['cv_spacing']:.3f}",
                f"{stats['mean_ci_width']:.3f}"
            ])
        else:
            table_data.append([
                level_names[level],
                "N/A",
                f"{stats['mean_spacing']:.3f}",
                f"{stats['cv_spacing']:.3f}",
                f"{stats['mean_ci_width']:.3f}"
            ])

    table = ax5.table(cellText=table_data,
                     colLabels=['Model', r'$\kappa$', 'Mean Spacing', 'Spacing CV', 'Mean CI Width'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.2, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color code the table rows
    for i, level in enumerate([l for l in levels if l in samples]):
        color = heterogeneity_colors[level]
        for j in range(len(table_data[0])):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)

    plt.suptitle('Hierarchical Spacing Stick-Breaking: Comprehensive Comparison',
                fontsize=14, y=0.98)

    if save_path:
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(output_dir / f'05_comparison_summary.{ext}',
                       dpi=300, bbox_inches='tight')
        print(f"Saved comparison summary plot to {save_path}")

    return fig


def main():
    """
    Main function to demonstrate the hierarchical spacing stick-breaking model.
    """
    print("🎯 Hierarchical Spacing Stick-Breaking Model Demonstration")
    print("=" * 60)

    # Set parameters
    n_cells = 20
    T_max = 10.0
    n_samples = 50
    seed = 42

    # Define heterogeneity levels (UPDATED with high κ values for convergence testing)
    kappa_values = {
        'uniform': None,    # Standard stick-breaking
        'low': 0.5,         # High heterogeneity (low kappa)
        'moderate': 2.0,    # Moderate heterogeneity
        'high': 10.0,       # Low heterogeneity (increased from 8.0)
        'very_high': 100.0  # Should approximate standard stick-breaking
    }

    print(f"Parameters:")
    print(f"  • Number of cells: {n_cells}")
    print(f"  • Maximum time: {T_max}")
    print(f"  • Samples per level: {n_samples}")
    print(f"  • Random seed: {seed}")
    print(f"  • Heterogeneity levels: {list(kappa_values.keys())}")
    print()

    # Validate convergence to standard stick-breaking
    print("🧪 Validating Mathematical Correctness...")
    convergence_distances = validate_convergence_to_standard(
        n_cells=n_cells,
        T_max=T_max,
        kappa_values=[0.5, 2.0, 10.0, 100.0, 1000.0],
        n_samples=500,  # Smaller sample for faster validation
        seed=seed
    )
    print()

    # Generate samples
    print("🔬 Generating samples...")
    samples = generate_multiple_samples(
        n_cells=n_cells,
        T_max=T_max,
        kappa_values=kappa_values,
        n_samples=n_samples,
        seed=seed
    )

    # Create output directory
    save_path = "reports/docs/hierarchical_stick_breaking"
    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {save_path}")
    print()

    # Generate plots
    print("📊 Generating visualizations...")

    # Plot 1: Sample trajectories
    print("  • Sample trajectories...")
    fig1 = plot_sample_trajectories(samples, n_cells, T_max, save_path=save_path)
    plt.close(fig1)

    # Plot 2: Spacing analysis
    print("  • Spacing analysis...")
    fig2 = plot_spacing_analysis(samples, save_path=save_path)
    plt.close(fig2)

    # Plot 3: Parameter analysis
    print("  • Parameter analysis...")
    fig3 = plot_parameter_analysis(samples, save_path=save_path)
    plt.close(fig3)

    # Plot 4: Uncertainty quantification
    print("  • Uncertainty quantification...")
    fig4 = plot_uncertainty_quantification(samples, save_path=save_path)
    plt.close(fig4)

    # Plot 5: Comprehensive comparison
    print("  • Comprehensive comparison...")
    fig5 = plot_comparison_summary(samples, save_path=save_path)
    plt.close(fig5)

    print()
    print("✅ Analysis complete!")
    print(f"📈 All plots saved to: {save_path}")

    # Combine all PDFs into a single document
    print("\n📄 Combining PDFs...")
    try:
        combine_pdfs(
            pdf_directory=save_path,
            output_filename="hierarchical_stick_breaking_complete_analysis.pdf",
            pdf_pattern="*.pdf",
            exclude_patterns=["hierarchical_stick_breaking_complete_analysis.pdf"]  # Don't include the output file itself
        )
        print(f"✅ Combined PDF created: {save_path}/hierarchical_stick_breaking_complete_analysis.pdf")
    except Exception as e:
        print(f"⚠️  Warning: Could not combine PDFs: {e}")
        print("   Individual PDF files are still available in the output directory.")

    # Print summary statistics
    print("\n📋 Summary Statistics:")
    print("-" * 40)

    for level, data in samples.items():
        coords = data['temporal_coords']
        spacings = torch.diff(coords, dim=1)

        kappa_str = f"κ={data['kappa']}" if level != 'uniform' else 'Standard'
        print(f"\n{level.upper()} ({kappa_str}):")
        print(f"  Mean spacing: {spacings.mean():.3f} ± {spacings.std():.3f}")
        print(f"  Spacing CV: {(spacings.std() / spacings.mean()):.3f}")

        # Compute credible intervals
        lower = torch.quantile(coords, 0.025, dim=0)
        upper = torch.quantile(coords, 0.975, dim=0)
        ci_widths = upper - lower
        print(f"  Mean CI width: {ci_widths.mean():.3f}")
        print(f"  Max CI width: {ci_widths.max():.3f}")

    print("\n🔬 Mathematical Validation Results:")
    print("-" * 40)

    # Compare very_high κ with standard stick-breaking
    if 'very_high' in samples and 'uniform' in samples:
        uniform_coords = samples['uniform']['temporal_coords']
        very_high_coords = samples['very_high']['temporal_coords']

        # Compute mean absolute difference
        uniform_mean = uniform_coords.mean(dim=0)
        very_high_mean = very_high_coords.mean(dim=0)
        mean_diff = torch.mean(torch.abs(uniform_mean - very_high_mean)).item()

        # Compute spacing correlation
        uniform_spacings = torch.diff(uniform_coords, dim=1).flatten()
        very_high_spacings = torch.diff(very_high_coords, dim=1).flatten()

        # Compute correlation coefficient
        uniform_centered = uniform_spacings - uniform_spacings.mean()
        very_high_centered = very_high_spacings - very_high_spacings.mean()
        correlation = torch.sum(uniform_centered * very_high_centered) / (
            torch.sqrt(torch.sum(uniform_centered**2)) * torch.sqrt(torch.sum(very_high_centered**2))
        )

        print(f"✅ Convergence Metrics:")
        print(f"   Mean trajectory difference (κ=100 vs uniform): {mean_diff:.4f}")
        print(f"   Spacing correlation (κ=100 vs uniform): {correlation:.4f}")

        if mean_diff < 0.5 and correlation > 0.8:
            print("   ✅ PASS: High κ successfully approximates standard stick-breaking")
        else:
            print("   ⚠️  PARTIAL: Some convergence observed but not complete")

    print("\n🎓 Educational Insights:")
    print("-" * 40)
    print("• CORRECTED: Position-dependent rates α_j ~ Gamma(κ·(N-j), κ)")
    print("• Lower κ values create more heterogeneous spacing patterns")
    print("• Higher κ values approach uniform stick-breaking behavior")
    print("• Position-specific rates α_j enable adaptive temporal resolution")
    print("• The model maintains biological ordering constraints")
    print("• Uncertainty quantification reveals model flexibility")
    print("• Mathematical fix ensures proper convergence as κ→∞")

    print(f"\n🔗 For detailed mathematical derivations, see:")
    print(f"   pyrovelocity-01032024/nbs/concepts/parameter-recovery-validation/")


if __name__ == "__main__":
    main()