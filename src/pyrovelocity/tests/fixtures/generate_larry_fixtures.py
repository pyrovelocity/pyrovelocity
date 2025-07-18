#!/usr/bin/env python
"""
Consolidated Larry fixture generation script.

This script generates all required Larry test fixtures with optimized sizes:
- larry_cospar_100_6.json (~8KB) - COSPAR compatibility fixture
- larry_mono_100_6.json (~117KB) - Mono dataset with clone structure
- larry_neu_100_6.json (~123KB) - Neu dataset with clone structure

All fixtures include cells from the multilineage fixture for cross-dataset compatibility.

Usage:
    python src/pyrovelocity/tests/fixtures/generate_larry_fixtures.py
"""

import sys
from pathlib import Path
from importlib.resources import files

import numpy as np
import pandas as pd
from anndata import AnnData
from beartype import beartype

from pyrovelocity.io.datasets import larry_cospar, larry_mono, larry_neu
from pyrovelocity.io.serialization import load_anndata_from_json, save_anndata_to_json
from pyrovelocity.utils import configure_logging

logger = configure_logging(__name__)

# Target genes that must be present in all fixtures
TARGET_GENES = ["Itgb2", "S100a9", "Fcer1g", "Lilrb4", "Vim", "Serbp1"]


@beartype
def get_multilineage_base_cells() -> dict[str, list[str]]:
    """
    Get base cell names from the existing multilineage fixture.
    
    All new fixtures must include these cells to ensure cross-dataset compatibility.
    
    Returns:
        Dict with cell names converted for each dataset type
    """
    logger.info("Loading multilineage fixture to get base cells")
    
    multilineage_path = files("pyrovelocity.tests.data") / "postprocessed_larry_multilineage_50_6.json"
    adata_multilineage = load_anndata_from_json(multilineage_path)
    
    multilineage_cells = list(adata_multilineage.obs_names)
    logger.info(f"Multilineage fixture has {len(multilineage_cells)} base cells")
    
    # Convert cell names for different datasets based on their naming conventions:
    # - COSPAR: remove -N-N suffix (e.g., "LSK_d6_2_1:TCCAGAAGTTCGTTCC-0-0" -> "LSK_d6_2_1:TCCAGAAGTTCGTTCC")
    # - Mono/Neu: convert -N-N to -N (e.g., "LSK_d6_2_1:TCCAGAAGTTCGTTCC-0-0" -> "LSK_d6_2_1:TCCAGAAGTTCGTTCC-0")
    
    cospar_cells = [cell.rsplit("-", 2)[0] for cell in multilineage_cells]
    mono_neu_cells = []
    
    for cell in multilineage_cells:
        if "-" in cell:
            parts = cell.rsplit("-", 2)
            if len(parts) == 3:  # Has -N-N pattern
                patched_cell = f"{parts[0]}-{parts[1]}"
            else:
                patched_cell = cell
            mono_neu_cells.append(patched_cell)
        else:
            mono_neu_cells.append(cell)
    
    logger.info(f"Converted {len(cospar_cells)} cell names for COSPAR")
    logger.info(f"Converted {len(mono_neu_cells)} cell names for mono/neu")
    
    return {
        "cospar": cospar_cells,
        "mono": mono_neu_cells,
        "neu": mono_neu_cells,
        "original": multilineage_cells
    }


@beartype
def analyze_clone_structure(adata: AnnData, dataset_name: str) -> dict:
    """
    Analyze clone-time structure for trajectory generation capability.
    
    Args:
        adata: AnnData object with X_clone and time_info
        dataset_name: Name for logging
        
    Returns:
        Dict with clone analysis results
    """
    logger.info(f"Analyzing clone structure for {dataset_name}")
    
    n_clones = adata.obsm['X_clone'].shape[1]
    good_clones = []
    clone_info = {}
    
    for i in range(n_clones):
        clone_cells = adata.obsm['X_clone'][:, i] >= 1
        n_cells = clone_cells.sum()
        
        if n_cells >= 2:  # At least 2 cells in clone
            clone_times = adata.obs.loc[clone_cells, 'time_info'].unique()
            if len(clone_times) >= 2:  # At least 2 time points
                cells_per_time = adata.obs.loc[clone_cells, 'time_info'].value_counts().sort_index()
                good_clones.append(i)
                clone_info[i] = {
                    'n_cells': n_cells,
                    'n_times': len(clone_times),
                    'cells_per_time': cells_per_time.to_dict()
                }
    
    logger.info(f"{dataset_name}: {len(good_clones)} good clones out of {n_clones}")
    
    return {
        'good_clones': good_clones,
        'clone_info': clone_info,
        'total_clones': n_clones
    }


@beartype
def select_cells_with_clone_structure(
    adata: AnnData, 
    required_cell_names: list[str],
    target_cells: int = 100,
    min_clones: int = 10
) -> list[int]:
    """
    Select cells that preserve clone trajectory structure while including required cells.
    
    Args:
        adata: Source AnnData object
        required_cell_names: Cell names that must be included
        target_cells: Target number of cells
        min_clones: Minimum number of clones to preserve
        
    Returns:
        List of cell indices to include
    """
    # Find indices of required cells
    available_cells = set(adata.obs_names)
    required_indices = []
    missing_required = []
    
    for cell_name in required_cell_names:
        if cell_name in available_cells:
            idx = list(adata.obs_names).index(cell_name)
            required_indices.append(int(idx))
        else:
            missing_required.append(cell_name)
    
    logger.info(f"Found {len(required_indices)} required cells out of {len(required_cell_names)}")
    if missing_required:
        logger.warning(f"Missing {len(missing_required)} required cells")
    
    if not required_indices:
        raise ValueError("No required cells found in dataset")
    
    # If we already have enough cells, return required indices
    if len(required_indices) >= target_cells:
        logger.info(f"Required cells ({len(required_indices)}) already meet target ({target_cells})")
        return required_indices[:target_cells]
    
    # Analyze clone structure to add more cells
    clone_analysis = analyze_clone_structure(adata, "selection")
    good_clones = clone_analysis['good_clones']
    
    if len(good_clones) < min_clones:
        logger.warning(f"Only {len(good_clones)} good clones available, less than minimum {min_clones}")
    
    # Score clones by cells * time_points
    clone_scores = []
    for clone_idx in good_clones:
        clone_cells = adata.obsm['X_clone'][:, clone_idx] >= 1
        n_cells = clone_cells.sum()
        times = adata.obs.loc[clone_cells, 'time_info'].unique()
        n_times = len(times)
        
        # Score: prefer clones with more cells and more time points
        score = n_cells * n_times
        clone_scores.append((score, clone_idx, n_cells))
    
    # Sort by score and collect additional cells
    clone_scores.sort(reverse=True)
    additional_indices = []
    
    for score, clone_idx, n_cells in clone_scores:
        clone_cells = np.where(adata.obsm['X_clone'][:, clone_idx] >= 1)[0]
        for cell_idx in clone_cells:
            cell_idx = int(cell_idx)
            if cell_idx not in required_indices and cell_idx not in additional_indices:
                additional_indices.append(cell_idx)
                if len(required_indices) + len(additional_indices) >= target_cells:
                    break
        if len(required_indices) + len(additional_indices) >= target_cells:
            break
    
    # Combine required and additional cells
    selected_cell_indices = required_indices + additional_indices[:target_cells - len(required_indices)]
    
    logger.info(f"Selected {len(selected_cell_indices)} cells:")
    logger.info(f"  - {len(required_indices)} required cells")
    logger.info(f"  - {len(additional_indices[:target_cells - len(required_indices)])} additional cells")
    
    return selected_cell_indices


@beartype
def create_cospar_fixture(
    base_cells: list[str],
    output_filename: str = "larry_cospar_100_6.json",
) -> Path:
    """
    Create minimal COSPAR fixture with compatibility cells only.
    
    Args:
        base_cells: Cell names that must be included
        output_filename: Name of output file
        
    Returns:
        Path to saved fixture
    """
    script_dir = Path(__file__).parent
    output_path = script_dir.parent / "data" / output_filename
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Creating COSPAR fixture")
    logger.info(f"Output: {output_path.absolute()}")
    
    # Load full COSPAR dataset
    logger.info("Loading full COSPAR dataset...")
    adata_cospar = larry_cospar()
    logger.info(f"Loaded COSPAR: {adata_cospar.shape}")
    
    # Find matching cells
    available_cells = set(adata_cospar.obs_names)
    matching_cells = [cell for cell in base_cells if cell in available_cells]
    missing_cells = [cell for cell in base_cells if cell not in available_cells]
    
    logger.info(f"Matching cells: {len(matching_cells)}/{len(base_cells)}")
    if missing_cells:
        logger.warning(f"Missing {len(missing_cells)} cells")
    
    if not matching_cells:
        raise ValueError("No matching cells found in COSPAR dataset")
    
    # Extract subset
    adata_subset = adata_cospar[matching_cells, :].copy()
    logger.info(f"Extracted {adata_subset.n_obs} cells")
    
    # Select target genes
    available_genes = [gene for gene in TARGET_GENES if gene in adata_subset.var_names]
    missing_genes = set(TARGET_GENES) - set(available_genes)
    
    if missing_genes:
        logger.warning(f"Missing genes: {missing_genes}")
    
    if available_genes:
        logger.info(f"Subsetting to {len(available_genes)} genes: {available_genes}")
        adata_subset = adata_subset[:, available_genes].copy()
    else:
        logger.warning("No target genes found! Using first 6 genes")
        adata_subset = adata_subset[:, :6].copy()
    
    # Create minimal AnnData with only required fields for COSPAR
    minimal_fields = ["fate_potency_transition_map"]
    available_fields = [f for f in minimal_fields if f in adata_subset.obs.columns]
    
    minimal_adata = AnnData(
        X=adata_subset.X,
        obs=adata_subset.obs[available_fields] if available_fields else adata_subset.obs,
        var=adata_subset.var[[]],  # Empty var dataframe to minimize size
        obsm={"X_emb": adata_subset.obsm["X_emb"]} if "X_emb" in adata_subset.obsm else {},
    )
    
    logger.info(f"Final shape: {minimal_adata.shape}")
    
    # Save fixture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_anndata_to_json(minimal_adata, output_path)
    
    # Report file size
    size_kb = output_path.stat().st_size / 1024
    logger.info(f"Saved {output_filename}: {size_kb:.1f} KB")
    
    return output_path


@beartype
def create_clone_aware_fixture(
    dataset_name: str,
    dataset_loader,
    base_cells: list[str],
    output_filename: str,
    target_cells: int = 100,
    min_clones: int = 10,
) -> Path:
    """
    Create fixture with clone trajectory capability.
    
    Args:
        dataset_name: Name of dataset
        dataset_loader: Function to load full dataset
        base_cells: Cell names that must be included
        output_filename: Name of output file
        target_cells: Target number of cells
        min_clones: Minimum number of clones to preserve
        
    Returns:
        Path to saved fixture
    """
    script_dir = Path(__file__).parent
    output_path = script_dir.parent / "data" / output_filename
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Creating clone-aware {dataset_name} fixture")
    logger.info(f"Output: {output_path.absolute()}")
    
    # Load full dataset
    logger.info(f"Loading full {dataset_name} dataset...")
    adata_full = dataset_loader()
    logger.info(f"Loaded {dataset_name}: {adata_full.shape}")
    
    # Select cells with clone structure preservation
    selected_indices = select_cells_with_clone_structure(
        adata_full,
        required_cell_names=base_cells,
        target_cells=target_cells,
        min_clones=min_clones
    )
    
    # Extract cell subset
    adata_subset = adata_full[selected_indices, :].copy()
    logger.info(f"Extracted {adata_subset.n_obs} cells")
    
    # Verify clone structure is preserved
    clone_analysis = analyze_clone_structure(adata_subset, f"{dataset_name}_subset")
    
    # Select target genes
    available_genes = [gene for gene in TARGET_GENES if gene in adata_subset.var_names]
    missing_genes = set(TARGET_GENES) - set(available_genes)
    
    if missing_genes:
        logger.warning(f"Missing genes: {missing_genes}")
    
    if available_genes:
        logger.info(f"Subsetting to {len(available_genes)} genes: {available_genes}")
        adata_subset = adata_subset[:, available_genes].copy()
    else:
        logger.warning("No target genes found! Using first 6 genes")
        adata_subset = adata_subset[:, :6].copy()
    
    logger.info(f"Final shape: {adata_subset.shape}")
    
    # Save fixture
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_anndata_to_json(adata_subset, output_path)
    
    # Report file size
    size_kb = output_path.stat().st_size / 1024
    logger.info(f"Saved {output_filename}: {size_kb:.1f} KB")
    
    return output_path


@beartype
def validate_fixtures():
    """Validate that all fixtures have compatible structure."""
    logger.info(f"\n{'='*60}")
    logger.info("Validating fixtures...")
    
    fixture_dir = Path(__file__).parent.parent / "data"
    fixtures = {
        "multilineage": fixture_dir / "postprocessed_larry_multilineage_50_6.json",
        "cospar": fixture_dir / "larry_cospar_100_6.json",
        "mono": fixture_dir / "larry_mono_100_6.json", 
        "neu": fixture_dir / "larry_neu_100_6.json",
    }
    
    # Load all fixtures
    loaded = {}
    for name, path in fixtures.items():
        if path.exists():
            loaded[name] = load_anndata_from_json(path)
            logger.info(f"✅ Loaded {name}: {loaded[name].shape}")
        else:
            logger.warning(f"❌ Missing {name}: {path}")
    
    # Check genes match
    if len(loaded) > 1:
        gene_sets = {name: set(adata.var_names) for name, adata in loaded.items()}
        common_genes = set.intersection(*gene_sets.values())
        logger.info(f"Common genes across fixtures: {sorted(common_genes)}")
        
        for name, genes in gene_sets.items():
            unique = genes - common_genes
            if unique:
                logger.warning(f"{name} has unique genes: {unique}")
    
    # Check cell counts
    cell_counts = {name: adata.n_obs for name, adata in loaded.items()}
    logger.info(f"Cell counts: {cell_counts}")
    
    # Check clone structure for mono/neu
    for name in ["mono", "neu"]:
        if name in loaded:
            adata = loaded[name]
            if "X_clone" in adata.obsm:
                clone_analysis = analyze_clone_structure(adata, name)
                logger.info(f"{name} clone structure: {len(clone_analysis['good_clones'])} good clones")
            else:
                logger.warning(f"{name} missing X_clone data")
    
    logger.info("✅ Validation complete")


def main():
    """Generate all Larry fixtures with optimized sizes and clone structure."""
    logger.info("Starting consolidated Larry fixture generation")
    
    try:
        # Get base cells from multilineage fixture
        base_cells = get_multilineage_base_cells()
        
        fixtures_created = []
        
        # 1. COSPAR fixture (minimal, ~8KB)
        cospar_path = create_cospar_fixture(
            base_cells=base_cells["cospar"],
            output_filename="larry_cospar_100_6.json",
        )
        fixtures_created.append(cospar_path)
        
        # 2. Mono fixture (with clone structure, ~117KB) 
        mono_path = create_clone_aware_fixture(
            dataset_name="larry_mono",
            dataset_loader=larry_mono,
            base_cells=base_cells["mono"],
            output_filename="larry_mono_100_6.json",
            target_cells=100,
            min_clones=10,
        )
        fixtures_created.append(mono_path)
        
        # 3. Neu fixture (with clone structure, ~123KB)
        neu_path = create_clone_aware_fixture(
            dataset_name="larry_neu",
            dataset_loader=larry_neu,
            base_cells=base_cells["neu"],
            output_filename="larry_neu_100_6.json",
            target_cells=100,
            min_clones=10,
        )
        fixtures_created.append(neu_path)
        
        # Validate all fixtures
        validate_fixtures()
        
        # Print summary
        print("\n" + "="*60)
        print("✅ Successfully created all Larry fixtures:")
        for path in fixtures_created:
            size_kb = path.stat().st_size / 1024
            print(f"  - {path.name}: {size_kb:.1f} KB")
        
        print("\nNext steps:")
        print("1. Run tests to verify fixtures work:")
        print("   pytest src/pyrovelocity/tests/plots/test_lineage_fate_correlation.py -v")
        print("   pytest src/pyrovelocity/tests/tasks/test_time_fate_correlation.py -v")
        print("")
        print("2. Update fixture hashes in conftest.py using:")
        print("   python src/pyrovelocity/tests/fixtures/get_fixture_hashes.py")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Failed to create fixtures: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()