#!/usr/bin/env python3
"""
Generate pancreas fixture from original raw data.

This script creates an optimized fixture by extracting the identified optimal
genes and cells from the original raw pancreas() dataset, not from preprocessed data.
The fixture is designed to be minimal while maintaining full preprocessing pipeline compatibility.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from pyrovelocity.io.datasets import pancreas
from pyrovelocity.io.serialization import save_anndata_to_json


def generate_pancreas_fixture(
    output_path: Path,
    selected_genes: list[str],
    n_cells_per_cluster: int = 12,
    random_state: int = 42,
) -> str:
    """
    Generate pancreas fixture from original raw data.
    
    Args:
        output_path: Path to save the fixture
        selected_genes: List of genes to include
        n_cells_per_cluster: Number of cells per cluster for stratified sampling
        random_state: Random state for reproducibility
        
    Returns:
        SHA256 hash of the generated fixture
    """
    print("Loading original pancreas dataset...")
    adata_raw = pancreas()
    
    print(f"Original dataset shape: {adata_raw.shape}")
    print(f"Available genes: {len(adata_raw.var_names)}")
    print(f"Selected genes: {selected_genes}")
    
    # Check if all selected genes are available
    missing_genes = [g for g in selected_genes if g not in adata_raw.var_names]
    if missing_genes:
        raise ValueError(f"Missing genes in dataset: {missing_genes}")
    
    # Filter to selected genes
    adata_genes = adata_raw[:, selected_genes].copy()
    print(f"After gene filtering: {adata_genes.shape}")
    
    # Stratified sampling of cells by cluster
    np.random.seed(random_state)
    
    # Get unique clusters
    clusters = adata_genes.obs['clusters'].unique()
    print(f"Available clusters: {clusters}")
    
    selected_indices = []
    for cluster in clusters:
        cluster_indices = np.where(adata_genes.obs['clusters'] == cluster)[0]
        if len(cluster_indices) >= n_cells_per_cluster:
            selected = np.random.choice(
                cluster_indices, 
                size=n_cells_per_cluster, 
                replace=False
            )
        else:
            selected = cluster_indices
        selected_indices.extend(selected)
    
    selected_indices = np.array(selected_indices)
    print(f"Selected {len(selected_indices)} cells across {len(clusters)} clusters")
    
    # Create optimized fixture with only essential data
    adata_minimal = adata_genes[selected_indices, :].copy()
    
    # Keep only essential obs columns (remove all preprocessing artifacts)
    essential_obs_columns = ['clusters_coarse', 'clusters']
    adata_minimal.obs = adata_minimal.obs[essential_obs_columns]
    
    # Keep only essential var columns (remove all preprocessing artifacts)
    essential_var_columns = []  # Start with no var columns
    if 'mt' in adata_minimal.var.columns:
        essential_var_columns.append('mt')
    if 'ribo' in adata_minimal.var.columns:
        essential_var_columns.append('ribo')
    
    if essential_var_columns:
        adata_minimal.var = adata_minimal.var[essential_var_columns]
    else:
        adata_minimal.var = pd.DataFrame(index=adata_minimal.var_names)
    
    # Keep only raw expression layers (remove all preprocessing artifacts)
    essential_layers = []
    if 'raw_spliced' in adata_minimal.layers:
        essential_layers.append('raw_spliced')
    if 'raw_unspliced' in adata_minimal.layers:
        essential_layers.append('raw_unspliced')
    if 'spliced' in adata_minimal.layers:
        essential_layers.append('spliced')
    if 'unspliced' in adata_minimal.layers:
        essential_layers.append('unspliced')
    
    # Create new layers dict with only essential layers
    new_layers = {}
    for layer in essential_layers:
        if layer in adata_minimal.layers:
            new_layers[layer] = adata_minimal.layers[layer]
    adata_minimal.layers = new_layers
    
    # Remove all obsm, varm, and uns data (preprocessing artifacts)
    adata_minimal.obsm = {}
    adata_minimal.varm = {}
    adata_minimal.uns = {}
    
    print(f"Final fixture shape: {adata_minimal.shape}")
    print(f"Obs columns: {len(adata_minimal.obs.columns)} - {list(adata_minimal.obs.columns)}")
    print(f"Var columns: {len(adata_minimal.var.columns)} - {list(adata_minimal.var.columns)}")
    print(f"Layers: {list(adata_minimal.layers.keys())}")
    
    # Save the fixture as JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_hash = save_anndata_to_json(
        adata=adata_minimal,
        filename=output_path
    )
    
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"Saved fixture: {output_path}")
    print(f"File size: {file_size:.2f} MB")
    print(f"SHA256 hash: {file_hash}")
    
    return file_hash


def main():
    # Optimal genes identified by bisection search
    selected_genes = [
        'Ins2', 'Pcsk2', 'Mboat4', 'Ppy', 'Bicc1', 
        'Meis2', 'Nnat', 'Ppp1r1a', 'Cenpf', 'Mapt'
    ]
    
    output_path = Path("src/pyrovelocity/tests/data/pancreas_raw_96_10.json")
    
    try:
        file_hash = generate_pancreas_fixture(
            output_path=output_path,
            selected_genes=selected_genes,
            n_cells_per_cluster=12,
            random_state=42
        )
        
        print(f"\n✅ SUCCESS: Generated pancreas fixture in JSON format")
        print(f"📁 Path: {output_path}")
        print(f"🔒 Hash: {file_hash}")
        print(f"\n🔧 Next steps:")
        print(f"1. Update conftest.py fixture name to 'pancreas_raw_96_10.json'")
        print(f"2. Update conftest.py with new hash: {file_hash}")
        print(f"3. Update fixture to use load_anndata_from_json instead of sc.read_h5ad")
        print(f"4. Run test: pytest src/pyrovelocity/tests/tasks/test_preprocess.py::test_preprocess_dataset_pancreas -v")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()