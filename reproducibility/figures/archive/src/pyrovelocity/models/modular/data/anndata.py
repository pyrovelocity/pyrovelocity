"""
AnnData integration utilities for PyroVelocity PyTorch/Pyro modular implementation.

This module contains utilities for AnnData integration, including:

- prepare_anndata: Convert AnnData to PyTorch tensors
- extract_layers: Extract layers from AnnData
- store_results: Store results in AnnData
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import torch
from beartype import beartype
from jaxtyping import Array, Float


@beartype
def prepare_anndata(
    adata: anndata.AnnData,
    spliced_layer: str = "spliced",
    unspliced_layer: str = "unspliced",
    use_raw: bool = False,
) -> Dict[str, torch.Tensor]:
    """Convert AnnData to PyTorch tensors.

    Args:
        adata: AnnData object
        spliced_layer: Name of the spliced layer
        unspliced_layer: Name of the unspliced layer
        use_raw: Whether to use raw data

    Returns:
        Dictionary of PyTorch tensors
    """
    # Extract layers
    u, s = extract_layers(adata, spliced_layer, unspliced_layer, use_raw)

    # Extract metadata
    cell_types = None
    if "cell_type" in adata.obs:
        # Check if cell_type is categorical
        if hasattr(adata.obs["cell_type"], "cat") and hasattr(
            adata.obs["cell_type"].cat, "categories"
        ):
            cell_types = torch.tensor(
                [
                    adata.obs["cell_type"].cat.categories.get_loc(ct)
                    for ct in adata.obs["cell_type"]
                ]
            )
        else:
            # If not categorical, convert to numerical representation
            unique_types = list(set(adata.obs["cell_type"]))
            cell_types = torch.tensor(
                [unique_types.index(ct) for ct in adata.obs["cell_type"]]
            )
    elif "clusters" in adata.obs:
        # Check if clusters is categorical
        if hasattr(adata.obs["clusters"], "cat") and hasattr(
            adata.obs["clusters"].cat, "categories"
        ):
            cell_types = torch.tensor(
                [
                    adata.obs["clusters"].cat.categories.get_loc(ct)
                    for ct in adata.obs["clusters"]
                ]
            )
        else:
            # If not categorical, convert to numerical representation
            unique_clusters = list(set(adata.obs["clusters"]))
            cell_types = torch.tensor(
                [unique_clusters.index(ct) for ct in adata.obs["clusters"]]
            )

    gene_names = torch.tensor(range(adata.n_vars))
    if "gene_name" in adata.var:
        gene_names = torch.tensor([i for i in range(adata.n_vars)])

    # Create dictionary of PyTorch tensors
    data_dict = {
        "X_unspliced": u,
        "X_spliced": s,
        "cell_types": cell_types,
        "gene_names": gene_names,
    }

    # Add library sizes
    u_lib_size, s_lib_size = get_library_size(
        adata, spliced_layer, unspliced_layer, use_raw
    )
    data_dict["u_lib_size"] = u_lib_size
    data_dict["s_lib_size"] = s_lib_size

    return data_dict


@beartype
def extract_layers(
    adata: anndata.AnnData,
    spliced_layer: str = "spliced",
    unspliced_layer: str = "unspliced",
    use_raw: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract layers from AnnData.

    Args:
        adata: AnnData object
        spliced_layer: Name of the spliced layer
        unspliced_layer: Name of the unspliced layer
        use_raw: Whether to use raw data

    Returns:
        Tuple of (unspliced, spliced) PyTorch tensors
    """
    if use_raw and adata.raw is not None:
        if spliced_layer in adata.raw.layers:
            s = torch.tensor(adata.raw.layers[spliced_layer].toarray(), dtype=torch.float32)
        else:
            s = torch.tensor(adata.raw.X.toarray(), dtype=torch.float32)

        if unspliced_layer in adata.raw.layers:
            u = torch.tensor(adata.raw.layers[unspliced_layer].toarray(), dtype=torch.float32)
        else:
            raise ValueError(
                f"Layer '{unspliced_layer}' not found in adata.raw.layers"
            )
    else:
        if spliced_layer in adata.layers:
            s = torch.tensor(
                adata.layers[spliced_layer].toarray()
                if hasattr(adata.layers[spliced_layer], "toarray")
                else adata.layers[spliced_layer],
                dtype=torch.float32
            )
        else:
            s = torch.tensor(
                adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                dtype=torch.float32
            )

        if unspliced_layer in adata.layers:
            u = torch.tensor(
                adata.layers[unspliced_layer].toarray()
                if hasattr(adata.layers[unspliced_layer], "toarray")
                else adata.layers[unspliced_layer],
                dtype=torch.float32
            )
        else:
            raise ValueError(
                f"Layer '{unspliced_layer}' not found in adata.layers"
            )

    return u, s


@beartype
def store_results(
    adata: anndata.AnnData,
    results: Dict[str, Union[torch.Tensor, np.ndarray]],
    model_name: str = "velocity_model",
) -> anndata.AnnData:
    """Store results in AnnData.

    Args:
        adata: AnnData object
        results: Dictionary of results
        model_name: Name of the model

    Returns:
        Updated AnnData object
    """
    # Create a copy of the AnnData object to avoid modifying the original
    adata_out = adata.copy()

    # Store each result in the appropriate location
    for key, value in results.items():
        # Convert PyTorch tensor to numpy array if needed
        if isinstance(value, torch.Tensor):
            value_np = value.detach().cpu().numpy()
        else:
            value_np = value

        # Special handling for alpha, beta, gamma - store in var
        if key in ["alpha", "beta", "gamma"]:
            # If value is multi-dimensional, take the mean across samples
            if value_np.ndim > 1:
                value_mean = value_np.mean(axis=0)
            else:
                value_mean = value_np

            # Ensure the value has the right shape for var
            if value_mean.shape[0] != adata.n_vars:
                # Transpose if needed
                if value_mean.shape[0] == adata.n_obs:
                    value_mean = value_mean.T
                else:
                    # Reshape if possible
                    try:
                        value_mean = value_mean.reshape(adata.n_vars)
                    except ValueError:
                        # Still store the original in uns
                        adata_out.uns[f"{model_name}_{key}"] = value_np
                        continue

            # Store in var dataframe
            adata_out.var[f"{model_name}_{key}"] = value_mean
            # Also store original in uns
            adata_out.uns[f"{model_name}_{key}"] = value_np
        # Store the result based on its shape
        elif (
            value_np.ndim == 2
            and value_np.shape[0] == adata.n_obs
            and value_np.shape[1] == adata.n_vars
        ):
            # Cell x gene matrices go in layers
            adata_out.layers[f"{model_name}_{key}"] = value_np
        elif value_np.ndim == 1 and value_np.shape[0] == adata.n_obs:
            # Cell vectors go in obs
            adata_out.obs[f"{model_name}_{key}"] = value_np
        elif value_np.ndim == 1 and value_np.shape[0] == adata.n_vars:
            # Gene vectors go in var
            adata_out.var[f"{model_name}_{key}"] = value_np
        elif value_np.ndim == 0 or (
            value_np.ndim == 1 and value_np.shape[0] == 1
        ):
            # Scalars go in uns
            adata_out.uns[f"{model_name}_{key}"] = (
                value_np.item() if value_np.size == 1 else value_np
            )
        else:
            # Other arrays go in uns
            adata_out.uns[f"{model_name}_{key}"] = value_np

    return adata_out


@beartype
def get_library_size(
    adata: anndata.AnnData,
    spliced_layer: str = "spliced",
    unspliced_layer: str = "unspliced",
    use_raw: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get library size for spliced and unspliced data.

    Args:
        adata: AnnData object
        spliced_layer: Name of the spliced layer
        unspliced_layer: Name of the unspliced layer
        use_raw: Whether to use raw data

    Returns:
        Tuple of (unspliced_lib_size, spliced_lib_size) PyTorch tensors
    """
    # Extract layers
    u, s = extract_layers(adata, spliced_layer, unspliced_layer, use_raw)

    # Calculate library sizes (sum across genes for each cell)
    # Ensure the result is a 1D tensor with shape (n_cells,)
    u_lib_size = torch.sum(u, dim=1).flatten()
    s_lib_size = torch.sum(s, dim=1).flatten()

    return u_lib_size, s_lib_size
