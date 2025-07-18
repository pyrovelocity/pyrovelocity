"""Tests for `pyrovelocity.tasks.preprocess` module."""
import pytest
from beartype.typing import List

from pyrovelocity.io.serialization import (
    load_anndata_from_json,
    save_anndata_to_json,
)
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.utils import (
    anndata_string,
    configure_logging,
    print_anndata,
    print_string_diff,
)
from pyrovelocity.workflows.main_configuration import (
    pancreas_summary_configuration,
)

logger = configure_logging(__name__)


def test_load_preprocess():
    from pyrovelocity.tasks import preprocess

    print(preprocess.__file__)


@pytest.mark.slow
@pytest.mark.integration
def test_preprocess_dataset(preprocess_dataset_output):
    return preprocess_dataset_output


@pytest.mark.slow
@pytest.mark.integration
def test_preprocess_dataset_pancreas(adata_pancreas_raw_96_10, tmp_path):
    data_set_name = "pancreas"
    data_processed_path = tmp_path / "data/processed"
    reports_processed_path = tmp_path / "reports/processed"

    # Use the ultra-optimized pancreas fixture with 10 top velocity genes
    adata = adata_pancreas_raw_96_10
    n_obs = 96
    n_vars = 10

    result = preprocess_dataset(
        data_set_name=data_set_name,
        adata=adata,
        data_processed_path=data_processed_path,
        reports_processed_path=reports_processed_path,
        use_obs_subset=False,  # Don't subset - use all 96 cells
        n_obs_subset=n_obs,
        use_vars_subset=False,  # Don't subset - use all 10 genes
        n_vars_subset=n_vars,
        process_cytotrace=True,
    )

    (
        adata,
        preprocessed_dataset_path,
        preprocessed_reports_path,
    ) = result

    assert preprocessed_dataset_path.exists()
    assert preprocessed_reports_path.exists()

    selected_genes: List[str] = pancreas_summary_configuration.selected_genes
    selected_genes_in_adata: List[str] = [
        gene for gene in selected_genes if gene in adata.var_names
    ]
    if len(selected_genes_in_adata) < len(selected_genes):
        logger.warning(
            f"\nSome selected genes are not in the downsampled data:\n"
            f"selected_genes: {selected_genes}\n"
            f"selected_genes_in_adata: {selected_genes_in_adata}\n"
        )

    genes_to_keep: List[str] = list(
        set(adata.var_names[:n_vars].tolist() + selected_genes_in_adata)
    )
    adata = adata[:, genes_to_keep]
    preprocessed_anndata_string = anndata_string(adata)
    output_path = data_processed_path / "pancreas_processed.json"

    print_anndata(adata)
    save_anndata_to_json(adata, output_path)

    logger.info(f"Test fixture saved to {output_path}")

    try:
        logger.info("Attempting to load the serialized AnnData object...")
        loaded_adata = load_anndata_from_json(output_path)
        loaded_anndata_string = anndata_string(loaded_adata)
        logger.info("Successfully loaded the serialized AnnData object.")
        print_string_diff(
            text1=preprocessed_anndata_string,
            text2=loaded_anndata_string,
            diff_title="Preprocessed vs Loaded AnnData",
        )
        print_anndata(loaded_adata)
    except Exception as e:
        logger.error(f"Error loading serialized AnnData object: {str(e)}")
