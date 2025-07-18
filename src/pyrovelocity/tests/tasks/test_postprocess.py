"""Tests for `pyrovelocity.tasks.postprocess` module."""

import os
import shutil

import pytest

from pyrovelocity.io.serialization import save_anndata_to_json
from pyrovelocity.tasks.postprocess import postprocess_dataset
from pyrovelocity.utils import load_anndata_from_path


def test_load_postprocess():
    from pyrovelocity.tasks import postprocess

    print(postprocess.__file__)


@pytest.mark.slow
@pytest.mark.integration
def test_postprocess_dataset(postprocess_dataset_output):
    return postprocess_dataset_output


@pytest.mark.slow
@pytest.mark.integration
def test_postprocess_dataset_model1(postprocess_dataset_model1_output):
    return postprocess_dataset_model1_output


@pytest.mark.slow
@pytest.mark.integration
def test_postprocess_dataset_pancreas(
    adata_trained_pancreas_50_7,
    pancreas_model2_path,
    pancreas_model2_model_path,
    pancreas_model2_posterior_samples_path,
    pancreas_model2_metrics_path,
    tmp_path,
):
    """Test postprocessing of the pancreas dataset using real model fixtures.

    This test uses the actual model, posterior samples, and metrics files
    from the fixtures to test the postprocess_dataset function.
    """
    data_set_name = "pancreas"
    model_identifier = "model2"
    data_model = f"{data_set_name}_{model_identifier}"

    data_model_path = tmp_path / "models" / data_model
    data_model_path.mkdir(parents=True, exist_ok=True)

    trained_data_path = data_model_path / "trained.h5ad"
    adata_trained_pancreas_50_7.write(trained_data_path)

    model_path = data_model_path / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    for item in pancreas_model2_model_path.iterdir():
        if item.is_file():
            shutil.copy(item, model_path / item.name)

    posterior_samples_path = data_model_path / "posterior_samples.pkl.zst"
    shutil.copy(pancreas_model2_posterior_samples_path, posterior_samples_path)

    metrics_path = data_model_path / "metrics.json"
    shutil.copy(pancreas_model2_metrics_path, metrics_path)

    result = postprocess_dataset(
        data_model=data_model,
        data_model_path=data_model_path,
        trained_data_path=trained_data_path,
        model_path=model_path,
        posterior_samples_path=posterior_samples_path,
        metrics_path=metrics_path,
        vector_field_basis="umap",
        number_posterior_samples=4,
    )

    pyrovelocity_data_path, postprocessed_data_path = result

    assert os.path.exists(pyrovelocity_data_path)
    assert os.path.exists(postprocessed_data_path)

    postprocessed_adata = load_anndata_from_path(postprocessed_data_path)
    save_anndata_to_json(
        postprocessed_adata, tmp_path / "postprocessed_adata.json"
    )

    return result
