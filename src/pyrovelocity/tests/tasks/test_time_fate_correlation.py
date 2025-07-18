import os
from importlib.resources import files

import pytest

from pyrovelocity.tasks.time_fate_correlation import (
    create_time_lineage_fate_correlation_plot,
)


def test_create_time_lineage_fate_correlation_plot(
    larry_multilineage_model2_pyrovelocity_data_path,
    larry_cospar_100_6,
    larry_mono_100_6,
    larry_neu_100_6,
    tmp_path,
):
    """Test the time_fate_correlation function with multiple model results.

    This test simulates how the function is used in combine_time_lineage_fate_correlation
    in main_workflow.py, where it processes multiple model results.
    """
    output_dir = tmp_path / "multi_model_time_fate"

    shared_fixture_path = (
        files("pyrovelocity.tests.data")
        / "postprocessed_larry_multilineage_50_6.json"
    )

    model_results = [
        {
            "data_model": "larry_neu_model2",
            "postprocessed_data": shared_fixture_path,
            "pyrovelocity_data": larry_multilineage_model2_pyrovelocity_data_path,
        },
        {
            "data_model": "larry_multilineage_model2",
            "postprocessed_data": shared_fixture_path,
            "pyrovelocity_data": larry_multilineage_model2_pyrovelocity_data_path,
        },
    ]

    dataset_label_map = {
        "larry_neu": "Neutrophils",
        "larry_multilineage": "Multilineage",
    }

    output_path = create_time_lineage_fate_correlation_plot(
        model_results=model_results,
        output_dir=output_dir,
        dataset_label_map=dataset_label_map,
        adata_cospar=larry_cospar_100_6,
        adata_mono=larry_mono_100_6,
        adata_neu=larry_neu_100_6,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    assert os.path.exists(f"{output_path}.png")
    assert os.path.getsize(f"{output_path}.png") > 0
