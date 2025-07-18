import os
from importlib.resources import files

import matplotlib.pyplot as plt
import pytest

from pyrovelocity.analysis.trajectory import get_clone_trajectory
from pyrovelocity.plots import plot_lineage_fate_correlation
from pyrovelocity.styles.colors import LARRY_CELL_TYPE_COLORS
from pyrovelocity.utils import load_anndata_from_path


def test_plot_lineage_fate_correlation(
    larry_multilineage_model2_pyrovelocity_data_path,
    larry_cospar_100_6,
    tmp_path,
):
    """Test plot_lineage_fate_correlation function works correctly."""
    fig, axes = plt.subplots(1, 7, figsize=(14, 3))

    # Use optimized fixture instead of downloading large dataset
    adata_cospar = larry_cospar_100_6

    postprocessed_data_path = (
        files("pyrovelocity.tests.data")
        / "postprocessed_larry_multilineage_50_6.json"
    )
    posterior_samples_path = larry_multilineage_model2_pyrovelocity_data_path
    output_path = tmp_path / "lineage_fate_correlation.pdf"

    adata_source = load_anndata_from_path(postprocessed_data_path)
    adata_input_clone = get_clone_trajectory(adata_source)

    plot_lineage_fate_correlation(
        posterior_samples_path=posterior_samples_path,
        adata_pyrovelocity=postprocessed_data_path,
        adata_cospar=adata_cospar,
        all_axes=axes,
        fig=fig,
        state_color_dict=LARRY_CELL_TYPE_COLORS,
        adata_input_clone=adata_input_clone,
        lineage_fate_correlation_path=output_path,
        save_plot=True,
        show_titles=True,
        show_colorbars=False,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    plt.close(fig)
