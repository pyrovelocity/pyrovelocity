from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from beartype.typing import Dict, List, Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial import distance
from scipy.stats import spearmanr

from pyrovelocity.analysis.trajectory import align_trajectory_diff
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.logging import configure_logging
from pyrovelocity.plots._common import set_colorbar
from pyrovelocity.plots._time import plot_posterior_time
from pyrovelocity.plots._uncertainty import (
    get_posterior_sample_angle_uncertainty,
)
from pyrovelocity.plots._vector_fields import plot_vector_field_uncertainty
from pyrovelocity.utils import load_anndata_from_path

__all__ = ["plot_lineage_fate_correlation"]

logger = configure_logging(__name__)


@beartype
def plot_lineage_fate_correlation(
    posterior_samples_path: str | Path | AnnData,
    adata_pyrovelocity: str | Path | AnnData,
    adata_cospar: str | Path | AnnData,
    all_axes: List[Axes] | np.ndarray,
    fig: Figure,
    state_color_dict: Dict,
    adata_input_clone: str | Path | AnnData,
    ylabel: str = "Monocyte lineage",
    dotsize: int = 3,
    scale: float = 0.35,
    arrow: float = 3.5,
    lineage_fate_correlation_path: str | Path = "lineage_fate_correlation.pdf",
    save_plot: bool = True,
    show_colorbars: bool = False,
    show_titles: bool = False,
    default_fontsize: int = 7,
    default_title_padding: int = 2,
    include_uncertainty_measures: bool = False,
    plot_individual_obs: bool = False,
) -> List[Axes] | np.ndarray:
    """
    Plot lineage fate correlation with shared latent time estimates.

    Args:
        posterior_samples_path (str | Path | AnnData): Path to the posterior samples.
        adata_pyrovelocity (str | Path | AnnData): Path to the Pyro-Velocity AnnData object.
        adata_cospar (str | Path | AnnData): AnnData object with COSPAR results.
        all_axes (List[Axes] | np.ndarray): List of matplotlib axes.
        fig (Figure): Matplotlib figure.
        state_color_dict (Dict): Dictionary with cell state colors.
        adata_input_clone (str | Path | AnnData): Pre-computed clone trajectory data.
        ylabel (str, optional): Label for y axis. Defaults to "Monocyte lineage".
        dotsize (int, optional): Size of plotted points. Defaults to 3.
        scale (float, optional): Plot scale. Defaults to 0.35.
        arrow (float, optional): Arrow size. Defaults to 3.5.
        lineage_fate_correlation_path (str | Path, optional): Path to save the plot.
            Defaults to "lineage_fate_correlation.pdf".
        save_plot (bool, optional): Whether to save the plot. Defaults to True.
        show_colorbars (bool, optional): Whether to show colorbars. Defaults to False.
        show_titles (bool, optional): Whether to show titles. Defaults to False.
        default_fontsize (int, optional): Default font size. Defaults to 7.
        default_title_padding (int, optional): Default title padding. Defaults to 2.
        include_uncertainty_measures (bool, optional): Whether to include uncertainty
            measures. Defaults to False.
        plot_individual_obs (bool, optional): Whether to plot individual observations
            instead of using hexbin. Defaults to False.

    Returns:
        List[Axes] | np.ndarray: The axes objects.

    Examples:
        >>> # xdoctest: +SKIP
        >>> import matplotlib.pyplot as plt
        >>> import scanpy as sc
        >>> from pyrovelocity.io.datasets import larry_cospar, larry_mono
        >>> from pyrovelocity.utils import load_anndata_from_path
        >>> from pyrovelocity.plots import plot_lineage_fate_correlation
        ...
        >>> fig, ax = plt.subplots(1, 9)
        >>> fig.set_size_inches(17, 2.75)
        >>> fig.subplots_adjust(
        ...     hspace=0.4, wspace=0.2, left=0.01, right=0.99, top=0.95, bottom=0.3
        >>> )
        ...
        >>> data_set_name = "larry_mono"
        >>> model_name = "model2"
        >>> data_set_model_pairing = f"{data_set_name}_{model_name}"
        >>> model_path = f"models/{data_set_model_pairing}"
        ...
        >>> adata_pyrovelocity = load_anndata_from_path(f"{model_path}/postprocessed.h5ad")
        >>> # color_dict = dict(
        ... #     zip(
        ... #         adata_pyrovelocity.obs.state_info.cat.categories,
        ... #         adata_pyrovelocity.uns["state_info_colors"],
        ... #     )
        ... # )
        >>> adata_dynamical = load_anndata_from_path(f"data/processed/larry_mono_processed.h5ad")
        >>> adata_cospar = load_anndata_from_path(f"data/external/larry_cospar.h5ad")
        >>> plot_lineage_fate_correlation(
        ...     posterior_samples_path=f"{model_path}/pyrovelocity.pkl.zst",
        ...     adata_pyrovelocity=adata_pyrovelocity,
        ...     adata_scvelo=adata_dynamical,
        ...     adata_cospar=adata_cospar,
        ...     ax=ax,
        ...     fig=fig,
        ... )
    """
    posterior_samples = CompressedPickle.load(posterior_samples_path)
    embed_mean = posterior_samples["vector_field_posterior_mean"]

    if isinstance(adata_pyrovelocity, str | Path):
        adata_pyrovelocity = load_anndata_from_path(adata_pyrovelocity)
    if isinstance(adata_cospar, str | Path):
        adata_cospar = load_anndata_from_path(adata_cospar)
    if isinstance(adata_input_clone, str | Path):
        adata_input_clone = load_anndata_from_path(adata_input_clone)

    adata_scvelo = adata_pyrovelocity.copy()

    # Calculate appropriate density for dataset size to avoid n_neighbors=0 in scvelo
    # For small datasets (< 200 cells), use larger density to ensure n_neighbors >= 1
    n_cells = adata_pyrovelocity.n_obs
    if n_cells < 200:
        density = max(0.5, 3.0 / n_cells)  # Ensure density gives at least 1-2 neighbors
    else:
        density = 0.35
    diff = align_trajectory_diff(
        [adata_input_clone, adata_scvelo, adata_scvelo],
        [
            adata_input_clone.obsm["clone_vector_emb"],
            adata_scvelo.obsm["velocity_emb"],
            embed_mean,
        ],
        embed="emb",
        density=density,
    )
    scvelo_cos = pd.DataFrame(diff).apply(
        lambda x: 1 - distance.cosine(x[2:4], x[4:6]), axis=1
    )
    pyro_cos = pd.DataFrame(diff).apply(
        lambda x: 1 - distance.cosine(x[2:4], x[6:8]), axis=1
    )
    scvelo_cos_mean = scvelo_cos.mean()
    pyro_cos_mean = pyro_cos.mean()
    logger.info(
        f"\nscVelo cosine similarity: {scvelo_cos_mean:.2f}\n"
        f"Pyro-Velocity cosine similarity: {pyro_cos_mean:.2f}\n\n"
    )

    current_axis_index = 0

    # SHIFT AXIS INDEX
    ax, current_axis_index = get_next_axis(all_axes, current_axis_index)

    if plot_individual_obs or adata_pyrovelocity.n_obs < 5000:
        plot_data = pd.DataFrame(
            {
                "X": adata_pyrovelocity.obsm["X_emb"][:, 0],
                "Y": adata_pyrovelocity.obsm["X_emb"][:, 1],
                "cell_type": adata_pyrovelocity.obs.state_info,
            }
        )
    else:
        cell_counts = adata_pyrovelocity.obs.state_info.value_counts()
        smallest_cluster_size = cell_counts.min()
        min_representation = max(50, int(smallest_cluster_size * 0.9))
        max_total_cells = 5000
        obs_indices = []

        for ct in adata_pyrovelocity.obs.state_info.cat.categories:
            mask = adata_pyrovelocity.obs.state_info == ct
            n_cells = np.sum(mask)
            if n_cells <= min_representation:
                sample_size = n_cells
            else:
                proportion = n_cells / adata_pyrovelocity.n_obs
                sample_size = max(
                    min_representation,
                    min(n_cells, int(max_total_cells * proportion)),
                )
            if sample_size >= n_cells:
                ct_indices = np.where(mask)[0]
            else:
                ct_indices = np.random.choice(
                    np.where(mask)[0], size=sample_size, replace=False
                )
            obs_indices.extend(ct_indices)

        plot_data = pd.DataFrame(
            {
                "X": adata_pyrovelocity.obsm["X_emb"][obs_indices, 0],
                "Y": adata_pyrovelocity.obsm["X_emb"][obs_indices, 1],
                "cell_type": adata_pyrovelocity.obs.state_info.values[
                    obs_indices
                ],
            }
        )

    sns.scatterplot(
        x="X",
        y="Y",
        data=plot_data,
        alpha=0.90,
        s=dotsize,
        linewidth=0,
        edgecolor="none",
        hue="cell_type",
        palette=state_color_dict,
        ax=ax,
        legend="brief",
    )

    ax.get_legend().remove()
    ax.axis("off")
    if show_titles:
        ax.set_title(
            "Cell types",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )
    else:
        ax.set_title(
            "",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )
    ax.set_ylabel(ylabel, fontsize=default_fontsize)

    # SHIFT AXIS INDEX
    ax, current_axis_index = get_next_axis(all_axes, current_axis_index)
    # Calculate safe n_neighbors for small datasets
    n_neighbors = max(1, min(10, adata_input_clone.n_obs // 10))
    
    scv.pl.velocity_embedding_grid(
        adata=adata_input_clone,
        scale=scale,
        autoscale=True,
        show=False,
        s=dotsize,
        density=density,
        n_neighbors=n_neighbors,
        arrow_size=arrow,
        linewidth=1,
        vkey="clone_vector",
        basis="emb",
        ax=ax,
        title="",
        color="gray",
        arrow_color="black",
        fontsize=default_fontsize,
    )
    ax.axis("off")
    if show_titles:
        ax.set_title(
            "Clonal progression",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )

    # SHIFT AXIS INDEX
    ax, current_axis_index = get_next_axis(all_axes, current_axis_index)
    scv.pl.velocity_embedding_grid(
        adata=adata_scvelo,
        show=False,
        s=dotsize,
        density=density,
        n_neighbors=n_neighbors,
        scale=scale,
        autoscale=True,
        arrow_size=arrow,
        linewidth=1,
        basis="emb",
        ax=ax,
        title="",
        fontsize=default_fontsize,
        color="gray",
        arrow_color="black",
    )
    ax.axis("off")
    if show_titles:
        # "scVelo cosine similarity: %.2f" % scvelo_cos_mean, fontsize=default_fontsize
        ax.set_title(
            f"scVelo ({scvelo_cos_mean:.2f})",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )
    else:
        ax.set_title(
            f"({scvelo_cos_mean:.2f})",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )

    # SHIFT AXIS INDEX
    ax, current_axis_index = get_next_axis(all_axes, current_axis_index)
    scv.pl.velocity_embedding_grid(
        adata=adata_pyrovelocity,
        basis="emb",
        vkey="velocity_pyro",
        show=False,
        s=dotsize,
        density=density,
        n_neighbors=n_neighbors,
        scale=scale,
        autoscale=True,
        arrow_size=arrow,
        linewidth=1,
        ax=ax,
        title="",
        fontsize=default_fontsize,
        color="gray",
        arrow_color="black",
    )
    ax.axis("off")
    if show_titles:
        # "Pyro-Velocity cosine similarity: %.2f" % pyro_cos_mean, fontsize=default_fontsize
        ax.set_title(
            rf"Pyro\thinspace-Velocity ({pyro_cos_mean:.2f})"
            if matplotlib.rcParams["text.usetex"]
            else f"Pyro\u2009-Velocity ({pyro_cos_mean:.2f})",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )
    else:
        ax.set_title(
            f"({pyro_cos_mean:.2f})",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )

    if include_uncertainty_measures:
        pca_angles = posterior_samples["pca_embeds_angle"]
        pca_cell_angles = pca_angles / np.pi * 180
        pca_angles_std = get_posterior_sample_angle_uncertainty(pca_cell_angles)
        logger.info(
            f"\nPCA angle uncertainty: {pca_angles_std.mean():.2f}"
            f"± {pca_angles_std.std():.2f}\n\n"
        )

        # SHIFT AXIS INDEX
        ax, current_axis_index = get_next_axis(all_axes, current_axis_index)
        plot_vector_field_uncertainty(
            adata_pyrovelocity,
            embed_mean,
            pca_angles_std,
            ax=ax,
            cbar=show_colorbars,
            fig=fig,
            basis="emb",
            scale=scale,
            arrow_size=arrow,
            p_mass_min=1,
            autoscale=True,
            density=density,
            only_grid=False,
            uncertain_measure="PCA angle",
            cmap="winter",
            cmax=None,
            color_vector_field_by_measure=False,
            show_titles=show_titles,
        )
        if show_titles:
            ax.set_title(
                r"Pyro\thinspace-Velocity angle $\sigma$"
                if matplotlib.rcParams["text.usetex"]
                else "Pyro\u2009-Velocity angle σ",
                fontsize=default_fontsize,
                pad=default_title_padding,
            )

    # The obs names in adata_pyrovelocity have a "-N" suffix that
    # is not present in the adata_cospar obs Index.
    patched_adata_pyrovelocity_obs_names = (
        adata_pyrovelocity.obs_names.str.replace(
            r"-\d",
            "",
            regex=True,
        )
    )
    adata_cospar_obs_subset = adata_cospar[
        patched_adata_pyrovelocity_obs_names, :
    ]
    scatter_dotsize_factor = 3
    # SHIFT AXIS INDEX
    ax, current_axis_index = get_next_axis(all_axes, current_axis_index)

    if plot_individual_obs:
        scv.pl.scatter(
            adata=adata_cospar_obs_subset,
            basis="emb",
            fontsize=default_fontsize,
            color="fate_potency_transition_map",
            cmap="inferno_r",
            show=False,
            ax=ax,
            s=dotsize * scatter_dotsize_factor,
            colorbar=show_colorbars,
            title="",
        )
    else:
        im = ax.hexbin(
            x=adata_cospar_obs_subset.obsm[f"X_emb"][:, 0],
            y=adata_cospar_obs_subset.obsm[f"X_emb"][:, 1],
            C=adata_cospar_obs_subset.obs["fate_potency_transition_map"],
            gridsize=100,
            cmap="inferno_r",
            linewidths=0,
            edgecolors="none",
            reduce_C_function=np.mean,
        )
        if show_colorbars:
            set_colorbar(
                im, ax, labelsize=default_fontsize, fig=fig, position="right"
            )

    ax.axis("off")
    if show_titles:
        ax.set_title(
            "Fate potency",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )
    gold_standard = adata_cospar_obs_subset.obs.fate_potency_transition_map
    select = ~np.isnan(gold_standard)
    scvelo_latent_time_correlation = spearmanr(
        -gold_standard[select],
        adata_scvelo.obs.latent_time.values[select],
    )[0]

    # SHIFT AXIS INDEX
    ax, current_axis_index = get_next_axis(all_axes, current_axis_index)

    if plot_individual_obs:
        scv.pl.scatter(
            adata=adata_scvelo,
            c="latent_time",
            basis="emb",
            s=dotsize * scatter_dotsize_factor,
            cmap="inferno",
            ax=ax,
            show=False,
            fontsize=default_fontsize,
            colorbar=show_colorbars,
            title="",
        )
    else:
        im = ax.hexbin(
            x=adata_scvelo.obsm[f"X_emb"][:, 0],
            y=adata_scvelo.obsm[f"X_emb"][:, 1],
            C=adata_scvelo.obs["latent_time"],
            gridsize=100,
            cmap="inferno",
            linewidths=0,
            edgecolors="none",
            reduce_C_function=np.mean,
        )
        if show_colorbars:
            set_colorbar(
                im, ax, labelsize=default_fontsize, fig=fig, position="right"
            )

    ax.axis("off")
    if show_titles:
        ax.set_title(
            f"scVelo time ({scvelo_latent_time_correlation:.2f})",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )
    else:
        ax.set_title(
            f"({scvelo_latent_time_correlation:.2f})",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )

    pyrovelocity_shared_time_correlation = spearmanr(
        -gold_standard[select],
        posterior_samples["cell_time"].mean(0).flatten()[select],
    )[0]
    # SHIFT AXIS INDEX
    ax, current_axis_index = get_next_axis(all_axes, current_axis_index)
    plot_posterior_time(
        posterior_samples,
        adata_pyrovelocity,
        ax=ax,
        basis="emb",
        fig=fig,
        addition=False,
        position="right",
        cmap="winter",
        s=dotsize,
        show_colorbar=show_colorbars,
        show_titles=show_titles,
    )
    if show_titles:
        ax.set_title(
            rf"Pyro\thinspace-Velocity time ({pyrovelocity_shared_time_correlation:.2f})"
            if matplotlib.rcParams["text.usetex"]
            else f"Pyro\u2009-Velocity time ({pyrovelocity_shared_time_correlation:.2f})",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )
    else:
        ax.set_title(
            f"({pyrovelocity_shared_time_correlation:.2f})",
            fontsize=default_fontsize,
            pad=default_title_padding,
        )

    if save_plot:
        for ext in ["", ".png"]:
            fig.savefig(
                fname=f"{lineage_fate_correlation_path}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
                transparent=False,
            )

    return all_axes


@beartype
def get_next_axis(
    axes: List[Axes] | np.ndarray,
    current_index: int = 0,
) -> Tuple[Axes, int]:
    return axes[current_index], current_index + 1

    # import ipdb

    # ipdb.set_trace()
    # colorbar_axes = [
    #     ax[5].images[0],
    #     ax[6].collections[0],
    #     ax[7].collections[0],
    # ]
    # return colorbar_axes

    # cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
    # cell_time_std = posterior_samples["cell_time"].std(0).flatten()
    # cell_time_cov = cell_time_std / cell_time_mean
    # print(cell_time_cov)

    # plot_vector_field_uncertainty(
    #     adata_pyrovelocity,
    #     embed_mean,
    #     cell_time_std,
    #     ax=ax[3],
    #     cbar=True,
    #     fig=fig,
    #     basis="emb",
    #     scale=scale,
    #     arrow_size=arrow,
    #     p_mass_min=1,
    #     autoscale=True,
    #     density=density,
    #     only_grid=False,
    #     uncertain_measure="shared time",
    #     cmap="winter",
    #     cmax=None,
    # )
    # ax[3].set_title("Shared time uncertainty", fontsize=default_fontsize)

    # cell_magnitudes = posterior_samples["original_spaces_embeds_magnitude"]
    # cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
    # cell_magnitudes_std = cell_magnitudes.std(axis=-2)
    # cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean
    # print(cell_magnitudes_cov)

    # plot_vector_field_uncertainty(
    #     adata_pyrovelocity,
    #     embed_mean,
    #     cell_magnitudes_cov,
    #     ax=ax[4],
    #     cbar=True,
    #     fig=fig,
    #     basis="emb",
    #     scale=scale,
    #     p_mass_min=1,
    #     density=density,
    #     arrow_size=arrow,
    #     only_grid=False,
    #     autoscale=True,
    #     uncertain_measure="base magnitude",
    #     cmap="summer",
    # )
    # ax[4].set_title(
    #     "Pyro-Velocity cosine similarity: %.2f" % pyro_cos_mean, fontsize=default_fontsize
    # )
