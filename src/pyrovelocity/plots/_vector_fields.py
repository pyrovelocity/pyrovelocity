from os import PathLike

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from anndata import AnnData
from beartype import beartype
from beartype.typing import Any, Dict, List, Optional, Tuple
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import FigureBase
from matplotlib.gridspec import SubplotSpec
from matplotlib.ticker import MaxNLocator
from scvelo.plotting.velocity_embedding_grid import default_arrow

from pyrovelocity.analysis.analyze import compute_mean_vector_field
from pyrovelocity.logging import configure_logging
from pyrovelocity.plots._common import calculate_adaptive_n_neighbors
from pyrovelocity.plots._time import plot_posterior_time
from pyrovelocity.plots._uncertainty import (
    get_posterior_sample_angle_uncertainty,
)
from pyrovelocity.styles import configure_matplotlib_style
from pyrovelocity.utils import quartile_coefficient_of_dispersion, setup_colors

__all__ = [
    "plot_vector_field_summary",
    "plot_vector_field_uncertainty",
    "plot_mean_vector_field",
]

logger = configure_logging(__name__)

configure_matplotlib_style()


@beartype
def create_vector_field_summary_layout(
    fig: Optional[FigureBase] = None,
    gs: Optional[SubplotSpec] = None,
    fig_width: int | float = 7.5,
    fig_height: int | float = 1.08,
) -> Tuple[FigureBase, List[Axes], Axes, List[Axes]]:
    if fig is None:
        fig = plt.figure(
            figsize=(fig_width, fig_height),
        )
    if gs is None:
        gsi = fig.add_gridspec(
            nrows=2,
            ncols=6,
            width_ratios=[1] * 6,
            height_ratios=[10, 1],
            wspace=0.1,
            hspace=0.0,
        )
    else:
        gsi = gs.subgridspec(
            nrows=2,
            ncols=6,
            width_ratios=[1] * 6,
            height_ratios=[10, 1],
            wspace=0.1,
            hspace=0.0,
        )
    main_axes = [fig.add_subplot(gsi[0, i]) for i in range(6)]
    legend_axes = fig.add_subplot(gsi[1, :3])
    colorbar_axes = [fig.add_subplot(gsi[1, i]) for i in range(3, 6)]
    all_axes = main_axes + [legend_axes] + colorbar_axes
    for ax in all_axes:
        ax.set_label("vector_field")

    return fig, main_axes, legend_axes, colorbar_axes


@beartype
def plot_vector_field_summary(
    adata: AnnData,
    posterior_samples: Dict[str, np.ndarray],
    vector_field_basis: str,
    plot_name: Optional[PathLike | str] = None,
    cell_state: str = "cell_type",
    state_color_dict: Optional[Dict[str, Any]] = None,
    fig: Optional[FigureBase] = None,
    gs: Optional[SubplotSpec] = None,
    default_fontsize: int = 7 if matplotlib.rcParams["text.usetex"] else 6,
    default_title_padding: int = 5,
    dotsize: int | float = 1,
    scale: float = 0.35,
    arrow_size: float = 3,
    density: float = 0.4,
    save_fig: bool = False,
    linewidth: float = 0.5,
    title_background_color: str = "#F0F0F0",
    force_complete_angular_scale: bool = False,
) -> FigureBase:
    posterior_time = posterior_samples["cell_time"]
    pca_embeds_angle = posterior_samples["pca_embeds_angle"]
    embed_mean = posterior_samples["vector_field_posterior_mean"]

    (
        fig,
        ax,
        legend_axes,
        colorbar_axes,
    ) = create_vector_field_summary_layout(fig=fig, gs=gs)

    ress = pd.DataFrame(
        {
            "cell_type": adata.obs[cell_state].values,
            "X1": adata.obsm[f"X_{vector_field_basis}"][:, 0],
            "X2": adata.obsm[f"X_{vector_field_basis}"][:, 1],
        }
    )

    if state_color_dict is None:
        state_color_dict = setup_colors(adata, cell_state)

    sns.scatterplot(
        x="X1",
        y="X2",
        hue="cell_type",
        s=dotsize,
        palette=state_color_dict,
        data=ress,
        alpha=0.9,
        linewidth=0,
        edgecolor="none",
        ax=ax[0],
        legend="brief",
    )
    ax[0].get_legend().remove()
    ax[0].set_xticklabels([])
    ax[0].set_xlabel("")
    ax[0].axis("off")
    ax[0].set_title(
        "Cell types",
        fontsize=default_fontsize,
        pad=default_title_padding,
    )

    scv.pl.velocity_embedding_grid(
        adata,
        basis=vector_field_basis,
        fontsize=default_fontsize,
        ax=ax[1],
        title="",
        color="gray",
        s=dotsize,
        show=False,
        alpha=0.25,
        min_mass=3.5,
        scale=scale,
        frameon=False,
        density=density,
        arrow_size=arrow_size,
        linewidth=linewidth,
    )
    ax[1].set_xticklabels([])
    ax[1].set_xlabel("")
    ax[1].axis("off")
    ax[1].set_title(
        "scVelo",
        fontsize=default_fontsize,
        pad=default_title_padding,
    )
    scv.pl.velocity_embedding_grid(
        adata,
        fontsize=default_fontsize,
        basis=vector_field_basis,
        title="",
        ax=ax[2],
        vkey="velocity_pyro",
        color="gray",
        s=dotsize,
        show=False,
        alpha=0.25,
        min_mass=3.5,
        scale=scale,
        frameon=False,
        density=density,
        arrow_size=arrow_size,
        linewidth=linewidth,
    )
    ax[2].set_xticklabels([])
    ax[2].set_xlabel("")
    ax[2].axis("off")
    ax[2].set_title(
        rf"Pyro\thinspace-Velocity"
        if matplotlib.rcParams["text.usetex"]
        else f"Pyro\u2009-Velocity",
        fontsize=default_fontsize,
        pad=default_title_padding,
        backgroundcolor=title_background_color,
    )

    pca_cell_angles = pca_embeds_angle / np.pi * 180
    pca_angles_std = get_posterior_sample_angle_uncertainty(pca_cell_angles)

    plot_vector_field_uncertainty(
        adata=adata,
        embed_mean=embed_mean,
        embeds_radian_or_magnitude=pca_angles_std,
        ax=ax[3],
        cbar=False,
        fig=fig,
        basis=vector_field_basis,
        scale=scale,
        arrow_size=arrow_size,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="PCA angle",
        cmap="inferno",
        cmax=None,
        show_titles=False,
        force_complete_angular_scale=force_complete_angular_scale,
    )
    ax[3].set_title(
        r"PCA angle uncertainty"
        if matplotlib.rcParams["text.usetex"]
        else "PCA angle σ",
        fontsize=default_fontsize,
        pad=default_title_padding,
        backgroundcolor=title_background_color,
    )

    plot_posterior_time(
        posterior_samples,
        adata,
        ax=ax[4],
        basis=vector_field_basis,
        fig=fig,
        addition=False,
        position="right",
        cmap="winter",
        s=dotsize,
        show_colorbar=False,
        show_titles=False,
        alpha=1,
    )
    ax[4].set_title(
        r"Shared time mean"
        # r"shared time $\hat{\mu}(t)$"
        if matplotlib.rcParams["text.usetex"]
        else "Shared time μ",
        fontsize=default_fontsize,
        pad=default_title_padding,
        backgroundcolor=title_background_color,
    )

    cell_time_mean = posterior_time.mean(0).flatten()
    cell_time_mean_max = cell_time_mean.max()
    cell_times = posterior_time / cell_time_mean_max
    # cell_time_mean = cell_times.mean(0).flatten()
    # cell_time_std = cell_times.std(0).flatten()
    # cell_time_cov = cell_time_std / cell_time_mean
    cell_time_qcd = quartile_coefficient_of_dispersion(cell_times).flatten()

    plot_vector_field_uncertainty(
        adata=adata,
        embed_mean=embed_mean,
        # embeds_radian_or_magnitude=cell_time_cov,
        embeds_radian_or_magnitude=cell_time_qcd,
        ax=ax[5],
        cbar=False,
        fig=fig,
        basis=vector_field_basis,
        scale=scale,
        arrow_size=arrow_size,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="shared time",
        cmap="winter",
        cmax=None,
        show_titles=False,
    )
    ax[5].set_title(
        r"Shared time uncertainty"
        if matplotlib.rcParams["text.usetex"]
        else "shared time σ/μ",
        fontsize=default_fontsize,
        pad=default_title_padding,
        backgroundcolor=title_background_color,
    )

    handles, labels = ax[0].get_legend_handles_labels()
    legend_axes.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        bbox_to_anchor=(-0.05, 2.0),
        # bbox_transform=bottom_axes[0].transAxes,
        bbox_transform=legend_axes.transAxes,
        ncol=5,
        frameon=False,
        fancybox=True,
        markerscale=3,
        columnspacing=0.05,
        handletextpad=-0.5,
        fontsize=default_fontsize * 0.9,
    )
    legend_axes.axis("off")

    for axi in ax:
        axi.set_aspect("equal", adjustable="box")

    colorbar_labels = [
        r"$\hat{\sigma}$",
        r"$\mu(t)$",
        r"$\left.\hat{\sigma}(t) \right/ \hat{\mu}(t)$",
    ]
    colorbar_ticks = [
        [0, 360] if force_complete_angular_scale else [],
        [0, 1],
        [],
    ]
    for axi, cax, clabel, cticks in zip(
        ax[3:],
        colorbar_axes,
        colorbar_labels,
        colorbar_ticks,
    ):
        ax_pos = axi.get_position()
        cax.axis("on")
        cbar = fig.colorbar(
            mappable=axi.collections[0],
            cax=cax,
            orientation="horizontal",
        )

        if len(cticks) > 0:
            vmin, vmax = axi.collections[0].get_clim()
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([rf"{cticks[0]}", rf"{cticks[1]}"])
        else:
            cbar.locator = MaxNLocator(nbins=3)
            cbar.update_ticks()

        cax.xaxis.set_ticks_position("bottom")
        cax.xaxis.set_label_position("bottom")
        cax.xaxis.set_tick_params(labelsize=default_fontsize * 0.8)
        # cbar.set_label(label=clabel, fontsize=default_fontsize, labelpad=0)
        # TODO: support colorbar with width specified as a fraction of the axis width
        # cbar_width = ax_pos.width * 0.6
        # cbar_height = ax_pos.height * 0.10
        # cax.set_position(
        #     [
        #         ax_pos.x0 + (ax_pos.width - cbar_width),
        #         ax_pos.y0 - cbar_height * 1.05,
        #         cbar_width,
        #         cbar_height,
        #     ]
        # )

    if save_fig:
        fig.tight_layout()

        for ext in ["", ".png"]:
            fig.savefig(
                f"{plot_name}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )
        plt.close(fig)
    return fig


def plot_vector_field_uncertainty(
    adata,
    embed_mean,
    embeds_radian_or_magnitude,
    fig=None,
    cbar=True,
    basis="umap",
    scale=0.002,
    cbar_pos=[0.22, 0.28, 0.5, 0.05],
    p_mass_min=3.5,
    only_grid=False,
    ax=None,
    autoscale=False,
    density=0.3,
    arrow_size=5,
    uncertain_measure="angle",
    cmap="winter",
    cmax=0.305,
    color_vector_field_by_measure=False,
    dotsize=1,
    show_titles: bool = True,
    default_fontsize: int = 7,
    plot_individual_obs: bool = False,
    force_complete_angular_scale: bool = False,
):
    if uncertain_measure == "angle":
        adata.obs["uncertain"] = get_posterior_sample_angle_uncertainty(
            embeds_radian_or_magnitude / np.pi * 180
        )
    elif uncertain_measure in ["base magnitude", "shared time", "PCA angle"]:
        adata.obs["uncertain"] = embeds_radian_or_magnitude
    else:
        adata.obs["uncertain"] = embeds_radian_or_magnitude.std(axis=0)

    if ax is None:
        ax = fig.subplots(1, 2)
    if isinstance(ax, list) and len(ax) == 2:
        if not only_grid:
            order = np.argsort(adata.obs["uncertain"].values)
            im = ax[0].scatter(
                adata.obsm[f"X_{basis}"][:, 0][order],
                adata.obsm[f"X_{basis}"][:, 1][order],
                c=adata.obs["uncertain"].values[order],
                cmap=cmap,
                norm=None,
                vmin=np.percentile(uncertain, 5),
                vmax=np.percentile(uncertain, 95),
                s=dotsize,
                linewidth=1,
                edgecolors="none",
            )
            ax[0].axis("off")
            if show_titles:
                ax[0].set_title(
                    # f"Single-cell\n {uncertain_measure} uncertainty ",
                    f"{uncertain_measure} uncertainty",
                    fontsize=default_fontsize,
                )
            ax = ax[1]

    if color_vector_field_by_measure:
        X_grid, V_grid, uncertain = project_grid_points(
            adata.obsm[f"X_{basis}"],
            embed_mean,
            adata.obs["uncertain"].values,
            p_mass_min=p_mass_min,
            autoscale=autoscale,
            density=density,
        )

        hl, hw, hal = default_arrow(arrow_size)
        quiver_kwargs = {"angles": "xy", "scale_units": "xy"}
        quiver_kwargs.update({"width": 0.001, "headlength": hl / 2})
        quiver_kwargs.update({"headwidth": hw / 2, "headaxislength": hal / 2})
        quiver_kwargs.update({"linewidth": 1, "zorder": 3})
        norm = Normalize()
        norm.autoscale(uncertain)
        ax.scatter(
            adata.obsm[f"X_{basis}"][:, 0],
            adata.obsm[f"X_{basis}"][:, 1],
            s=1,
            linewidth=0,
            color="gray",
            alpha=0.22,
        )
        im = ax.quiver(
            X_grid[:, 0],
            X_grid[:, 1],
            V_grid[:, 0],
            V_grid[:, 1],
            uncertain,
            norm=None,
            cmap=cmap,
            edgecolors="face",
            scale=scale,
            clim=(
                np.percentile(uncertain, 5),
                np.percentile(uncertain, 95) if cmax is None else cmax,
            ),
            **quiver_kwargs,
        )
        if show_titles:
            ax.set_title(
                f"Averaged\n {uncertain_measure} uncertainty ",
                fontsize=default_fontsize,
            )
        ax.axis("off")
    else:
        order = np.argsort(adata.obs["uncertain"].values)
        ordered_uncertainty_measure = adata.obs["uncertain"].values[order]
        if plot_individual_obs:
            im = ax.scatter(
                adata.obsm[f"X_{basis}"][:, 0][order],
                adata.obsm[f"X_{basis}"][:, 1][order],
                c=ordered_uncertainty_measure,
                cmap=cmap,
                norm=None,
                vmin=0
                if "angle" in uncertain_measure
                else min(ordered_uncertainty_measure),
                vmax=360
                if force_complete_angular_scale
                else max(ordered_uncertainty_measure),
                s=dotsize,
                linewidth=1,
                edgecolors="none",
            )
        else:
            im = ax.hexbin(
                adata.obsm[f"X_{basis}"][:, 0][order],
                adata.obsm[f"X_{basis}"][:, 1][order],
                C=ordered_uncertainty_measure,
                gridsize=100,
                cmap=cmap,
                vmin=0
                if "angle" in uncertain_measure
                else min(ordered_uncertainty_measure),
                vmax=360
                if force_complete_angular_scale
                else max(ordered_uncertainty_measure),
                linewidths=0,
                edgecolors="none",
            )

        ax.axis("off")
        if show_titles:
            ax.set_title(
                # f"Single-cell\n {uncertain_measure} uncertainty ", fontsize=default_fontsize
                f"{uncertain_measure} uncertainty",
                fontsize=default_fontsize,
            )
    if cbar:
        pos = ax.get_position()
        cax = fig.add_axes(
            [pos.x0 + 0.05, pos.y0 - 0.05, pos.width * 0.6, pos.height / 17]
        )

        cbar = fig.colorbar(
            im, cax=cax, orientation="horizontal"
        )  # fraction=0.046, pad=0.04
        cbar.ax.tick_params(axis="x", labelsize=default_fontsize * 0.8)
        cbar.ax.locator = MaxNLocator(nbins=2, integer=True)
        # cbar.ax.set_xlabel(f"{uncertain_measure} uncertainty", fontsize=default_fontsize)


def plot_mean_vector_field(
    posterior_samples,
    adata,
    ax,
    basis="umap",
    n_jobs=1,
    scale=0.2,
    density=0.4,
    spliced="spliced_pyro",
    raw=False,
    random_seed: int = 99,
):
    compute_mean_vector_field(
        posterior_samples=posterior_samples,
        adata=adata,
        basis=basis,
        n_jobs=n_jobs,
        spliced=spliced,
        raw=raw,
        random_seed=random_seed,
    )
    scv.pl.velocity_embedding_grid(
        adata,
        basis=basis,
        vkey="velocity_pyro",
        linewidth=1,
        ax=ax,
        show=False,
        legend_loc="on data",
        density=density,
        scale=scale,
        arrow_size=3,
    )
    return adata.obsm[f"velocity_pyro_{basis}"]


def project_grid_points(
    emb,
    velocity_emb,
    uncertain=None,
    p_mass_min=1.0,
    density=0.3,
    autoscale=False,
):
    from scipy.stats import norm as normal
    from scvelo.tools.velocity_embedding import quiver_autoscale
    from sklearn.neighbors import NearestNeighbors

    X_grid = []
    grs = []
    grid_num = 50 * density
    smooth = 0.5

    for dim_i in range(2):
        m, M = np.min(emb[:, dim_i]), np.max(emb[:, dim_i])
        m = m - 0.01 * np.abs(M - m)
        M = M + 0.01 * np.abs(M - m)
        gr = np.linspace(m, M, int(grid_num))
        grs.append(gr)
    meshes_tuple = np.meshgrid(*grs)
    scale = np.mean([(g[1] - g[0]) for g in grs]) * smooth
    X_grid = np.vstack([i.flat for i in meshes_tuple]).T

    n_neighbors = calculate_adaptive_n_neighbors(emb.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(emb)
    dists, neighs = nn.kneighbors(X_grid)
    weight = normal.pdf(x=dists, scale=scale)
    p_mass = weight.sum(1)

    if len(velocity_emb.shape) == 2:
        V_grid = (velocity_emb[:, :2][neighs] * weight[:, :, None]).sum(
            1
        ) / np.maximum(1, p_mass)[:, None]
    else:
        V_grid = (velocity_emb[:, :2][neighs] * weight[:, :, None, None]).sum(
            1
        ) / np.maximum(1, p_mass)[:, None, None]

    p_mass_min *= np.percentile(p_mass, 99) / 100
    if autoscale:
        V_grid /= 3 * quiver_autoscale(X_grid, V_grid)

    if uncertain is None:
        return X_grid[p_mass > p_mass_min], V_grid[p_mass > p_mass_min]
    else:
        uncertain = (uncertain[neighs] * weight).sum(1) / np.maximum(1, p_mass)
        return (
            X_grid[p_mass > p_mass_min],
            V_grid[p_mass > p_mass_min],
            uncertain[p_mass > p_mass_min],
        )


# TODO: remove unused code
# cell_magnitudes = posterior_samples["original_spaces_embeds_magnitude"]
# cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
# cell_magnitudes_std = cell_magnitudes.std(axis=-2)
# cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean
# plot_vector_field_uncertainty(
#     adata,
#     embed_mean,
#     cell_magnitudes_cov,
#     ax=ax[4],
#     cbar=False,
#     fig=fig,
#     basis=vector_field_basis,
#     scale=scale,
#     arrow_size=arrow_size,
#     p_mass_min=1,
#     autoscale=True,
#     density=density,
#     only_grid=False,
#     uncertain_measure="base magnitude",
#     cmap="summer",
#     cmax=None,
#     show_titles=False,
# )
# ax[4].set_title(
#     r"base magnitude $\sigma$"
#     if matplotlib.rcParams["text.usetex"]
#     else "base magnitude σ",
#     fontsize=default_fontsize,
#     pad=default_title_padding,
# )


# def plot_arrow_examples(
#     adata,
#     v_maps,
#     embeds_radian,
#     embed_mean,
#     ax=None,
#     fig=None,
#     cbar=True,
#     basis="umap",
#     n_sample=30,
#     scale=0.0021,
#     alpha=0.02,
#     index=19,
#     index2=0,
#     scale2=0.04,
#     num_certain=3,
#     num_total=4,
#     p_mass_min=1.0,
#     density=0.3,
#     arrow_size=4,
#     customize_uncertain=None,
# ):
#     X_grid, V_grid, uncertain = project_grid_points(
#         adata.obsm[f"X_{basis}"],
#         v_maps,
#         get_posterior_sample_angle_uncertainty(embeds_radian / np.pi * 180)
#         if customize_uncertain is None
#         else customize_uncertain,
#         p_mass_min=p_mass_min,
#         density=density,
#     )
#     # print(X_grid.shape, V_grid.shape, uncertain.shape)
#     norm = Normalize()
#     norm.autoscale(uncertain)
#     colormap = cm.inferno

#     indexes = np.argsort(uncertain)[::-1][
#         index : (index + num_total - num_certain)
#     ]
#     hl, hw, hal = default_arrow(arrow_size)
#     # print(hl, hw, hal)
#     quiver_kwargs = {"angles": "xy", "scale_units": "xy"}
#     # quiver_kwargs = {"angles": "xy", "scale_units": "width"}
#     quiver_kwargs = {"width": 0.002, "zorder": 0}
#     quiver_kwargs.update({"headlength": hl / 2})
#     quiver_kwargs.update({"headwidth": hw / 2, "headaxislength": hal / 2})

#     ax.scatter(
#         adata.obsm[f"X_{basis}"][:, 0],
#         adata.obsm[f"X_{basis}"][:, 1],
#         s=1,
#         linewidth=0,
#         color="gray",
#         alpha=alpha,
#     )

#     # normalize arrow size the constant
#     V_grid[:, 0] = V_grid[:, 0] / np.sqrt(V_grid[:, 0] ** 2 + V_grid[:, 1] ** 2)
#     V_grid[:, 1] = V_grid[:, 1] / np.sqrt(V_grid[:, 1] ** 2 + V_grid[:, 1] ** 2)

#     for i in range(n_sample):
#         for j in indexes:
#             # ax.quiver(
#             #    X_grid[j, 0],
#             #    X_grid[j, 1],
#             #    embed_mean[j, 0],
#             #    embed_mean[j, 1],
#             #    ec='black',
#             #    scale=scale,
#             #    color=colormap(norm(uncertain))[j],
#             #    **quiver_kwargs,
#             # )
#             ax.quiver(
#                 X_grid[j, 0],
#                 X_grid[j, 1],
#                 V_grid[j][0][i],
#                 V_grid[j][1][i],
#                 ec="face",
#                 norm=Normalize(vmin=0, vmax=360),
#                 scale=scale,
#                 color=colormap(norm(uncertain))[j],
#                 linewidth=0,
#                 alpha=0.3,
#                 **quiver_kwargs,
#             )
#         ax.quiver(
#             X_grid[j, 0],
#             X_grid[j, 1],
#             V_grid[j][0].mean(),
#             V_grid[j][1].mean(),
#             ec="black",
#             alpha=1,
#             norm=Normalize(vmin=0, vmax=360),
#             scale=scale,
#             linewidth=0,
#             color=colormap(norm(uncertain))[j],
#             **quiver_kwargs,
#         )
#     indexes = np.argsort(uncertain)[index2 : (index2 + num_certain)]
#     for i in range(n_sample):
#         for j in indexes:
#             # ax.quiver(
#             #    X_grid[j, 0],
#             #    X_grid[j, 1],
#             #    embed_mean[j, 0],
#             #    embed_mean[j, 1],
#             #    ec='black',
#             #    scale=scale,
#             #    color=colormap(norm(uncertain))[j],
#             #    **quiver_kwargs,
#             # )
#             ax.quiver(
#                 X_grid[j, 0],
#                 X_grid[j, 1],
#                 V_grid[j][0][i],
#                 V_grid[j][1][i],
#                 # ec=colormap(norm(uncertain))[j],
#                 ec="face",
#                 scale=scale2,
#                 alpha=0.3,
#                 linewidth=0,
#                 color=colormap(norm(uncertain))[j],
#                 norm=Normalize(vmin=0, vmax=360),
#                 **quiver_kwargs,
#             )
#         ax.quiver(
#             X_grid[j, 0],
#             X_grid[j, 1],
#             V_grid[j][0].mean(),
#             V_grid[j][1].mean(),
#             ec="black",
#             alpha=1,
#             linewidth=0,
#             norm=Normalize(vmin=0, vmax=360),
#             scale=scale2,
#             color=colormap(norm(uncertain))[j],
#             **quiver_kwargs,
#         )
#     ax.axis("off")
