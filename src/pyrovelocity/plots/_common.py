import matplotlib
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from beartype import beartype

from pyrovelocity.logging import configure_logging

__all__ = ["calculate_adaptive_n_neighbors", "set_colorbar", "set_font_size"]

logger = configure_logging(__name__)


@beartype
def calculate_adaptive_n_neighbors(
    n_obs: int, 
    min_neighbors: int = 3, 
    max_fraction: float = 0.05
) -> int:
    """Calculate adaptive n_neighbors based on dataset size.
    
    Uses scVelo's default formula (n_obs/50) as the base calculation,
    with additional constraints for very small or very large datasets.
    
    Args:
        n_obs: Number of observations/cells in the dataset.
        min_neighbors: Minimum number of neighbors to ensure stable velocity estimation.
            For very small datasets (<150 cells), at least 3 neighbors 
            are needed for meaningful results. Defaults to 3.
        max_fraction: Maximum fraction of total cells to use as neighbors.
            Prevents using too large a neighborhood (e.g., 5% of 100k cells = 5k neighbors max).
            Defaults to 0.05.
        
    Returns:
        Appropriate number of neighbors for the dataset size.
        
    Examples:
        >>> calculate_adaptive_n_neighbors(50)
        3
        >>> calculate_adaptive_n_neighbors(500)
        10
        >>> calculate_adaptive_n_neighbors(5000)
        100
        >>> calculate_adaptive_n_neighbors(50000)
        1000
        >>> calculate_adaptive_n_neighbors(200000, max_fraction=0.02)
        4000
        >>> calculate_adaptive_n_neighbors(10, min_neighbors=5)
        5
    """
    # Use scVelo's default formula as base
    n_neighbors = int(n_obs / 50)
    
    # Ensure minimum for small datasets
    n_neighbors = max(min_neighbors, n_neighbors)
    
    # Cap at reasonable fraction of total cells for very large datasets
    # but never go below the minimum
    max_neighbors = max(int(n_obs * max_fraction), min_neighbors)
    n_neighbors = min(n_neighbors, max_neighbors)
    
    return n_neighbors


def set_font_size(size: int):
    matplotlib.rcParams.update({"font.size": size})


def set_colorbar(
    smp,
    ax,
    orientation="vertical",
    labelsize=None,
    fig=None,
    position="right",
    rainbow=False,
    axes_label=None,
):
    if position == "right" and (not rainbow):
        cax = inset_axes(ax, width="2%", height="30%", loc=4, borderpad=0)
        cb = fig.colorbar(smp, orientation=orientation, cax=cax)
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position, size="8%", pad=0.08)
        cb = fig.colorbar(smp, cax=cax, orientation=orientation, shrink=0.4)
    if axes_label:
        cax.set_label(axes_label)

    cb.ax.tick_params(labelsize=labelsize)

    # TODO: remove cb.draw_all()
    #
    # MatplotlibDeprecationWarning: The draw_all function was deprecated in
    # Matplotlib 3.6 and will be removed two minor releases later. Use
    # fig.draw_without_rendering() instead. cbar.draw_all()
    #
    # draw_all is not required with cb.solids.set_alpha(1)
    # https://matplotlib.org/stable/api/colorbar_api.html#matplotlib.colorbar.Colorbar.set_alpha
    cb.solids.set_alpha(1)
    # cb.set_alpha(1)
    # cb.draw_all()

    cb.locator = MaxNLocator(nbins=2, integer=True)

    if position == "left":
        cb.ax.yaxis.set_ticks_position("left")
    cb.update_ticks()
