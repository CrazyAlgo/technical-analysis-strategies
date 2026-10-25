
"""Reusable Matplotlib plotting utilities."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def new_axes( ax: Axes | None = None, figsize: tuple[float, float] = (12, 6)) -> Axes:
    """Return the supplied Axes or create a new one."""

    if ax is not None:
        if not isinstance(ax, Axes):
            raise TypeError("ax must be a Matplotlib Axes or None.")

        return ax

    if ( not isinstance(figsize, (tuple, list)) or len(figsize) != 2 ):
        raise ValueError("figsize must contain two positive dimensions.")

    if any( isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf") for value in figsize ):
        raise ValueError("figsize dimensions must be positive and finite.")

    _, ax = plt.subplots(figsize=figsize)

    return ax
