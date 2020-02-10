# ray drawing test

import itertools
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.TFRayTrace as tfrt
import tfrt.drawing as drawing
import tfrt.OpticsUtilities as outl
from tfrt.spectrumRGB import rgb

STYLES = itertools.cycle(["-", "--", "-.", ":"])
COLORMAPS = itertools.cycle(
    [
        mpl.colors.ListedColormap(rgb()),
        plt.get_cmap("viridis"),
        plt.get_cmap("seismic"),
        plt.get_cmap("spring"),
        plt.get_cmap("winter"),
        plt.get_cmap("brg"),
        plt.get_cmap("gist_ncar"),
    ]
)


def get_rays(count=50):
    wavelengths = np.linspace(drawing.VISIBLE_MIN, drawing.VISIBLE_MAX, count)
    rays = np.array([[w, 0.1, w, 0.9, w] for w in wavelengths])

    return rays


def on_key(event, drawer):
    # Message loop called whenever a key is pressed on the figure

    if event.key == "t":
        drawer.rays = get_rays()
    elif event.key == "c":
        drawer.rays = None
    elif event.key == "n":
        drawer.set_wavelength_limits(0.450, 0.650)
    elif event.key == "m":
        drawer.set_wavelength_limits(drawing.VISIBLE_MIN, drawing.VISIBLE_MAX)
    elif event.key == "i":
        drawer.style = next(STYLES)
    elif event.key == "u":
        drawer.colormap = next(COLORMAPS)
    elif event.key == "d":
        drawer.draw()

    drawing.redraw_current_figure()


if __name__ == "__main__":
    drawing.disable_figure_key_commands()

    # set up the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))

    # configure axes
    ax.set_aspect("equal")
    ax.set_xbound(0, 1)
    ax.set_ybound(0, 1)

    # set up drawer
    drawer = drawing.RayDrawer(
        ax, rays=get_rays(), style=next(STYLES), colormap=next(COLORMAPS)
    )
    drawer.draw()

    # hand over to user
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, drawer))
    plt.show()
