# arc drawing test

import itertools
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.TFRayTrace as tfrt
import tfrt.drawing as drawing
import tfrt.OpticsUtilities as outl

PI = math.pi

COLORS = itertools.cycle(
    ["black", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
)
STYLES = itertools.cycle(["-", "--", "-.", ":"])


def get_random_arcs(count=10):
    centers = np.random.uniform(-1.0, 1.0, (count, 2))
    angles = np.random.uniform(0, 2 * PI, (count, 2))
    radii = np.random.uniform(-1.0, 1.0, (count, 1))
    arcs = np.concatenate([centers, angles, radii], axis=1)

    return arcs


def on_key(event, drawer):
    # Message loop called whenever a key is pressed on the figure

    if event.key == "t":
        drawer.arcs = get_random_arcs()
    elif event.key == "c":
        drawer.arcs = None
    elif event.key == "n":
        drawer.toggle_norm_arrow_visibility()
    elif event.key == "a":
        drawer.norm_arrow_count += 1
    elif event.key == "z":
        drawer.norm_arrow_count -= 1
    elif event.key == "k":
        drawer.norm_arrow_length *= 1.1
    elif event.key == "m":
        drawer.norm_arrow_length *= 0.9
    elif event.key == "u":
        drawer.color = next(COLORS)
    elif event.key == "i":
        drawer.style = next(STYLES)
    elif event.key == "d":
        drawer.draw()

    drawing.redraw_current_figure()


if __name__ == "__main__":
    drawing.disable_figure_key_commands()

    # set up the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))

    # configure axes
    ax.set_aspect("equal")
    ax.set_xbound(-2, 2)
    ax.set_ybound(-2, 2)

    # set up drawer
    drawer = drawing.ArcDrawer(
        ax,
        arcs=get_random_arcs(),
        color=next(COLORS),
        style=next(STYLES),
        draw_norm_arrows=True,
    )
    drawer.draw()

    # hand over to user
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, drawer))
    plt.show()
