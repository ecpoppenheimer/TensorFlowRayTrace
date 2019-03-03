# arc drawing test

import itertools
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.TFRayTrace as tfrt
import tfrt.drawing as rtdraw
import tfrt.OpticsUtilities as outl

PI = math.pi

COLORS = itertools.cycle(
    ["black", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
)
STYLES = itertools.cycle(["-", "--", "-.", ":"])


def get_random_arcs(count):
    centers = np.random.uniform(-1.0, 1.0, (count, 2))
    angles = np.random.uniform(0, 2 * PI, (count, 2))
    radii = np.random.uniform(-1.0, 1.0, (count, 1))
    arcs = np.concatenate([centers, angles, radii], axis=1)

    return arcs


def on_key(event, drawer):
    # Message loop called whenever a key is pressed on the figure

    if event.key == "t":
        drawer.arcs = get_random_arcs(10)
    elif event.key == "c":
        drawer.arcs = None
    elif event.key == "n":
        drawer.toggle_norm_arrow_visibility()
    elif event.key == "a":
        drawer.arrow_count += 1
    elif event.key == "z":
        drawer.arrow_count -= 1
    elif event.key == "u":
        drawer.color = next(COLORS)
    elif event.key == "i":
        drawer.style = next(STYLES)

    drawer.draw()
    plt.gcf().canvas.draw()


if __name__ == "__main__":
    # set up the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    # plt.show(block=False)

    # configure axes
    ax.set_aspect("equal")
    ax.set_xbound(-2, 2)
    ax.set_ybound(-2, 2)

    # set up drawer
    drawer = rtdraw.ArcDrawer(
        ax,
        arcs=get_random_arcs(10),
        color=next(COLORS),
        style=next(STYLES),
        draw_norm_arrows=True,
    )
    drawer.draw()

    # hand over to user
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, drawer))
    plt.show()