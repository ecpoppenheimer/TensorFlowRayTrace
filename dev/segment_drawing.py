# segment drawing test

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


def get_random_segments(count=10):
    return np.random.uniform(-1.8, 1.8, (count, 4))


def on_key(event, drawer):
    # Message loop called whenever a key is pressed on the figure

    if event.key == "t":
        drawer.segments = get_random_segments()
    elif event.key == "c":
        drawer.segments = None
    elif event.key == "n":
        drawer.toggle_norm_arrow_visibility()
    elif event.key == "u":
        drawer.color = next(COLORS)
    elif event.key == "i":
        drawer.style = next(STYLES)

    drawer.draw()
    plt.gcf().canvas.draw()


if __name__ == "__main__":
    # set up the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))

    # configure axes
    ax.set_aspect("equal")
    ax.set_xbound(-2, 2)
    ax.set_ybound(-2, 2)

    # set up drawer
    drawer = drawing.SegmentDrawer(
        ax,
        segments=get_random_segments(),
        color=next(COLORS),
        style=next(STYLES),
        draw_norm_arrows=True,
    )
    drawer.draw()

    # hand over to user
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, drawer))
    plt.show()
