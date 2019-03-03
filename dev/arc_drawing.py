# arc drawing test

import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.TFRayTrace as tfrt
import tfrt.drawing as rtdraw
import tfrt.OpticsUtilities as outl

PI = math.pi

# set up the figure and axes
fig, ax = plt.subplots(1, 1, figsize=(15, 9))
plt.show(block=False)

# configure axes
ax.set_aspect("equal")
ax.set_xbound(-2, 2)
ax.set_ybound(-2, 2)

arcDrawer = rtdraw.ArcDrawer(ax, color=(0, 0, 1), include_norm_arrows=True)


def on_key(event):
    # Message loop called whenever a key is pressed on the figure

    if event.key == "t":
        arcCount = 10
        centers = np.random.uniform(-1.0, 1.0, (arcCount, 2))
        angles = np.random.uniform(0, 2 * PI, (arcCount, 2))
        radi = np.random.uniform(-1.0, 1.0, (arcCount, 1))
        arcs = np.concatenate([centers, angles, radi], axis=1)
        arcDrawer.update(arcs)
    elif event.key == "c":
        arcDrawer.clear()
    elif event.key == "r":
        arcDrawer.color = (0, 1, 1)
    elif event.key == "n":
        arcDrawer.toggle_norm_arrow_visibility()
    else:
        return

    plt.gcf().canvas.draw()


fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
