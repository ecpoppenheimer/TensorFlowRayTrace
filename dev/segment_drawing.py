# segment drawing test

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

segmentDrawer = rtdraw.SegmentDrawer(ax, color=(0, 0, 1), include_norm_arrows=True)

segmentCount = 10
segments = np.random.uniform(-1.0, 1.0, (segmentCount, 4))


def on_key(event):
    # Message loop called whenever a key is pressed on the figure

    if event.key == "t":
        segmentDrawer.update(segments)
        # segmentDrawer.update(np.random.uniform(-1.0, 1.0, (segmentCount, 4)))
    elif event.key == "c":
        segmentDrawer.clear()
    elif event.key == "r":
        segmentDrawer.line_collection.set_color((0, 1, 1))
    elif event.key == "n":
        segmentDrawer.toggle_norm_arrow_visibility()
    else:
        return

    plt.gcf().canvas.draw()


fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
