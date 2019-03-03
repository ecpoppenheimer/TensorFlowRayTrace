# ray drawing test

import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.TFRayTrace as tfrt
import tfrt.drawing as rtdraw
import tfrt.OpticsUtilities as outl

# set up the figure and axes
fig, ax = plt.subplots(1, 1, figsize=(15, 9))
plt.show(block=False)

# configure axes
ax.set_aspect("equal")
ax.set_xbound(-0, 1)
ax.set_ybound(-0, 1)

rayDrawer = rtdraw.RayDrawer(ax, style="-")

rayCount = 50
wavelengths = np.linspace(rtdraw.VISIBLE_MIN, rtdraw.VISIBLE_MAX, rayCount)
rays = np.array([[w, 0.1, w, 0.9, w] for w in wavelengths])


def on_key(event):
    # Message loop called whenever a key is pressed on the figure

    if event.key == "t":
        rayDrawer.update(rays)
    elif event.key == "c":
        rayDrawer.clear()
    elif event.key == "n":
        rayDrawer.line_collection.norm = plt.Normalize(450, 600)
        rayDrawer.update(rays)
    else:
        return

    plt.gcf().canvas.draw()


fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
