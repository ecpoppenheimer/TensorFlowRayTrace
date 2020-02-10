from math import pi as PI
import itertools

import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.sources as sources
import tfrt.distributions as distributions
import tfrt.drawing as drawing
from tfrt.spectrumRGB import rgb

start_samples = itertools.cycle([3, 5, 7])
start_ends = itertools.cycle([((0, -1), (0, 1)), ((0, -.5), (0, .5)), ((0, 0), (-1, 1))])
end_samples = itertools.cycle([3, 5, 7])
end_ends = itertools.cycle([((1, -1), (1, 1)), ((1, -.5), (1, .5)), ((1, -1), (1, 0))])

# build the source rays
se = next(start_ends)
start_points = distributions.StaticUniformAperaturePoints(se[0], se[1], next(start_samples))
ee = next(end_ends)
end_points = distributions.StaticUniformAperaturePoints(ee[0], ee[1], next(end_samples))
source = sources.AperatureSource(
    2, start_points, end_points, [drawing.YELLOW], dense=True
)

print(f"source fields: {source.keys()}")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_aspect("equal")
ax.set_xbound(-4, 4)
ax.set_ybound(-4, 4)
drawer = drawing.RayDrawer2D(ax)
drawing.disable_figure_key_commands()

def redraw():
    source.update()
    drawer.rays = source
    drawer.draw()
    drawing.redraw_current_figure()
redraw()

def toggle_start_samples():
    val = next(start_samples)
    print(f"set start_samples to {val}")
    start_points.sample_count = val
    redraw()

def toggle_start_ends():
    val = next(start_ends)
    print(f"set start_ends to {val}")
    start_points.start_point = val[0]
    start_points.end_point = val[1]
    redraw()
    
def toggle_end_samples():
    val = next(end_samples)
    print(f"set end_samples to {val}")
    end_points.sample_count = val
    redraw()

def toggle_end_ends():
    val = next(end_ends)
    print(f"set end_ends to {val}")
    end_points.start_point = val[0]
    end_points.end_point = val[1]
    redraw()

def on_key(event):
    if event.key == 'q':
        toggle_start_samples()
    elif event.key == 'a':
        toggle_start_ends()
    elif event.key == 'e':
        toggle_end_samples()
    elif event.key == 'd':
        toggle_end_ends()
        
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
