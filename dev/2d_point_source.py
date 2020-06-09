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

angular_size = itertools.cycle(np.array([PI/2, PI/3, PI/4, PI/8, PI/12, PI/48], dtype=np.float64))
sample_count = itertools.cycle([5, 9, 25])
center = itertools.cycle([(0, 0), (3, 0), (0, 3)])
central_angle = itertools.cycle([0, PI/4, PI/2, PI, -PI/2])

# build the source rays
start_angle = next(angular_size)
angles = distributions.StaticUniformAngularDistribution(
    -start_angle, start_angle, next(sample_count)
)
source = sources.PointSource(
    2, next(center), next(central_angle), angles, [drawing.YELLOW], dense=True
)

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
    print("Source field printout:")
    for key in source.keys():
        print(f"{key}: {source[key].shape}")
    print("----------------------")
redraw()

def toggle_angular_size():
    val = next(angular_size)
    print(f"set angular_size to {val}")
    angles.min_angle = -val
    angles.max_angle = val
    redraw()
    
def toggle_sample_count():
    val = next(sample_count)
    print(f"set sample_count to {val}")
    angles.sample_count = val
    source.resize()
    redraw()
    
def toggle_center():
    val = next(center)
    print(f"set center to {val}")
    source.center = val
    redraw()
    
def toggle_central_angle():
    val = next(central_angle)
    print(f"set central_angle to {val}")
    source.central_angle = val
    redraw()

def on_key(event):
    if event.key == 'a':
        toggle_angular_size()
    elif event.key == 's':
        toggle_sample_count()
    elif event.key == 'c':
        toggle_center()
    elif event.key == 'r':
        toggle_central_angle()
        
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
