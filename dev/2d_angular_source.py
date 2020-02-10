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
angle_samples = itertools.cycle([5, 9, 25])
center = itertools.cycle([(0, 0), (3, 0), (0, 3)])
central_angle = itertools.cycle([0, PI/4, PI/2, PI, -PI/2])
beam_samples = itertools.cycle([5, 9, 25])

# build the source rays
start_angle = next(angular_size)
angles = distributions.StaticUniformAngularDistribution(
    -start_angle, start_angle, next(angle_samples)
)
base_points = distributions.StaticUniformBeam(-.5, .5, next(beam_samples))
#base_points = distributions.StaticUniformSquare(.1,5,y_size=.3,y_res=10)
#base_points = distributions.StaticUniformCircle(next(beam_samples))
source = sources.AngularSource(
    2, next(center), next(central_angle), angles, base_points, [drawing.YELLOW], dense=True
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
redraw()

def toggle_angular_size():
    val = next(angular_size)
    print(f"set angular_size to {val}")
    angles.min_angle = -val
    angles.max_angle = val
    redraw()
    
def toggle_angle_samples():
    val = next(angle_samples)
    print(f"set angle_samples to {val}")
    angles.sample_count = val
    redraw()
    
def toggle_beam_samples():
    val = next(beam_samples)
    print(f"set beam_samples to {val}")
    base_points.sample_count = val
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
        toggle_angle_samples()
    elif event.key == 'c':
        toggle_center()
    elif event.key == 'r':
        toggle_central_angle()
    elif event.key == 'b':
        toggle_beam_samples()
        
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
