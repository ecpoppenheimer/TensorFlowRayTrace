from math import pi as PI
import itertools

import pyvista as pv
import numpy as np
import tensorflow as tf

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
    3, start_points, end_points, [drawing.YELLOW], dense=True
)

plot = pv.Plotter()
plot.add_axes()
drawer = drawing.RayDrawer3D(plot)
central_angle_drawer = drawing.RayDrawer3D(plot)

def redraw():
    source.update()
    drawer.rays = source
    drawer.draw()
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

plot.add_key_event("w", toggle_start_samples)
plot.add_key_event("s", toggle_start_ends)
plot.add_key_event("a", toggle_end_samples)
plot.add_key_event("d", toggle_end_ends)
plot.show()
