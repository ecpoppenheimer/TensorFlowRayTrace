from math import pi as PI
import itertools

import pyvista as pv
import numpy as np
import tensorflow as tf

import tfrt.sources as sources
import tfrt.distributions as distributions
import tfrt.drawing as drawing
from tfrt.spectrumRGB import rgb

start_samples = itertools.cycle([10, 40, 150])
start_translation = itertools.cycle([None, (0, 0, 0), (-1, 0, 0), (0, 1, 0)])
end_samples = itertools.cycle([10, 40, 150])
end_translation = itertools.cycle([(1, 0, 0), (2, 0, 0), (1, 1, 0), (1, 1, 1)])
rotation = itertools.cycle([None, (1, 0, 0), (1, 1, 0), (-1, 0, 0), (1, 1, 1), (0, 1, 3)])

# build the source rays
start_base = distributions.StaticUniformCircle(next(start_samples), radius=.5)
end_base = distributions.StaticUniformCircle(next(end_samples), radius=1.0)
initial_rotation = next(rotation)
start_points = distributions.TransformableBasePoints(
    start_base, rotation=initial_rotation, translation=next(start_translation)
)
end_points = distributions.TransformableBasePoints(
    end_base, rotation=initial_rotation, translation=next(end_translation)
)
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
    """print("Source field printout:")
    for key in source.keys():
        print(f"{key}: {source[key].shape}")
    print("----------------------")"""
redraw()

def toggle_start_samples():
    val = next(start_samples)
    print(f"set start_samples to {val}")
    start_base.sample_count = val
    source.resize()
    redraw()

def toggle_start_translation():
    val = next(start_translation)
    print(f"set start_translation to {val}")
    start_points.translation = val
    redraw()
    
def toggle_end_samples():
    val = next(end_samples)
    print(f"set end_samples to {val}")
    end_base.sample_count = val
    source.resize()
    redraw()

def toggle_end_translation():
    val = next(end_translation)
    print(f"set end_translation to {val}")
    end_points.translation = val
    redraw()
    
def toggle_rotation():
    val = next(rotation)
    print(f"set rotation to {val}")
    start_points.rotation = val
    end_points.rotation = val
    redraw()

plot.add_key_event("w", toggle_start_samples)
plot.add_key_event("s", toggle_start_translation)
plot.add_key_event("a", toggle_end_samples)
plot.add_key_event("d", toggle_end_translation)
plot.add_key_event("r", toggle_rotation)
plot.show()
