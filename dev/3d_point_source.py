from math import pi as PI
import itertools

import pyvista as pv
import numpy as np
import tensorflow as tf

import tfrt.sources as sources
import tfrt.distributions as distributions
import tfrt.drawing as drawing
from tfrt.spectrumRGB import rgb

angular_size = itertools.cycle(np.array([PI/2, PI/3, PI/4, PI/8, PI/12, PI/48], dtype=np.float64))
sample_count = itertools.cycle([100, 500, 2500])
center = itertools.cycle([(0, 0, 0), (3, 0, 0), (0, 3, 0)])
central_angle = itertools.cycle([(1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 1), (-1, 0, 0)])

# build the source rays
angles = distributions.StaticUniformSphere(next(angular_size), next(sample_count))
source = sources.PointSource(
    3, next(center), next(central_angle), angles, [drawing.YELLOW], dense=True
)

plot = pv.Plotter()
plot.add_axes()
drawer = drawing.RayDrawer3D(plot)
central_angle_drawer = drawing.RayDrawer3D(plot)
class ManualSource:
    def __init__(self):
        self._fields = {}
        
    def __getitem__(self, key):
        return self._fields[key]

    def __setitem__(self, key, item):
        self._fields[key] = item
        
    @property
    def dimension(self):
        return 3   
        
    @property    
    def keys(self):
        return self._fields.keys

central_angle_ray = ManualSource()
central_angle_ray["x_start"] = np.array([0], dtype=np.float64)
central_angle_ray["y_start"] = np.array([0], dtype=np.float64)
central_angle_ray["z_start"] = np.array([0], dtype=np.float64)
central_angle_ray["wavelength"] = np.array([drawing.RED], dtype=np.float64)

def redraw():
    source.update()
    drawer.rays = source
    drawer.draw()
    
    central_angle_ray["x_end"], central_angle_ray["y_end"], central_angle_ray["z_end"] = \
        tf.unstack(3 * tf.cast(tf.reshape(source.central_angle, (3, 1)), tf.float64))
    central_angle_drawer.rays = central_angle_ray
    central_angle_drawer.draw()
redraw()

def toggle_angular_size():
    val = next(angular_size)
    print(f"set angular_size to {val}")
    angles.angular_size = val
    redraw()
    
def toggle_sample_count():
    val = next(sample_count)
    print(f"set sample_count to {val}")
    angles.sample_count = val
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

plot.add_key_event("a", toggle_angular_size)
plot.add_key_event("s", toggle_sample_count)
plot.add_key_event("c", toggle_center)
plot.add_key_event("r", toggle_central_angle)
plot.show()
