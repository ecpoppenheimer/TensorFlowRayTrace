from math import pi as PI
import itertools

import pyvista as pv
import numpy as np
import tensorflow as tf
import tfquaternion as tfq

import tfrt.sources as sources
import tfrt.distributions as distributions
import tfrt.drawing as drawing
from tfrt.spectrumRGB import rgb
import tfrt.geometry as geometry

angular_size = itertools.cycle(np.array([PI/2, PI/3, PI/4, PI/8, PI/12, PI/48], dtype=np.float64))
sample_count = itertools.cycle([100, 500, 2500])
center = itertools.cycle([(0, 0, 0), (3, 0, 0), (0, 3, 0)])
central_angle = tf.constant((1.0, 0.0, 0.0), dtype=tf.float64)

angle_step_size = PI/12
a = np.cos(angle_step_size / 2.0)
b = np.sin(angle_step_size / 2.0)
x_rotation = tfq.Quaternion((a, b, 0.0, 0.0), dtype=tf.float64)
y_rotation = tfq.Quaternion((a, 0.0, b, 0.0), dtype=tf.float64)
z_rotation = tfq.Quaternion((a, 0.0, 0.0, b), dtype=tf.float64)

# build the source rays
angles = distributions.StaticUniformSphere(next(angular_size), next(sample_count))
source = sources.PointSource(
    3, next(center), central_angle, angles, [drawing.YELLOW], dense=True
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
    source.central_angle = central_angle
    source.update()
    drawer.rays = source
    drawer.draw()
    
    x, y, z = tf.unstack(central_angle)
    central_angle_ray["x_end"] = np.array([x], dtype=np.float64)
    central_angle_ray["y_end"] = np.array([y], dtype=np.float64)
    central_angle_ray["z_end"] = np.array([z], dtype=np.float64)
    central_angle_drawer.rays = central_angle_ray
    central_angle_drawer.draw()
    
    """print("Source field printout:")
    for key in source.keys():
        print(f"{key}: {source[key].shape}")
    print("----------------------")"""
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
    source.resize()
    redraw()
    
def toggle_center():
    val = next(center)
    print(f"set center to {val}")
    source.center = val
    redraw()
    
def x_rotate():
    global central_angle
    central_angle = tfq.rotate_vector_by_quaternion(x_rotation, central_angle)
    redraw()

def y_rotate():
    global central_angle
    central_angle = tfq.rotate_vector_by_quaternion(y_rotation, central_angle)
    redraw()

def z_rotate():
    global central_angle
    central_angle = tfq.rotate_vector_by_quaternion(z_rotation, central_angle)
    redraw()

def flip_rotate():
    global central_angle
    central_angle = tf.constant((-1.0, 0.0, 0.0), dtype=tf.float64)
    redraw()

plot.add_key_event("a", toggle_angular_size)
plot.add_key_event("s", toggle_sample_count)
plot.add_key_event("c", toggle_center)
plot.add_key_event("x", x_rotate)
plot.add_key_event("y", y_rotate)
plot.add_key_event("z", z_rotate)
plot.add_key_event("f", flip_rotate)
plot.show()
