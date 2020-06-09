from math import pi as PI
import itertools

import pyvista as pv
import numpy as np
import tensorflow as tf

import tfrt.sources as sources
import tfrt.distributions as distributions
import tfrt.drawing as drawing
from tfrt.spectrumRGB import rgb

angular_size = itertools.cycle(np.array([PI/2, PI/3, PI/4, PI/8, PI/12, PI/48, PI/100], dtype=np.float64))
angle_samples = itertools.cycle([20, 100, 1000])
center = itertools.cycle([(0, 0, 0), (3, 0, 0), (0, 3, 0)])
central_angle = itertools.cycle([(1, 0, 0), (0, 1, 0), (0, -1, 0), (1, 1, 1), (-1, 0, 0)])
beam_samples = itertools.cycle([5, 9, 25, 100])
bp_theta_start = itertools.cycle([0, PI/4, PI/2, PI, 2*PI])
bp_theta_end = itertools.cycle([2*PI, 1.5*PI, PI, PI/2, PI/4])
source_theta_start = itertools.cycle([0, PI/4, PI/2, PI, 2*PI])
source_theta_end = itertools.cycle([2*PI, 3*PI/4, PI, PI/2, PI/4])

# build the source rays
angles = distributions.StaticUniformSphere(
    next(angular_size),
    next(angle_samples),
    theta_start=next(source_theta_start),
    theta_end=next(source_theta_end)
)
#base_points = distributions.StaticUniformBeam(-.5, .5, next(beam_samples))
#base_points = distributions.StaticUniformSquare(.1,5,y_size=.3,y_res=10)
base_points = distributions.StaticUniformCircle(
    next(beam_samples),
    theta_start=next(bp_theta_start),
    theta_end=next(bp_theta_end)
)
source = sources.AngularSource(
    3, next(center), next(central_angle), angles, base_points, [drawing.YELLOW], dense=True
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
    
def toggle_angle_samples():
    val = next(angle_samples)
    print(f"set angle_samples to {val}")
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
    
def toggle_beam_samples():
    val = next(beam_samples)
    print(f"set beam_samples to {val}")
    base_points.sample_count = val
    source.resize()
    redraw()
    
def toggle_bp_theta_start():
    val = next(bp_theta_start)
    print(f"set bp_theta_start to {val}")
    base_points.theta_start = val
    redraw()
    
def toggle_bp_theta_end():
    val = next(bp_theta_end)
    print(f"set bp_theta_end to {val}")
    base_points.theta_end = val
    redraw()
    
def toggle_source_theta_start():
    val = next(source_theta_start)
    print(f"set source_theta_start to {val}")
    angles.theta_start = val
    redraw()
    
def toggle_source_theta_end():
    val = next(source_theta_end)
    print(f"set source_theta_end to {val}")
    angles.theta_end = val
    redraw()

plot.add_key_event("a", toggle_angular_size)
plot.add_key_event("s", toggle_angle_samples)
plot.add_key_event("c", toggle_center)
plot.add_key_event("r", toggle_central_angle)
plot.add_key_event("b", toggle_beam_samples)
plot.add_key_event("u", toggle_bp_theta_start)
plot.add_key_event("i", toggle_bp_theta_end)
plot.add_key_event("j", toggle_source_theta_start)
plot.add_key_event("k", toggle_source_theta_end)
plot.show()
