from math import pi

import pyvista as pv
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.sources as sources
import tfrt.distributions as distributions
import tfrt.drawing as drawing
import tfrt.boundaries as boundaries
import tfrt.engine as engine
import tfrt.operation as operation
import tfrt.materials as materials

PI = tf.constant(pi, dtype=tf.float64)

# set up the imaging problem
source_distance = 2
magnification = 5

target_distance = source_distance * magnification

# build the source rays
bp_count = 32
ray_count = bp_count**2
random_base_points = distributions.RandomUniformSquare(.2, bp_count)
random_angles = distributions.RandomUniformSphere(PI/8.0, ray_count)
random_source = sources.AngularSource(
    3,
    (-source_distance, 0, 0),
    (1, 0, 0),
    random_angles, 
    random_base_points, 
    [drawing.YELLOW]*ray_count, 
    dense=False,
    rank_type="base_point"
)

# build the boundaries
zero_points = pv.read("./stl/processed_disk.stl")
zero_points.rotate_y(90)
vg = boundaries.FromVectorVG((1, 0, 0))
lens = boundaries.ParametricTriangleBoundary(
    zero_points,
    vg,
    flip_norm = False,
    auto_update_mesh=True,
    material_dict={"mat_in": 1, "mat_out": 0}
)
target = boundaries.ManualTriangleBoundary(mesh=pv.Plane(
    center=(target_distance, 0, 0),
    direction=(1, 0, 0),
    i_size = 100,
    j_size = 100,
    i_resolution = 1,
    j_resolution = 1
).triangulate())
target.frozen=True

# build the optical system
system = engine.OpticalSystem3D()
system.optical = [lens]
system.targets = [target]
system.sources = [random_source]
system.materials = [{"n": materials.vacuum}, {"n": materials.acrylic}]
system.update()

# draw the boundary
plot = pv.Plotter()
plot.add_axes()
first_drawer = drawing.TriangleDrawer(plot, color="cyan", show_edges=True)

first_drawer.surface = lens
first_drawer.draw()

# build the engine the trace
trace_engine = engine.OpticalEngine(
    3,
    [operation.StandardReaction()],
    simple_ray_inheritance = {"wavelength", "rank"}
)
trace_engine.optical_system = system
trace_engine.validate_system()
trace_engine.ray_trace(6)

# define various optimization steps
def process_gradient(engine, learning_rate, parameters, grad_clip):
    """
    Calculate and process the gradient.
    
    Returns the processed gradient and the error.
    """
    with tf.GradientTape() as tape:
        engine.optical_system.update()
        engine.clear_ray_history()
        engine.ray_trace(3)
        #output = tf.stack(
        #    [engine.finished_rays["y_end"], engine.finished_rays["z_end"]],
        #    axis=1
        #)
        #goal = engine.finished_rays["rank"]
        #goal *= - magnification
        #error = tf.squared_difference(output - goal)
        error = engine.finished_rays["y_end"]**2 + engine.finished_rays["z_end"]**2
        
        grad = tape.gradient(error, parameters)
        
        grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        grad = tf.clip_by_value(grad, -grad_clip, grad_clip)
        grad *= learning_rate
        
        return grad, tf.reduce_sum(error)
        
class MomentumHolder():
    def __init__(self, momentum):
        self.value = momentum
        
    @property
    def get_value(self):
        return lambda: self.value
momentum = MomentumHolder(.8)

def set_momentum(val):
    momentum.value = val

def single_step(engine, learning_rate, momentum, parameters, optimizer, grad_clip=.1):
    """
    Does the mathmatical parts of optimization, but does not update the display.
    """
    #set_momentum(momentum)
    grads, error = process_gradient(engine, learning_rate, parameters, grad_clip)
    print(f"step {optimizer.iterations.numpy()} error: {error.numpy()}")
    optimizer.apply_gradients([(grads, parameters)])
    
optimizer = tf.optimizers.SGD(
    learning_rate=1.0,
    momentum=momentum.get_value(),
    nesterov=True
)

# draw the rays
drawer = drawing.RayDrawer3D(plot)
drawer.draw()

def step_with_redraw():
    single_step(trace_engine, .1, .8, lens.parameters, optimizer)
    drawer.rays = trace_engine.all_rays
    drawer.draw()

plot.add_key_event("s", step_with_redraw)

plot.show()
