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
import tfrt.graph as graph

PI = tf.constant(pi, dtype=tf.float64)

# build the source rays
angles = distributions.ManualAngularDistribution([(1, 0, 0)])
source = sources.PointSource(
    3,
    (-5, 0, 0),
    (1, 0, 0),
    angles, 
    [drawing.YELLOW], 
    dense=False,
    rank_type=None
)

# build the boundaries
zero_points = pv.PolyData(
    np.array([(0, 0, 1), (0, -1, -1), (0, 1, -1)]),
    np.array([3, 0, 1, 2])
)

# do the mesh tricks
top_parent = graph.get_closest_point(zero_points, (0, 0, 0))
vertex_update_map, accumulator = graph.mesh_parametrization_tools(zero_points, top_parent)

print(f"accumulator: {accumulator}")
print(f"vertex update map: {vertex_update_map}")

vg = boundaries.FromVectorVG((1, 0, 0))
lens = boundaries.ParametricTriangleBoundary(
    zero_points,
    vg,
    True,
    auto_update_mesh=True,
    material_dict={"mat_in": 1, "mat_out": 0},
    vertex_update_map=vertex_update_map
)
target = boundaries.ManualTriangleBoundary(mesh=pv.Plane(
    center=(5, 0, 0),
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
system.sources = [source]
system.materials = [{"n": materials.vacuum}, {"n": materials.acrylic}]
system.update()

# draw the boundary
plot = pv.Plotter()
plot.add_axes()
lens_drawer = drawing.TriangleDrawer(
    plot, color="cyan", show_edges=True, draw_norm_arrows=True, norm_arrow_visibility=True,
    draw_parameter_arrows=True, parameter_arrow_visibility=True
)
lens_drawer.surface = lens
lens_drawer.draw()

# build the engine the trace
trace_engine = engine.OpticalEngine(
    3,
    [operation.StandardReaction()],
    simple_ray_inheritance = {"wavelength"}
)
trace_engine.optical_system = system
trace_engine.validate_system()
trace_engine.ray_trace(6)

# define various optimization steps
def process_gradient(engine, lr, parameters, grad_clip, accumulator):
    """
    Calculate and process the gradient.
    
    Returns the processed gradient and the error.
    """
    with tf.GradientTape() as tape:
        engine.optical_system.update()
        engine.clear_ray_history()
        engine.ray_trace(2)
        output = tf.stack(
            [engine.finished_rays["y_end"], engine.finished_rays["z_end"]],
            axis=1
        )
        goal = tf.constant([(0, -.25)], dtype=tf.float64)
        error = tf.math.squared_difference(output, goal)
        grad = tape.gradient(error, parameters)
        
        print(f"raw gradient: {grad}")
        
        try:
            grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        except(ValueError):
            grad = tf.zeros_like(parameters, dtype=tf.float64)
        grad *= lr
        grad = tf.clip_by_value(grad, -grad_clip, grad_clip)
        print(f"gradient after scaling: {grad}")
        grad = tf.reshape(grad, (-1, 1))
        grad = tf.matmul(accumulator, grad)
        grad = tf.reshape(grad, (-1,))
        print(f"gradient after accumulation: {grad}")
        
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

def smooth(surface, smoother):
    params = tf.reshape(surface.parameters, (-1, 1))
    params = tf.matmul(smoother, params)
    params = tf.reshape(params, (-1,))
    surface.parameters.assign(params)
    surface.update()

def single_step(
    engine,
    lr,
    momentum,
    parameters,
    optimizer,
    accumulator,
    grad_clip=1e-3,
):
    """
    Does the mathmatical parts of optimization, but does not update the display.
    """
    set_momentum(momentum)
    grads, error = process_gradient(
        engine, lr, parameters, grad_clip, accumulator
    )
    print(f"step {optimizer.iterations.numpy()} error: {error.numpy()}")
    optimizer.apply_gradients([(grads, parameters)])
    redraw()
    
optimizer = tf.optimizers.SGD(
    learning_rate=1.0,
    momentum=momentum.get_value(),
    nesterov=True
)
    
def training_routine():
    for i in range(25):
        single_step(
            trace_engine, .2, .8, lens.parameters, optimizer, accumulator, 
            grad_clip=.2
        )
    for i in range(25):
        single_step(
            trace_engine, .05, .9, lens.parameters, optimizer, accumulator, 
            grad_clip=.2
        )

# draw the rays
ray_drawer = drawing.RayDrawer3D(plot)
trace_engine.ray_trace(2)
def redraw():
    system.update()
    ray_drawer.rays = trace_engine.all_rays
    lens_drawer.draw()
    ray_drawer.draw()
redraw()

plot.add_key_event("t", training_routine)

plot.show()
