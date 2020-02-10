from math import pi
import pickle
import time

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
import tfrt.mesh_tools as graph

PI = tf.constant(pi, dtype=tf.float64)

# set up the imaging problem
source_distance = 4
magnification = 2
target_distance = source_distance * magnification

object_size = .2

# build the source rays
bp_count = 45
ray_count = bp_count**2
random_base_points = distributions.RandomUniformSquare(object_size, bp_count)
random_angles = distributions.RandomUniformSphere(PI/16.0, ray_count)
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

bp_count = 5
ray_count = bp_count**2
display_base_points = distributions.RandomUniformSquare(.2, bp_count)
display_angles = distributions.RandomUniformSphere(PI/16.0, ray_count)
display_source = sources.AngularSource(
    3,
    (-source_distance, 0, 0),
    (1, 0, 0),
    display_angles, 
    display_base_points, 
    [drawing.YELLOW]*ray_count, 
    dense=False,
    rank_type="base_point"
)


# build the boundaries
zero_points = pv.read("./stl/processed_disk_large.stl")
zero_points.rotate_y(90)

# do the mesh tricks
top_parent = graph.get_closest_point(zero_points, (0, 0, 0))
vertex_update_map, accumulator = graph.mesh_parametrization_tools(zero_points, top_parent)

vg = boundaries.FromVectorVG((1, 0, 0))
lens = boundaries.ParametricMultiTriangleBoundary(
    zero_points,
    vg,
    [
        boundaries.ThicknessConstraint(0.0, "min"),
        boundaries.ThicknessConstraint(0.2, "min"),
    ],
    [True, False],
    auto_update_mesh=True,
    material_list=[{"mat_in": 1, "mat_out": 0}] * 2,
    vertex_update_map=vertex_update_map
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
system.optical = lens.surfaces
system.targets = [target]
system.sources = [random_source]
system.materials = [{"n": materials.vacuum}, {"n": materials.acrylic}]
system.update()

# draw the boundary
plot = pv.Plotter()
plot.add_axes()
first_drawer = drawing.TriangleDrawer(plot, color="cyan", show_edges=True)
second_drawer = drawing.TriangleDrawer(plot, color="cyan", show_edges=True)

first_drawer.surface = lens.surfaces[0]
first_drawer.draw()
second_drawer.surface = lens.surfaces[1]
second_drawer.draw()

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
def process_gradient(engine, lr1, lr2, parameters, grad_clip, accumulator):
    """
    Calculate and process the gradient.
    
    Returns the processed gradient and the error.
    """
    with tf.GradientTape() as tape:
        engine.optical_system.update()
        engine.clear_ray_history()
        engine.ray_trace(3)
        output = tf.stack(
            [engine.finished_rays["y_end"], engine.finished_rays["z_end"]],
            axis=1
        )
        goal = engine.finished_rays["rank"]
        goal *= -(magnification * object_size)
        error = tf.math.squared_difference(output, goal)
        #error = engine.finished_rays["y_end"]**2 + engine.finished_rays["z_end"]**2
        grad1, grad2 = tape.gradient(error, parameters)
        
        try:
            grad1 = tf.where(tf.math.is_finite(grad1), grad1, tf.zeros_like(grad1))
        except(ValueError):
            grad1 = tf.zeros_like(parameters[0], dtype=tf.float64)
        grad1 *= lr1
        grad1 = tf.clip_by_value(grad1, -grad_clip, grad_clip)
        if accumulator is not None:
            grad1 = tf.reshape(grad1, (-1, 1))
            grad1 = tf.matmul(accumulator, grad1)
            grad1 = tf.reshape(grad1, (-1,))
        
        try:
            grad2 = tf.where(tf.math.is_finite(grad2), grad2, tf.zeros_like(grad2))
        except(ValueError):
            grad2 = tf.zeros_like(parameters[1], dtype=tf.float64)
        grad2 *= lr2
        grad2 = tf.clip_by_value(grad2, -grad_clip, grad_clip)
        if accumulator is not None:
            grad2 = tf.reshape(grad2, (-1, 1))
            grad2 = tf.matmul(accumulator, grad2)
            grad2 = tf.reshape(grad2, (-1,))
        
        return [grad1, grad2], tf.reduce_sum(error)
        
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
    lr1,
    lr2,
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
        engine, lr1, lr2, parameters, grad_clip, accumulator
    )
    print(f"step {optimizer.iterations.numpy()} error: {error.numpy()}")
    optimizer.apply_gradients([(grads[0], parameters[0]), (grads[1], parameters[1])])
    
optimizer = tf.optimizers.SGD(
    learning_rate=1.0,
    momentum=momentum.get_value(),
    nesterov=True
)

# draw the rays
drawer = drawing.RayDrawer3D(plot)
smoother = graph.mesh_smoothing_tool(
    zero_points,
    [300, 50, 20, 10, 5]
)

def step_with_redraw():
    smoother = graph.mesh_smoothing_tool(
        zero_points,
        [400, 20, 7, 2]
    )
    for i in range(50):
        single_step(
            trace_engine, 2e-6, 2e-6, .9, lens.parameters, optimizer, None, 
            grad_clip=1e-3
        )
        #smooth(lens.surfaces[0], smoother)
        #smooth(lens.surfaces[1], smoother)
        #if i % 10 == 0:
        #    simple_trace_redraw()
        #first_drawer.draw()
    
def training_routine():
    parameter_history = []
    print("starting training routine.")
    start_time = time.time()
    def save_history():
        parameter_history.append(
            (lens.surfaces[0].parameters.numpy(), lens.surfaces[1].parameters.numpy())
        )

    smoother = graph.mesh_smoothing_tool(
        zero_points,
        [300, 50, 20, 10, 5]
    )
    for i in range(75):
        single_step(
            trace_engine, 5e-7, 5e-7, .8, lens.parameters, optimizer, accumulator, 
            grad_clip=1e-3
        )
        smooth(lens.surfaces[0], smoother)
        smooth(lens.surfaces[1], smoother)
        if i % 10 == 0:
            simple_trace_redraw()
            save_history()
        first_drawer.draw()
        
    smoother = graph.mesh_smoothing_tool(
        zero_points,
        [400, 40, 15, 5, 2]
    )
    for i in range(50):
        single_step(
            trace_engine, 1e-5, 1e-5, .9, lens.parameters, optimizer, None, 
            grad_clip=1e-3
        )
        smooth(lens.surfaces[0], smoother)
        smooth(lens.surfaces[1], smoother)
        if i % 10 == 0:
            simple_trace_redraw()
            save_history()
        first_drawer.draw()
       
    for i in range(50):
        single_step(
            trace_engine, 2e-6, 2e-6, .95, lens.parameters, optimizer, None, 
            grad_clip=1e-3
        )
        if i % 10 == 0:
            simple_trace_redraw()
            save_history()
        first_drawer.draw()
    
    with open("./stl/saved_parameters.dat", 'wb') as outFile:
        pickle.dump(parameter_history, outFile, pickle.HIGHEST_PROTOCOL)
    
    end_time = time.time()    
    print(f"Completed training routine.  Took {end_time-start_time} seconds.")
    
def simple_trace_redraw():
    system.sources = [display_source]
    system.update()
    trace_engine.ray_trace(3)
    drawer.rays = trace_engine.all_rays
    first_drawer.draw()
    second_drawer.draw()
    drawer.draw()
    system.sources = [random_source]
    system.update()
    
def save():
    lens.surfaces[0].save("./stl/optimized_simple_first.stl")
    lens.surfaces[1].save("./stl/optimized_simple_second.stl")
    print("Saved mesh.")
    
def load_parameters():
    with open("./stl/saved_parameters_2.dat", 'rb') as inFile:
        parameter_history = pickle.load(inFile)
    p1, p2 = parameter_history[-1]
    lens.surfaces[0].parameters.assign(p1)
    lens.surfaces[1].parameters.assign(p2)
    system.update()
    first_drawer.draw()
    second_drawer.draw()
    print("Loaded from file.")
    
def save_parameters():
    with open("./stl/saved_parameters_3.dat", 'wb') as outFile:
        pickle.dump([(
            lens.surfaces[0].parameters.numpy(),
            lens.surfaces[1].parameters.numpy()
        )], outFile, pickle.HIGHEST_PROTOCOL)
    print("Saved parameters.")

plot.add_key_event("s", step_with_redraw)
plot.add_key_event("d", simple_trace_redraw)
plot.add_key_event("t", training_routine)
plot.add_key_event("v", save)
plot.add_key_event("l", load_parameters)
plot.add_key_event("p", save_parameters)

plot.show()
