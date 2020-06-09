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
import tfrt.mesh_tools as mt

PI = tf.constant(pi, dtype=tf.float64)

precompiled_start_filename = "./data/PCF_precompiled_start_points.dat"
do_precompile = False

ray_sample_count = 2000
start_point_perturbation_factor = .02

# set up the imaging problem
object_distance = 10
magnification = 1
object_size = .2

lens_res_scale = .06
lens_aperature = 1
theta_start = 0
theta_end = PI/6

# calculated parameters
image_distance = object_distance * magnification
image_size = magnification * object_size
start_point_perturbation = start_point_perturbation_factor * object_size

if do_precompile:
    # build the source and precompile it, then exit.
    
    # generate the target points.  This will make a square of side length image_size centered
    # on the origin, in 2D (yz plane)
    target_point_distribution = distributions.ImageBasePoints(
        "./data/full_hex_blob.png",
        image_size
    )
    target_point_count = target_point_distribution.points.shape[0]
    
    # print statistics on the generated points
    print("===============================")
    print("===============================")
    print("Performing source precompilation.")
    print(f"Generated {target_point_count} points from image.")
    print(f"Sanity Check: image has {target_point_distribution.grey_levels} grey levels.")

    # generate the start points of the rays
    start_points = distributions.RandomUniformCircle(
        target_point_count,
        object_size
    )
    start_point_transform = distributions.BasePointTransformation(
        start_points,
        translation=(-object_distance, 0, 0)
    )

    # Perform transform_map to rearrange the points in target_point_distribution so that
    # they match the order of the start_points.
    #
    # Start points will be used as the fixed distribution.  If it's points are taken now,
    # the transformation will not be applied.  But this is actually what we want; we want
    # them 2D, and the transformation moves them orthagonal to the directions we care about.
    
    def get_mean_difference(first, second):
        return tf.reduce_mean(tf.abs(first - second))
    
    target_points = target_point_distribution.points * -magnification
    print(
        "Before transform_map, mean difference between input and output distributions is:"
    )
    print(
        f"{get_mean_difference(start_points.points, target_points)}"
    )
    print("starting transform_map on the target points...")
    target_points = distributions.transform_map(
        start_points.points,
        target_points
    )
    print("completed transform_map!")
    print(
        f"After transform_map, mean difference between input and output distributions is:"
    )
    print(
        f"is {get_mean_difference(start_points.points, target_points)}"
    )

    # build the precompiled start point distribution
    start_points.update()
    precompiled_start_points = distributions.PrecompiledBasePoints(
        start_points,
        do_downsample=False
    )
    precompiled_start_points.ranks = target_points
    precompiled_start_points.save(precompiled_start_filename)
    
    print("===============================")
    print("Sucessfully Finished!")
    print(
        "Quitting.  Toggle the 'do_precompile' flag in the script to use the precompiled "
        "start points to run the optimization."
    )
    exit()
else:
    # else load the precompiled source from the file
    print("Loading precompiled start points.")
    start_points = distributions.PrecompiledBasePoints(
        precompiled_start_filename,
        sample_count=ray_sample_count,
        perturbation=(0, start_point_perturbation, start_point_perturbation)
    )
    
    end_points = distributions.RandomUniformCircle(
        ray_sample_count,
        .99*lens_aperature,
        theta_start=theta_start,
        theta_end=theta_end
    )
    # The only effect of this is to transform end points to 3D
    end_point_transform = distributions.BasePointTransformation(end_points)
    
    source = sources.AperatureSource(
        3,
        start_points, 
        end_points, 
        [drawing.YELLOW], 
        dense=False,
        extra_fields={
            "goal": ("start_point", start_points, "ranks")
        }
    )
    """print("===============================")
    print("Source printout")
    for key in source.keys():
        print(f"{key} shape: {source[key].shape}")"""

# build the boundaries
points, faces = mt.circular_mesh(
    lens_aperature,
    lens_res_scale,
    theta_start=theta_start,
    theta_end=theta_end
)
zero_points = pv.PolyData(points, faces)
zero_points.rotate_y(90)
zero_points.rotate_x(90)
print("Parametric mesh properties:")
print(f"vertices/parameters: {2 * points.shape[0]}")
# div by 2 because faces is 1D here => each face accounts for 4 array elements, and there
# are two meshes
print(f"faces: {int(faces.shape[0] / 2)}")

# do the mesh tricks
top_parent = mt.get_closest_point(zero_points, (0, 0, 0))
vertex_update_map, accumulator = mt.mesh_parametrization_tools(zero_points, top_parent)

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
    center=(image_distance, 0, 0),
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
system.sources = [source]
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

# build the trace engine
trace_engine = engine.OpticalEngine(
    3,
    [operation.StandardReaction()],
    simple_ray_inheritance = {"wavelength", "goal"}
)
trace_engine.optical_system = system
trace_engine.validate_system()

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
        goal = engine.finished_rays["goal"]
        error = tf.math.squared_difference(output, goal)
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

smoother = mt.mesh_smoothing_tool(
    zero_points,
    [300, 50, 20, 10, 5]
)
    
def training_routine():
    parameter_history = []
    print("starting training routine.")
    start_time = time.time()
    def save_history():
        parameter_history.append(
            (lens.surfaces[0].parameters.numpy(), lens.surfaces[1].parameters.numpy())
        )

    smoother = mt.mesh_smoothing_tool(
        zero_points,
        [500, 50, 20, 10, 5]
    )
    for i in range(50):
        single_step(
            trace_engine, 2e-8, 2e-8, .6, lens.parameters, optimizer, accumulator, 
            grad_clip=1e-3
        )
        smooth(lens.surfaces[0], smoother)
        smooth(lens.surfaces[1], smoother)
        if i % 10 == 0:
            #simple_trace_redraw()
            save_history()
        first_drawer.draw()
        
    smoother = mt.mesh_smoothing_tool(
        zero_points,
        [500, 10, 4, 2]
    )
    for i in range(25):
        single_step(
            trace_engine, 4e-8, 4e-8, .9, lens.parameters, optimizer, None, 
            grad_clip=1e-3
        )
        smooth(lens.surfaces[0], smoother)
        smooth(lens.surfaces[1], smoother)
        if i % 10 == 0:
            #simple_trace_redraw()
            save_history()
        first_drawer.draw()
       
    for i in range(50):
        single_step(
            trace_engine, 9e-8, 9e-8, .95, lens.parameters, optimizer, None, 
            grad_clip=1e-3
        )
        if i % 10 == 0:
            #simple_trace_redraw()
            save_history()
        first_drawer.draw()
        
    for i in range(100):
        single_step(
            trace_engine, 2e-8, 2e-8, .98, lens.parameters, optimizer, None, 
            grad_clip=1e-3
        )
        if i % 10 == 0:
            #simple_trace_redraw()
            save_history()
        first_drawer.draw()
    
    with open("./data/PCF_parameters.dat", 'wb') as outFile:
        pickle.dump(parameter_history, outFile, pickle.HIGHEST_PROTOCOL)
    
    end_time = time.time()    
    print(f"Completed training routine.  Took {end_time-start_time} seconds.")

# draw the rays
ray_drawer = drawing.RayDrawer3D(plot)    
def simple_trace_redraw():
    system.update()
    trace_engine.ray_trace(3)
    ray_drawer.rays = trace_engine.all_rays
    first_drawer.draw()
    second_drawer.draw()
    ray_drawer.draw()
    
def clear_rays():
    ray_drawer.rays = None
    ray_drawer.draw()
    
def save():
    lens.surfaces[0].save("./stl/PCF_first.stl")
    lens.surfaces[1].save("./stl/PCF_second.stl")
    print("Saved mesh.")
    
def load_parameters():
    with open("./data/PCF_parameters.dat", 'rb') as inFile:
        parameter_history = pickle.load(inFile)
    p1, p2 = parameter_history[-1]
    lens.surfaces[0].parameters.assign(p1)
    lens.surfaces[1].parameters.assign(p2)
    system.update()
    first_drawer.draw()
    second_drawer.draw()
    print("Loaded from file.")
    
def save_parameters():
    with open("./data/PCF_parameters.dat", 'wb') as outFile:
        pickle.dump([(
            lens.surfaces[0].parameters.numpy(),
            lens.surfaces[1].parameters.numpy()
        )], outFile, pickle.HIGHEST_PROTOCOL)
    print("Saved parameters.")
    
"""# draw the target points as spheres.
target_points.update()
plot.add_mesh(
    pv.PolyData(target_points.points.numpy()),
    color="Red",
    point_size=10,
    render_points_as_spheres=True
)"""

plot.add_key_event("d", simple_trace_redraw)
plot.add_key_event("t", training_routine)
plot.add_key_event("v", save)
plot.add_key_event("l", load_parameters)
plot.add_key_event("p", save_parameters)
plot.add_key_event("c", clear_rays)

plot.show()
