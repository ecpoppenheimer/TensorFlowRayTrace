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
source_distance = 4
target_distance = 8

# build the optical system
target = boundaries.ManualTriangleBoundary(mesh=pv.Plane(
    center=(target_distance, 0, 0),
    direction=(1, 0, 0),
    i_size = 100,
    j_size = 100,
    i_resolution = 1,
    j_resolution = 1
).triangulate())
target.frozen=True

system = engine.OpticalSystem3D()
system.targets = [target]
system.materials = [{"n": materials.vacuum}, {"n": materials.acrylic}]

# build the optical surfaces
first_surface = boundaries.ManualTriangleBoundary(
    file_name="./stl/optimized_simple_first.stl",
    material_dict={"mat_in": 1, "mat_out": 0}
)
second_surface = boundaries.ManualTriangleBoundary(
    file_name="./stl/optimized_simple_second.stl",
    material_dict={"mat_in": 1, "mat_out": 0}
)
system.optical = [first_surface, second_surface]

# build the trace engine
trace_engine = engine.OpticalEngine(
    3,
    [operation.StandardReaction()],
    simple_ray_inheritance = {"wavelength", "rank"}
)
trace_engine.optical_system = system

# draw the surfaces
plot = pv.Plotter()
plot.add_axes()
first_drawer = drawing.TriangleDrawer(plot, color="cyan", show_edges=True)
second_drawer = drawing.TriangleDrawer(plot, color="cyan", show_edges=True)

first_drawer.surface = first_surface
first_drawer.draw()
second_drawer.surface = second_surface
second_drawer.draw()

# build the source
if False: # build the f points
    backbone_z = np.linspace(-.2, .2, 20)
    backbone = np.stack(
        [np.zeros_like(backbone_z), -.1*np.ones_like(backbone_z), backbone_z],
        axis=1
    )
    middle_y = np.linspace(-.1, .1, 10)
    middle = np.stack(
        [np.zeros_like(middle_y), middle_y, np.zeros_like(middle_y)],
        axis=1
    )
    upper_y = np.linspace(-.1, .2, 15)
    upper = np.stack(
        [np.zeros_like(upper_y), upper_y, .2*np.ones_like(upper_y)],
        axis=1
    )
    f_points = np.concatenate([backbone, middle, upper])
    f_object = pv.PolyData(f_points)
    f_object.save("./stl/f_points.vtk")
else: # load the f points from the file
    f_object = pv.read("./stl/f_points.vtk")
moved_f_object = f_object.copy()
moved_f_object.translate([-source_distance, 0, 0])
plot.add_mesh(
    moved_f_object,
    color="yellow",
    point_size=10,
    render_points_as_spheres=True
)

bp_count = f_object.points.shape[0]
angle_count = 200
source_ray_count = bp_count * angle_count

# tile the f points so we can use a non-dense source to make it more random
f_points = f_object.points
f_points = np.tile(f_points, (angle_count, 1))

base_points = distributions.ManualBasePointDistribution(3, points=f_points)
base_points.ranks = f_points
angles = distributions.RandomUniformSphere(PI/12, source_ray_count)
source = sources.AngularSource(
    3,
    (-source_distance, 0, 0),
    (1, 0, 0),
    angles,
    base_points,
    [drawing.YELLOW] * source_ray_count,
    rank_type="base_point",
    dense=False
)
system.sources = [source]

# display the optimization goal
system.update()
trace_engine.ray_trace(3)

goal = trace_engine.finished_rays["rank"]
goal *= - 2
goal = goal.numpy()
goal[:,0] = target_distance*np.ones_like(goal[:,0])
plot.add_mesh(
    goal,
    color="red",
    point_size=10,
    render_points_as_spheres=True
)

# display the rays
ray_drawer = drawing.RayDrawer3D(plot)
def draw_rays():
    system.update()
    trace_engine.clear_ray_history()
    trace_engine.ray_trace(3)
    ray_drawer.rays = trace_engine.all_rays
    ray_drawer.draw()
    
def clear_rays():
    ray_drawer.rays = None
    ray_drawer.draw()

# generate the image samples
def get_samples(msg_counter=None):
    if msg_counter:
        print(f"sampling iteration {msg_counter}")
    system.update()
    trace_engine.clear_ray_history()
    trace_engine.ray_trace(3)
    output = tf.stack(
        [trace_engine.finished_rays["y_end"], trace_engine.finished_rays["z_end"]],
        axis=1
    )
    return output.numpy()
image_samples = [get_samples(i) for i in range(10)]
image_samples = np.reshape(image_samples, (-1, 2))

# display a the sampled image as a histogram
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
plt.hist2d(image_samples[:,0], image_samples[:,1], bins=50, range=((-.6, .6), (-.6, .6)))
plt.show()

plot.add_key_event("d", draw_rays)
plot.add_key_event("c", clear_rays)
plot.show()








