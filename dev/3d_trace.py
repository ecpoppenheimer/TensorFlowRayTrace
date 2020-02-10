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

# build the source rays
#base_points = distributions.StaticUniformSquare(1, 5)
angles = distributions.StaticUniformSphere(PI/8.0, 25)
source = sources.PointSource(
    3, (-1, 0, 0), (1, 0, 0), angles, drawing.RAINBOW_6, dense=True
)
source.frozen=True

# build the boundary
surface1 = boundaries.ManualTriangleBoundary(
    file_name="./stl/short_pyramid.stl",
    material_dict={"mat_in": 1, "mat_out": 0}
)
surface2 = boundaries.ManualTriangleBoundary(
    mesh=pv.Sphere(radius=.5, center=(1, 0, 0)),
    material_dict={"mat_in": 1, "mat_out": 0}
)
surface3 = boundaries.ManualTriangleBoundary(mesh=pv.Plane(
    center=(3, 0, 0),
    direction=(1, 0, 0),
    i_size = 7,
    j_size = 7,
    i_resolution = 1,
    j_resolution = 1
).triangulate())

# build the optical system
system = engine.OpticalSystem3D()
system.optical = [surface1, surface2]
system.targets = [surface3]
system.sources = [source]
system.materials = [{"n": materials.vacuum}, {"n": materials.acrylic}]
system.update()

# draw the boundary
plot = pv.Plotter()
surface1.mesh.rotate_y(-90)
surface1.update_from_mesh()
system.update()

first_drawer = drawing.TriangleDrawer(plot, color="cyan")
second_drawer = drawing.TriangleDrawer(plot, color="green")
third_drawer = drawing.TriangleDrawer(plot, color="brown")

first_drawer.surface = surface1
first_drawer.draw()
second_drawer.surface = surface2
second_drawer.draw()
third_drawer.surface = surface3
third_drawer.draw()

# test the trace
trace_engine = engine.OpticalEngine(
    3,
    [operation.StandardReaction()],
    compile_dead_rays=True,
    dead_ray_length=10
)
trace_engine.optical_system = system
trace_engine.validate_system()
trace_engine.ray_trace(6)

# draw the rays
drawer = drawing.RayDrawer3D(plot)

def source_rays():
    drawer.rays = system.sources
    drawer.draw()

def traced_rays():
    drawer.rays = trace_engine.all_rays
    drawer.draw()
    
source_rays()

plot.add_key_event("u", source_rays)
plot.add_key_event("i", traced_rays)

plot.show()
