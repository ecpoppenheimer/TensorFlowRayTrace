import math

import numpy as np
import pyvista as pv
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.drawing as drawing
import tfrt.distributions as distributions
import tfrt.sources as sources
import tfrt.engine as engine
import tfrt.materials as materials
import tfrt.boundaries as boundaries
import tfrt.operation as operation

PI = tf.constant(math.pi, dtype=tf.float64)
sample_count = 1000
angular_cutoff = PI/2

plot = pv.Plotter()
plot.add_axes()

# build and draw the source
visual_dist = distributions.SquareRankLambertianSphere(sample_count, angular_cutoff)
visual_source = sources.PointSource(
    3,
    (5, 0, 0),
    (0, 0, 1),
    visual_dist,
    [drawing.YELLOW]
)

test_dist = distributions.SquareRankLambertianSphere(1000*sample_count, angular_cutoff)
test_source = sources.PointSource(
    3,
    (5, 0, 0),
    (0, 0, 1),
    test_dist,
    [drawing.YELLOW]
)

ray_drawer = drawing.RayDrawer3D(plot, visual_source)
ray_drawer.draw()

# draw the ranks and circle points, with lines between them, showing the square-circle
# transformation
fake_rays = {
    "x_start": visual_dist.ranks[:,0],
    "y_start": visual_dist.ranks[:,1],
    "z_start": tf.zeros((sample_count,), dtype=tf.float64),
    "x_end": visual_dist._circle_x,
    "y_end": visual_dist._circle_y,
    "z_end": tf.ones((sample_count,), dtype=tf.float64),
    "wavelength": tf.ones((sample_count,), dtype=tf.float64) * drawing.YELLOW
}
sq_to_circ_drawer = drawing.RayDrawer3D(plot, fake_rays)
sq_to_circ_drawer.draw()

# build the trace engine to measure the actual angular dependance of the source
def get_test_square(angle):
    plane = pv.Plane(
        (5, np.sin(angle), np.cos(angle)),
        (0, np.sin(angle), np.cos(angle)),
        .1,
        .1,
        1,
        1
    ).triangulate()
    return boundaries.ManualTriangleBoundary(mesh=plane)
system = engine.OpticalSystem3D()
system.sources = [test_source]

square_drawer = drawing.TriangleDrawer(
    plot,
    color="red"
)
square_drawer.surface = get_test_square(0)
square_drawer.draw()

# build the trace engine
trace_engine = engine.OpticalEngine(
    3,
    [operation.StandardReaction()],
    simple_ray_inheritance = {"wavelength"}
)
trace_engine.optical_system = system
trace_engine.validate_system()

if True:
    # histogram the circle points to check the uniformity of the Arbitrary_Distribution
    plt.hist2d(test_dist._circle_x, test_dist._circle_y, bins=100)
    plt.show()

if True:
    # histogram phi
    plt.hist(test_dist._phi, bins=100)
    plt.show()

if True:
    # iterate through test angles to test the angular dependance
    test_angles = np.linspace(0.0, PI/2, 25)
    ray_counts = []
    for test_angle in test_angles:
        system.targets = [get_test_square(test_angle)]
        system.update()
        trace_engine.ray_trace(1)
        ray_counts.append(trace_engine.finished_rays['x_start'].shape[0])
        
    max_ray_counts = np.amax(ray_counts)
    plt.plot(test_angles, ray_counts)
    plt.plot(test_angles, max_ray_counts * np.cos(test_angles))
    plt.show()

if True:
    plot.show()
