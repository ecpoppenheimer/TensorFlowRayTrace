from math import pi

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np

import tfrt.distributions as dist
import tfrt.drawing as drawing
import tfrt.sources as sources
import tfrt.boundaries as boundaries
import tfrt.engine as engine

PI = tf.constant(pi, dtype=tf.float64)

# Set up the source
angular_size = PI/8
print(f"Angular Size: {np.degrees(angular_size)}")
#sphere = dist.StaticUniformSphere(angular_size, 100000)
#sphere = dist.RandomUniformSphere(angular_size, 100000)
#sphere = dist.StaticLambertianSphere(angular_size, 100000)
sphere = dist.RandomLambertianSphere(angular_size, 100000)
source = sources.PointSource(3, (0, 0, 0), (1, 0, 0), sphere, [drawing.YELLOW])
source.frozen = True
print(f"Minimum phi generated: {np.degrees(tf.reduce_min(sphere.ranks[:,0]))}")
print(f"Maximum phi generated: {np.degrees(tf.reduce_max(sphere.ranks[:,0]))}")

# Set up the target
angular_step_size = 5
plane_mesh = pv.Plane((1, 0, 0), (1, 0, 0), i_size=.1, j_size=.1).triangulate()
target = boundaries.ManualTriangleBoundary(mesh=plane_mesh)
    
# set up the system
system = engine.OpticalSystem3D()
system.sources = [source]
system.targets = [target]

eng = engine.OpticalEngine(3, [], optical_system=system)

def rotate_target():
    target.mesh.rotate_y(angular_step_size)

if True:
    # plot the histogram
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # scroll through the angles
    ray_counts = []
    angles = range(0, 95, angular_step_size)
    for angle in angles:
        system.update()
        eng.ray_trace(2)
        finished_count = tf.shape(eng.finished_rays["x_start"])[0].numpy()
        print(f"angle {angle}: {finished_count} rays")
        ray_counts.append(finished_count)
        rotate_target()
        
    plt.plot(angles, ray_counts)
    max_count = ray_counts[1]
    plt.plot(angles, max_count * np.cos(np.radians(angles)))
    
    

    """# Couldn't figure out a better way to measure the angular density, so I am
    # taking a small constant width strip through the center if the distribution.
    y_points = sphere.points[:,1]
    choose_strip = tf.less(tf.abs(y_points), .025)
    phi = tf.boolean_mask(sphere.ranks[:,0], choose_strip)
    n, b, _ = plt.hist(phi, bins=25)
    x = b[:-1]
    y = tf.cos(x) * tf.reduce_max(n)
    plt.plot(x, y)"""

    plt.show()

if False:
    # plot the points
    pv_plot = pv.Plotter()
    pv_plot.add_axes()
    
    target_actor = pv_plot.add_mesh(target.mesh)
    ray_drawer = drawing.RayDrawer3D(pv_plot)
    
    def test_angle():
        rotate_target()
        system.update()
        eng.ray_trace(2)
        ray_drawer.rays = eng.finished_rays
        ray_drawer.draw()
        
    def whole_distribution():
        system.update()
        ray_drawer.rays = system.sources
        ray_drawer.draw()
    whole_distribution()
    
    points_only_mesh = pv.PolyData(np.zeros_like(sphere.points, dtype=np.float64))
    pv_plot.add_mesh(points_only_mesh)
    def points_only():
        ray_drawer.rays = None
        ray_drawer.draw()
        pv_plot.remove_actor(target_actor)
        
        sphere.update()
        points_only_mesh.points = sphere.points.numpy()
    
    pv_plot.add_key_event("r", test_angle)
    pv_plot.add_key_event("w", whole_distribution)
    pv_plot.add_key_event("p", points_only)
    pv_plot.show()
    
