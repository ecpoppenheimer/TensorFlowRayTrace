"""
Test whether source rotation induces unwanted roll.

Try adjusting source_y_offset with source.angle_type 'vector' (the default) and you will
notice an unwanted roll appearing in the source.
"""
import math

import pyvista as pv
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation.quaternion as quaternion

import tfrt.sources as sources
import tfrt.distributions as distributions
import tfrt.drawing as drawing

source_size = 1.0
target_z_distance = 20.0
target_x_size = 40.0
target_y_size = 40.0

# offset between the center of the source and the center of the target in the x-y plane.
source_x_offset = 10.0
source_y_offset = 20.0
source_angular_cutoff = 15.0

ray_sample_count = 35*35
wavelengths = [drawing.YELLOW]

# =============================================================================
# Re-interpret some of the above parameters

# convert to radians
PI = tf.constant(math.pi, dtype=tf.float64)
source_angular_cutoff *= PI/180.0

# reduce ray sample count since it needs to be a square number
sqrt_ray_sample_count = math.floor(math.sqrt(ray_sample_count))
ray_sample_count = sqrt_ray_sample_count*sqrt_ray_sample_count

# actually want the center-to-edge distance, i.e. the 'square radius'.
source_size /= 2

# =============================================================================
# Make the source.
angle_type = "quaternion"
if angle_type == "vector":
    source_angle = np.array(
        (
            0.0,
            -source_y_offset,
            target_z_distance
        ),
        dtype=np.float64
    )
else:
    rot1 = quaternion.from_euler((0.0, -PI/2, 0.0))
    rot2 = quaternion.from_euler(tf.cast(
        (
            math.atan2(source_y_offset, target_z_distance),
            0.0,
            0.0
        ),
        dtype=tf.float64
    ))
    source_angle = quaternion.multiply(rot2, rot1)
    
source_center = np.array(
    (
        source_x_offset,
        source_y_offset,
        -target_z_distance
    ),
    dtype=np.float64
)
angular_distribution = distributions.SquareRankLambertianSphere(
    ray_sample_count,
    source_angular_cutoff
)
base_point_distribution = distributions.RandomUniformSquare(
    source_size,
    sqrt_ray_sample_count
)
source = sources.AngularSource(
    3,
    source_center,
    source_angle,
    angular_distribution,
    base_point_distribution,
    wavelengths,
    dense=False,
    ray_length=100,
    angle_type=angle_type
)

# =============================================================================
# Set up the plot

plot = pv.Plotter()
plot.add_axes()
ray_drawer = drawing.RayDrawer3D(plot)

# draw a visual target in the x-y plane, for reference
visual_target = pv.Plane(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    i_size = target_x_size,
    j_size = target_y_size,
    i_resolution = 1,
    j_resolution = 1
)
plot.add_mesh(visual_target, color="green")

# draw a visual backing that the source will slide along, at the proper
# rotation and offset
visual_backing = pv.Plane(
    center=(0.0, .99*source_y_offset, -.99*target_z_distance),
    direction=(0.0, -source_y_offset, target_z_distance),
    i_size = target_y_size,
    j_size = 5*source_size,
    i_resolution = 1,
    j_resolution = 1
)
plot.add_mesh(visual_backing, color="green")

# =============================================================================
# Define the plot interface

def redraw():
    source.update()
    ray_drawer.rays = source
    ray_drawer.draw()
redraw()

plot.show()



