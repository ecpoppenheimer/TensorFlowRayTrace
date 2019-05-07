import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

import tfrt.drawing as drawing
import tfrt.sources as sources
import tfrt.geometry as geometry

PI = math.pi

def plot_angle_test(session, ax, x, angle):
    # generate and draw the starting rays
    ray_count = 11
    angular_distribution = sources.StaticUniformAngularDistribution(
        -0.9*PI/2.0,
        0.9*PI/2.0,
        ray_count
    )
    wavelengths = np.linspace(drawing.VISIBLE_MIN, drawing.VISIBLE_MAX, ray_count)
    
    external_rays = sources.PointSource(
        (x, 0.25),
        angle,
        angular_distribution,
        wavelengths,
        start_on_center=False,
        dense=False,
        ray_length=0.1
    )
    external_drawer = drawing.RayDrawer(ax, rays=session.run(external_rays.rays))
    external_drawer.draw()
    
    internal_rays = sources.PointSource(
        (x, -0.25),
        angle + PI,
        angular_distribution,
        wavelengths,
        start_on_center=False,
        dense=False,
        ray_length=0.1
    )
    internal_drawer = drawing.RayDrawer(ax, rays=session.run(internal_rays.rays))
    internal_drawer.draw()
    
    # generate and draw the boundary
    segment_length = 0.2
    segment_angle = angle + PI/2.0
    segments = np.array([
        [
            x + segment_length * math.cos(segment_angle),
            0.25 + segment_length * math.sin(segment_angle),
            x - segment_length * math.cos(segment_angle),
            0.25 - segment_length * math.sin(segment_angle)
        ],
        [
            x + segment_length * math.cos(segment_angle),
            -0.25 + segment_length * math.sin(segment_angle),
            x - segment_length * math.cos(segment_angle),
            -0.25 - segment_length * math.sin(segment_angle)
        ]
    ])
    segment_drawer = drawing.SegmentDrawer(
        ax,
        segments=segments,
        color=(0,1,1),
        draw_norm_arrows=True
    )
    segment_drawer.draw()
    
    # react the rays and draw the reactions
    external_reacted = geometry.ray_reaction(
        tf.cast(external_rays.rays, tf.float32),
        angle,
        1.5,
        1.0,
        new_ray_length=0.2
    )
    ex_reacted_drawer = drawing.RayDrawer(
        ax,
        rays=session.run(external_reacted),
        style="--"
    )
    ex_reacted_drawer.draw()
    
    internal_reacted = geometry.ray_reaction(
        tf.cast(internal_rays.rays, tf.float32),
        angle,
        1.5,
        1.0,
        new_ray_length=0.2
    )
    in_reacted_drawer = drawing.RayDrawer(
        ax,
        rays=session.run(internal_reacted),
        style="--"
    )
    in_reacted_drawer.draw()
    
if __name__ == "__main__":
    drawing.disable_figure_key_commands()

    # set up the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))

    # configure axes
    ax.set_aspect("equal")
    ax.set_xbound(-2, 2)
    ax.set_ybound(-.5, .5)
    
    with tf.Session() as session:
        angles = np.arange(0, 2*PI, PI/4.0, dtype=np.float32)
        xs = np.linspace(-1.75, 1.75, angles.shape[0], dtype=np.float32)
        for x, angle in zip(xs, angles):
            plot_angle_test(session, ax, x, angle)
        
    plt.show()
