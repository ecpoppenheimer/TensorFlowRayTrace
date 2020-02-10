import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

import tfrt.sources as sources
import tfrt.drawing as drawing
import tfrt.materials as materials
import tfrt.engine as engine
import tfrt.distributions as distributions

PI = sources.PI


def on_key(
    event,
    drawer,
    optical_system,
    source,
    angular_distribution,
    base_point_distribution
):
    # Message loop called whenever a key is pressed on the figure

    # Extract the center and central_angle
    center = source.center
    central_angle = source.central_angle

    if event.key == "w":
        # move the center up
        center[1] = center[1] + 0.1
    elif event.key == "s":
        # move the center down
        center[1] = center[1] - 0.1
    elif event.key == "a":
        # move the center left
        center[0] = center[0] - 0.1
    elif event.key == "d":
        # move the center right
        center[0] = center[0] + 0.1
    elif event.key == "q":
        # rotate source ccw
        central_angle += 0.1
    elif event.key == "e":
        # rotate source cw
        central_angle -= 0.1
    elif event.key == "r":
        # reduce angle sample count
        if angular_distribution.sample_count > 1:
            angular_distribution.sample_count -= 1
    elif event.key == "f":
        # increase angle sample count
        angular_distribution.sample_count += 1
    elif event.key == "t":
        # reduce base point sample count
        if base_point_distribution.sample_count > 1:
            base_point_distribution.sample_count -= 1
    elif event.key == "g":
        # increase base point sample count
        base_point_distribution.sample_count += 1
        
    source.center = center
    source.central_angle = central_angle
    optical_system.update()
    
    drawer.rays = optical_system.sources
    drawer.draw()
    drawing.redraw_current_figure()


if __name__ == "__main__":
    drawing.disable_figure_key_commands()
    # set up the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    # configure axes
    ax.set_aspect("equal")
    ax.set_xbound(-2, 2)
    ax.set_ybound(-2, 2)

    # build the source rays
    angular_distribution = distributions.StaticUniformAngularDistribution(
        -PI / 4.0, PI / 4.0, 6, name="StaticUniformAngularDistribution"
    )
    base_point_distribution = distributions.StaticUniformBeam(
        -1.0, 1.0, 6
    )
    movable_source = sources.AngularSource(
        [0.0, 0.0],
        0.0,
        angular_distribution,
        base_point_distribution,
        drawing.RAINBOW_6,
        name="AngularSource",
        dense=True,
        start_on_base=True,
        ray_length=1.0,
    )
    
    angles_2 = distributions.RandomUniformAngularDistribution(
        -PI / 4.0, PI / 4.0, 6, name="StaticUniformAngularDistribution"
    )
    fixed_source = sources.PointSource(
        [-1.0, 0.0],
        PI,
        angles_2,
        drawing.RAINBOW_6,
        dense=False
    )
    
    optical_system = engine.OpticalSystem2D()
    optical_system.sources = [movable_source, fixed_source]
    optical_system.update()

    """# Make an optical surface, so we can check that the rays are oriented
    # correctly
    segment_boundary = np.array([[1, -2, 1, 2, 1, 0]], dtype=np.float64)
    material_list = [materials.vacuum, materials.acrylic]

    # Make a ray tracer.  Concat all the ray types into traced_rays, for ease of
    # use
    reactedRays, activeRays, finishedRays, deadRays, counter = ray_trace.rayTrace(
        source_rays.rays, segment_boundary, None, None, None, material_list, 1
    )
    traced_rays = tf.concat(
        [reactedRays, activeRays, finishedRays, deadRays], axis=0
    )

    # initialize global variables
    session.run(tf.global_variables_initializer())"""

    # set up drawer
    drawer = drawing.RayDrawer2D(ax)
    drawer.rays = optical_system.sources
    drawer.draw()

    """segment_drawer = drawing.SegmentDrawer(
        ax, segments=segment_boundary, color=(0, 1, 1)
    )
    segment_drawer.draw()"""

    # hand over to user
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_key(
            event,
            drawer,
            optical_system,
            movable_source,
            angular_distribution,
            base_point_distribution
        ),
    )
    plt.show()
