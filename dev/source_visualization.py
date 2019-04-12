import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

import tfrt.sources as sources
import tfrt.drawing as drawing
import tfrt.materials as materials
import tfrt.TFRayTrace as ray_trace

PI = sources.PI


def on_key(event, session, drawer, source_rays, traced_rays):
    # Message loop called whenever a key is pressed on the figure

    # Extract the center and central_angle
    center = session.run(source_rays.center)
    central_angle = session.run(source_rays.central_angle)

    if event.key == "w":
        # move the center up
        center[1] = center[1] + 0.1
        session.run(
            source_rays.assign_center,
            feed_dict={source_rays.center_placeholder: center},
        )
    elif event.key == "s":
        # move the center down
        center[1] = center[1] - 0.1
        session.run(
            source_rays.assign_center,
            feed_dict={source_rays.center_placeholder: center},
        )
    elif event.key == "a":
        # move the center left
        center[0] = center[0] - 0.1
        session.run(
            source_rays.assign_center,
            feed_dict={source_rays.center_placeholder: center},
        )
    elif event.key == "d":
        # move the center right
        center[0] = center[0] + 0.1
        session.run(
            source_rays.assign_center,
            feed_dict={source_rays.center_placeholder: center},
        )
    elif event.key == "q":
        # rotate source ccw
        central_angle += 0.1
        session.run(
            source_rays.assign_central_angle,
            feed_dict={source_rays.central_angle_placeholder: central_angle},
        )
    elif event.key == "e":
        # rotate source cw
        central_angle -= 0.1
        session.run(
            source_rays.assign_central_angle,
            feed_dict={source_rays.central_angle_placeholder: central_angle},
        )

    if event.key == "t":
        drawer.rays = session.run(traced_rays)
    else:
        drawer.rays = session.run(source_rays._rays)
    drawer.draw()
    drawing.redraw_current_figure()


if __name__ == "__main__":
    drawing.disable_figure_key_commands()
    with tf.Session() as session:
        # set up the figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

        # configure axes
        ax.set_aspect("equal")
        ax.set_xbound(-2, 2)
        ax.set_ybound(-2, 2)

        # Build the variables that allow the source to be parametric
        center = tf.get_variable("source_center", initializer=(0.0, 0.0))
        central_angle = tf.get_variable("source_central_angle", initializer=0.0)

        # build the source rays
        angular_distribution = sources.StaticUniformAngularDistribution(
            -PI / 4.0, PI / 4.0, 10, name="StaticUniformAngularDistribution"
        )
        base_point_distribution = sources.StaticUniformBeam(
            0.1, -0.1, 5, name="StaticUniformBeam"
        )
        source_rays = sources.AngularSource(
            center,
            central_angle,
            angular_distribution,
            base_point_distribution,
            drawing.RAINBOW_6,
            name="AngularSource",
            dense=True,
            start_on_center=True,
            ray_length=1.0,
        )

        # Store center setting machinery inside source_rays
        source_rays.center = center
        source_rays.center_placeholder = tf.placeholder(
            tf.float32, shape=(2,), name="center_placeholder"
        )
        source_rays.assign_center = tf.assign(center, source_rays.center_placeholder)

        source_rays.central_angle = central_angle
        source_rays.central_angle_placeholder = tf.placeholder(
            tf.float32, shape=[], name="central_angle_placeholder"
        )
        source_rays.assign_central_angle = tf.assign(
            central_angle, source_rays.central_angle_placeholder
        )

        # Make an optical surface, so we can check that the rays are oriented
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
        session.run(tf.global_variables_initializer())

        # set up drawer
        drawer = drawing.RayDrawer(ax, rays=session.run(source_rays._rays))
        drawer.draw()

        segment_drawer = drawing.SegmentDrawer(
            ax, segments=segment_boundary, color=(0, 1, 1)
        )
        segment_drawer.draw()

        # Print the the public helper attributes
        print("angles")
        print(np.degrees(session.run(source_rays.angles)))
        print("angle ranks")
        print(session.run(source_rays.angle_ranks))
        print("base points")
        try:
            print(session.run(source_rays.base_points))
        except:
            print("None")
        print("base point ranks")
        try:
            print(session.run(source_rays.base_point_ranks))
        except:
            print("None")

        # Set up the summary writer, so we can visualize the graph with tensorboard
        # start it with the terminal command (from the /dev folder)
        # tensorboard --logdir=./summary
        if tf.gfile.Exists("./summary"):
            tf.gfile.DeleteRecursively("./summary")
        with tf.summary.FileWriter("./summary") as summaryWriter:
            summaryWriter.add_graph(session.graph)

        # hand over to user
        fig.canvas.mpl_connect(
            "key_press_event",
            lambda event: on_key(event, session, drawer, source_rays, traced_rays),
        )
        plt.show()
