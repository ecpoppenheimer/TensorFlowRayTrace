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

        # build the source rays
        start_points = sources.StaticUniformAperaturePoints(
            (-1, -0.5), (-1, 0.5), 5, name="StartPoints"
        )
        end_points = sources.ManualBasePointDistribution(
            (-0.5, -0.4, -0.3, -0.4, -0.5), (-0.4, -0.1, 0, 0.1, 0.4), name="EndPoints"
        )
        source_rays = sources.AperatureSource(
            start_points,
            end_points,
            [drawing.YELLOW] * 5,
            name="AperatureSource",
            dense=False,
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
