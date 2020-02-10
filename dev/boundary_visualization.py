import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

import tfrt.distributions as distributions
import tfrt.boundaries as boundaries
import tfrt.drawing as drawing
import tfrt.materials as materials
import tfrt.engine as engine


def on_key(event, drawer, boundary, system):
    # Message loop called whenever a key is pressed on the figure
    p0 = boundary.surfaces[0].parameters
    p1 = boundary.surfaces[1].parameters
    p2 = boundary.surfaces[2].parameters
    dp0 = np.linspace(-0.2, 0.2, p0.shape[0], dtype=np.float64)**2
    dp1 = np.linspace(-0.2, 0.2, p1.shape[0], dtype=np.float64)**2
    dp2 = np.linspace(-0.2, 0.2, p2.shape[0], dtype=np.float64)**2
    if event.key == "n":
        drawer.toggle_norm_arrow_visibility()
    elif event.key == "q":
        p0.assign_sub(dp0)
    elif event.key == "e":
        p0.assign_add(dp0)
    elif event.key == "a":
        p1.assign_sub(dp1)
    elif event.key == "d":
        p1.assign_add(dp1)
    elif event.key == "z":
        p2.assign_sub(dp2)
    elif event.key == "c":
        p2.assign_add(dp2)
    """elif event.key == '0':
        boundary.constraints[0].target_vertex = 0
        boundary.constraints[1].target_vertex = 0
    elif event.key == '7':
        boundary.constraints[0].target_vertex = 7
        boundary.constraints[1].target_vertex = 7
    elif event.key == 'm':
        boundary.constraints[0].target_vertex = 7
        boundary.constraints[1].parent_vertex = 0"""
          
    system.update()
    drawer.segments = system.optical_segments
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

    # build the boundary
    point_count = 15
    zero_points = distributions.StaticUniformAperaturePoints(
        (0.0, -1.0), (0.0, 1.0), point_count
    )
    one_points = distributions.StaticUniformAperaturePoints(
        (1.0, -1.0), (1.0, 1.0), point_count
    )
    boundary = boundaries.ParametricMultiSegmentBoundary(
        zero_points,
        one_points,
        constraints=[
            boundaries.ThicknessConstraint(0.0, "min"),
            boundaries.ThicknessConstraint(0.5, "min"),
            boundaries.PointConstraint(0.0, 7, parent="zero")
            #boundaries.ThicknessConstraint(0.0, "min", parent="zero")
            #boundaries.PointConstraint(0.0, 0),
            #boundaries.PointConstraint(0.5, 7)
        ],
        flip_norm=[False, True, True]
    )
    system = engine.OpticalSystem2D()
    system.optical_segments = boundary.surfaces
    system.update()

    # set up drawer
    drawer = drawing.SegmentDrawer(ax, color="cyan", draw_norm_arrows=True)
    drawer.segments = system.optical_segments
    drawer.draw()

    # hand over to user
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_key(event, drawer, boundary, system),
    )
    plt.show()
