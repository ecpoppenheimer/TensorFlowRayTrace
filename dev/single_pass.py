from math import pi as PI

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

import tfrt.distributions as distributions
import tfrt.boundaries as boundaries
import tfrt.drawing as drawing
import tfrt.materials as materials
import tfrt.TFRayTrace as ray_trace
import tfrt.engine as eng
import tfrt.sources as sources
import tfrt.operation as op


def on_key(event, system):
    # Message loop called whenever a key is pressed on the figure      
    system.update()
    
    drawer.segments = system.optical_segments
    drawer.draw()
    ray_drawer.rays = system.sources
    ray_drawer.draw()
    drawing.redraw_current_figure()


if __name__ == "__main__":
    drawing.disable_figure_key_commands()
    # set up the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    
    # build the arc boundary
    arc_boundary = boundaries.ManualArcBoundary()
    arc_boundary["x_center"] = np.array([5], dtype=np.float64)
    arc_boundary["y_center"] = np.array([0], dtype=np.float64)
    arc_boundary["angle_start"] = np.array([3*PI/4], dtype=np.float64)
    arc_boundary["angle_end"] = np.array([5*PI/4], dtype=np.float64)
    arc_boundary["radius"] = np.array([5], dtype=np.float64)
    eng.annotation_helper(arc_boundary, "mat_in", 1, "x_center", dtype=tf.int64)
    eng.annotation_helper(arc_boundary, "mat_out", 0, "x_center", dtype=tf.int64)
    
    # build the source rays
    beam_points = distributions.StaticUniformBeam(-1.5, 1.5, 10)
    angles = distributions.StaticUniformAngularDistribution(0, 0, 1)
    source = sources.AngularSource(
        (-1.0, 0.0), 0.0, angles, beam_points, drawing.RAINBOW_6
    )
    
    # build the system
    system = eng.OpticalSystem2D()
    system.optical_arcs = [arc_boundary]
    system.sources = [source]
    system.materials = [
        {"n": materials.vacuum},
        {"n": materials.acrylic}
    ]
    
    trace_engine = eng.OpticalEngine(
        2,
        [op.StandardReaction()],
        compile_dead_rays=True,
        dead_ray_length=10,
        simple_ray_inheritance={"angle_ranks", "wavelength"}
    )
    trace_engine.optical_system = system
    system.update()
    trace_engine.validate_system()
    trace_engine.validate_output()
    
    # test stuff
    new_rays = trace_engine.single_pass(system._amalgamated_sources.copy())
    print(f"projected result printout")
    eng.recursive_dict_key_print(trace_engine.last_projection_result)
    print("------------")
    print(f"norm: {trace_engine.last_projection_result['optical']['norm']}")
    print(f"new rays printout")
    eng.recursive_dict_key_print(new_rays)
    print("------------")
    trace_engine.validate_output()

    # set up drawers
    segment_drawer = drawing.SegmentDrawer(ax, color="cyan", draw_norm_arrows=True)
    segment_drawer.segments = system.optical_segments
    segment_drawer.draw()
    
    ray_drawer = drawing.RayDrawer(ax)
    ray_drawer.rays = eng.amalgamate([trace_engine.all_rays, new_rays])
    ray_drawer.draw()

    arc_drawer = drawing.ArcDrawer(ax, color="cyan", draw_norm_arrows=True)
    arc_drawer.arcs = system.optical_arcs
    arc_drawer.draw()
    
    # configure axes
    ax.set_aspect("equal")
    ax.set_xbound(-2, 12)
    ax.set_ybound(-7, 7)

    # hand over to user
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_key(event, system),
    )
    plt.show()
