import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

import tfrt.distributions as distributions
import tfrt.boundaries as boundaries
import tfrt.drawing as drawing
import tfrt.materials as materials
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

    # build the segment boundary
    segment_boundary = boundaries.ManualSegmentBoundary()
    segment_boundary.feed_segments(np.array(
        [
            [.0, -.2, .0, .2],
            [.2, .3, .4, .6],
            [.6, -.6, .7, -.4]
        ],
        dtype=np.float64
    ))
    #segment_boundary["seg_foo"] = [1, 2, 3]
    #segment_boundary["both_foo"] = [10, 20, 30]
    
    # build the arc boundary
    arc_boundary = boundaries.ManualArcBoundary()
    arc_boundary["x_center"] = np.array([.3, .55], dtype=np.float64)
    arc_boundary["y_center"] = np.array([-.45, .5], dtype=np.float64)
    arc_boundary["angle_start"] = np.array([-1, -1], dtype=np.float64)
    arc_boundary["angle_end"] = np.array([2.5, 2.5], dtype=np.float64)
    arc_boundary["radius"] = np.array([.15, .15], dtype=np.float64)
    
    #arc_boundary["arc_foo"] = [1, 2]
    #arc_boundary["both_foo"] = [10, 20]
    
    # build the source rays
    beam_points = distributions.StaticUniformBeam(-.5, .5, 3)
    angles = distributions.StaticUniformAngularDistribution(-.2, .2, 7)
    source = sources.AngularSource((-1.0, 0.0), 0.0, angles, beam_points, [drawing.YELLOW])
    
    # build the system
    system = eng.OpticalSystem2D()
    system.optical_segments = [segment_boundary]
    system.optical_arcs = [arc_boundary]
    system.sources = [source]
    system.update()
    
    trace_engine = eng.OpticalEngine(
        2,
        [op.OldestAncestor()],
        compile_geometry_specific_result=True,
        compile_technical_intersections=True,
        compile_stopped_rays=True,
        compile_dead_rays=True,
        dead_ray_length=10
    )
    trace_engine.optical_system = system
    
    # test stuff
    result = trace_engine.process_projection(system._amalgamated_sources.copy())
    print("result printout")
    eng.recursive_dict_key_print(result)
    print("-------------------")

    # set up drawers
    segment_drawer = drawing.SegmentDrawer(ax, color="cyan", draw_norm_arrows=True)
    segment_drawer.segments = system.optical_segments
    segment_drawer.draw()
    
    ray_drawer = drawing.RayDrawer2D(ax)
    ray_drawer.rays = trace_engine.all_rays
    ray_drawer.draw()

    arc_drawer = drawing.ArcDrawer(ax, color="cyan", draw_norm_arrows=True)
    arc_drawer.arcs = system.optical_arcs
    arc_drawer.draw()
    
    # configure axes
    ax.set_aspect("equal")
    ax.set_xbound(-2, 2)
    ax.set_ybound(-2, 2)

    # hand over to user
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_key(event, system),
    )
    plt.show()
