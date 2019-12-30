import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import math

import tfrt.distributions as distributions
import tfrt.sources as sources
import tfrt.drawing as drawing
import tfrt.TFRayTrace as ray_trace
import tfrt.engine as eng
import tfrt.operation as op

PI = math.pi

def on_key(event, drawer, source, system):
    # Message loop called whenever a key is pressed on the figure
    central_angle = source.central_angle
    if event.key == 'a':
        central_angle -= .1
    elif event.key == 'd':
        central_angle += .1
    
    source.central_angle = central_angle 
    system.update()
    
    drawer.rays = source
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

    angles = distributions.StaticUniformAngularDistribution(-PI/4.0, PI/4.0, 5)
    source = sources.PointSource((0.0, 0.0), 0.0, angles, [drawing.YELLOW])
    
    engine = eng.OpticalEngine(2, [op.OldestAncestor(), op.StandardReaction()])
    system = eng.OpticalSystem2D()
    system.sources = [source]
    system.update()

    # set up drawer
    drawer = drawing.RayDrawer(ax)
    drawer.rays = source
    drawer.draw()

    # hand over to user
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_key(event, drawer, source, system),
    )
    plt.show()
