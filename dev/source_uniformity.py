from math import pi as PI

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

import tfrt.distributions as distributions
import tfrt.drawing as drawing
import tfrt.sources as sources
import tfrt.engine as engine

if __name__ == "__main__":
    drawing.disable_figure_key_commands()
    # set up the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    
    # build the source rays
    beam_points = distributions.StaticUniformBeam(-.5, .5, 10)
    angles1 = distributions.StaticUniformAngularDistribution(-PI/3.0, PI/3.0, 20)
    source1 = sources.AngularSource(
        (1.0, 0.0), 0.0, angles1, beam_points, [drawing.YELLOW], ray_length=10
    )
    source1.frozen=True
    
    angles2 = distributions.StaticLambertianAngularDistribution(-PI/3.0, PI/3.0, 20)
    source2 = sources.AngularSource(
        (-1.0, 0.0), PI, angles2, beam_points, [drawing.YELLOW], ray_length=10
    )
    source2.frozen=True

    # set up drawers    
    ray_drawer = drawing.RayDrawer2D(ax)
    ray_drawer.rays = engine.amalgamate([source1, source2])
    ray_drawer.draw()
    #drawing.redraw_current_figure()
    
    # configure axes
    ax.set_aspect("equal")
    ax.set_xbound(-2, 12)
    ax.set_ybound(-7, 7)

    # display the plot
    plt.show() 
