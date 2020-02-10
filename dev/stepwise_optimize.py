from math import pi as PI

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


def redraw(engine, ray_drawer, arc_drawer):
    """ handles redrawing the display."""
    system = engine.optical_system
    system.update()
    
    arc_drawer.arcs = system.optical_arcs
    arc_drawer.draw()
    if engine.all_rays:
        ray_drawer.rays = engine.all_rays
    else:
        ray_drawer.rays = system.sources
    ray_drawer.draw()
    drawing.redraw_current_figure()
   
def process_gradient(engine, learning_rate, parameter, grad_clip):
    """
    Calculate and process the gradient.
    
    Returns the processed gradient and the error.
    """
    with tf.GradientTape() as tape:
        engine.optical_system.update()
        engine.clear_ray_history()
        engine.ray_trace(2)
        output = engine.finished_rays["y_end"]
        error = output ** 2
        grad = tape.gradient(error, parameter)
        grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        grad = tf.clip_by_value(grad, -grad_clip, grad_clip)
        grad *= learning_rate
        
        return grad, tf.reduce_sum(error)
        
class MomentumHolder():
    def __init__(self, momentum):
        self.value = momentum
        
    @property
    def get_value(self):
        return lambda: self.value
momentum = MomentumHolder(.8)

def set_momentum(val):
    momentum.value = val

def single_step(engine, learning_rate, momentum, parameter, optimizer, grad_clip=.1):
    """
    Does the mathmatical parts of optimization, but does not update the display.
    """
    #set_momentum(momentum)
    grad, error = process_gradient(engine, learning_rate, parameter, grad_clip)
    print(f"step {optimizer.iterations.numpy()} error: {error.numpy()}")
    optimizer.apply_gradients([(grad, parameter)])
    
def self_scaling_step(engine, ray_drawer, arc_drawer, parameter, optimizer):
    step_count = optimizer.iterations
    if step_count < 20:
        single_step(engine, 1.0, .8, parameter, optimizer)
        redraw(engine, ray_drawer, arc_drawer)
    else:
        single_step(engine, .1, .9, parameter, optimizer)
        redraw(engine, ray_drawer, arc_drawer)
        
def on_key(event, engine, ray_drawer, arc_drawer, parameter, optimizer):
    if event.key == "n":
        self_scaling_step(engine, ray_drawer, arc_drawer, parameter, optimizer)
    

if __name__ == "__main__":
    drawing.disable_figure_key_commands()
    # set up the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    
    # build the arc boundary
    parameter = tf.Variable((5,), dtype=tf.float64)
    arc_boundary = boundaries.ManualArcBoundary()
    arc_boundary["x_center"] = parameter
    arc_boundary["y_center"] = np.array([0], dtype=np.float64)
    arc_boundary["angle_start"] = np.array([3*PI/4], dtype=np.float64)
    arc_boundary["angle_end"] = np.array([5*PI/4], dtype=np.float64)
    arc_boundary["radius"] = parameter
    eng.annotation_helper(arc_boundary, "mat_in", 1, "x_center", dtype=tf.int64)
    eng.annotation_helper(arc_boundary, "mat_out", 0, "x_center", dtype=tf.int64)
    
    # build the target
    target = boundaries.ManualSegmentBoundary()
    target.feed_segments(np.array([[10, -5, 10, 5]], dtype=np.float64))
    target.frozen=True
    
    # build the source rays
    beam_points = distributions.StaticUniformBeam(-1.5, 1.5, 10)
    angles = distributions.StaticUniformAngularDistribution(0, 0, 1)
    source = sources.AngularSource(
        (-1.0, 0.0), 0.0, angles, beam_points, drawing.RAINBOW_6
    )
    source.frozen=True
    
    # build the system
    system = eng.OpticalSystem2D()
    system.optical_arcs = [arc_boundary]
    system.sources = [source]
    system.target_segments = [target]
    system.materials = [
        {"n": materials.vacuum},
        {"n": materials.acrylic}
    ]
    
    trace_engine = eng.OpticalEngine(
        2,
        [op.StandardReaction()],
        simple_ray_inheritance={"angle_ranks", "wavelength"}
    )
    trace_engine.optical_system = system
    system.update()
    trace_engine.validate_system()

    # set up drawers    
    target_drawer = drawing.SegmentDrawer(ax, color="black", draw_norm_arrows=False)
    target_drawer.segments = system.target_segments
    target_drawer.draw()
    
    ray_drawer = drawing.RayDrawer2D(ax)

    arc_drawer = drawing.ArcDrawer(ax, color="cyan", draw_norm_arrows=True)
    
    # configure axes
    ax.set_aspect("equal")
    ax.set_xbound(-2, 12)
    ax.set_ybound(-7, 7)

    # display the plot
    redraw(trace_engine, ray_drawer, arc_drawer)
    plt.show(block=False)
    
    # build the optimizer
    optimizer = tf.optimizers.SGD(
        learning_rate=1.0,
        momentum=momentum.get_value(),
        nesterov=True
    )
    
    # initial state draw    
    system.update()
    redraw(trace_engine, ray_drawer, arc_drawer)
    
    # hand over to user
    fig.canvas.mpl_connect(
        "key_press_event",
        lambda event: on_key(
            event, trace_engine, ray_drawer, arc_drawer, parameter, optimizer
        ),
    )
    plt.show() 
