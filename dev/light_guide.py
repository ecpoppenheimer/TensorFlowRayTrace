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

drawing.disable_figure_key_commands()
# set up the figure and axes
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
ax.set_xbound(-2, 12)
ax.set_ybound(-7, 7)

# build the segment boundary
boundary = boundaries.ManualSegmentBoundary()
"""boundary.feed_segments([
    (-.1, -4, -.08, 4),
    (-.08, 4, .08, 4),
    (.08, 4, .1, -4),
    (.1, -4, -.1, -4)
])"""
boundary.feed_segments([
    (-.1, -4, 0, 4),
    (0, 4, .1, -4),
    (.1, -4, -.1, -4)
])
boundary["mat_in"] = np.array((1, 1, 1, 1), dtype=np.int64)
boundary["mat_out"] = np.array((0, 0, 0, 0), dtype=np.int64)

# build the source rays
"""angles = distributions.StaticUniformAngularDistribution(0, 0, 1)
source = sources.PointSource(
    2, (0, -4.02), .7, angles, [drawing.YELLOW], rank_type=None
)"""
sample_count = 100
angles = distributions.RandomLambertianAngularDistribution(-.4*PI, .4*PI, sample_count)
beam_points = distributions.RandomUniformBeam(-.09, .09, sample_count)
source = sources.AngularSource(
    2, (0, -4.001), PI/2, angles, beam_points, [drawing.YELLOW] * sample_count, rank_type=None, dense=False
)

# build the system
system = eng.OpticalSystem2D()
system.optical_segments = [boundary]
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
    simple_ray_inheritance={"wavelength"}
)
trace_engine.optical_system = system
system.update()
trace_engine.validate_system()
trace_engine.ray_trace(max_iterations=50)

# set up drawers
segment_drawer = drawing.SegmentDrawer(
    ax, color="cyan", draw_norm_arrows=True, norm_arrow_visibility=True
)
segment_drawer.segments = system.optical_segments
segment_drawer.draw()

ray_drawer = drawing.RayDrawer2D(ax)
ray_drawer.rays = trace_engine.all_rays
ray_drawer.draw()

plt.show()
