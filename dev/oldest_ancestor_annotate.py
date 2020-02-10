import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import math

import tfrt.distributions as distributions
import tfrt.sources as sources
import tfrt.drawing as drawing
import tfrt.engine as eng
import tfrt.operation as op

PI = math.pi

a1 = distributions.StaticUniformAngularDistribution(-PI/4.0, PI/4.0, 5)
s1 = sources.PointSource((0.0, 0.0), 0.0, a1, [drawing.YELLOW])

a2 = distributions.StaticUniformAngularDistribution(-PI/4.0, PI/4.0, 1)
s2 = sources.PointSource((0.0, 1.0), 0.0, a2, [drawing.YELLOW])

a3 = distributions.StaticUniformAngularDistribution(-PI/4.0, PI/4.0, 7)
s3 = sources.PointSource((0.0, 2.0), 0.0, a3, [drawing.YELLOW])

engine = eng.OpticalEngine(2, [op.OldestAncestor()])
system = eng.OpticalSystem2D()
engine.optical_system = system
system.sources = [s1, s2, s3]
engine.annotate()
engine.update()

print(f"system validation:")
engine.validate_system()
print("passed")

print("source[oldest_ancestor] after annotation:")
print(f"system: {system.sources['oldest_ancestor']}")
print(f"s1: {s1['oldest_ancestor']}")
print(f"s2: {s2['oldest_ancestor']}")
print(f"s3: {s3['oldest_ancestor']}")

# drawing
drawing.disable_figure_key_commands()
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.set_aspect("equal")
ax.set_xbound(-2, 2)
ax.set_ybound(-2, 2)

drawer = drawing.RayDrawer2D(ax)
drawer.rays = system.sources
drawer.draw()

plt.show()
