import pyvista as pv
import numpy as np
import tensorflow as tf

import tfrt.drawing as drawing
import tfrt.boundaries as boundaries
import tfrt.graph as graph

# generate the accumulator
zero_points = pv.read("./stl/processed_disk.stl")
#zero_points = pv.read("./stl/short_pyramid.stl")

smoother = graph.mesh_smoothing_tool(
    zero_points,
    [100, 50, 20, 10]
)
if False:
    print("smoother")
    print(smoother)
    print("------------")
row_sums = [np.sum(row) for row in smoother]
sum_to_1 = np.equal(row_sums, 1)
print(f"check rows sum to 1: {np.all(sum_to_1)}")

print(f"shape: {smoother.shape}")

# build the boundary
vg = boundaries.FromVectorVG((0, 0, 1))
point_count = zero_points.points.shape[0]
lens = boundaries.ParametricTriangleBoundary(
    zero_points,
    vg,
    auto_update_mesh=True,
    initial_parameters=tf.random.normal((point_count,), stddev=.1, dtype=tf.float64)
)

# build the plotter
plot = pv.Plotter()
plot.add_axes()
drawer = drawing.TriangleDrawer(plot, color="cyan", show_edges=True)
drawer.surface = lens
drawer.draw()

def smooth(lens, smoother):
    params = tf.reshape(lens.parameters, (-1, 1))
    params = tf.matmul(smoother, params)
    params = tf.reshape(params, (-1,))
    lens.parameters.assign(params)
    lens.update()

plot.add_key_event("s", smooth)

plot.show()

