import pyvista as pv
import numpy as np
import tensorflow as tf

import tfrt.drawing as drawing
import tfrt.boundaries as boundaries
import tfrt.graph as graph

# generate the accumulator
#zero_points = pv.read("./stl/processed_disk.stl")
zero_points = pv.read("./stl/short_pyramid.stl")

top_parent = graph.get_closest_point(zero_points, (0, 0, 0))
vertex_update_map, accumulator = graph.mesh_parametrization_tools(zero_points, top_parent)
print("accumulator: ")
print(accumulator)

# build the boundary
vg = boundaries.FromVectorVG((0, 0, 1))
lens = boundaries.ParametricTriangleBoundary(
    zero_points,
    vg,
    auto_update_mesh=True,
    vertex_update_map=vertex_update_map
)

# build the plotter
plot = pv.BackgroundPlotter()
plot.add_axes()
drawer = drawing.TriangleDrawer(plot, color="cyan", show_edges=True)
drawer.surface = lens
drawer.draw()
plot.show()

# input loop
point_count = accumulator.shape[0]
grad_step = .1
print(f"type a number between 0 and {point_count - 1} to adjust surface.")
print(f"type anything else to quit.")

while True:
    try:
        point = int(input())
    except:
        print("quitting.")
        break
        
    if point >= point_count:
        print("Too big!")
    else:
        grad = np.where(np.equal(range(point_count), point), grad_step, 0)
        grad = tf.matmul(accumulator, tf.reshape(grad, (-1, 1)))
        grad = tf.reshape(grad, (-1,))
        lens.parameters.assign_sub(grad)
        lens.update()
