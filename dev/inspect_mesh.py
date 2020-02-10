import tensorflow as tf
import numpy as np
import pyvista as pv

import tfrt.boundaries as boundaries

plot = pv.Plotter()

b = pv.read("./stl/processed_disk_large.stl")
plot.add_mesh(
    b, show_edges=True, color="cyan"
)
#plot.add_mesh(
#    pv.PolyData(b.points), color='red', point_size=5, render_points_as_spheres=True
#)

print(b)

plot.show()
