import numpy as np
import pyvista as pv

import tfrt.mesh_tools as mt

base1 = mt.hexagonal_mesh(0.08)
inp1 = pv.read("./stl/remesh_input_1.stl")
inp1.rotate_y(-90)

out1 = mt.planar_interpolated_remesh(
    inp1,
    base1,
    flatten=False
)

base2 = mt.hexagonal_mesh(0.2)
inp2 = pv.read("./stl/remesh_input_2.stl")
inp2.rotate_y(-90)

out2 = mt.planar_interpolated_remesh(
    inp2,
    base2,
    flatten=False
)

plot = pv.Plotter()
plot.add_axes()

base1.translate((-.2, 0, 0))
plot.add_mesh(base1, color="green", show_edges=True)
plot.add_mesh(inp1, color="green", show_edges=True)
out1.translate((.2, 0, 0))
plot.add_mesh(out1, color="green", show_edges=True)

base2.translate((-.4, .4, 0))
plot.add_mesh(base2, color="cyan", show_edges=True)
inp2.translate((0, .4, 0))
plot.add_mesh(inp2, color="cyan", show_edges=True)
out2.translate((.4, .4, 0))
plot.add_mesh(out2, color="cyan", show_edges=True)
plot.show()
