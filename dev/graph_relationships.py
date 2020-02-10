import numpy as np
import pyvista as pv

import tfrt.mesh_tools as graph

plot = pv.Plotter()

mesh = pv.read("./stl/processed_disk_large.stl")
#mesh = pv.read("./stl/short_pyramid.stl")
print(f"raw mesh stats: {mesh}")
plot.add_mesh(mesh, show_edges=True, color="cyan")

target_center_point = np.array((0, 0, 0), dtype=np.float64)
top_parent = graph.get_closest_point(mesh, target_center_point)
print(f"chosen center point: {top_parent}")

face_updates, vertex_ancestors, vertex_parents, missed_vertices = graph.raw_mesh_parametrization_tools(mesh, top_parent)
print(f"face_updates: {face_updates}")
graph.visualize_face_updates(plot, mesh, graph.movable_to_updatable(mesh, face_updates))
graph.visualize_connections(plot, mesh, vertex_parents)
try:
    graph.visualize_generations(plot, mesh, [missed_vertices], colors=["blue"])
except(RuntimeError):
    pass

plot.show()










