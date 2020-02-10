import numpy as np
import pyvista as pv

import tfrt.graph as graph

plot = pv.Plotter()

mesh = pv.read("./stl/processed_disk.stl")
#mesh = pv.read("./stl/short_pyramid.stl")
print(f"raw mesh stats: {mesh}")
plot.add_mesh(mesh, show_edges=True, color="cyan")

target_center_point = np.array((0, 0, 0), dtype=np.float64)
top_parent = graph.get_closest_point(mesh, target_center_point)
print(f"chosen center point: {top_parent}")

"""
# plot the top parent as a red dot.
plot.add_mesh(
    pv.PolyData(mesh.points[top_parent]), color='red', point_size=10, render_points_as_spheres=True
)"""

unique_edges = graph.get_unique_edges(mesh)
    
"""
# plot the children of top_parent as yellow dots.
first_children = get_children(0, unique_edges)
first_children_points = np.take(mesh.points, list(first_children), axis=0)
plot.add_mesh(
    pv.PolyData(first_children_points), color='yellow', point_size=10, render_points_as_spheres=True
)"""
    
descendant_list, child_list, parent_list, ancestor_list = graph.find_all_relationships_1p(
    top_parent, mesh, unique_edges
)

gradient_accumulator = graph.connections_to_array(descendant_list)
#print(gradient_accumulator)
    
graph.visualize_connections(plot, child_list, mesh)
#visualize_connections(plot, parent_list, mesh)
        

plot.show()










