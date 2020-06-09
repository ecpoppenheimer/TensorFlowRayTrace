import itertools
import math

import numpy as np
import pyvista as pv

import tfrt.mesh_tools as mt
import tfrt.drawing as drawing
import tfrt.boundaries as boundaries

PI = math.pi

radius_iter = itertools.cycle([1, .7, .5, .3])
target_edge_size_iter = itertools.cycle([1, .8, .6, .4, .2, .1, .05])
starting_radius_iter = itertools.cycle([0, .05, .2, .5, .7, .9])
angle_start_iter = itertools.cycle([0, PI/12, PI/3, PI/2, .95*PI, PI, 1.5*PI, 2*PI])
angle_end_iter = itertools.cycle([2*PI, 1.9*PI, 1.5*PI, 1.1*PI, PI, .6*PI, PI/2, PI/3, 2.5*PI, 3*PI])

mesh_params = {
    "radius": next(radius_iter),
    "target_edge_size": next(target_edge_size_iter),
    "starting_radius": next(starting_radius_iter),
    "angle_start": next(angle_start_iter),
    "angle_end": next(angle_end_iter)
}

# build the plotter
plot = pv.Plotter()
plot.add_axes()
plot.camera_position = [(0, 0, 5), (0, 0, 0), (0, 1, 0)]

points, faces = mt.circular_mesh(
    mesh_params["radius"],
    mesh_params["target_edge_size"],
    starting_radius=mesh_params["starting_radius"],
    theta_start=mesh_params["angle_start"],
    theta_end=mesh_params["angle_end"]
)
mesh = pv.PolyData(points, faces)

vg = boundaries.FromVectorVG((1, 0, 0))
boundary = boundaries.ManualTriangleBoundary(mesh=mesh)

drawer = drawing.TriangleDrawer(
    plot,
    draw_norm_arrows=True,
    norm_arrow_visibility=True,
    norm_arrow_length=.2
)
drawer.surface = boundary

def update(mesh, mesh_params):
    mesh.points, mesh.faces = mt.circular_mesh(
        mesh_params["radius"],
        mesh_params["target_edge_size"],
        starting_radius=mesh_params["starting_radius"],
        theta_start=mesh_params["angle_start"],
        theta_end=mesh_params["angle_end"]
    )
    boundary.update_from_mesh()
    drawer.draw(show_edges=True, color="cyan")
    #plot.render()
update(mesh, mesh_params)

# plot a point at the origin
plot.add_mesh(
    pv.PolyData(np.zeros((1, 3))),
    color="red",
    point_size=10,
    render_points_as_spheres=True
)

def update_radius():
    mesh_params["radius"] = next(radius_iter)
    print(f"set radius to {mesh_params['radius']}")
    update(mesh, mesh_params)
    
def update_starting_radius():
    mesh_params["starting_radius"] = next(starting_radius_iter)
    print(f"set starting_radius to {mesh_params['starting_radius']}")
    update(mesh, mesh_params)
    
def update_target_edge_size():
    mesh_params["target_edge_size"] = next(target_edge_size_iter)
    print(f"set target_edge_size to {mesh_params['target_edge_size']}")
    update(mesh, mesh_params)
    
def update_angle_start():
    mesh_params["angle_start"] = next(angle_start_iter)
    print(f"set angle_start to {mesh_params['angle_start']}")
    update(mesh, mesh_params)
    
def update_angle_end():
    mesh_params["angle_end"] = next(angle_end_iter)
    print(f"set angle_end to {mesh_params['angle_end']}")
    update(mesh, mesh_params)

plot.add_key_event("u", update_radius)
plot.add_key_event("i", update_starting_radius)
plot.add_key_event("t", update_target_edge_size)
plot.add_key_event("j", update_angle_start)
plot.add_key_event("k", update_angle_end)

plot.show()
