import pyvista as pv
import tensorflow as tf

import tfrt.mesh_tools as mt
import tfrt.boundaries as boundaries
import tfrt.drawing as drawing

plot = pv.Plotter()
plot.add_axes()
boundary = boundaries.ParametricCylindricalGuide(
    (0, 0, 0),
    (1, 0, 0), 
    .2, 
    theta_res=20, 
    z_res=10,
    start_cap=True,
    initial_taper=(.3, 0.0),
    end_cap=True,
    rotationally_symmetric=False,
    auto_update_mesh=True
)
drawer = drawing.TriangleDrawer(plot, color="cyan", show_edges=True)
#drawer = drawing.TriangleDrawer(
#    plot, color="cyan", show_edges=True, draw_norm_arrows=True, norm_arrow_visibility=True
#)

def randomize():
    boundary.parameters.assign_add(tf.abs(tf.random.normal(
        boundary.parameters.shape,
        stddev=.01,
        dtype=tf.float64
    )))
    boundary.update()
    #drawer.surface = boundary
    drawer.draw()
randomize()

drawer.surface = boundary
drawer.draw()
plot.add_key_event("r", randomize)
plot.show()
