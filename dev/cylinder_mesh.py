import pyvista as pv

import tfrt.mesh_tools as mt
import tfrt.boundaries as boundaries
import tfrt.drawing as drawing

plot = pv.Plotter()
plot.add_axes()
mesh = mt.cylindrical_mesh(
    (0, 0, 0),
     (1, 0, 0), 
     radius=.2, 
     theta_res=20, 
     z_res=10, 
     use_twist=False,
     start_cap=True,
     end_cap=True
)
boundary = boundaries.ManualTriangleBoundary(mesh=mesh)
#drawer = drawing.TriangleDrawer(plot, color="cyan", show_edges=True)
drawer = drawing.TriangleDrawer(
    plot, color="cyan", show_edges=True, draw_norm_arrows=True, norm_arrow_visibility=True
)
drawer.surface = boundary
drawer.draw()
plot.show()
