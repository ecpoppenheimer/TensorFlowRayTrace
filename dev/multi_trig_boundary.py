import tfrt.boundaries as boundaries
import tfrt.drawing as drawing
import pyvista as pv

plot = pv.Plotter()
plot.add_axes()
drawer1 = drawing.TriangleDrawer(
    plot,
    color="cyan",
    parameter_arrow_length=.2,
    draw_parameter_arrows=True,
    parameter_arrow_visibility=False,
    norm_arrow_length=.2,
    draw_norm_arrows=True,
    norm_arrow_visibility=False
)
drawer2 = drawing.TriangleDrawer(
    plot,
    color="cyan",
    parameter_arrow_length=.2,
    draw_parameter_arrows=True,
    parameter_arrow_visibility=False,
    norm_arrow_length=.2,
    draw_norm_arrows=True,
    norm_arrow_visibility=False
)

vg = boundaries.FromVectorVG((0, 0, 1))
multi_boundary = boundaries.ParametricMultiTriangleBoundary(
    "./stl/processed_disk.stl",
    vg,
    [
        boundaries.ThicknessConstraint(0.0, "min"),
        boundaries.ThicknessConstraint(0.5, "min"),
    ],
    [True, False],
    auto_update_mesh=True    
)

multi_boundary.update()

def draw():
    drawer1.surface = multi_boundary.surfaces[0]
    drawer1.draw()
    drawer2.surface = multi_boundary.surfaces[1]
    drawer2.draw()
draw()

def toggle_vectors():
    drawer1.toggle_parameter_arrow_visibility()
    drawer2.toggle_parameter_arrow_visibility()
    
def toggle_norm():
    drawer1.toggle_norm_arrow_visibility()
    drawer2.toggle_norm_arrow_visibility()

plot.add_key_event("v", toggle_vectors)
plot.add_key_event("n", toggle_norm)
plot.show()
