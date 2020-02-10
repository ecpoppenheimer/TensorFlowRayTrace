import tfrt.boundaries as boundaries
import tfrt.drawing as drawing
import pyvista as pv

plot = pv.Plotter()
plot.add_axes()
drawer = drawing.TriangleDrawer(
    plot,
    color="cyan",
    parameter_arrow_length=.2,
    draw_parameter_arrows=True,
    parameter_arrow_visibility=True,
    norm_arrow_length=.2,
    draw_norm_arrows=True,
    norm_arrow_visibility=True
)

#surface = boundaries.ManualTriangleBoundary(file_name="./stl/short_pyramid.stl")
#surface = boundaries.ManualTriangleBoundary(mesh=pv.Sphere())
vg = boundaries.FromAxisVG((0, 0, 0), direction=(0, 0, -1))
surface = boundaries.ParametricTriangleBoundary("./stl/short_pyramid.stl", vg)

def draw():
    drawer.surface = surface
    drawer.draw()
draw()

#print(f"surface vectors: {surface.vectors}")
plot.show()
