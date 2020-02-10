import tensorflow as tf
import numpy as np
import pyvista as pv

import tfrt.boundaries as boundaries

b = boundaries.ParametricTriangleBoundary("./stl/short_pyramid.stl", None)
#b.mesh.plot(color="cyan")
