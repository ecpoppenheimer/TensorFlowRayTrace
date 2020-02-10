"""
Utilities for drawing optical elements with matplotlib.

This module assists in visualizing the raysets and optical systems built with tfrt.
These classes act as a nice interface that connects ndarrays formatted like tfrt
optical elements to a set of matplotlib axes so that they can be displayed in a
matplotlib figure.

Please note that the optical system data objects fed to these classes do not have to 
be numpy ndarrays, but it is highly recommended that they be.  They must at least 
have a shape attribute and the proper shape requirements to represent that kind of 
object (see tfrt.raytrace for details on the required shapes).  TensorFlow tensors 
are not acceptable inputs to these classes, but the arrays returned by session.run
calls are.

This module defines some helpful constants which are the values of the wavelength of
visible light of various colors, in um.  These values give nice results when using
the default colormap to display rays, but they are not used anywhere in this module.
They exist only as convenient reference points for users of this module, and if you
need different wavelenghts for your application, you can use whatever values you want.

Most changes made to the drawing classes defined in this module will require the 
mpl canvas to be redrawn in order to see the change.  A convenience function,
redraw_current_figure() is provided to do this.  Multiple changes to the drawing
classes can be chained with only a single canvas redraw, in general you should use
as few canvas redraws as possible.

All drawing classes define a draw method, which updates the mpl artists controlled
by the drawing class.  Changes to some class attributes of the drawing classes 
require that the draw method be called again to visualize the change, others do not.
Each attribute is explicitly labeled whether it requires a redraw or not.  Calling a 
class draw method does not redraw the mpl canvas.

"""

import math
from abc import ABC, abstractmethod

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista as pv
import tensorflow as tf

from .spectrumRGB import rgb
from tfrt.boundaries import TriangleBoundaryBase

PI = math.pi

VISIBLE_MIN = 380
VISIBLE_MAX = 780

RED = 680
ORANGE = 620
YELLOW = 575
GREEN = 510
BLUE = 450
PURPLE = 400

RAINBOW_6 = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]

UNIT_TO_NUMBER = {"nm": 1, "um": 0.001}

ZEROS = np.zeros(0)

# ------------------------------------------------------------------------------------


def form_mpl_line_syntax(rays):
    return [
        [(start_x, start_y), (end_x, end_y)] \
        for start_x, start_y, end_x, end_y, *_ in \
        zip(rays["x_start"], rays["y_start"], rays["x_end"], rays["y_end"])
    ]
   

# -------------------------------------------------------------------------------------

class RayDrawer2D:
    """
    Class for drawing a rayset.

    This class makes it easy to draw a set of rays to a matplotlib.axes.Axes.  By 
    default this class will use the spectrumRGB colormap to color the rays by 
    wavelength, but a different colormap can be chosen if desired.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A handle to the mpl axis into which the rays will be drawn.
    rays : np.ndarray
        An object that provides the proper signature to store rays.  Meaning it is can
        be indexed with keys "x_start", "y_start", "x_end", "y_end", "wavelength".
    min_wavelength : float, optional
        The minimum wavelength, used only to normalize the colormap.
    max_wavelength : float, optional
        The maximum wavelength, used only to normalize the colormap.
    units : string, optional
        The units of wavelength.  Default is 'um', micrometers, but can also accept 
        'nm', nanometers.  Used to adjust the wavelength, for compatability with the 
        spectrumRGB colormap, which uses nm.  If you want to use a different colormap,
        set the units to nm which causes RayDrawer to do wavelength as is to the 
        colormap.  If um is selected, RayDrawer will multiply wavelength by 1000 to 
        convert into nm before passing wavelength to the colormap.
    style : string, optional
        The linestyle used to draw the rays.  Defaults to a solid line.  See 
        matplotlib.lines.Line2D linestyle options for a list of the valid values.
    colormap : matplotlib.colors.Colormap, optional
        The colormap to use for coloring the rays.  Defaults to the spectrumRGB map.
    
    Public attributes
    -----------------
    rays : dict
        An object that provides the proper signature to store rays.  Meaning it is can
        be indexed with keys "x_start", "y_start", "x_end", "y_end", "wavelength".
        Requires class redraw.
    ax : matplotlib.axes.Axes
        A handle to the mpl axis into which the rays will be drawn.
        Requires class redraw.
    colormap : matplotlib.colors.Colormap
        The colormap to use for coloring the rays.
        Does not require class redraw.
    style : string
        The linestyle used to draw the rays.
        Does not require class redraw.
        
    Public members
    --------------
    set_wavelength_limits(min, max)
        Change the minimum and maximum wavelengths for colormap normalization.
        Requires class redraw.
    """

    def __init__(
        self,
        ax,
        rays=None,
        min_wavelength=VISIBLE_MIN,
        max_wavelength=VISIBLE_MAX,
        units="um",
        style="-",
        colormap=mpl.colors.ListedColormap(rgb()),
    ):

        self.ax = ax
        self._ray_signature = set(["x_start", "y_start", "x_end", "y_end", "wavelength"])
        self.rays = rays
        self._style = style

        try:
            self._wavelength_unit_factor = UNIT_TO_NUMBER[units]
        except KeyError as e:
            raise ValueError(
                f"RayDrawer: Invalid units: {units}.  Allowed values are 'um', 'nm'."
            ) from e

        self._line_collection = mpl.collections.LineCollection(
            [], linestyles=self.style, cmap=colormap
        )
        self.set_wavelength_limits(min_wavelength, max_wavelength)
        self.ax.add_collection(self._line_collection)

    def draw(self):
        """Redraw the mpl artists controlled by this class."""
        if bool(self.rays):
            self._line_collection.set_segments(
                form_mpl_line_syntax(self.rays)
            )
            self._line_collection.set_array(
                self._wavelength_unit_factor * self.rays["wavelength"]
            )
            
    @property
    def rays(self):
        return self._rays

    @rays.setter
    def rays(self, rays):
        if not bool(rays):
            # if the rays are empty, give a valid empty state
            self._rays = {key: ZEROS for key in self._ray_signature}
        else:
            try:
                if self._ray_signature <= rays.keys():
                    self._rays = rays
                else:
                    raise ValueError(
                        f"RayDrawer: Rays does not have the proper"
                        " signature."
                    )
                    
            except AttributeError as e:
                raise ValueError(
                    f"RayDrawer: Rays doesn't have a signature."
            ) from e
            
    @property
    def colormap(self):
        return self._line_collection.cmap

    @colormap.setter
    def colormap(self, colormap):
        self._line_collection.set_cmap(colormap)

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style):
        self._line_collection.set_linestyle(style)

    def set_wavelength_limits(self, min, max):
        """Change the minimum and maximum wavelengths for colormap normalization."""
        self._line_collection.norm = plt.Normalize(
            self._wavelength_unit_factor * min, self._wavelength_unit_factor * max
        )
        
# -----------------------------------------------------------------------------------

class RayDrawer3D:
    """
    Class for drawing a rayset.

    This class makes it easy to draw a set of rays to a pyvista.Plotter.  By 
    default this class will use the spectrumRGB colormap to color the rays by 
    wavelength, but a different colormap can be chosen if desired.

    Parameters
    ----------
    plot : pyvista.Plotter
        A handle to the pyvista plotter into which the rays will be drawn.
    rays : np.ndarray
        An object that provides the proper signature to store rays.  Meaning it is can
        be indexed with keys "x_start", "y_start", "z_start", "x_end", "y_end", "z_end", 
        "wavelength".
    min_wavelength : float, optional
        The minimum wavelength, used only to normalize the colormap.
    max_wavelength : float, optional
        The maximum wavelength, used only to normalize the colormap.
    colormap : matplotlib.colors.Colormap, optional
        The colormap to use for coloring the rays.  Defaults to the spectrumRGB map.
    
    Public attributes
    -----------------
    rays : dict
        An object that provides the proper signature to store rays.  Meaning it is can
        be indexed with keys "x_start", "y_start", "z_start", "x_end", "y_end", "z_end", 
        "wavelength".  Requires class redraw.
    plot : matplotlib.axes.Axes
        A handle to the pyvista plotter into which the rays will be drawn.
    colormap : matplotlib.colors.Colormap
        The colormap to use for coloring the rays.
        
    Public members
    --------------
    set_wavelength_limits(min, max)
        Change the minimum and maximum wavelengths for colormap normalization.
        Requires class redraw.
    """

    def __init__(
        self,
        plot,
        rays=None,
        min_wavelength=VISIBLE_MIN,
        max_wavelength=VISIBLE_MAX,
        colormap=mpl.colors.ListedColormap(rgb()),
    ):

        self.plot = plot
        self._rays = rays
        self._min_wavelength = min_wavelength
        self._max_wavelength = max_wavelength
        self._colormap = colormap
        self._ray_signature = set([
            "x_start", "y_start", "z_start", "x_end", "y_end", "z_end", "wavelength"
        ])

        self._mesh = None
        self._actor = None
        
        
    @property
    def rays(self):
        return self._rays

    @rays.setter
    def rays(self, rays):
        if bool(rays):
            try:
                if self._ray_signature <= rays.keys():
                    self._rays = rays
                else:
                    raise ValueError(
                        f"RayDrawer: Rays does not have the proper"
                        " signature."
                    )
                    
            except AttributeError as e:
                raise ValueError(
                    f"RayDrawer: Rays doesn't have a signature."
            ) from e
        else:
            self._rays = {}

    def draw(self):
        """Redraw the pyvista actor controlled by this class."""
        if bool(self.rays) and tf.greater(tf.shape(self.rays['x_start'])[0], 0):
            start_points = tf.stack(
                [self.rays[field] for field in ("x_start", "y_start", "z_start")],
                axis=1
            )
            end_points = tf.stack(
                [self.rays[field] for field in ("x_end", "y_end", "z_end")], 
                axis=1
            )
            all_points = tf.concat([start_points, end_points], 0)
            line_count = tf.shape(self.rays["x_start"])[0]
            cell_range = tf.range(2 * line_count)
            cells = tf.stack([
                2 * tf.ones((line_count,), dtype=tf.int32),
                cell_range[:line_count],
                cell_range[line_count:]
            ], axis=1)

            if self._mesh is None:
                self._mesh = pv.PolyData()
            self._mesh.points = all_points.numpy()
            self._mesh.lines = cells.numpy()
            self._mesh["wavelength"] = self.rays["wavelength"]
            
            self._actor = self.plot.add_mesh(
                self._mesh,
                cmap=self._colormap,
                clim=(self._min_wavelength, self._max_wavelength)
            )
        else: # nothing to draw
            if self._actor is not None:
                self.plot.remove_actor(self._actor)
                self._mesh = None


# ------------------------------------------------------------------------------------


class ArcDrawer:
    """
    Class for drawing a set of optical arc surfaces.

    This class makes it easy to draw a set of arcs formatted like the optical 
    surfaces used by tfrt.raytrace to a matplotlib.axes.Axes.  One notable 
    restriction to this class is that all of the arcs drawn must have the same color 
    and style.  If you want differently styled optical surfaces in your 
    visualization, use multiple ArcDrawer instances.

    When designing an optical system with refractive surfaces, it is very important 
    to understand which direction the surface normal points, so that the ray tracer 
    can decide whether a ray interaction is an internal or external refraction.  To 
    assist with this, ArcDrawer provides support for visualizing the orientation of 
    the surface by drawing arrows along the surface that point in the direction of 
    the norm.  I find it convenient to toggle the display of the norm arrows,  
    turning them on when I want to inspect a surface to ensure that it is correctly 
    oriented, and then hiding them after ensuring the surface is correctly 
    oriented to unclutter the display.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A handle to the MPL axis into which the arcs will be drawn.
    arcs : np.ndarray
        An array that encodes information about the arcs to be drawn.  Must be rank 
        2.  The first dimension indexes arcs.  The second dimension must have length 
        >= 5, whose first five elements are [center_x, center_y, angle_start, 
        angle_end, radius].
    color : color_like, optional
        The color of all arcs and norm arrows drawn.  See
        https://matplotlib.org/api/colors_api.html for acceptable color formats.
    style : string, optional
        The linestyle used to draw the arcs.  Defaults to a solid line.  See 
        matplotlib.lines.Line2D linestyle options for a list of the valid values.
    draw_norm_arrows : bool, optional
        If true, adds arrows along the arc surfaces that depict the direction of the 
        surface norm, for visualizing the surface orientation.
    norm_arrow_visibility : bool, optional
        The initial state of the norm arrow visibility.  Defaults to true, so the 
        norm arrows start visible.
    norm_arrow_count : int, optional
        How many norm arrows to draw along each surface.
    norm_arrow_length : float, optional
        The length (in ax coords) of the norm arrows.
        
    Public attributes
    -----------------
    arcs : np.ndarray
        An array that encodes information about the arcs to be drawn.
        Requires class redraw.
    ax : matplotlib.axes.Axes
        A handle to the mpl axis into which the rays will be drawn.
        Requires class redraw.
    color : color_like, optional
        The color of all arcs and norm arrows drawn.
        Does not require class redraw.
    style : string
        The linestyle used to draw the rays.
        Does not require class redraw.
    draw_norm_arrows : bool
        Whether to include norm arrows for this surface.
        Requires class redraw.
    norm_arrow_visibility : bool
        Whether norm arrows are displayed (True) or hidden (False).
        Does not require class redraw.
    norm_arrow_count : int
        How many norm arrows to draw along each surface.
        Requires class redraw.
    norm_arrow_length : float
        The length (in ax coords) of the norm arrows.
        Requires class redraw.
    
    """

    def __init__(
        self,
        ax,
        arcs=None,
        color="black",
        style="-",
        draw_norm_arrows=False,
        norm_arrow_visibility=True,
        norm_arrow_count=5,
        norm_arrow_length=0.05,
    ):
        self.ax = ax
        self._color = color
        self._style = style
        self._arcs = arcs
        self._arc_signature = set([
            "x_center",
            "y_center",
            "angle_start",
            "angle_end",
            "radius"
        ])
        self._arc_patches = []
        self._norm_arrows = []
        self.draw_norm_arrows = draw_norm_arrows
        self._norm_arrow_visibility = norm_arrow_visibility
        self.norm_arrow_count = norm_arrow_count
        self.norm_arrow_length = norm_arrow_length

    @property
    def arcs(self):
        return self._arcs

    @arcs.setter
    def arcs(self, arcs):
        if not bool(arcs):
            # if the arcs are empty, give a valid empty state
            self._arcs = {key: ZEROS for key in self._arc_signature}
        else:
            try:
                if self._arc_signature <= arcs.keys():
                    self._arcs = arcs
                else:
                    raise ValueError(
                        f"ArcDrawer: Arcs does not have the proper signature."
                    )
                    
            except AttributeError as e:
                raise ValueError(
                    f"ArcDrawer: Arcs doesn't have a signature."
            ) from e

    def draw(self):
        """Redraw the mpl artists controlled by this class."""
        if bool(self._arcs):
            for arc_patch in self._arc_patches:
                arc_patch.remove()
            self._arc_patches = []

            for norm_arrow in self._norm_arrows:
                norm_arrow.remove()
            self._norm_arrows = []

            # these stupid numpy calls seem to fix MPL trying to index tensors when it 
            # shouldn't
            x_center = self._arcs["x_center"]
            y_center = self._arcs["y_center"]
            angle_start = self._arcs["angle_start"]
            angle_end = self._arcs["angle_end"]
            radius = self._arcs["radius"]
            try:
                x_center = x_center.numpy()
                y_center = y_center.numpy()
                angle_start = angle_start.numpy()
                angle_end = angle_end.numpy()
                radius = radius.numpy()
            except(AttributeError):
                pass
            for xc, yc, ang_s, ang_e, r in zip(
                x_center, y_center, angle_start, angle_end, radius
            ):
                self._draw_arc(xc, yc, ang_s, ang_e, r)

    def _draw_arc(self, center_x, center_y, angle_start, angle_end, radius):
        self._arc_patches.append(
            self.ax.add_patch(
                mpl.patches.Arc(
                    (center_x, center_y),
                    abs(2 * radius),
                    abs(2 * radius),
                    theta1=math.degrees(angle_start),
                    theta2=math.degrees(angle_end),
                    color=self.color,
                    linestyle=self.style,
                )
            )
        )

        if self.draw_norm_arrows:
            self._draw_norm_arrows_for_arc(
                center_x, center_y, angle_start, angle_end, radius
            )

    def _draw_norm_arrows_for_arc(
        self, center_x, center_y, angle_start, angle_end, radius
    ):
        if angle_start >= angle_end:
            angle_end += 2 * PI
        angles = np.linspace(angle_start, angle_end, self.norm_arrow_count)

        for theta in angles:
            self._norm_arrows.append(
                self.ax.add_patch(
                    mpl.patches.Arrow(
                        center_x + abs(radius) * math.cos(theta),
                        center_y + abs(radius) * math.sin(theta),
                        self.norm_arrow_length * math.cos(theta) * np.sign(radius),
                        self.norm_arrow_length * math.sin(theta) * np.sign(radius),
                        width=0.4 * self.norm_arrow_length,
                        color=self.color,
                        visible=self._norm_arrow_visibility,
                    )
                )
            )

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color
        for arc in self._arc_patches:
            arc.set_color(color)
        for arrow in self._norm_arrows:
            arrow.set_color(color)

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style):
        self._style = style
        for arc in self._arc_patches:
            arc.set_linestyle(style)

    @property
    def norm_arrow_visibility(self):
        return self._norm_arrow_visibility

    @norm_arrow_visibility.setter
    def norm_arrow_visibility(self, val):
        if not isinstance(val, bool):
            raise TypeError("norm_arrow_visibility must be bool")
        self._norm_arrow_visibility = val

        for arrow in self._norm_arrows:
            arrow.set_visible(self._norm_arrow_visibility)

    def toggle_norm_arrow_visibility(self):
        """Toggle the visibility of the norm arrows."""
        self.norm_arrow_visibility = not self.norm_arrow_visibility

    @property
    def norm_arrow_count(self):
        return self._arrow_count

    @norm_arrow_count.setter
    def norm_arrow_count(self, count):
        if not isinstance(count, int):
            raise TypeError("arrow_count must be int")

        if count < 0:
            count = 0

        self._arrow_count = count

    @property
    def norm_arrow_length(self):
        return self._norm_arrow_length

    @norm_arrow_length.setter
    def norm_arrow_length(self, length):
        if length < 0:
            length = 0

        self._norm_arrow_length = length


class SegmentDrawer:
    """
    Class for drawing a set of optical line segment surfaces.

    This class makes it easy to draw a set of line segments formatted like the 
    optical surfaces used by tfrt.raytrace to a matplotlib.axes.Axes.  All of the 
    lines drawn by a SegmentDrawer instance should have the same color and style.  If 
    you want differently styled optical surfaces in your visualization, use multiple 
    SegmentDrawer instances.

    When designing an optical system with refractive surfaces, it is very important 
    to understand which direction the surface normal points, so that the ray tracer 
    can decide whether a ray interaction is an internal or external refraction.  To 
    assist with this, SegmentDrawer provides support for visualizing the orientation 
    of the surface by drawing an arrow at the midpoint of the surface that points in 
    the direction of the norm.  I find it convenient to toggle the display of the 
    norm arrows, turning them on when I want to inspect a surface to ensure that it 
    is correctly oriented, and then hiding them after ensuring the surface is 
    correctly oriented to unclutter the display.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A handle to the mpl axis into which the segments will be
        drawn.
    segments : np.ndarray
        An object that provides the proper signature to store rays.  Meaning it is can
        be indexed with keys "x_start", "y_start", "x_end", "y_end".  Requires class
        redraw.
    color : color_like, optional
        The color of all segments and norm arrows drawn.  See
        https://matplotlib.org/api/colors_api.html for acceptable color formats.
    style : string, optional
        The linestyle used to draw the segments.  Defaults to a solid line.  See 
        matplotlib.lines.Line2D linestyle options for a list of the valid values.
    draw_norm_arrows : bool, optional
        If true, add an arrow at the midpoint of the segment that depicts the 
        direction of the surface norm, for visualizing the surface orientation.
    norm_arrow_visibility : bool, optional
        The initial state of the norm arrow visibility.  Defaults to true, so the 
        norm arrows start visible.
    norm_arrow_length : float, optional
        The length (in ax coords) of the norm arrows.
    
    Public attributes
    -----------------
    segments : np.ndarray
        An array that encodes information about the segments to be drawn.
        Requires class redraw.
    ax : matplotlib.axes.Axes
        A handle to the mpl axis into which the rays will be drawn.
        Requires class redraw.
    color : color_like, optional
        The color of all arcs and norm arrows drawn.
        Does not Require class redraw.
    style : string
        The linestyle used to draw the rays.
        Does not Require class redraw.
    draw_norm_arrows : bool
        Whether to include norm arrows for this surface.
        Requires class redraw.
    norm_arrow_visibility : bool
        Whether norm arrows are displayed (True) or hidden (False).
        Does not require class redraw.
    norm_arrow_length : float
        The length (in ax coords) of the norm arrows.
        Requires class redraw.
        
    """

    def __init__(
        self,
        ax,
        segments=None,
        color="black",
        style="-",
        draw_norm_arrows=False,
        norm_arrow_visibility=True,
        norm_arrow_length=0.05,
    ):
        self.ax = ax
        self._segments = segments
        self._segment_signature = set(["x_start", "y_start", "x_end", "y_end"])
        self._color = color
        self._style = style
        self._norm_arrows = []
        self.draw_norm_arrows = draw_norm_arrows
        self.norm_arrow_visibility = norm_arrow_visibility
        self.norm_arrow_length = norm_arrow_length
        self._line_collection = mpl.collections.LineCollection(
            [], colors=self.color, linestyles=self.style
        )
        self.ax.add_collection(self._line_collection)

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, segments):
        if not bool(segments):
            # if the segments are empty, give a valid empty state
            self._segments = {key: ZEROS for key in self._segment_signature}
        else:
            try:
                if self._segment_signature <= segments.keys():
                    self._segments = segments
                else:
                    raise ValueError(
                        f"SegmentDrawer: Segments does not have the proper signature."
                    )
            except AttributeError as e:
                raise ValueError(
                    f"SegmentDrawer: Segments doesn't have a signature."
            ) from e

    def draw(self):
        """Redraw the mpl artists controlled by this class."""
        if bool(self.segments):
            for norm_arrow in self._norm_arrows:
                norm_arrow.remove()
            self._norm_arrows = []

            segments = []
            for start_x, start_y, end_x, end_y, in zip(
                self.segments["x_start"],
                self.segments["y_start"],
                self.segments["x_end"],
                self.segments["y_end"]
            ):
                theta = np.arctan2(end_y - start_y, end_x - start_x) + PI / 2.0
                segments.append([(start_x, start_y), (end_x, end_y)])

                if self.draw_norm_arrows:
                    self._norm_arrows.append(
                        self.ax.add_patch(
                            mpl.patches.Arrow(
                                (start_x + end_x) / 2.0,
                                (start_y + end_y) / 2.0,
                                self.norm_arrow_length * math.cos(theta),
                                self.norm_arrow_length * math.sin(theta),
                                width=0.4 * self._norm_arrow_length,
                                color=self._color,
                                visible=self._norm_arrow_visibility,
                            )
                        )
                    )

            self._line_collection.set_segments(segments)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color
        self._line_collection.set_color(color)
        for arrow in self._norm_arrows:
            arrow.set_color(color)

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style):
        self._style = style
        self._line_collection.set_linestyle(style)

    @property
    def norm_arrow_visibility(self):
        return self._norm_arrow_visibility

    @norm_arrow_visibility.setter
    def norm_arrow_visibility(self, val):
        if not isinstance(val, bool):
            raise TypeError("visibility must be bool")

        self._norm_arrow_visibility = val
        for arrow in self._norm_arrows:
            arrow.set_visible(self._norm_arrow_visibility)

    def toggle_norm_arrow_visibility(self):
        """Toggle the visibility of the norm arrows."""
        self.norm_arrow_visibility = not self.norm_arrow_visibility

    @property
    def norm_arrow_length(self):
        return self._norm_arrow_length

    @norm_arrow_length.setter
    def norm_arrow_length(self, length):
        if length < 0:
            length = 0

        self._norm_arrow_length = length


# ------------------------------------------------------------------------------------

class TriangleDrawer:
    """
    Class for drawing pyvista meshes to a pyvista plot.
    
    Contains utilities for drawing norm arrows, and possibly parameter vectors, if the
    boundary is parametric.
    
    Unlike the other drawers, this class will draw the pyvista.PolyData mesh held by the 
    boundary given to it - it will not interpret data from the boundary's fields.  Which 
    means you may need to call update_mesh_from_vertices() before drawing the boundary if 
    the boundary is parametric, in order to get the properly updated version of the 
    boundary.  This also means it is impossible to draw an amalgamated boundary with a
    single drawer - you will have to use multiple drawers to draw multiple boundaries, or
    multiple layers of a multi-boundary.
    
    Unlike the other drawers, you don't necessairly need to call draw() every time you need
    to update the display, since pyplot automatically redraws when one of its plotted meshes
    change.  But if you are using a parametric boundary, you will need to use the boundary's 
    update_mesh_from_vertices() every time the parameter changes, to update the mesh.
    """
    def __init__(
        self,
        plot,
        draw_norm_arrows=False, 
        norm_arrow_visibility=False,
        norm_arrow_length=0.1,
        draw_parameter_arrows=False, 
        parameter_arrow_visibility=False,
        parameter_arrow_length=0.1,
        **kwargs
    ):
        self.plot = plot
        self._surface = None
        self._norm_actor = None
        self._parameter_actor = None
        self.draw_norm_arrows=draw_norm_arrows
        self.norm_arrow_visibility=norm_arrow_visibility
        self.norm_arrow_length = norm_arrow_length
        self.draw_parameter_arrows=draw_parameter_arrows
        self.parameter_arrow_visibility=parameter_arrow_visibility
        self.parameter_arrow_length = parameter_arrow_length
        self._constructor_kwargs = kwargs
        self._actor = None
        
        
    @property
    def surface(self):
        return self._surface
        
    @surface.setter
    def surface(self, val):
        if issubclass(type(val), (type(None), TriangleBoundaryBase)):
            self._surface = val
        else:
            raise ValueError(
                "TriangleDrawer: surface must be None or a subclass of "
                "TriangleBoundaryBase."
            )
            
    @property
    def norm_arrow_visibility(self):
        return self._norm_arrow_visibility
        
    @norm_arrow_visibility.setter
    def norm_arrow_visibility(self, val):
        self._norm_arrow_visibility = val
        self._draw_norm_arrows()
                
    def _draw_norm_arrows(self):
        if self._surface is not None:
            if self.draw_norm_arrows:
                if self._norm_arrow_visibility:
                    faces = self._surface._faces
                    vertices = self._surface._vertices
                    _, first_index, pivot_index, second_index = tf.unstack(faces, axis=1)
                    pivot_points = tf.gather(vertices, pivot_index)
                    first_points = tf.gather(vertices, first_index)
                    second_points = tf.gather(vertices, second_index)
                    points = (pivot_points + first_points + second_points)/3
                    
                    vectors = self._surface["norm"]
                    self._norm_actor = self.plot.add_arrows(
                        points.numpy(),
                        vectors.numpy(),
                        mag=self.norm_arrow_length,
                        **self._constructor_kwargs
                    )
                    return
        self.plot.remove_actor(self._norm_actor)
        
    def toggle_norm_arrow_visibility(self):
        self.norm_arrow_visibility = not self.norm_arrow_visibility
        
    @property
    def parameter_arrow_visibility(self):
        return self._parameter_arrow_visibility
        
    @parameter_arrow_visibility.setter
    def parameter_arrow_visibility(self, val):
        self._parameter_arrow_visibility = val
        self._draw_parameter_arrows()
                
    def _draw_parameter_arrows(self):
        if self._surface is not None:
            if self.draw_parameter_arrows and hasattr(self._surface, "_vectors"):
                if self._parameter_arrow_visibility:
                    points = self._surface._vertices
                    vectors = self._surface._vectors
                    self._parameter_actor = self.plot.add_arrows(
                        points.numpy(),
                        vectors.numpy(),
                        mag=self.parameter_arrow_length,
                        **self._constructor_kwargs
                    )
                    return
        self.plot.remove_actor(self._parameter_actor)
        
    def toggle_parameter_arrow_visibility(self):
        self.parameter_arrow_visibility = not self.parameter_arrow_visibility
            
    def draw(self, **kwargs):
        self.plot.remove_actor(self._norm_actor)
        self.plot.remove_actor(self._parameter_actor)
        self.plot.remove_actor(self._actor)
        if self._surface is not None:
            if self._surface._mesh is not None:
                # draw the mesh itself
                plot_kwargs = dict(self._constructor_kwargs, **kwargs)
                self._actor = self.plot.add_mesh(self._surface._mesh, **plot_kwargs)
                
                # draw the norm arrows
                self._draw_norm_arrows()
                self._draw_parameter_arrows()
                
                return
        self._actor = None
            
            
# ------------------------------------------------------------------------------------


def disable_figure_key_commands():
    """Disable all keyboard shortcuts in the mpl figure."""
    for key, value in plt.rcParams.items():
        if "keymap" in key:
            plt.rcParams[key] = ""


def redraw_current_figure():
    """Redraws the mpl canvas."""
    plt.gcf().canvas.draw()
