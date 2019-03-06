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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .spectrumRGB import rgb

PI = math.pi

VISIBLE_MIN = 0.38
VISIBLE_MAX = 0.78

RED = 0.68
ORANGE = 0.62
YELLOW = 0.575
GREEN = 0.51
BLUE = 0.45
PURPLE = 0.4

RAINBOW_6 = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]

UNIT_TO_NUMBER = {"nm": 1000, "um": 1}

# ------------------------------------------------------------------------------------


class RayDrawer:
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
        An array that encodes information about the rays to be drawn, formatted like 
        the raysets used by tfrt.raytrace.  Must be rank 2.  The first dimension 
        indexes rays.  The second dimension must have length >= 5, whose first five 
        elements are [start_x, start_y, end_x, end_y, wavelength].
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
    rays : np.ndarray
        An array that encodes information about the rays to be drawn.
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

    @property
    def rays(self):
        return self._rays

    @rays.setter
    def rays(self, rays):
        if rays is None:
            self._rays = np.zeros((0, 5))
        else:
            try:
                shape = rays.shape
            except AttributeError as e:
                raise AttributeError(
                    f"{self.__class__.__name__}: Invalid rays.  Could not retrieve "
                    "rays's shape."
                ) from e

            if len(shape) != 2:
                raise ValueError(
                    f"{self.__class__.__name__}: Invalid rays.  Tensor rank must be "
                    "2, but is rank {len(shape)}."
                )
            if shape[1] < 5:
                raise ValueError(
                    f"{self.__class__.__name__}: Invalid rays.  Dim 1 must have at "
                    "least 5 elements, but has {shape[1]}."
                )

            self._rays = rays

    def draw(self):
        """Redraw the mpl artists controlled by this class."""
        self._line_collection.set_segments(
            [
                [(start_x, start_y), (end_x, end_y)]
                for start_x, start_y, end_x, end_y, *_ in self.rays
            ]
        )
        self._line_collection.set_array(
            self._wavelength_unit_factor * self.rays[:, 4]
        )

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
        if arcs is None:
            self._arcs = np.zeros((0, 5))
        else:
            try:
                shape = arcs.shape
            except AttributeError as e:
                raise ValueError(
                    f"{self.__class__.__name__}: Invalid arcs.  Could not retrieve "
                    "arcs's shape."
                ) from e

            if len(shape) != 2:
                raise ValueError(
                    f"{self.__class__.__name__}: Invalid arcs.  Rank must be 2, but "
                    "is rank {len(shape)}."
                )
            if shape[1] < 5:
                raise ValueError(
                    f"{self.__class__.__name__}: Invalid arcs.  Dim 1 must have at "
                    "least 5 elements, but has {shape[1]}."
                )

            self._arcs = arcs

    def draw(self):
        """Redraw the mpl artists controlled by this class."""
        for arc_patch in self._arc_patches:
            arc_patch.remove()
        self._arc_patches = []

        for norm_arrow in self._norm_arrows:
            norm_arrow.remove()
        self._norm_arrows = []

        for arc in self._arcs:
            self._draw_arc(*arc[:5])

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
        An array that encodes information about the segments to be drawn.  Must be 
        rank 2.  The first dimension indexes segments.  The second dimension must 
        have length >= 4, whose first four elements are [start_x, stary_y, end_x, 
        end_y].
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
        self.segments = segments
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
        if segments is None:
            self._segments = np.zeros((0, 4))
        else:
            try:
                shape = segments.shape
            except AttributeError as e:
                raise AttributeError(
                    f"{self.__class__.__name__}: Invalid segments.  Could not "
                    "retrieve segments's shape."
                ) from e

            if len(shape) != 2:
                raise ValueError(
                    f"{self.__class__.__name__}: Invalid segments.  Rank must be 2, "
                    "but is rank {len(shape)}."
                )
            if shape[1] < 4:
                raise ValueError(
                    f"{self.__class__.__name__}: Invalid segments.  Dim 1 must have "
                    "at least 4 elements, but has {shape[1]}."
                )

            self._segments = segments

    def draw(self):
        """Redraw the mpl artists controlled by this class."""
        for norm_arrow in self._norm_arrows:
            norm_arrow.remove()
        self._norm_arrows = []

        segments = []
        for start_x, start_y, end_x, end_y, *_ in self.segments:
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


def disable_figure_key_commands():
    """Disable all keyboard shortcuts in the mpl figure."""
    for key, value in plt.rcParams.items():
        if "keymap" in key:
            plt.rcParams[key] = ""


def redraw_current_figure():
    """Redraws the mpl canvas."""
    plt.gcf().canvas.draw()
