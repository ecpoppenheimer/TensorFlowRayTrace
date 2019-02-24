"""
OpticsPlotting.py
Eric Poppenheimer
December 2018

Module with some utilities for plotting optical elements that are formatted as they need to be to be used in
TFRayTrace.py
"""

import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .spectrumRGB import rgb

PI = math.pi

# The following constants define some useful wavelengths (units in um)
VISIBLE_MIN = .38
VISIBLE_MAX = .78

RED = .68
ORANGE = .62
YELLOW = .575
GREEN = .51
BLUE = .45
PURPLE = .4

RAINBOW_6 = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]

rgb = rgb()

# -----------------------------------------------------------------------------

"""
TODO:

1) Should I define a mechanism for allowing the user to set new min/max wavelength?  Use can already do it by calling
something like "RayDrawer.line_collection.norm = plt.Normalize(min, max)".  This flexibility (and other like it is why
I am leaving rayDrawer.line_collection as a public member.
"""

"""
RayDrawer: A class for drawing rays from tensors with MPL.
"""


class RayDrawer(object):
    """
    __init__()
        ax: A handle to the MPL axis object, into which the rays will be drawn
        min/maxWavelength: The minimum and maximum wavelength, used to normalize the colormap
        style: The linestyle used to draw the rays.  Defaults to a solid line
        colormap: The colormap to use for coloring the rays.  Defaults to the spectrumRGB map
        b_autoRedrawCanvas: If true, redraws the MPL figure after updating.  If false, does not, and you have
            to manually redraw the canvas.  Useful to avoid unnecessary redrawing if you are using many drawing
            functions at once.
        units: The units of wavelength.  Default is um, micrometers, but can also accept nm, nanometers.
            Need to internally convert to nm, since that is what spectrumRGB uses
    """

    def __init__(self, ax, min_wavelength=VISIBLE_MIN,
        max_wavelength=VISIBLE_MAX, style="-",
        colormap=mpl.colors.ListedColormap(rgb), b_auto_redraw_canvas=False,
        units="um"):

        # Initialize class variables.
        self.b_auto_redraw_canvas = b_auto_redraw_canvas

        if units == "um":
            self._wavelength_unit_factor = 1000.0
        elif units == "nm":
            self._wavelength_unit_factor = 1.0
        else:
            raise ValueError("RayDrawer: Invalid units: {}.  Allowed "
                             "values are 'um', 'nm'.".format(units))

        # Build the line collection and add it to the axes.
        self.line_collection = mpl.collections.LineCollection(
            [],
            linestyles=style,
            cmap=colormap,
            norm=plt.Normalize(
                self._wavelength_unit_factor * min_wavelength,
                self._wavelength_unit_factor * max_wavelength))
        ax.add_collection(self.line_collection)

    """
    update()
    Updates and draws the rays.
    """

    def update(self, ray_data):
        # validate ray_data's shape
        try:
            shape = ray_data.shape
        except BaseException:
            raise ValueError("RayDrawer: Invalid ray_data.  Could not "
                "retrieve ray_data's shape.")
        if len(shape) != 2:
            raise ValueError("RayDrawer: Invalid ray_data.  Tensor rank must "
                "be 2, but is rank {}.".format(len(shape)))
        if shape[1] < 5:
            raise ValueError("RayDrawer: Invalid ray_data.  Dim 1 must have "
                "at least 5 elements, but has {}.".format(shape[1]))

        # transfer the ray_data into the line collection
        self.line_collection.set_segments([
            [(each[0], each[1]), (each[2], each[3])] for each in ray_data])
        self.line_collection.set_array(
            self._wavelength_unit_factor * ray_data[:, 4])

        # redraw the canvas
        if self.b_auto_redraw_canvas:
            plt.gcf().canvas.draw()

    def clear(self):
        self.update(np.zeros((0, 5)))

# -----------------------------------------------------------------------------

"""
ArcDrawer: A class for drawing arcs from tensors with MPL.  Contains support for displaying and hiding norm arrows.
"""

"""
TODO:
1) In the surface drawer classes, I am leaving everything except norm_arrow_visibility as public, because I can imagine some validity to changing those values during the run of the program.  It would be pretty crazy to change the axis the surfaces are being drawn to, but like, I think it wouldn't throw errors or anything.  norm_arrow_visibility is private because I have already implemented a getter/setter system for that one, since I know how I want the class to behave when that value is changed, and it is useful to be able to toggle that value during runtime.  If you change any of the other members, the drawing will not update until the user calls update again.  I suppose I could implement getter/setter for all of these other values, but... I am not sure if it is worth the trouble.

2) mpl.collections.LineCollection is a nice container that I can use for segments (and rays) but not arcs or norm arrows.  But I suppose I can use a mpl.collections.PatchCollection.  May clean things up.  Documentation claims that it would also be faster to store many patches in a patch collection, rather than have a list of many patches.  I see there is also a CircleCollection, but on skimming the documentation, that looks like the wrong thing to use.  Like, its for dots.  Not obvious how to even use it.
"""


class ArcDrawer(object):
    """
    __init__()
        ax: A handle to a MPL axis object, into which the arcs will be drawn
        color: The color to draw all arcs handled by this object
        b_includeNormArrows: If true, include norm arrows for this arc
        b_autoRedrawCanvas: If true, redraws the MPL figure after updating.  If false, does not, and you have
            to manually redraw the canvas.  Useful to avoid unnecessary redrawing if you are using many drawing
            functions at once.
        b_normArrowVisibility: The starting state of the norm arrow visibility.  Only meaningful if norm arrows are
            already included
    """

    def __init__(self, ax, color=(0, 1, 1), style="-",
        b_include_norm_arrows=False, b_auto_redraw_canvas=False,
        b_norm_arrow_visibility=True, arrow_length=0.05, arrow_count=5):

        self.ax = ax
        self.color = color
        self.style = style
        self.b_include_norm_arrows = b_include_norm_arrows
        self.b_auto_redraw_canvas = b_auto_redraw_canvas
        if self.b_include_norm_arrows:
            self._norm_arrow_visibility = b_norm_arrow_visibility
            self.arrow_length = arrow_length
            self.arrow_count = arrow_count

    """
    update()
        Runs the session to fetch the tensor's data and adds the arc patches to the canvas
        arcData: A np array formatted like boundaryArc (xcenter, ycenter, angleStart, angleEnd, radius).  May
            include the material indices after that.  Shape must be (..., n>=5).  Elements after the first five
            are ignored.
    """

    def update(self, arc_data):
        # validate arc_data's shape
        try:
            shape = arc_data.shape
        except BaseException:
            raise ValueError("ArcDrawer: Invalid arc_data.  Could not "
                "retrieve arc_data's shape.")

        if len(shape) != 2:
            raise ValueError("ArcDrawer: Invalid arc_data.  Rank must be 2, "
                "but is rank {}.".format(len(shape)))
        if shape[1] < 5:
            raise ValueError("ArcDrawer: Invalid arc_data.  Dim 1 must have "
                "at least 5 elements, but has {}.".format(shape[1]))

        # remove the old arc_patches
        try:
            for each in self._arc_patches:
                each.remove()
        except BaseException:
            pass  # Do nothing, there aren't any to remove
        self._arc_patches = []

        # remove the old arrow_patches
        if self.b_include_norm_arrows:
            try:
                for each in self._norm_arrows:
                    each.remove()
            except BaseException:
                pass  # Do nothing, there aren't any to remove
            self._norm_arrows = []

        # draw the new patches
        for each in arc_data:
            xcenter = each[0]
            ycenter = each[1]
            angle_start = each[2]
            angle_end = each[3]
            radius = each[4]

            # add the arc patch
            self._arc_patches.append(self.ax.add_patch(mpl.patches.Arc(
                (xcenter, ycenter),
                abs(2 * radius),
                abs(2 * radius),
                theta1=math.degrees(angle_start),
                theta2=math.degrees(angle_end),
                color=self.color,
                linestyle=self.style)))

            if self.b_include_norm_arrows:
                # add the norm arrows
                if angle_start < angle_end:
                    angles = np.linspace(
                        angle_start,
                        angle_end,
                        self.arrow_count)
                else:
                    angles = np.linspace(
                        angle_start,
                        angle_end + 2 * PI,
                        self.arrow_count)
                for theta in angles:
                    self._norm_arrows.append(
                        self.ax.add_patch(mpl.patches.Arrow(
                            xcenter + abs(radius) * math.cos(theta),
                            ycenter + abs(radius) * math.sin(theta),
                            self.arrow_length*math.cos(theta)*np.sign(radius),
                            self.arrow_length*math.sin(theta)*np.sign(radius),
                            width=0.4 * self.arrow_length,
                            color=self.color,
                            visible=self._norm_arrow_visibility)))

        # redraw the canvas
        if self.b_auto_redraw_canvas:
            plt.gcf().canvas.draw()

    def clear(self):
        self.update(np.zeros((0, 5)))

    # the next three parts allow to toggle the visibility of arrows that
    # visually depict the norm of the surface
    @property
    def norm_arrow_visibility(self):
        return self._norm_arrow_visibility

    @norm_arrow_visibility.setter
    def norm_arrow_visibility(self, val):
        if not isinstance(val, bool):
            raise TypeError("visibility must be bool")
        self._norm_arrow_visibility = val

        try:
            for arrow in self._norm_arrows:
                arrow.set_visible(self._norm_arrow_visibility)
        except BaseException:
            pass  # Do nothing, there aren't any to update

        # redraw the canvas
        if self.b_auto_redraw_canvas:
            plt.gcf().canvas.draw()

    def toggle_norm_arrow_visibility(self):
        self.norm_arrow_visibility = not self.norm_arrow_visibility


# -----------------------------------------------------------------------------

"""
SegmentDrawer: A class for drawing segment boundaries from tensors with MPL.  Contains support for displaying and
hiding norm arrows.
"""

"""
TODO
So in this one, if I change the color of the line_collection at runtime, it
updates the colors of the lines on a canvas redraw, and does not need a SegmentDrawer.update(), unlike the case with ArcDrawer.  But it does not update the color of the norm arrows.  I understand why this happens.  Question is, is this behavior important enough (and abberrant/inconsistent enough) that I need to deal with getter/setters for these kinds of properties.
"""


class SegmentDrawer(object):
    """
    __init__()
        ax: A handle to a MPL axis object, into which the arcs will be drawn
        color: The color to draw all arcs handled by this object
        b_includeNormArrows: If true, include norm arrows for this surface
        b_autoRedrawCanvas: If true, redraws the MPL figure after updating.  If false, does not, and you have
            to manually redraw the canvas.  Useful to avoid unnecessary redrawing if you are using many drawing
            functions at once.
        b_normArrowVisibility: The starting state of the norm arrow visibility.  Only meaningful if norm arrows are
            already included
    """

    def __init__(self, ax, color=(0, 1, 1), style="-",
        b_include_norm_arrows=False, b_auto_redraw_canvas=False,
        b_norm_arrow_visibility=True):

        self.ax = ax
        self.b_include_norm_arrows = b_include_norm_arrows
        self.b_auto_redraw_canvas = b_auto_redraw_canvas
        self._norm_arrow_visibility = b_norm_arrow_visibility
        self.arrowLength = .05
        self.arrowCount = 5

        # Build the line collection, and add it to the axes
        self.line_collection = mpl.collections.LineCollection(
            [], colors=color, linestyles=style)
        self.ax.add_collection(self.line_collection)

    """
    update()
        Runs the session to fetch the tensor's data and adds the arc patches to the canvas
        arcData: A np array formatted like boundaryArc (xcenter, ycenter, xend, yend).  May
            include the material indices after that.  Shape must be (..., n>=4).  Elements after the first four
            are ignored.
    """

    def update(self, segment_data):
        # validate the segment_data shape
        try:
            shape = segment_data.shape
        except BaseException:
            raise ValueError("SegmentDrawer: Invalid segment_data.  Could "
                "not retrieve segment_data's shape.")

        if len(shape) != 2:
            raise ValueError("SegmentDrawer: Invalid segment_data.  Rank "
                "must be 2, but is rank {}.".format(len(shape)))
        if shape[1] < 4:
            raise ValueError("SegmentDrawer: Invalid segment_data.  Dim 1 "
                "must have at least 4 elements, but has {}.".format(shape[1]))

        # remove the old arrowPatches
        if self.b_include_norm_arrows:
            try:
                for each in self._norm_arrows:
                    each.remove()
            except BaseException:
                pass  # There are none, so do nothing
            self._norm_arrows = []

        line_color = self.line_collection.get_colors()[0, :-1]
        print(self.line_collection.get_colors())

        segments = []

        # compose the new line collection
        for each in segment_data:
            xstart = each[0]
            ystart = each[1]
            xend = each[2]
            yend = each[3]
            theta = np.arctan2(yend - ystart, xend - xstart) + PI / 2.0

            # build the collections segments
            segments.append([(xstart, ystart), (xend, yend)])

            if self.b_include_norm_arrows:
                # add the norm arrows
                self._norm_arrows.append(self.ax.add_patch(mpl.patches.Arrow(
                    (xstart + xend) / 2.0,
                    (ystart + yend) / 2.0,
                    self.arrowLength * math.cos(theta),
                    self.arrowLength * math.sin(theta),
                    width=0.4 * self.arrowLength,
                    color=line_color,
                    visible=self._norm_arrow_visibility)))

        self.line_collection.set_segments(segments)

        # redraw the canvas
        if self.b_auto_redraw_canvas:
            plt.gcf().canvas.draw()

    def clear(self):
        self.update(np.zeros((0, 4)))

    # the next three parts allow to toggle the visibility of arrows that visually depict the norm of the
    # surface
    @property
    def norm_arrow_visibility(self):
        return self._norm_arrow_visibility

    @norm_arrow_visibility.setter
    def norm_arrow_visibility(self, val):
        if not isinstance(val, bool):
            raise TypeError("visibility must be bool")

        self._norm_arrow_visibility = val
        try:
            for arrow in self._norm_arrows:
                arrow.set_visible(self._norm_arrow_visibility)
        except BaseException:
            pass

        # redraw the canvas
        if self.b_auto_redraw_canvas:
            plt.gcf().canvas.draw()

    def toggle_norm_arrow_visibility(self):
        self.norm_arrow_visibility = not self.norm_arrow_visibility

#-----------------------------------------------------------------------------


def disable_figure_key_commands():
    for key, value in plt.rcParams.items():
        if "keymap" in key:
            plt.rcParams[key] = ''
