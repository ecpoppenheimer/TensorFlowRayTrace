"""
OpticsPlotting.py
Eric Poppenheimer
December 2018

Module with some utilities for plotting optical elements that are formatted as they need to be to be used in
TFRayTrace.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .spectrumRGB import rgb
import math

PI = math.pi
rgb=rgb()
VISIBLE_MIN = .38
VISIBLE_MAX = .78

RED = .68
ORANGE = .62
YELLOW = .575
GREEN = .51
BLUE = .45
PURPLE = .4

RAINBOW_6 = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]

#-------------------------------------------------------------------------------------------------------------------

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
        b_visible: The starting value for the visibility of this object.
        b_autoRedrawCanvas: If true, redraws the MPL figure after updating.  If false, does not, and you have
            to manually redraw the canvas.  Useful to avoid unnecessary redrawing if you are using many drawing
            functions at once.
        units: The units of wavelength.  Default is um, micrometers, but can also accept nm, nanometers.
            Need to internally convert to nm, since that is what spectrumRGB uses
    """
    def __init__(self, ax, minWavelength=VISIBLE_MIN, maxWavelength=VISIBLE_MAX, style="-",
        colormap=mpl.colors.ListedColormap(rgb), b_visible=False, b_autoRedrawCanvas=False, units="um"):

        # initialize various variables
        self.b_autoRedrawCanvas = b_autoRedrawCanvas
        self._b_visible = b_visible
        self.minWavelength = minWavelength
        self.maxWavelength = maxWavelength
        self.colormap = colormap

        self.units=units
        if units == "um":
            self.wavelengthUnitFactor = 1000.0
        elif units == "nm":
            self.wavelengthUnitFactor = 1.0
        else:
            raise ValueError("RayDrawer: Invalid units: {}.  Allowed values are 'um', 'nm'.".format(units))

        # build the colormap and add it to the axes
        self.lineCollection = mpl.collections.LineCollection(
            [],
            linestyles=style,
            cmap=self.colormap,
            norm = plt.Normalize(self.wavelengthUnitFactor * self.minWavelength,
                                 self.wavelengthUnitFactor * self.maxWavelength),
            visible=self._b_visible
            )
        ax.add_collection(self.lineCollection)

    """
    update()
        Updates and draws the rays.
    """
    def update(self, rayData):
        # validate rayData's shape
        try:
            shape = rayData.shape
        except:
            raise ValueError("RayDrawer: Invalid rayTensor.  Could not retrieve rayTensor's shape.")

        if len(shape) != 2:
            raise ValueError("RayDrawer: Invalid rayTensor.  Tensor rank must be 2, got rank {}."
                .format(len(shape)))
        if shape[1] < 5:
            raise ValueError("RayDrawer: Invalid rayTensor.  Dim 1 must have at least 5 elements, got {}."
                .format(shape[1]))

        self.lineCollection.set_segments([[(each[0], each[1]), (each[2], each[3])] for each in rayData])
        self.lineCollection.set_array(self.wavelengthUnitFactor*rayData[:,4])

        # redraw the canvas
        if self.b_autoRedrawCanvas:
            plt.gcf().canvas.draw()

    # the next three parts allow to toggle the visibility of the lines
    @property
    def visibility(self):
        return self._b_visible

    @visibility.setter
    def visibility(self, val):
        if type(val) != bool:
            raise TypeError("visibility must be bool")

        self._b_visible = val
        self.lineCollection.set_visible(self._b_visible)

        # redraw the canvas
        if self.b_autoRedrawCanvas:
            plt.gcf().canvas.draw()

    def toggleVisibility(self):
        self.visibility = not self.visibility

#-------------------------------------------------------------------------------------------------------------------

"""
ArcDrawer: A class for drawing arcs from tensors with MPL.  Contains support for displaying and hiding norm arrows.
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

    def __init__(self, ax, color = (0, 1, 1), style = "-", b_includeNormArrows = False, b_autoRedrawCanvas = False,
                 b_normArrowVisibility=True):

        self.ax = ax
        self.color = color
        self.style = style
        self.b_includeNormArrows = b_includeNormArrows
        self.b_autoRedrawCanvas = b_autoRedrawCanvas
        self._normArrowVisibility = b_normArrowVisibility
        self.arrowLength = .05
        self.arrowCount = 5

    """
    update()
        Runs the session to fetch the tensor's data and adds the arc patches to the canvas
        arcData: A np array formatted like boundaryArc (xcenter, ycenter, angleStart, angleEnd, radius).  May 
            include the material indices after that.  Shape must be (..., n>=5).  Elements after the first five
            are ignored.
    """
    def update(self, arcData):
        # validate the arcTensor shape
        try:
            shape = arcData.shape
        except:
            raise ValueError("ArcDrawer: Invalid arcData.  Could not retrieve arcTensor's shape.")

        if len(shape) != 2:
            raise ValueError("ArcDrawer: Invalid arcData.  Rank must be 2, got rank {}."
                .format(len(shape)))
        if shape[1] < 5:
            raise ValueError("ArcDrawer: Invalid arcData.  Dim 1 must have at least 5 elements, got {}."
                .format(shape[1]))

        # remove the old arcPatches
        try:
            for each in self._arcPatches:
                each.remove()
        except:
            pass
        self._arcPatches = []

        # remove the old arrowPatches
        if self.b_includeNormArrows:
            try:
                for each in self._normArrows:
                    each.remove()
            except:
                pass
            self._normArrows = []

        # draw the new patches
        for each in arcData:
            xcenter = each[0]
            ycenter = each[1]
            angleStart = each[2]
            angleEnd = each[3]
            radius = each[4]

            # add the arc patch
            self._arcPatches.append(self.ax.add_patch(mpl.patches.Arc(
                (xcenter, ycenter),
                abs(2*radius),
                abs(2 * radius),
                theta1=math.degrees(angleStart),
                theta2=math.degrees(angleEnd),
                color=self.color,
                linestyle=self.style)))

            if self.b_includeNormArrows:
                # add the norm arrows
                if angleStart < angleEnd:
                    angles = np.linspace(angleStart, angleEnd, self.arrowCount)
                else:
                    angles = np.linspace(angleStart, angleEnd + 2*PI, self.arrowCount)
                for theta in angles:
                    self._normArrows.append(self.ax.add_patch(mpl.patches.Arrow(
                        xcenter + abs(radius) * math.cos(theta),
                        ycenter + abs(radius) * math.sin(theta),
                        self.arrowLength * math.cos(theta) * np.sign(radius),
                        self.arrowLength * math.sin(theta) * np.sign(radius),
                        width=0.4*self.arrowLength,
                        color=self.color,
                        visible=self._normArrowVisibility
                        )))

        # redraw the canvas
        if self.b_autoRedrawCanvas:
            plt.gcf().canvas.draw()

    # the next three parts allow to toggle the visibility of arrows that visually depict the norm of the
    # surface
    @property
    def normArrowVisibility(self):
        return self._normArrowVisibility

    @normArrowVisibility.setter
    def normArrowVisibility(self, val):
        if type(val) != bool:
            raise TypeError("visibility must be bool")

        self._normArrowVisibility = val
        try:
            for arrow in self._normArrows:
                arrow.set_visible(self._normArrowVisibility)
        except:
            pass

    def toggleNormArrowVisibility(self):
        self.normArrowVisibility = not self.normArrowVisibility


#-------------------------------------------------------------------------------------------------------------------

"""
SegmentDrawer: A class for drawing segment boundaries from tensors with MPL.  Contains support for displaying and 
hiding norm arrows.
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

    def __init__(self, ax, color = (0, 1, 1), style = "-", b_includeNormArrows = False, b_autoRedrawCanvas = False,
                 b_normArrowVisibility=True):

        self.ax = ax
        self.color = color
        self.b_includeNormArrows = b_includeNormArrows
        self.b_autoRedrawCanvas = b_autoRedrawCanvas
        self._normArrowVisibility = b_normArrowVisibility
        self.arrowLength = .05
        self.arrowCount = 5
        self._lineCollection = mpl.collections.LineCollection([], colors=color, linestyles=style)
        self.ax.add_collection(self._lineCollection)

    """
    update()
        Runs the session to fetch the tensor's data and adds the arc patches to the canvas
        arcData: A np array formatted like boundaryArc (xcenter, ycenter, xend, yend).  May 
            include the material indices after that.  Shape must be (..., n>=4).  Elements after the first four
            are ignored.
    """
    def update(self, segmentData):
        # validate the arcTensor shape
        try:
            shape = segmentData.shape
        except:
            raise ValueError("SegmentDrawer: Invalid segmentData.  Could not retrieve segmentTensor's shape.")

        if len(shape) != 2:
            raise ValueError("SegmentDrawer: Invalid segmentData.  Rank must be 2, got rank {}."
                .format(len(shape)))
        if shape[1] < 4:
            raise ValueError("SegmentDrawer: Invalid segmentData.  Dim 1 must have at least 4 elements, got {}."
                .format(shape[1]))

        # remove the old arrowPatches
        if self.b_includeNormArrows:
            try:
                for each in self._normArrows:
                    each.remove()
            except:
                pass
            self._normArrows = []

        segments = []

        # compose the new line collection
        for each in segmentData:
            xstart = each[0]
            ystart = each[1]
            xend = each[2]
            yend = each[3]
            theta = np.arctan2(yend-ystart, xend-xstart) + PI/2.0

            # build the collections segments
            segments.append([(xstart, ystart), (xend, yend)])

            if self.b_includeNormArrows:
                # add the norm arrows
                self._normArrows.append(self.ax.add_patch(mpl.patches.Arrow(
                    (xstart+xend)/2.0,
                    (ystart+yend)/2.0,
                    self.arrowLength * math.cos(theta),
                    self.arrowLength * math.sin(theta),
                    width=0.4*self.arrowLength,
                    color=self.color,
                    visible=self._normArrowVisibility
                    )))

        self._lineCollection.set_segments(segments)

        # redraw the canvas
        if self.b_autoRedrawCanvas:
            plt.gcf().canvas.draw()

    # the next three parts allow to toggle the visibility of arrows that visually depict the norm of the
    # surface
    @property
    def normArrowVisibility(self):
        return self._normArrowVisibility

    @normArrowVisibility.setter
    def normArrowVisibility(self, val):
        if type(val) != bool:
            raise TypeError("visibility must be bool")

        self._normArrowVisibility = val
        try:
            for arrow in self._normArrows:
                arrow.set_visible(self._normArrowVisibility)
        except:
            pass

    def toggleNormArrowVisibility(self):
        self.normArrowVisibility = not self.normArrowVisibility


#===================================================================================================================

# draw lines, all of a single color, from an array formatted like rays or boundary segments
def drawLines(lineData, color=(0,0,0)):
    xdata = []
    ydata = []
    for each in lineData:
        xdata += [each[0], each[2], None]
        ydata += [each[1], each[3], None]

    ax.add_line(mpl.lines.Line2D(xdata, ydata, color=color))


#-------------------------------------------------------------------------------------------------------------------

# draw lines from data formatted like rays.  Can choose the line style.  Colors the lines with wavelength by using
# the hsv_r colormap from MPL
def drawLinesW(lineData, minWavelength, maxWavelength, ax, style="-"):
    rayLineCollection = mpl.collections.LineCollection(
        [((each[0], each[1]), (each[2], each[3])) for each in lineData],
        linestyles=style,
        cmap = plt.get_cmap("hsv_r"),
        norm=plt.Normalize(minWavelength, maxWavelength))
    rayLineCollection.set_array(np.array([each[4] for each in lineData]))

    ax.add_collection(rayLineCollection)


# draw lines from data formatted like rays.  Can choose the line style.  Colors the lines with wavelength by
# indexing colors out of the dictionary passed to argument color
def drawLinesC(lineData, ax, style="-", color={.4: (0,0,1), .5: (1,1,0), .6: (1,0,0)}):
    for each in lineData:
        ax.add_line(mpl.lines.Line2D(
            (each[0], each[2]),
            (each[1], each[3]),
            linestyle=style,
            color=color[each[4]]))


#-------------------------------------------------------------------------------------------------------------------

def disableFigureKeyCommands():
    for key, value in plt.rcParams.iteritems():
        if "keymap" in key:
            plt.rcParams[key] = ''
