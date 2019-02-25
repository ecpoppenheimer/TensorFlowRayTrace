"""
Utilities for drawing optical elements with matplotlib.

This module assists in visualizing the raysets and optical systems built
with tfrt.  These classes act as a nice interface that connects ndarrays
formatted like tfrt optical elements to a set of matplotlib axes so that
they can be displayed in a matplotlib figure.

Please note that the optical system data objects fed to these classes do
not have to be numpy ndarrays, but it is highly recommended that they be.
They must at least have a shape attribute and the proper shape 
requirements to represent that kind of object (see tfrt.raytrace for 
details on the required shapes).  TensorFlow tensors are not acceptable
inputs to these classes, but the arrays returned by session.run calls are.

This module defines some helpful constants which are the values of the 
wavelength of visible light of various colors, in um.  These values give
nice results when using the default colormap to display rays, but they are
not used anywhere in this module.  They exist only as convenient reference
points for users of this module, and if you need different wavelenghts for 
your application, you can use whatever values you want and either rescale
spectrumRGB or use a different colormap altogether.

"""

import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .spectrumRGB import rgb

PI = math.pi

VISIBLE_MIN = .38
VISIBLE_MAX = .78

RED = .68
ORANGE = .62
YELLOW = .575
GREEN = .51
BLUE = .45
PURPLE = .4

RAINBOW_6 = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]

# ----------------------------------------------------------------------------

"""
TODO:

1) Should I define a mechanism for allowing the user to set new min/max wavelength?  User can already do it by calling
something like "RayDrawer.line_collection.norm = plt.Normalize(min, max)".  This flexibility (and other like it is why
I am leaving rayDrawer.line_collection as a public member.
"""

class RayDrawer(object):
    """
    Class for drawing a rayset.
    
    This class makes it easy to draw a set of rays to a 
    matplotlib.axes.Axes.  By default this class will use the spectrumRGB
    colormap to color the rays by wavelength, but a different colormap
    can be chosen if desired.  If you desire to change the display 
    properties on the fly, you can get access to the underlying 
    mpl.collections.LineCollection used to actually plot the rays via
    the attribute RayDrawer.line_collection, though for most use cases
    this class should handle everything for you.
    
    Attributes
    ----------
    
    b_auto_redraw_canvas : bool
        If true, the class will automatically redraw the mpl figure after
        an update operation is performed.  Otherwise you will have to
        manually redraw.  Default behavior is for this attribute to be
        false, as there is no reason to redraw the canvas if multiple 
        drawing operations will need to be executed at once, and the user
        will better know when to redraw the canvas, but this functionalty
        is added as a convenience if desired.
    line_collection : mpl.collections.LineCollection
        The line collection internally used to draw the rays.  This
        attribute is exposed to the public API in case the user wants
        more control over the drawing, but for most use cases this 
        attribute won't need to be touched.
        
    Methods
    -------
    
    update(array_like)
        Draw the rays fed to this function.
    clear()
        Erase all rays controlled by this class instance.
        
    """

    def __init__(self, ax, min_wavelength=VISIBLE_MIN,
        max_wavelength=VISIBLE_MAX, style="-",
        colormap=mpl.colors.ListedColormap(rgb()), b_auto_redraw_canvas=False,
        units="um"):
        """
        Build the ray drawer, but do not feed it rays to draw.
        
        The constructor packages various style and behavior options
        and sets up a system for drawing rays, but does not itself
        accept any rays or draw anything.  Use method update() to
        actually draw rays.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            A handle to the mpl axis into which the rays will be 
            drawn.
        minimum_wavelength : float, optional
            The minimum wavelength, used only to normalize the 
            colormap.
        maximum_wavelength : float, optional
            The maximum wavelength, used only to normalize the
            colormap.
        style : string, optional
            The linestyle used to draw the rays.  Defaults to a solid
            line.  See matplotlib.lines.Line2D linestyle options for
            a list of the valid values.
        colormap : matplotlib.colors.Colormap, optional
            The colormap to use for coloring the rays.  Defaults to 
            the spectrumRGB map.
        b_autoRedrawCanvas : bool, optional
            If true, redraws the MPL figure after updating.  If 
            false, does not, and you have to call the canvas redraw
            yourself to see the effect of updating the drawer.
        units : string, optional
            The units of wavelength.  Default is 'um', micrometers, but 
            can also accept 'nm', nanometers.  Used to adjust the 
            wavelength, for compatability with the spectrumRGB colormap,
            which uses nm.  If you want to use a different colormap, 
            set the units to nm which causes RayDrawer to do wavelength
            as is to the colormap.  If um is selected, RayDrawer will
            multiply wavelength by 1000 to convert into nm before passing
            wavelength to the colormap.
            
        """

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

    def update(self, ray_data):
        """
        Feed ray data and draw the rays
        
        Any previously drawn rays will be discarded when this method is
        called to add new rays.  The LineCollection will be updated, but
        the canvas will only be redrawn by this method if 
        b_auto_redraw_canvas is true.
        
        Parameters
        ----------
        ray_data : np.ndarray
            An array that encodes information about the rays to be drawn,
            formatted like the raysets used by tfrt.raytrace.  Must be 
            rank 2.  The first dimension indexes rays.  The second 
            dimension must have shape >= 5, whose five elements are
            [xStart, yStart, xEnd, yEnd, wavelength].  Any additional 
            elements are ignored.
            
        """
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
        """ Erase all rays drawn by the last call to update(). """
        self.update(np.zeros((0, 5)))

# ----------------------------------------------------------------------------

"""
TODO:
1) In the surface drawer classes, I am leaving everything except norm_arrow_visibility as public, because I can imagine some validity to changing those values during the run of the program.  It would be pretty crazy to change the axis the surfaces are being drawn to, but like, I think it wouldn't throw errors or anything.  norm_arrow_visibility is private because I have already implemented a getter/setter system for that one, since I know how I want the class to behave when that value is changed, and it is useful to be able to toggle that value during runtime.  If you change any of the other members, the drawing will not update until the user calls update again.  I suppose I could implement getter/setter for all of these other values, but... I am not sure if it is worth the trouble.

1.5) I can imagine there could be a bug if you change b_include_norm_arrows
during runtime.  Should this one be private?

2) mpl.collections.LineCollection is a nice container that I can use for segments (and rays) but not arcs or norm arrows.  But I suppose I can use a mpl.collections.PatchCollection.  May clean things up.  Documentation claims that it would also be faster to store many patches in a patch collection, rather than have a list of many patches.  I see there is also a CircleCollection, but on skimming the documentation, that looks like the wrong thing to use.  Like, its for dots.  Not obvious how to even use it.
"""


class ArcDrawer(object):
    """
    Class for drawing a set of optical arc surfaces.
    
    This class makes it easy to draw a set of arcs formatted like the
    optical surfaces used by tfrt.raytrace to a matplotlib.axes.Axes.
    One notable restriction to this class is that all of the arcs drawn
    must have the same color and style.  If you want differently styled
    optical surfaces in your visualization, use multiple ArcDrawer 
    instances.  Color and style can be changed after instantiation, but
    you will need to call update afterward to see the change.
    
    When designing an optial system with refractive surfaces, it is very
    important to understand which direction the surface normal points, so
    that the ray tracer can decide whether a ray interaction is an 
    internal or external refraction.  To assist with this, ArcDrawer 
    provides support for visualizing the orientation of the surface by 
    drawing arrows along the surface that point in the direction of the
    norm.  I find it convenient to toggle the display of the norm arrows,
    turning them on when I want to inspect a surface to ensure that it is
    correctly oriented, and then turning them off after ensuring the 
    surface is correctly oriented to unclutter the display.
    
    Attributes
    ----------
    ax : matplotlib.axes.Axes
        A handle to the mpl axis into which the arcs will be drawn.
    color : color_like
        The color of all arcs and norm arrows drawn.  See 
        https://matplotlib.org/api/colors_api.html for acceptable
        color formats.
    style : string
        The linestyle used to draw the rays.  Defaults to a solid
        line.  See matplotlib.lines.Line2D linestyle options for
        a list of the valid values.
    b_auto_redraw_canvas : bool
        If true, the class will automatically redraw the mpl figure after
        an update operation is performed.  Otherwise you will have to
        manually redraw.  Default behavior is for this attribute to be
        false, as there is no reason to redraw the canvas if multiple 
        drawing operations will need to be executed at once, and the user
        will better know when to redraw the canvas, but this functionalty
        is added as a convenience if desired.
    b_include_norm_arrows : bool
        If True, will include norm arrows with the surface when updating.
        If changed, will not have an effect until the next call to 
        update().
    arrow_length : float
        The length (in ax coords) of the norm arrows.  If changed, will 
        not have an effect until the next call to update().
    arrow_count : int
        How many norm arrows to draw along the surface.  If changed, will
        not have an effect until the next call to update().
        
    Methods
    -------
    update(array_like)
        Draw the arcs fed to this function.
    clear()
        Erase all arcs controlled by this class instance.
    toggle_norm_arrow_visibility()
        Toggles display of the norm arrows.
    
    """

    def __init__(self, ax, color=(0, 1, 1), style="-",
        b_include_norm_arrows=False, b_auto_redraw_canvas=False,
        b_norm_arrow_visibility=True, arrow_length=0.05, arrow_count=5):
        """
        Build the arc drawer, but do not feed it arcs to draw.
        
        The constructor packages various style and behavior options
        and sets up a system for drawing arcs, but does not itself
        accept any arcs or draw anything.  Use method update() to
        actually draw arcs.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            A handle to the mpl axis into which the arcs will be 
            drawn.
        color : color_like, optional
            The color of all arcs and norm arrows drawn.  See 
            https://matplotlib.org/api/colors_api.html for acceptable
            color formats.
        style : string, optional
            The linestyle used to draw the arcs.  Defaults to a solid
            line.  See matplotlib.lines.Line2D linestyle options for
            a list of the valid values.
        b_include_norm_arrows : bool, optional
            If true, adds arrows along the arc surfaces that depict the
            direction of the surface norm, for visualizing the surface
            orientation.
        b_autoRedrawCanvas : bool, optional
            If true, redraws the MPL figure after updating.  If 
            false, does not, and you have to call the canvas redraw
            yourself to see the effect of updating the drawer.
        b_norm_arrow_visibility : bool, optional
            The initial state of the norm arrow visibility.  Defaults to
            true, so the norm arrows start visible.
        arrow_length : float, optional
            The length (in ax coords) of the norm arrows.
        arrow_count : int, optional
            How many norm arrows to draw along the surface.
            
        """

        self.ax = ax
        self.color = color
        self.style = style
        self.b_auto_redraw_canvas = b_auto_redraw_canvas
        
        self.b_include_norm_arrows = b_include_norm_arrows
        self._norm_arrow_visibility = b_norm_arrow_visibility
        if self.b_include_norm_arrows:
            self.arrow_length = arrow_length
            self.arrow_count = arrow_count

    def update(self, arc_data):
        """
        Feed arc data and draw the arcs
        
        Any previously drawn arcs will be discarded when this method is
        called to add new ones.  Arcs will be added to the axis, but
        the canvas will only be redrawn by this method if 
        b_auto_redraw_canvas is true.
        
        Parameters
        ----------
        arc_data : np.ndarray
            An array that encodes information about the arcs to be drawn.
            Must be rank 2.  The first dimension indexes arcs.  The
            second dimension must have shape >= 5, whose first five 
            elements are [xcenter, ycenter, angle_start, angle_end, 
            radius].  Any additional elements are ignored.
            
        """
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
        """ Erase all arcs drawn by the last call to update(). """
        self.update(np.zeros((0, 5)))

    # the next three parts allow to toggle the visibility of arrows that
    # visually depict the norm of the surface
    @property
    def norm_arrow_visibility(self):
        """ The visibility of the norm arrows. """
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
        """ Toggle the visibility of the norm arrows. """
        self.norm_arrow_visibility = not self.norm_arrow_visibility


# ----------------------------------------------------------------------------

"""
TODO
So in this one, if I change the color of the line_collection at runtime, it updates the colors of the lines on a canvas redraw, and does not need a SegmentDrawer.update(), unlike the case with ArcDrawer.  But it does not update the color of the norm arrows.  I understand why this happens.  Question is, is this behavior important enough (and abberrant/inconsistent enough) that I need to deal with getter/setters for these kinds of properties.
"""

class SegmentDrawer(object):
    """
    Class for drawing a set of optical line segment surfaces.
    
    This class makes it easy to draw a set of line segments formatted like
    the optical surfaces used by tfrt.raytrace to a matplotlib.axes.Axes.
    All of the lines drawn by a SegmentDrawer instance should have the 
    same color and style.  The underlying mpl.collections.LineCollection
    is exposed to the public API of this class, so it is actually possible
    to set the color of individual segments, but this behavior isn't
    recommended.  If you want differently styled optical surfaces
    in your visualization, use multiple SegmentDrawer instances.  
    
    When designing an optial system with refractive surfaces, it is very
    important to understand which direction the surface normal points, so
    that the ray tracer can decide whether a ray interaction is an 
    internal or external refraction.  To assist with this, SegmentDrawer 
    provides support for visualizing the orientation of the surface by 
    drawing an arrow at the midpoint of the surface that points in the
    direction of the norm.  I find it convenient to toggle the display of
    the norm arrows, turning them on when I want to inspect a surface to 
    ensure that it is correctly oriented, and then turning them off after
    ensuring the surface is correctly oriented to unclutter the display.
    
    Attributes
    ----------
    ax : matplotlib.axes.Axes
        A handle to the mpl axis into which the arcs will be drawn.
    b_auto_redraw_canvas : bool
        If true, the class will automatically redraw the mpl figure after
        an update operation is performed.  Otherwise you will have to
        manually redraw.  Default behavior is for this attribute to be
        false, as there is no reason to redraw the canvas if multiple 
        drawing operations will need to be executed at once, and the user
        will better know when to redraw the canvas, but this functionalty
        is added as a convenience if desired.
    line_collection : mpl.collections.LineCollection
        The line collection internally used to draw the line segments.
        This attribute is exposed to the public API in case the user wants
        more control over the drawing, but for most use cases this 
        attribute won't need to be touched.
    b_include_norm_arrows : bool
        If True, will include norm arrows with the surface when updating.
        If changed, will not have an effect until the next call to 
        update().
    arrow_length : float
        The length (in ax coords) of the norm arrows.  If changed, will 
        not have an effect until the next call to update().
        
    Methods
    -------
    update(array_like)
        Draw the segments fed to this function.
    clear()
        Erase all segments controlled by this class instance.
    toggle_norm_arrow_visibility()
        Toggles display of the norm arrows.
    
    """

    def __init__(self, ax, color=(0, 1, 1), style="-",
        b_include_norm_arrows=False, b_auto_redraw_canvas=False,
        b_norm_arrow_visibility=True, arrow_length=0.05):
        """
        Build the segment drawer, but do not feed it segments to draw.
        
        The constructor packages various style and behavior options
        and sets up a system for drawing segments, but does not itself
        accept any segments or draw anything.  Use method update() to
        actually draw segments.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            A handle to the mpl axis into which the segments will be 
            drawn.
        color : color_like, optional
            The color of all segments and norm arrows drawn.  See 
            https://matplotlib.org/api/colors_api.html for acceptable
            color formats.
        style : string, optional
            The linestyle used to draw the segments.  Defaults to a solid
            line.  See matplotlib.lines.Line2D linestyle options for
            a list of the valid values.
        b_include_norm_arrows : bool, optional
            If true, add an arrow at the midpoint of the segment that 
            depicts the direction of the surface norm, for visualizing the
            surface orientation.
        b_autoRedrawCanvas : bool, optional
            If true, redraws the MPL figure after updating.  If 
            false, does not, and you have to call the canvas redraw
            yourself to see the effect of updating the drawer.
        b_norm_arrow_visibility : bool, optional
            The initial state of the norm arrow visibility.  Defaults to
            true, so the norm arrows start visible.
        arrow_length : float, optional
            The length (in ax coords) of the norm arrows.
            
        """

        self.ax = ax
        self.b_auto_redraw_canvas = b_auto_redraw_canvas
        
        self.b_include_norm_arrows = b_include_norm_arrows
        if self.b_include_norm_arrows:
            self._norm_arrow_visibility = b_norm_arrow_visibility
            self.arrow_length = arrow_length

        # Build the line collection, and add it to the axes
        self.line_collection = mpl.collections.LineCollection(
            [], colors=color, linestyles=style)
        self.ax.add_collection(self.line_collection)

    def update(self, segment_data):
        """
        Feed segment data and draw the segments
        
        Any previously drawn segments will be discarded when this method
        is called to add new ones.  Segments will be added to the axis,
        but the canvas will only be redrawn by this method if 
        b_auto_redraw_canvas is true.
        
        Parameters
        ----------
        segment_data : np.ndarray
            An array that encodes information about the segments to be 
            drawn.  Must be rank 2.  The first dimension indexes segments.
            The second dimension must have shape >= 4, whose first four 
            elements are [xstart, ystart, xend, yend].  Any additional 
            elements are ignored.
            
        """
        
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

        """
        TODO
        So I am extracting the color of the first line in the line collection
        and using that to color all of the norm arrows.  I could probably build
        the segments in one step, then extract the colors of all lines and
        use that to color the norm arrows.  But this will only ever matter
        if the user directly accesses the line_collection and uses a colormap
        to set the line colors, which is an odd use case.  Not sure what is
        worth doing here.
        """
        line_color = self.line_collection.get_colors()[0, :-1]

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
                    self.arrow_length * math.cos(theta),
                    self.arrow_length * math.sin(theta),
                    width=0.4 * self.arrow_length,
                    color=line_color,
                    visible=self._norm_arrow_visibility)))

        self.line_collection.set_segments(segments)

        # redraw the canvas
        if self.b_auto_redraw_canvas:
            plt.gcf().canvas.draw()

    def clear(self):
        """ Erase all segments drawn by the last call to update(). """
        self.update(np.zeros((0, 4)))

    # the next three parts allow to toggle the visibility of arrows that 
    # visually depict the norm of the surface
    @property
    def norm_arrow_visibility(self):
        """ The visibility of the norm arrows. """
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
        """ Toggle the visibility of the norm arrows. """
        self.norm_arrow_visibility = not self.norm_arrow_visibility

#-----------------------------------------------------------------------------

def disable_figure_key_commands():
    """ Disable all keyboard shortcuts in the mpl figure. """
    for key, value in plt.rcParams.items():
        if "keymap" in key:
            plt.rcParams[key] = ''
