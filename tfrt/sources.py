"""
Classes to help make source light ray sets that can be fed to the ray tracer.

Rays are stored as line segments, defined as being between two endpoints.  But they are
interpreted as semi-infinite rays; the location of the end point only determines the 
direction of the ray but the location of the start point determines where the ray 
originates, and hence which surfaces it can intersect with.

Sources store their data like a dictionary; the class is read/write indexable.
By convention keys should be strings, but this is not forced.  Sources will only
expose fields relevant to them.  Possible data fields are x_start, y_start, z_start,
x_end, y_end, z_end, wavelength, rank.  Each field is a 1-D tf.float64 tensor, and each 
should have exactly the same length.

Sources inherit from tfrt.RecursivelyUpdatable, defined in tfrt/update.py.  Since these
are intended to be a mid-level construction, typical use will mean that these sources are
fed to an OpticalSystem, which can take care of updating them.

Each source requires one or more distribution, defined in tfrt/distributions.py.  Input 
Distributions are typically assumed to be relative to zero, and the source is able to 
translate and rotate them.  But this is not true for AperatureSource, which takes its
distributions to be absolute.

A dense source generates a set of rays that takes each combination of its 
inputs.  An un-dense source requires its input all have the same size and 
produces that many rays by matching data in its inputs 1:1.  Dense sources are  
convenient for static sources (sources that use static distributions) since it 
generates rays that span the entire range of specified inputs.  It is not
recommended that sources that use random distributions choose to be dense, simply 
because they will be less random.  But the source will work either way.

Sources may be in 2D or 3D.  2D sources can only accept 2D distributions, but 3D sources
may take 2D base point distributions, in which case the base points will be assumed to lie
in the y-z plane.

In 2D, angles are scalars, defined counterclockwise from the x-axis.  In 3D, angles are
vectors, and must be 3D.  The vectors need not be normalized.  Base points are always 2- 
or 3D vectors.

The 'rank' field is a utility that can convey geometric information about where each ray
originates, which can be used to determine where the ray should end up if your script is
using an optimizer.  It is on by default, but you don't need it or don't intend to use
this source with an optimizer, it can be turned off by passing None to the 'rank_type'
parameter in the constructor.  If not none, this parameter conveys which type of input
distribution the rank goes with (necessary for densifying the source).  The 
'external_rank_source' parameter of the constructor allows for specifying an external
source for the rank data.  Whatever object is put here must have an attribute 'ranks'.  By 
default this parameter is None, in which case rank data will be taken from one of the
distributions.  Rank generation behavior can be called on the fly by calling the
process_rank_params() member, and feeding it these two parameters.

"""

import math
import itertools
from abc import ABC, abstractmethod
from tfrt.update import RecursivelyUpdatable

import tensorflow as tf
import numpy as np
import tensorflow_graphics.geometry.transformation.quaternion as quaternion
import tensorflow_graphics.geometry.transformation.rotation_matrix_2d as rotate_2d
PI = math.pi
COUNTER = itertools.count(0)

# ====================================================================================


class SourceBase(RecursivelyUpdatable, ABC):
    """
    Abstract implementation of a source, which defines the bare minimum interface
    that all sources should implement.
    
    Parameters
    ----------
    name : string, optional
        The name of the distribution.
    dense : bool, optional
        True if the source is dense.
    update_handles : list, optional
        A list of functions that should be called when this source is update, before
        it updates itself.  Defaults to None, in which case a default is used that is
        simply the update function of the distributions this source consumes.  If 
        anything other than None is passed to this parameter, then the default update 
        handles will NOT be added to the handler list.  But if you know that one 
        distribution needs to be updated and the other does not, you can pass this 
        parameter a list containing just the update method of the distribution that 
        needs to be updated, and
        that distribution will be updated by a call to the source's update method.
        These update methods will NOT be called if recursively_update is False.
    recursively_update : bool, optional
        If true, will call all methods in update_handles before updating itself, each
        time update is called.  Otherwise will only call the source's own update 
        method.
        
    Public read-only attributes
    ---------------------------
    name : string
        The name of the distribution.
    keys : set of strings
        The keys that can be used to index data out of this source.
    
    Public read-write attributes
    ----------------------------
    dense : bool
        True if the source is dense.
    recursively_update : bool
        If true, will call all methods in update_handles before updating itself, each
        time update is called.  Otherwise will only call the source's own update 
        method.
    update_handles : list
        A list of functions that should be called when this source is update, before
        it updates itself.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of light generated by this source
        
    Public methods
    --------------
    update()
        Will cause the source to re-calculate all of its values.  If 
        recursively_update is True, will first call all of the update methods in 
        update_handles.
    Indexing
        A source is indexable.  The numbers that define the rays are accessed by 
        indexing
        the source with any of the keys in signature.  Can also add new keys, to store
        extra data inside the source.  Typically this extra data should be 
        broadcastable to the ray data.  For example, you can add an intensity field 
        to a source which describes the intensity of each starting ray in the source, 
        and this intensity can be used by various ray operations in the ray engine.

    """

    def __init__(
        self,
        name=None,
        dense=True,
        **kwargs
    ):
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self.dense = dense
        self._fields = {}
        super().__init__(**kwargs)
            
    def _set_dimension(self, dimension):
        if dimension not in {2, 3}:
            raise ValueError("Source: dimension must be 2 or 3")
        else:
            self._dimension = dimension

    @property
    def name(self):
        return self._name
        
    @property
    def dimension(self):
        return self._dimension

    def __getitem__(self, key):
        return self._fields[key]

    def __setitem__(self, key, item):
        self._fields[key] = item
    
    @property    
    def keys(self):
        return self._fields.keys
        
    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value
        
# =======================================================================================

class ManualSource(SourceBase):
    """
    A do-nothing source which can be manually filled with data.
    
    Updating this class will do nothing by default, but update functions can be registered
    to an instance by appending functions to the self.update_handles or 
    self.post_update_handles list.
    """
    def __init__(self, dimension, **kwargs):
        self._set_dimension(dimension)
        super().__init__(**kwargs)
        
    def _generate_update_handles(self):
        return []
        
    def update(self):
        pass

# ========================================================================================

class PointSource(SourceBase):
    """
    Rays eminating from a single point, going in some specified direction(s).
    
    This source is built from one distribution: an angular distribution that is assumed to 
    be relative to the origin.  The source can move and rotate the angles defined in
    the angular distribution.
    
    Parameters
    ----------
    dimension : int scalar
        Must be either 2 or 3.  The dimension of the source.
    center : tf.float64 tensor of shape (2,)
        The x and y coordinates of the center of the source.
    central_angle : scalar tf.float64 tensor
        The angle which the center of the source faces.
    angular_distribution : a child of AngularDistributionBase
        The angular distribution that describes the angles of rays generated by the source.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of the rays generated by the source.
    start_on_center : bool, optional
        True if rays should start on the center point, for a diverging source.
        False if rays should end on the center point, for a converging source.
    ray_length : tf.float64 tensor broadcastable to (None,), optional
        The length of all generated rays.  Usually a scalar, but could possibly also be 1-D
        and have the exact length of the total number of rays generated by the source.
    rank_type : None or string, optional
        For this source, may be None to not generate ranks, or must be 'angle' to generate
        ranks that are related to the angle.  This means only that whatever generates the
        ranks must generate as many elements as there are angles.  Not currently supported
        to generate ranks like wavelengths.
    external_rank_source : None or Object, optional
        Only matters if rank_type is not None.  If None, ranks will come from   
        angular_distribution.  Otherwise, this parameter must be an object that has an  
        attribute named 'ranks' from which the ranks will be taken.
    Plus kwargs consumed by SourceBase or its inherited classes.
    
        
    Public read-only attributes
    ---------------------------
    angular_distribution : a child of AngularDistributionBase
        The angular distribution that describes the angles of rays generated by the source.
    angles : tf.float64 tensor of shape (None,)
        The angles of each ray in the distribution.  Differs from the angles described
        by the angular_distribution if the source is dense, since the angles will be
        combined with the wavelengths.
    keys : python view object
        The keys that can be used to index data out of this source.
    dimension: int
        Either 2 or 3, the dimension of the space in which the source is embedded.
    name : string
        The name of the distribution.
    Plus attribues exposed by SourceBase or its inherited classes 
        (update.RecursivelyUpdatable).
    
    Public read-write attributes
    ----------------------------
    center : tf.float64 tensor of shape (2,)
        The x and y coordinates of the center of the source.
    central_angle : scalar tf.float64 tensor
        The angle which the center of the source faces.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of the rays generated by the source.
    start_on_center : bool, optional
        True if rays should start on the center point, for a diverging source.
        False if rays should end on the center point, for a converging source.
    ray_length : tf.float64 tensor broadcastable to (None,), optional
        The length of all generated rays.
    dense : bool
        True if the source is dense.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of light generated by this source
    Plus attribues exposed by SourceBase or its inherited classes 
        (update.RecursivelyUpdatable).
        
    Public methods
    --------------
    update()
        Will cause the source to re-calculate all of its values.  If recursively_update is
        True, will first call all of the update methods in update_handles.
    Indexing
        A source is indexable.  The numbers that define the rays are accessed by indexing
        the source with any of the keys in signature.  Can also add new keys, to store
        extra data inside the source.  Typically this extra data should be broadcastable
        to the ray data.  For example, you can add an intensity field to a source which
        describes the intensity of each starting ray in the source, and this intensity
        can be used by various ray operations in the ray engine.
    process_rank_params(rank_type, external_rank_source)
        Should usually not be needed by the user, but no reason why you can't call it.
        Takes and process the kwargs of the same name defined in the constructor, could be
        used to change the rank generation behavior on the fly.

    """

    def __init__(
        self,
        dimension,
        center,
        central_angle,
        angular_distribution,
        wavelengths,
        start_on_center=True,
        ray_length=1.0,
        rank_type="angle",
        external_rank_source=None,
        **kwargs,
    ):
        self._set_dimension(dimension)
        self.center = center
        self.central_angle = central_angle
        self._angular_distribution = angular_distribution
        self._wavelengths = wavelengths
        self.start_on_center = start_on_center
        self.ray_length = ray_length
        self.process_rank_params(rank_type, external_rank_source)

        super().__init__(**kwargs)
        
    def process_rank_params(self, rank_type, external_rank_source):
        if rank_type is None:
            self._use_rank = False
            self._external_rank_source = None
            self._use_external_rank = False
        elif rank_type == "angle":
            self._use_rank = True
            if external_rank_source is not None:
                self._use_external_rank = True
                self._external_rank_source = external_rank_source
            else:
                self._use_external_rank = False
        else:
            raise ValueError("PointSource: rank_type must be None or 'angle'.")

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(2,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.bool),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ]
    )"""
    @staticmethod
    def _build_2d(
        center, central_angle, wavelengths, angles, start_on_center, ray_length
    ):
        ray_count = tf.shape(angles)
        angles = angles + central_angle
        x_start = tf.broadcast_to(center[0], ray_count)
        y_start = tf.broadcast_to(center[1], ray_count)
        x_end = x_start + ray_length * tf.cos(angles)
        y_end = y_start + ray_length * tf.sin(angles)

        if start_on_center:
            return x_start, y_start, x_end, y_end, wavelengths
        else:
            return x_end, y_end, x_start, y_start, wavelengths
            
    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(3,), dtype=tf.float64),
            tf.TensorSpec(shape=(3,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.bool),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ]
    )"""
    """
    had to remove the @tf.function decorator because of a bug in the tf.geometry module.
    """
    @staticmethod
    def _build_3d(
        center, central_angle, wavelengths, angles, start_on_center, ray_length
    ):
        center = tf.cast(center, tf.float64)
        central_angle = tf.cast(central_angle, tf.float64)
        wavelengths = tf.cast(wavelengths, tf.float64)
        angles = tf.cast(angles, tf.float64)
        ray_length = tf.cast(ray_length, tf.float64)
    
        x_axis = tf.constant((1, 0, 0), dtype=tf.float64)
        rotation = quaternion.between_two_vectors_3d(x_axis, central_angle)
        
        # there seems to be a bug in between_two_vectors_3d when doing a 180 rotation.
        # I am writing a hack solution here
        rotation = tf.where(
            tf.less(tf.reduce_max(tf.abs(rotation)), .01),
            tf.constant((0, 0, 1, 0), dtype=tf.float64),
            rotation
        )
        
        angles = quaternion.rotate(angles, rotation)
        start = tf.broadcast_to(center, tf.shape(angles))
        end = start + ray_length * angles
        
        x_start, y_start, z_start = tf.unstack(start, axis=1)
        x_end, y_end, z_end = tf.unstack(end, axis=1)

        if start_on_center:
            return x_start, y_start, z_start, x_end, y_end, z_end, wavelengths
        else:
            return x_end, y_end, z_end, x_start, y_start, z_start, wavelengths

    def _update(self):
        try:
            angles = self._angular_distribution.angles
        except(AttributeError):
            angles = self._angular_distribution.points
            
        # collect the value to use for ranks.
        if self._use_rank:
            if self._use_external_rank:
                ranks = self._external_rank_source.ranks
            else:
                ranks = self._angular_distribution.ranks
        else:
            ranks = tf.zeros_like(angles, dtype=tf.float64)
        
        # optionally densify the distributions, and check shapes.
        if self.dense:
            self._angles, self._processed_wavelengths, self._ranks = \
                self._make_dense(
                    angles,
                    self._wavelengths,
                    ranks
                )
        else:
            self._angles, self._processed_wavelengths, self._ranks = \
                self._make_undense(
                    angles,
                    self._wavelengths,
                    ranks
                )
            
        # build the source and set the fields 
        if self.dimension == 2:        
            self["x_start"], self["y_start"], self["x_end"], self["y_end"], \
                self["wavelength"] = self._build_2d(
                self.center,
                self.central_angle,
                self._processed_wavelengths,
                self._angles,
                self.start_on_center,
                self.ray_length,
            )
        else: # self.dimension == 3:
            self["x_start"], self["y_start"], self["z_start"], self["x_end"], \
                self["y_end"], self["z_end"], self["wavelength"] = self._build_3d(
                self.center,
                self.central_angle,
                self._processed_wavelengths,
                self._angles,
                self.start_on_center,
                self.ray_length,
            )
        
        # set the ranks    
        if self._use_rank:
            self["rank"] = self._ranks

    def _generate_update_handles(self):
        return [self._angular_distribution.update]

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64)
        ]
    )"""
    @staticmethod
    def _make_dense(angles, wavelengths, ranks):
        angles = tf.cast(angles, tf.float64)
        wavelengths = tf.cast(wavelengths, tf.float64)
        ranks = tf.cast(ranks, tf.float64)
        
        angle_count = tf.shape(angles)[0]
        angle_range = tf.range(tf.cast(angle_count, tf.float64), dtype=tf.float64)
        angle_gather, out_wavelengths = tf.meshgrid(
            angle_range, wavelengths
        )
        
        angle_gather = tf.cast(angle_gather, tf.int64)
        angle_gather = tf.reshape(angle_gather, (-1,))
        
        out_angles = tf.gather(angles, angle_gather, axis=0)
        out_ranks = tf.gather(ranks, angle_gather, axis=0)
        out_wavelengths = tf.reshape(out_wavelengths, (-1,))
        return out_angles, out_wavelengths, out_ranks

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64)
        ]
    )"""
    @staticmethod
    def _make_undense(angles, wavelengths, ranks):
        angles = tf.cast(angles, tf.float64)
        wavelengths = tf.cast(wavelengths, tf.float64)
        ranks = tf.cast(ranks, tf.float64)
        
        angle_shape = tf.shape(angles)[0]
        wavelength_shape = tf.shape(wavelengths)[0]
        tf.debugging.assert_equal(
            angle_shape,
            wavelength_shape,
            message=f"For un dense source, need "
            f"exactly as many angles as wavelengths.",
        )

        return angles, wavelengths, ranks
        
    @property
    def center(self):
        return self._center
        
    @center.setter
    def center(self, val):
        if self.dimension == 2:
            tf.debugging.assert_shapes(
                {val: (2,)}, 
                message="PointSource: center must be size (2,)."
            )
        else:
            tf.debugging.assert_shapes(
                {val: (3,)}, 
                message="PointSource: center must be size (3,)."
            )
        self._center = val
        
    @property
    def central_angle(self):
        return self._central_angle
        
    @central_angle.setter
    def central_angle(self, val):
        if self.dimension == 2:
            tf.debugging.assert_scalar(
                val,
                message="PointSource: central_angle must be scalar."
            )
        else:
            tf.debugging.assert_shapes(
                {val: (3,)}, 
                message="PointSource: central_angle must be size (3,)."
            )
        self._central_angle = val

    @property
    def angles(self):
        return self._angles

    @property
    def angular_distribution(self):
        return self._angular_distribution


class AngularSource(SourceBase):
    """
    Rays eminating from a multiple points, going in some specified direction(s).
    
    This source is built from two distributions: an angular distribution and a point    
    distribution both of which are assumed to be relative to the origin.  The source can 
    move and rotate the distributions' data.
    
    Parameters
    ----------
    dimension : int scalar
        Must be either 2 or 3.  The dimension of the source.
    center : tf.float64 tensor of shape (2,)
        The x and y coordinates of the center of the source.
    central_angle : scalar tf.float64 tensor
        The angle which the center of the source faces.
    angular_distribution : a child of AngularDistributionBase
        The angular distribution that describes the angles of rays generated by the source.
    base_point_distribution : a child of BasePointDistributionBase
        The point distribution that describes the locations where rays will origionate 
        from.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of the rays generated by the source.
    start_on_base : bool, optional
        True if rays should start on the base points, for a diverging source.
        False if rays should end on the base points, for a converging source.
    ray_length : tf.float64 tensor broadcastable to (None,), optional
        The length of all generated rays.  Usually a scalar, but could possibly also be 1-D
        and have the exact length of the total number of rays generated by the source.
    rank_type : None or string, optional
        For this source, may be None to not generate ranks, may be 'angle' to generate
        ranks that are related to the angles, or may be 'base_point' to generate ranks 
        that are related to the base points.  This means only that whatever generates the
        ranks must generate as many elements as there are angles or base points.  Not 
        currently supported to generate ranks like wavelengths.
    external_rank_source : None or Object, optional
        Only matters if rank_type is not None.  If None, ranks will come from   
        the selected distribution (selected by rank_type).  Otherwise, this parameter must 
        be an object that has an attribute named 'ranks' from which the ranks will be 
        taken.
    Plus kwargs consumed by SourceBase or its inherited classes.
    
        
    Public read-only attributes
    ---------------------------
    angular_distribution : a child of AngularDistributionBase
        The angular distribution that describes the angles of rays generated by the source.
    base_point_distribution : a child of BasePointDistributionBase
        The point distribution that describes the locations where rays will origionate 
        from.
    angles : tf.float64 tensor of shape (None,) or (None, 3)
        The angles of each ray in the distribution.  Differs from the angles described
        by the angular_distribution if the source is dense, since the angles will be
        combined with the wavelengths and base points.
    base_points : tf.float64 tensor of shape (None, 2) or (None, 3)
        The base point of each ray in the distribution.  Differs from the base points 
        described by the base_point_distribution if the source is dense, since the base 
        points will be combined with the wavelengths and angles.
    keys : python view object
        The keys that can be used to index data out of this source.
    dimension: int
        Either 2 or 3, the dimension of the space in which the source is embedded.
    name : string
        The name of the distribution.
    Plus attribues exposed by SourceBase or its inherited classes 
        (update.RecursivelyUpdatable).
    
    Public read-write attributes
    ----------------------------
    center : tf.float64 tensor of shape (2,)
        The x and y coordinates of the center of the source.
    central_angle : scalar tf.float64 tensor
        The angle which the center of the source faces.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of the rays generated by the source.
    start_on_center : bool, optional
        True if rays should start on the center point, for a diverging source.
        False if rays should end on the center point, for a converging source.
    ray_length : tf.float64 tensor broadcastable to (None,), optional
        The length of all generated rays.
    dense : bool
        True if the source is dense.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of light generated by this source.
    Plus attribues exposed by SourceBase or its inherited classes 
        (update.RecursivelyUpdatable).
        
    Public methods
    --------------
    update()
        Will cause the source to re-calculate all of its values.  If recursively_update is
        True, will first call all of the update methods in update_handles.
    Indexing
        A source is indexable.  The numbers that define the rays are accessed by indexing
        the source with any of the keys in signature.  Can also add new keys, to store
        extra data inside the source.  Typically this extra data should be broadcastable
        to the ray data.  For example, you can add an intensity field to a source which
        describes the intensity of each starting ray in the source, and this intensity
        can be used by various ray operations in the ray engine.
    process_rank_params(rank_type, external_rank_source)
        Should usually not be needed by the user, but no reason why you can't call it.
        Takes and process the kwargs of the same name defined in the constructor, could be
        used to change the rank generation behavior on the fly.

    """

    def __init__(
        self,
        dimension,
        center,
        central_angle,
        angular_distribution,
        base_point_distribution,
        wavelengths,
        start_on_base=True,
        ray_length=1.0,
        rank_type="angle",
        external_rank_source=None,
        **kwargs
    ):
        self._set_dimension(dimension)
        self.center = center
        self.central_angle = central_angle
        self._angular_distribution = angular_distribution
        self._base_point_distribution = base_point_distribution
        self._wavelengths = wavelengths
        self.start_on_base=start_on_base
        self.ray_length = ray_length
        self.process_rank_params(rank_type, external_rank_source)
        
        super().__init__(**kwargs)
    
    def process_rank_params(self, rank_type, external_rank_source):
        if rank_type is None:
            self._use_rank = False
            self._use_external_rank = False
            self._external_rank_source = None
            self._use_angle_rank = False
        elif rank_type == "angle" or rank_type == "base_point":
            self._use_rank = True
            if external_rank_source is not None:
                self._use_external_rank = True
                self._external_rank_source = external_rank_source
            else:
                self._use_external_rank = False
            if rank_type == "angle":
                self._use_angle_rank = True
            else: #rank_type == "base_point"
                self._use_angle_rank = False
        else:
            raise ValueError("PointSource: rank_type must be None, 'angle', or"
                " 'base_point'."
            )

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(2,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.bool),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float64)
        ]
    )"""
    @staticmethod
    def _build_2d(
        center,
        central_angle,
        wavelengths,
        angles,
        start_on_base,
        ray_length,
        base_points
    ):
        # cast everything to float64, since we don't get to use the function decorator
        center = tf.cast(center, tf.float64)
        central_angle = tf.cast(central_angle, tf.float64)
        wavelengths = tf.cast(wavelengths, tf.float64)
        angles = tf.cast(angles, tf.float64)
        ray_length = tf.cast(ray_length, tf.float64)
        base_points = tf.cast(base_points, tf.float64)
        
        angles = angles + central_angle
        rotation = rotate_2d.from_euler(tf.reshape(central_angle, (1,)))
        base_points = rotate_2d.rotate(base_points, rotation)
        
        start = center + base_points
        x_start, y_start = tf.unstack(start, axis=1)
        x_end = x_start + ray_length * tf.cos(angles)
        y_end = y_start + ray_length * tf.sin(angles)

        if start_on_base:
            return x_start, y_start, x_end, y_end, wavelengths
        else:
            return x_end, y_end, x_start, y_start, wavelengths
    
    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(2,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.bool),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64)
        ]
    )"""  
    """
    had to remove the @tf.function decorator because of a bug in the tf.geometry module.
    """      
    @staticmethod
    def _build_3d(
        center,
        central_angle,
        wavelengths,
        angles,
        start_on_base,
        ray_length,
        base_points
    ):
        # cast everything to float64, since we don't get to use the function decorator
        center = tf.cast(center, tf.float64)
        central_angle = tf.cast(central_angle, tf.float64)
        wavelengths = tf.cast(wavelengths, tf.float64)
        angles = tf.cast(angles, tf.float64)
        ray_length = tf.cast(ray_length, tf.float64)
        base_points = tf.cast(base_points, tf.float64)
        
        # rotate the angles
        x_axis = tf.constant((1, 0, 0), dtype=tf.float64)
        rotation = quaternion.between_two_vectors_3d(x_axis, central_angle)
        # there seems to be a bug in between_two_vectors_3d when doing a 180 rotation.
        # I am writing a hack solution here
        rotation = tf.where(
            tf.less(tf.reduce_max(tf.abs(rotation)), .01),
            tf.constant((0, 0, 1, 0), dtype=tf.float64),
            rotation
        )
        angles = quaternion.rotate(angles, rotation)
        
        # rotate the base points
        # base points can be either 3D or 2D.  If 2D, assume in y-z plane and convert.
        bp_shape = tf.shape(base_points)
        if tf.equal(bp_shape[1], 2):
            base_points = tf.concat(
                [tf.zeros((bp_shape[0], 1), dtype=tf.float64), base_points], 
                1
            )
        base_points = quaternion.rotate(base_points, rotation)
        
        # calculate the ray endpoints
        start = center + base_points
        end = start + ray_length * angles
        
        x_start, y_start, z_start = tf.unstack(start, axis=1)
        x_end, y_end, z_end = tf.unstack(end, axis=1)
        if start_on_base:
            return x_start, y_start, z_start, x_end, y_end, z_end, wavelengths
        else:
            return x_end, y_end, z_end, x_start, y_start, z_start, wavelengths
            
    def _update(self):
        try:
            angles = self._angular_distribution.angles
        except(AttributeError):
            angles = self._angular_distribution.points
            
        # collect the value to use for ranks.
        if self._use_rank:
            if self._use_external_rank:
                if self._use_angle_rank:
                    angle_ranks = self._external_rank_source.ranks
                    bp_ranks = self._base_point_distribution.ranks
                else:
                    angle_ranks = self._angular_distribution.ranks
                    bp_ranks = self._external_rank_source.ranks
            else:
                angle_ranks = self._angular_distribution.ranks
                bp_ranks = self._base_point_distribution.ranks
        else:
            angle_ranks = tf.zeros_like(angles, dtype=tf.float64)
            bp_ranks = tf.zeros_like(self._base_points, dtype=tf.float64)
        
        # optionally densify the distributions, and shape check
        if self.dense:
            self._angles, self._base_points, self._processed_wavelengths, \
                self._angle_ranks, self._point_ranks = self._make_dense(
                    angles,
                    self._base_point_distribution.points,
                    self._wavelengths,
                    angle_ranks,
                    bp_ranks
                )
        else:
            self._angles, self._base_points, self._processed_wavelengths, \
                self._angle_ranks, self._point_ranks = self._make_undense(
                    angles,
                    self._base_point_distribution.points,
                    self._wavelengths,
                    angle_ranks,
                    bp_ranks
                )
                
        if self.dimension == 2:
            self["x_start"], self["y_start"], self["x_end"], self["y_end"], \
                self["wavelength"] = self._build_2d(
                    self.center,
                    self.central_angle,
                    self._processed_wavelengths,
                    self._angles,
                    self.start_on_base,
                    self.ray_length,
                    self._base_points
                )
        else: #self.dimension == 3:
            self["x_start"], self["y_start"], self["z_start"], self["x_end"], \
                self["y_end"], self["z_end"], self["wavelength"] = self._build_3d(
                    self.center,
                    self.central_angle,
                    self._processed_wavelengths,
                    self._angles,
                    self.start_on_base,
                    self.ray_length,
                    self._base_points
                )
                
        # set the ranks
        if self._use_rank:
            if self._use_angle_rank:
                self["rank"] = self._angle_ranks
            else:
                self["rank"] = self._point_ranks
        
    def _generate_update_handles(self):
        return [self._angular_distribution.update, self._base_point_distribution.update]

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(None, None), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64)
        ]
    )"""
    @staticmethod
    def _make_dense(
        angles,
        base_points,
        wavelengths,
        angle_ranks,
        base_point_ranks
    ):
        angles = tf.cast(angles, tf.float64)
        base_points = tf.cast(base_points, tf.float64)
        wavelengths = tf.cast(wavelengths, tf.float64)
        angle_ranks = tf.cast(angle_ranks, tf.float64)
        base_point_ranks = tf.cast(base_point_ranks, tf.float64)
    
        angle_count = tf.shape(angles)[0]
        angle_range = tf.range(tf.cast(angle_count, tf.float64), dtype=tf.float64)
        base_count = tf.shape(base_points)[0]
        base_range = tf.range(tf.cast(base_count, tf.float64), dtype=tf.float64)
        
        angle_gather, wavelengths, base_gather = tf.meshgrid(
            angle_range, wavelengths, base_range
        )
        angle_gather = tf.cast(angle_gather, tf.int64)
        angle_gather = tf.reshape(angle_gather, (-1,))
        base_gather = tf.cast(base_gather, tf.int64)
        base_gather = tf.reshape(base_gather, (-1,))
        wavelengths = tf.reshape(wavelengths, (-1,))
        
        angles = tf.gather(angles, angle_gather, axis=0)
        base_points = tf.gather(base_points, base_gather, axis=0)
        angle_ranks = tf.gather(angle_ranks, angle_gather, axis=0)
        base_point_ranks = tf.gather(base_point_ranks, base_gather, axis=0)
        
        return angles, base_points, wavelengths, angle_ranks, base_point_ranks

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(None, None), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64)
        ]
    )"""
    @staticmethod
    def _make_undense(
        angles,
        base_points,
        wavelengths,
        angle_ranks,
        base_point_ranks
    ):
        angles = tf.cast(angles, tf.float64)
        base_points = tf.cast(base_points, tf.float64)
        wavelengths = tf.cast(wavelengths, tf.float64)
        angle_ranks = tf.cast(angle_ranks, tf.float64)
        base_point_ranks = tf.cast(base_point_ranks, tf.float64)
    
        angle_count = tf.shape(angles)[0]
        wavelength_count = tf.shape(wavelengths)[0]
        base_count = tf.shape(base_points)[0]
        tf.debugging.assert_equal(
            angle_count,
            wavelength_count,
            message=f"For un dense source, need "
            f"exactly as many angles as wavelengths.",
        )
        tf.debugging.assert_equal(
            base_count,
            wavelength_count,
            message=f"For un dense source, need "
            f"exactly as many base points as wavelengths.",
        )
        return angles, base_points, wavelengths, angle_ranks, base_point_ranks
            
    @property
    def center(self):
        return self._center
        
    @center.setter
    def center(self, val):
        if self.dimension == 2:
            tf.debugging.assert_shapes(
                {val: (2,)}, 
                message="PointSource: center must be size (2,)."
            )
        else:
            tf.debugging.assert_shapes(
                {val: (3,)}, 
                message="PointSource: center must be size (3,)."
            )
        self._center = val
        
    @property
    def central_angle(self):
        return self._central_angle
        
    @central_angle.setter
    def central_angle(self, val):
        if self.dimension == 2:
            tf.debugging.assert_scalar(
                val,
                message="PointSource: central_angle must be scalar."
            )
        else:
            tf.debugging.assert_shapes(
                {val: (3,)}, 
                message="PointSource: central_angle must be size (3,)."
            )
        self._central_angle = val

    @property
    def angles(self):
        return self._angles
        
    @property
    def angular_distribution(self):
        return self._angular_distribution

    @property
    def base_points(self):
        return self._base_points

    @property
    def base_point_distribution(self):
        return self._base_point_distribution


class AperatureSource(SourceBase):
    """
    A set of rays that span two sets of endpoints.
    
    This source does not use an angular distribution, and instead makes rays between
    two sets of points.  Useful if you know your input light is bounded by two
    apertures, and you don't want to calculate angles.
    
    AperatureSource takes two point distributions.  The most important thing that makes
    AperatureSource different than PointSource or AngularSource is that AperatureSource
    treats the coordinates of the point distributions as absolute rather than relative; 
    AperatureSource does not have a center or central angle.
    
    For the 2D case, a tfrt.distributions contains several aperature point distributions
    that go well with this class.  For the 3D case, things are a little more complicated.
    The identity of this class is that it doesn't care about a center or aiming, so I don't
    want to add parameters like that to the 2D distributions to accomodate AperatureSource.
    But then there aren't any convenient distributions to use for the 3D AperatureSource.
    So my recommendation if you want to use AperatureSource in 3D is to create your points
    in a pyvista.PolyData object, either created with pyvista or created in an external 
    CAD program and loaded through pyvista.  The PolyData objects can be fed to a
    distributions.ManualBasePointDistribution whose from_mesh kwarg was set to True.  But
    you will still need to build your own rank handler if you need to use ranks.    
        
    Parameters
    ----------
    dimension : int scalar
        Must be either 2 or 3.  The dimension of the source.
    start_point_distribution : a child of BasePointDistributionBase
        The base point distribution that determines where rays generated by the source
        start.
    end_point_distribution : a child of BasePointDistributionBase
        The base point distribution that determines where rays generated by the source
        end.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of the rays generated by the source.
    rank_type : None or string, optional
        For this source, may be None to not generate ranks, may be 'start' to generate
        ranks that are related to the start points, or may be 'end' to generate ranks 
        that are related to the end points.  This means only that whatever generates the
        ranks must generate as many elements as there are points.  Not currently supported 
        to generate ranks like wavelengths.
    external_rank_source : None or Object, optional
        Only matters if rank_type is not None.  If None, ranks will come from   
        the selected distribution (selected by rank_type).  Otherwise, this parameter must 
        be an object that has an attribute named 'ranks' from which the ranks will be 
        taken.
    Plus kwargs consumed by SourceBase or its inherited classes.
        
    Public read-only attributes
    ---------------------------
    start_point_distribution : a child of BasePointDistributionBase
        The base point distribution that determines where rays generated by the source
        start.
    start_points : a 2-tuple of tf.float64 tensor of shape (None,)
        The x and y coordinates of the start point of each ray in the distribution.  
        Differs from the base points described by the start_point_distribution if the 
        source is dense, since the start points will be combined with the wavelengths and 
        end points.
    end_point_distribution : a child of BasePointDistributionBase
        The base point distribution that determines where rays generated by the source
        end.
    end_points : a 2-tuple of tf.float64 tensor of shape (None,)
        The x and y coordinates of the end point of each ray in the distribution.  
        Differs from the base points described by the end_point_distribution if the 
        source is dense, since the end points will be combined with the wavelengths and 
        start points.
    keys : python view object
        The keys that can be used to index data out of this source.
    dimension: int
        Either 2 or 3, the dimension of the space in which the source is embedded.
    Plus attribues exposed by SourceBase or its inherited classes 
        (update.RecursivelyUpdatable).
    
    Public read-write attributes
    ----------------------------
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of the rays generated by the source.
    dense : bool
        True if the source is dense.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of light generated by this source.
    Plus attribues exposed by SourceBase or its inherited classes 
        (update.RecursivelyUpdatable).
        
    Public methods
    --------------
    update()
        Will cause the source to re-calculate all of its values.  If recursively_update is
        True, will first call all of the update methods in update_handles.
    Indexing
        A source is indexable.  The numbers that define the rays are accessed by indexing
        the source with any of the keys in signature.  Can also add new keys, to store
        extra data inside the source.  Typically this extra data should be broadcastable
        to the ray data.  For example, you can add an intensity field to a source which
        describes the intensity of each starting ray in the source, and this intensity
        can be used by various ray operations in the ray engine.
    process_rank_params(rank_type, external_rank_source)
        Should usually not be needed by the user, but no reason why you can't call it.
        Takes and process the kwargs of the same name defined in the constructor, could be
        used to change the rank generation behavior on the fly.
        
    """

    def __init__(
        self,
        dimension,
        start_point_distribution,
        end_point_distribution,
        wavelengths,
        rank_type="start",
        external_rank_source=None,
        **kwargs
    ):
        self._set_dimension(dimension)
        self._start_point_distribution = start_point_distribution
        self._end_point_distribution = end_point_distribution
        self._wavelengths = wavelengths
        self.process_rank_params(rank_type, external_rank_source)
        super().__init__(**kwargs)
        
    def process_rank_params(self, rank_type, external_rank_source):
        if rank_type is None:
            self._use_rank = False
            self._external_rank_source = None
            self._use_external_rank = False
            self._use_start_rank = False
        elif rank_type == "start" or rank_type == "end":
            self._use_rank = True
            if external_rank_source is not None:
                self._use_external_rank = True
                self._external_rank_source = external_rank_source
            else:
                self._use_external_rank = False
            if rank_type == "start":
                self._use_start_rank = True
            else: #rank_type == "end"
                self._use_start_rank = False
        else:
            raise ValueError("AperatureSource: rank_type must be None, 'start', or"
                " 'end'."
            )
        
    def _update(self):
        start_points = self._start_point_distribution.points
        start_points_shape = tf.shape(start_points)
        end_points = self._end_point_distribution.points
        end_points_shape = tf.shape(end_points)
    
        # collect the value to use for ranks.
        if self._use_rank:
            if self._use_external_rank:
                if self._use_start_rank:
                    start_ranks = self._external_rank_source.ranks
                    end_ranks = self._end_point_distribution.ranks
                else:
                    start_ranks = self._start_point_distribution.ranks
                    end_ranks = self._external_rank_source.ranks
            else:
                start_ranks = self._start_point_distribution.ranks
                end_ranks = self._end_point_distribution.ranks
        else:
            start_ranks = tf.zeros_like(start_points, dtype=tf.float64)
            end_ranks = tf.zeros_like(end_points, dtype=tf.float64)
        
        # provide the option of expanding from 2D to 3D.  Assume they are in the y-z plane.
        if self.dimension == 3:
            if tf.equal(start_points_shape[1], 2):
                start_points = tf.concat(
                    [
                        tf.zeros((start_points_shape[0], 1), dtype=tf.float64), 
                        start_points
                    ], 
                    1
                )
            if tf.equal(end_points_shape[1], 2):
                end_points = tf.concat(
                    [
                        tf.zeros((end_points_shape[0], 1), dtype=tf.float64), 
                        end_points
                    ], 
                    1
                )
        
        # optionally densify the distributions, and shape check
        if self.dense:
            self._start_points, self._end_points, self["wavelength"], self._start_ranks, \
                self._end_ranks = self._make_dense(
                    start_points,
                    end_points,
                    self._wavelengths,
                    start_ranks,
                    end_ranks
                )
        else:
            self._start_points, self._end_points, self["wavelength"], self._start_ranks, \
                self._end_ranks = self._make_undense(
                    start_points,
                    end_points,
                    self._wavelengths,
                    start_ranks,
                    end_ranks
                )
                
        # set the fields
        if self.dimension == 2:
            self["x_start"], self["y_start"] = tf.unstack(self._start_points, axis=1)
            self["x_end"], self["y_end"] = tf.unstack(self._end_points, axis=1)
        else: #self.dimension == 3
            self["x_start"], self["y_start"], self["z_start"] = tf.unstack(
                self._start_points, axis=1
            )
            self["x_end"], self["y_end"], self["z_end"] = tf.unstack(
                self._end_points, axis=1
            )
            
        # set the ranks
        if self._use_rank:
            if self._use_start_rank:
                self["rank"] = self._start_ranks
            else:
                self["rank"] = self._end_ranks
            
    def _generate_update_handles(self):
        return [
            self.start_point_distribution.update,
            self.end_point_distribution.update
        ]

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64)
        ]
    )"""
    @staticmethod
    def _make_dense(start, end, wavelengths, start_ranks, end_ranks):
        start = tf.cast(start, tf.float64)
        end = tf.cast(end, tf.float64)
        wavelengths = tf.cast(wavelengths, tf.float64)
        start_ranks = tf.cast(start_ranks, tf.float64)
        end_ranks = tf.cast(end_ranks, tf.float64)
    
        start_count = tf.shape(start)[0]
        start_range = tf.range(tf.cast(start_count, tf.float64), dtype=tf.float64)
        end_count = tf.shape(end)[0]
        end_range = tf.range(tf.cast(end_count, tf.float64), dtype=tf.float64)
        
        start_gather, wavelengths, end_gather = tf.meshgrid(
            start_range, wavelengths, end_range
        )
        start_gather = tf.cast(start_gather, tf.int64)
        start_gather = tf.reshape(start_gather, (-1,))
        end_gather = tf.cast(end_gather, tf.int64)
        end_gather = tf.reshape(end_gather, (-1,))
        wavelengths = tf.reshape(wavelengths, (-1,))
        
        start = tf.gather(start, start_gather, axis=0)
        end = tf.gather(end, end_gather, axis=0)
        start_ranks = tf.gather(start_ranks, start_gather, axis=0)
        end_ranks = tf.gather(end_ranks, end_gather, axis=0)
        
        return start, end, wavelengths, start_ranks, end_ranks

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=None, dtype=tf.float64)
        ]
    )"""
    @staticmethod
    def _make_undense(start, end, wavelengths, start_ranks, end_ranks):
        start = tf.cast(start, tf.float64)
        end = tf.cast(end, tf.float64)
        wavelengths = tf.cast(wavelengths, tf.float64)
        start_ranks = tf.cast(start_ranks, tf.float64)
        end_ranks = tf.cast(end_ranks, tf.float64)
        
        start_count = tf.shape(start)[0]
        wavelength_count = tf.shape(wavelengths)[0]
        end_count = tf.shape(end)[0]
        tf.debugging.assert_equal(
            start_count,
            wavelength_count,
            message=f"For un dense source, need "
            f"exactly as many start points as wavelengths.",
        )
        tf.debugging.assert_equal(
            end_count,
            wavelength_count,
            message=f"For un dense source, need "
            f"exactly as many end points as wavelengths.",
        )
        return start, end, wavelengths, start_ranks, end_ranks

    @property
    def start_points(self):
        return self._start_points

    @property
    def end_points(self):
        return self._end_points

    @property
    def start_point_distribution(self):
        return self._start_point_distribution

    @property
    def end_point_distribution(self):
        return self._end_point_distribution
