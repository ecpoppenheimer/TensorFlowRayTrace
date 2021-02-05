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
import pickle
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
        extra_fields={},
        standard_domains=set(),
        dense=True,
        always_resize=False,
        **kwargs
    ):
        self.extra_fields = extra_fields
        self._standard_domains = standard_domains | {"whole", "wavelength"}
        self.dense = dense
        self._fields = {}
        self._domain_gathers = {}
        self._domain_sizes = {}
        self._needs_resize = True
        self.always_resize = always_resize
        super().__init__(**kwargs)
            
    def _set_dimension(self, dimension):
        if dimension not in {2, 3}:
            raise ValueError("Source: dimension must be 2 or 3")
        else:
            self._dimension = dimension
    
    @staticmethod        
    def validate_extra_fields(extra_fields):
        for field, args in extra_fields.items():
            if type(field) is not str:
                raise ValueError("Source extra fields: keys must be strings.")
            arg_count = len(args)
            if arg_count not in (2, 3):
                raise ValueError(
                    "Source extra fields: every entry must be either a 2-tuple of "
                    "(domain, value), or a 3-tuple of (domain, object, attribute)."
                )
                    
    def make_vars(self, internal_vars):
        expanded_vars = {}
        for varname, item in internal_vars.items():
            domain, var = item
            if self.dense:
                var = tf.gather(var, self._domain_gathers[domain], axis=0)
            elif tf.rank(var) < 2:
                # if not dense, we can try broadcasting to enable us to properly shape
                # scalars and such.
                var = tf.broadcast_to(var, (self._domain_sizes["whole"],))
                
            expanded_vars[varname] = var
        return expanded_vars
    
    def resize(self):
        self._needs_resize = True
        
    def _resize(self):
        self._needs_resize = False
        def add_shape(domain_sizes, domain, value):
            # check the shape of value, and add it to domain_sizes
            try:
                shape = tf.shape(value).numpy()
            except Exception as e:
                raise ValueError(
                    f"Source resize: could not obtain shape of internal variable {name}."
                ) from e
            
            # value could possibly be scalar, which is ok, but needs a special case.
            try:
                size = shape[0]
            except:
                size = 1
                
            try:
                domain_sizes[domain].append(size)
            except(KeyError):
                domain_sizes[domain] = [size]
        
        all_domain_sizes = {}
        # get the size of each of the internal vars, and store it in a dict by domain.
        for name, items in self._internal_vars().items():
            domain, value = items
            add_shape(all_domain_sizes, domain, value)
        
        # get the size of each extra field, and store it in the same dict as above.        
        for field, items in self._extra_fields.items():
            if len(items) == 2:
                domain, value = items
            else: #len(items) must be 3, this is already checked by validate_extra_fields
                domain, cls, attrb = items
                value = getattr(cls, attrb)
            add_shape(all_domain_sizes, domain, value)
        
        self._domain_sizes = {}
        # compress all_domain_sizes down into a single size for each domain
        for domain, sizes in all_domain_sizes.items():
            # ensure that the sizes in a domain are all broadcastable (the same, or 1)
            size = set(sizes)
            if size == {1}:
                self._domain_sizes[domain] = 1
            else:
                size -= {1}
                if len(size) == 1:
                    self._domain_sizes[domain] = size.pop()
                else:
                    raise ValueError(
                        f"Source resize: found incompatible shapes in the same domain."
                    )

        # build the gather indices for each domain:
        # I am creating a list of the domains here because I need to be absolutely sure
        # that I preserve the order of domains for the next two steps.
        if self.dense:
            domain_list = list(self._domain_sizes.keys())
            try:
                # need to treat whole separately, and not generate a domain gather for it
                domain_list.remove("whole")
            except(ValueError):
                # don't do anything if whole wasn't found
                pass
            range_list = [tf.range(self._domain_sizes[domain]) for domain in domain_list]
            gather_list = [tf.reshape(x, (-1,)) for x in tf.meshgrid(*range_list)]
            self._domain_gathers = {
                domain: gather for domain, gather in zip(domain_list, gather_list)
            }
        else:
            self._domain_gathers = {}
        
        if self.dense:
            # if dense, compute the size of whole
            self._domain_sizes["whole"] = tf.reduce_prod([
                self._domain_sizes[domain] for domain in domain_list]
            ).numpy()
        else:
            # if undense, ensure that all the domain sizes are the same        
            var_size = 1
            for size in self._domain_sizes.values():
                if size == 1:
                    # ok
                    continue
                if var_size == 1:
                    # if the var_size was one and the current one isn't, then this size
                    # should be the size to compare everything to
                    var_size = size
                if var_size != size:
                    # will be true only when var_size was not 1 and size != var_size
                    raise ValueError(
                        "Source resize: found incompatibly sized variables with an "
                        f"undense source."
                    )
            self._domain_sizes["whole"] = var_size
                
    def publish_extra_fields(self):
        for field, items in self._extra_fields.items():
            # dereference the class attribute, if desired
            if len(items) == 2:
                domain, raw_value = items
            else:
                domain, cls, attrb = items
                try:
                    raw_value = cls[attrb]
                except(TypeError, KeyError):
                    raw_value = getattr(cls, attrb)
            
            # if value is a function, evaluate it    
            try:
                value = raw_value()
            except(TypeError):
                value = raw_value
            
            # broadcast to the correct size, unless the rank is large, in which case
            # it had better already be the correct size.    
            if tf.rank(value) < 2:
                value = tf.broadcast_to(value, (self._domain_sizes[domain],))
            
            # may need to gather the field    
            if domain != "whole" and self.dense:
                value = tf.gather(value, self._domain_gathers[domain], axis=0)
                
            self[field] = value                
            
    def _update(self):
        if self._needs_resize or self.always_resize:
            self._resize()
        self._internal_update(self.make_vars(self._internal_vars()))
        self.publish_extra_fields()
        
    def snapshot(self, do_update=True):
        if do_update:
            self.update()
        return {field: tf.convert_to_tensor(data) for field, data in self.items()}
            
    @abstractmethod
    def _internal_vars(self):
        raise NotImplementedError
        
    @abstractmethod
    def _internal_update(self, expanded_vars):
        raise NotImplementedError
        
    @property
    def extra_fields(self):
        return self._extra_fields
        
    @extra_fields.setter
    def extra_fields(self, val):
        self.validate_extra_fields(val)
        self._extra_fields = val
            
    @property
    def standard_domains(self):
        return self._standard_domains
        
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
    def items(self):
        return self._fields.items
        
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
        
    def _internal_vars(self):
        return {}
        
    def _internal_update(self, expanded_vars):
        pass

# ========================================================================================
        
class RotationBase:
    """
    Class to handle rotation stuff common to PointSource and AngularSource.
    
    In 2D, rotation angles are just scalar angles, and the value of angle_type is 
    ignored.  In 3D, rotations are a lot more difficult and I have had difficulty
    engineering a more convenient interface.  Sorry.
    
    You can choose to use angle_type = 'vector', in which case the central_angle is
    interpreted as a vector and the source is oriented along it.  I have found that
    this mode can induce unwanted roll in the source, I believe as an unavoidable problem
    of underspecification (there are many rotations that rotate one vector into another,
    but not preserving orientation in the plane of the vector itself.)
    To avoid this problem, may set angle_type to 'quaternion', in which case the 
    elements in central_angle must be the quaternion that performs the desired
    rotation.  Rotations are terrible, so it is up to the user to compose the
    quaternion themselves.  Sorry.
    """
    def __init__(self, central_angle, angle_type):
        if angle_type == "vector" or angle_type == "quaternion":
            self._angle_type = angle_type
        else:
            raise ValueError("Source: angle_type must be 'vector' or 'quaternion'.")
        self.central_angle = central_angle
        
    @property
    def central_angle(self):
        return self._central_angle
        
    @central_angle.setter
    def central_angle(self, val):
        val = tf.cast(val, tf.float64)
        if self.dimension == 2:
            if val.shape != ():
                raise ValueError("Source: central_angle must be scalar.")
            else:
                self._central_angle = val
        else:
            if self._angle_type == "vector":
                if val.shape != (3,):
                    raise ValueError("Source: central_angle must be size (3,).")
                else:
                    self._central_angle = quaternion.between_two_vectors_3d(
                        self._x_axis,
                        val
                    )
                    # there seems to be a bug in between_two_vectors_3d when doing
                    # a 180 rotation.
                    # I am writing a hack solution here
                    self._central_angle = tf.where(
                        tf.less(tf.reduce_max(tf.abs(self._central_angle)), .01),
                        tf.constant((0, 0, 1, 0), dtype=tf.float64),
                        self._central_angle
                    )
            else:
                if val.shape != (4,):
                    raise ValueError("Source: central_angle must be size (4,).")
                else:
                    self._central_angle = val
            
    def _rotate_angles(self, angles):
        if self._dimension == 2:
            return angles + self._central_angle
        else:
            return quaternion.rotate(angles, self._central_angle)
        
    def _rotate_points(self, points):
        if self._dimension == 2:
            rotation = rotate_2d.from_euler(tf.reshape(self._central_angle, (1,)))
            return rotate_2d.rotate(points, rotation)
        else:
            # rotate the base points
            # base points can be either 3D or 2D.  If 2D, assume in y-z plane and convert.
            p_shape = tf.shape(points)
            if tf.equal(p_shape[1], 2):
                points = tf.concat(
                    [tf.zeros((p_shape[0], 1), dtype=tf.float64), points], 
                    1
                )
            return quaternion.rotate(points, self._central_angle)
    
    _x_axis = tf.constant((1, 0, 0), dtype=tf.float64)

# ========================================================================================

class PointSource(SourceBase, RotationBase):
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
    central_angle : tf.float64 tensor
        The angle which the center of the source faces.  See RotationBase for more details.
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
    angle_type : str, optional
        Defaults to 'vector', in which case the central_angle is interpreted as a vector
        and the source will be rotated to use this vector as its normal.  I have found that
        this mode can induce unwanted roll in the source, I believe as an unavoidable problem
        of underspecification (there are many rotations that rotate one vector into another,
        but not preserving orientation in the plane of the vector itself.)
        To avoid this problem, may set angle_type to 'quaternion', in which case the 
        elements in central_angle must be the quaternion that performs the desired
        rotation.  Rotations are terrible, so it is up to the user to compose the
        quaternion themselves.  Sorry.
        
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
        angle_type="vector",
        **kwargs,
    ):
        self._set_dimension(dimension)
        RotationBase.__init__(self, central_angle, angle_type)
        self.center = center
        self._angular_distribution = angular_distribution
        if wavelengths is not None:
            wavelengths = tf.cast(wavelengths, tf.float64)
        self._wavelengths = wavelengths
        self.start_on_center = start_on_center
        self.ray_length = ray_length

        SourceBase.__init__(self, standard_domains={"angle"}, **kwargs)
        
    def _internal_update(self, expanded_vars):
        angles = expanded_vars["angles"]
        center = self._center
        ray_length = self.ray_length
        
        # perform rotations about central_angle
        angles = self._rotate_angles(angles)
        
        if self.dimension == 2:
            ray_count = tf.shape(angles)
            x_start = tf.broadcast_to(center[0], ray_count)
            y_start = tf.broadcast_to(center[1], ray_count)
            x_end = x_start + ray_length * tf.cos(angles)
            y_end = y_start + ray_length * tf.sin(angles)

            
        else: # self.dimension == 3
            start = tf.broadcast_to(center, tf.shape(angles))
            end = start + ray_length * angles
            
            x_start, y_start, z_start = tf.unstack(start, axis=1)
            x_end, y_end, z_end = tf.unstack(end, axis=1)
            
            if self.start_on_center:
                self["z_start"] = z_start
                self["z_end"] = z_end
            else:
                self["z_start"] = z_end
                self["z_end"] = z_start
        
        if self.start_on_center:
            self["x_start"] = x_start
            self["y_start"] = y_start
            self["x_end"] = x_end
            self["y_end"] = y_end
        else:
            self["x_start"] = x_end
            self["y_start"] = y_end
            self["x_end"] = x_start
            self["y_end"] = y_start
            
        try:
            self["wavelength"] = expanded_vars["wavelengths"]
        except(KeyError):
            # do nothing if wavelength was not defined
            pass
            
    def _internal_vars(self):
        try:
            angles = self._angular_distribution.angles
        except(AttributeError):
            angles = self._angular_distribution.points
            
        output = {
            "angles": ("angle", angles),
        }
        if self._wavelengths is not None:    
            output["wavelengths"] = ("wavelength", self._wavelengths)
        return output
        
    def _generate_update_handles(self):
        return [self._angular_distribution.update]
        
    @property
    def center(self):
        return self._center
        
    @center.setter
    def center(self, val):
        val = tf.cast(val, tf.float64)
        if self.dimension == 2:
            if val.shape != ():
                raise ValueError("PointSource: center must be size (2,).")
        else:
            if val.shape != (3,):
                raise ValueError("PointSource: center must be size (3,).")
        self._center = val

    @property
    def angles(self):
        return self._angles

    @property
    def angular_distribution(self):
        return self._angular_distribution

# ========================================================================================

class AngularSource(SourceBase, RotationBase):
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
    central_angle : tf.float64 tensor
        The angle which the center of the source faces.  See RotationBase for more details.
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
    angle_type : str, optional
        Defaults to 'vector', in which case the central_angle is interpreted as a vector
        and the source will be rotated to use this vector as its normal.  I have found that
        this mode can induce unwanted roll in the source, I believe as an unavoidable problem
        of underspecification (there are many rotations that rotate one vector into another,
        but not preserving orientation in the plane of the vector itself.)
        To avoid this problem, may set angle_type to 'quaternion', in which case the 
        elements in central_angle must be the quaternion that performs the desired
        rotation.  Rotations are terrible, so it is up to the user to compose the
        quaternion themselves.  Sorry.
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
        angle_type="vector",
        **kwargs
    ):
        self._set_dimension(dimension)
        RotationBase.__init__(self, central_angle, angle_type)
        self.center = center
        self._angular_distribution = angular_distribution
        self._base_point_distribution = base_point_distribution
        if wavelengths is not None:
            wavelengths = tf.cast(wavelengths, tf.float64)
        self._wavelengths = wavelengths
        self.start_on_base=start_on_base
        self.ray_length = ray_length
        
        SourceBase.__init__(self, standard_domains={"base_point", "angle"}, **kwargs)
            
    def _internal_update(self, expanded_vars):
        angles = expanded_vars["angles"]
        base_points = expanded_vars["base_points"]
        center = self._center
        ray_length = self.ray_length
        
        # perform rotations about central_angle
        angles = self._rotate_angles(angles)
        base_points = self._rotate_points(base_points)
    
        if self._dimension == 2:
            start = center + base_points
            x_start, y_start = tf.unstack(start, axis=1)
            x_end = x_start + ray_length * tf.cos(angles)
            y_end = y_start + ray_length * tf.sin(angles)
            
        else: #self._dimension == 3
            # calculate the ray endpoints
            start = center + base_points
            end = start + ray_length * angles
            
            x_start, y_start, z_start = tf.unstack(start, axis=1)
            x_end, y_end, z_end = tf.unstack(end, axis=1)
            
            if self.start_on_base:
                self["z_start"] = z_start
                self["z_end"] = z_end
            else:
                self["z_start"] = z_end
                self["z_end"] = z_start
        
        if self.start_on_base:
            self["x_start"] = x_start
            self["y_start"] = y_start
            self["x_end"] = x_end
            self["y_end"] = y_end
        else:
            self["x_start"] = x_end
            self["y_start"] = y_end
            self["x_end"] = x_start
            self["y_end"] = y_start
            
        try:
            self["wavelength"] = expanded_vars["wavelengths"]
        except(KeyError):
            # do nothing if wavelength was not defined
            pass
        
    def _internal_vars(self):
        try:
            angles = self._angular_distribution.angles
        except(AttributeError):
            angles = self._angular_distribution.points
            
        output = {
            "angles": ("angle", angles),
            "base_points": ("base_point", self._base_point_distribution.points)
        }
        if self._wavelengths is not None:    
            output["wavelengths"] = ("wavelength", self._wavelengths)
        return output
    
    def _generate_update_handles(self):
        return [self._angular_distribution.update, self._base_point_distribution.update]
            
    @property
    def center(self):
        return self._center
        
    @center.setter
    def center(self, val):
        val = tf.cast(val, tf.float64)
        if self.dimension == 2:
            if val.shape != ():
                raise ValueError("AngularSource: center must be size (2,).")
        else:
            if val.shape != (3,):
                raise ValueError("AngularSource: center must be size (3,).")
        self._center = val

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
        
# ========================================================================================

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
    
    For the 2D case, tfrt.distributions contains several aperature point distributions
    that go well with this class.  For the 3D case, things are a little more complicated.
    The identity of this class is that it doesn't care about a center or aiming, so I don't
    want to add parameters like that to the 2D distributions to accomodate AperatureSource.
    But then there aren't any convenient distributions to use for the 3D AperatureSource.
    
    tfrt.distributions does have a variety of distribution options that, while they generate
    points in 2D, are intended to be used for 3D sources.  These sources may need to be 
    transformed using tfrt.distributions.BasePointTransformation().  This class when used
    on its own will simply convert a 2D distribution to 3D (assuming the two input
    dimensions are y and z), but it can also apply an offset or other simple transformation.
    
    Or you can create your points in a pyvista.PolyData object, either created with pyvista 
    or created in an external CAD program and loaded through pyvista.  The PolyData objects 
    can be fed to a distributions.ManualBasePointDistribution whose from_mesh kwarg was set 
    to True.  But you will still need to build your own rank handler if you need to use 
    ranks for this case.    
        
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
        **kwargs
    ):
        self._set_dimension(dimension)
        self._start_point_distribution = start_point_distribution
        self._end_point_distribution = end_point_distribution
        if wavelengths is not None:
            wavelengths = tf.cast(wavelengths, tf.float64)
        self._wavelengths = wavelengths
        super().__init__(standard_domains={"start_point", "end_point"}, **kwargs)
        
    def _internal_update(self, expanded_vars):
        start_points = expanded_vars["start_points"]
        start_points_shape = tf.shape(start_points)
        end_points = expanded_vars["end_points"]
        end_points_shape = tf.shape(end_points)
                
        # set the fields
        if self.dimension == 2:
            self["x_start"], self["y_start"] = tf.unstack(start_points, axis=1)
            self["x_end"], self["y_end"] = tf.unstack(end_points, axis=1)
        else: #self.dimension == 3
            self["x_start"], self["y_start"], self["z_start"] = \
                tf.unstack(start_points, axis=1)
            self["x_end"], self["y_end"], self["z_end"] = tf.unstack(end_points, axis=1)
            
        try:
            self["wavelength"] = expanded_vars["wavelengths"]
        except(KeyError):
            # do nothing if wavelength was not defined
            pass
                
    def _internal_vars(self):
        output = {
            "start_points": ("start_point", self._start_point_distribution.points),
            "end_points": ("end_point", self._end_point_distribution.points)
        }
        if self._wavelengths is not None:    
            output["wavelengths"] = ("wavelength", self._wavelengths)
        return output
            
    def _generate_update_handles(self):
        return [
            self.start_point_distribution.update,
            self.end_point_distribution.update
        ]

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
      
# =========================================================================================

class PrecompiledSource(RecursivelyUpdatable):
    """
    A source that has been precompiled and stored in a file, for performance reasons.
    
    This source is intended to satisfy two performance related needs: 1) If you want to 
    trace a static optic before a parametric one, and don't want to spend computation time
    retracing the fixed optic every step, but do want to use a random source.  2) If you
    want to use a random optic but you require an extra field that is very time-consuming
    to compute, like distribution-to-distribution remapping.  In this case, you can perform
    these computation-intensive steps once, sample a very large number of rays, and then
    store the results in a file.  This source can accomodate randomly sampling a small
    subset of its rays, via the sample_count attribute.
    
    The constructor takes one required argument.  If it is a string, then it should be a 
    filename, and the source will load itself from that file.  If it is an int, the 
    constructor will interpret that as the dimension of the source, and initialize an
    empty source, to be populated later.  Otherwise the source will interpret arg as a 
    SourceBase that has already been fully updated and annotated, and will populate by 
    copying the fields out of the provided source.
    
    Thus to precompile a source, generate the source as normal, and once it is fully
    updated, pass it to PrecompiledSource and call save to save the source data.  And to
    use a precompiled source, pass the filename previously used to store the data.
    
    You can also call from_samples() to fill the source from a list of samples
    generate some other way, for instance as the output rayset from the tracer.  This
    function takes a list of dicts, and concatenates all the fields from each dict
    to generate the data for the source.
    
    This source naturally stores its data as np arrays, rather than tensors, but you
    can convert the underlying data to tensors if you want with the function to_tensors.
    """
    def __init__(
        self, 
        arg, 
        sample_count=100, 
        do_downsample=True,
        start_perturbation=None,
        end_perturbation=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        arg :
            The thing to initialize the source from, with several different options for
            initialization based on the type of this arg:
            
            str : arg is interpreted as a filename, and the source is loaded from this
                file.
            int : arg is interpreted as the dimension of the source, and creates
                a new empty source with this dimension.
            default :
                arg is interpreted as a source object, and the source is initialized from
                arg.
        sample_count : int, optional
            Only has an effect if do_downsample is True.  The number of rays to generate
            for this source each time it is updated.  Defaults to 100
        do_downsample : bool, optional
            If true, will select sample_count rays randomly out of all the rays stored
            in the source.  Random selection is performed with replacement.  This number
            may be larger than the actual number of rays stored in the source, but will
            usually be smaller.  Defaults to True.
        start_perturbation, end_perturbation : float tensor, optional
            Defaults to None, in which case each update will generate rays matching the
            internally stored rays with no perturbation.  But if one of these parameters
            is not None, it must be broadcastable to the number of dimensions, and will
            add a random value to the endpoints of each ray, sampled from a normal 
            distribution with a standard deviation equal to the value of each element of
            this parameter.  Useful for generating a slightly randomized source from a
            large set of precompiled rays.  For example, if the source is 3D, and 
            start_perturbation is (.1, .1, 0), then the x and y coordinates of each ray
            will be moved, but the z coordinate will remain unchanged.  May just be a 
            scalar, to perturbute each dimension equally.
        """
        arg_type = type(arg)
        if arg_type is str:
            # arg is a string, so interpret it as a filename
            with open(arg, 'rb') as in_file:
                in_data = pickle.load(in_file)
                self._dimension = in_data["dimension"]
                self._standard_domains = in_data["standard_domains"]
                self._full_fields = in_data["fields"]
        elif arg_type is not int:
            # interpret arg as a source
            self._dimension = arg._dimension
            self._standard_domains = arg._standard_domains
            self._full_fields = {
                field: np.array(value) for field, value in arg._fields.items()
            }
        else:
            # arg was an int, interpret that as the dimension and initialize an
            # empty source
            self._dimension = arg
            self._standard_domains = set()
            self._full_fields = {}
        
        self.start_perturbation = start_perturbation
        self.end_perturbation = end_perturbation
        self._fields = {}
        try:
            self._sampling_domain_size = tf.shape(self._full_fields["x_start"])[0]
        except(KeyError):
            self._sampling_domain_size = 0
        self.sample_count = sample_count
        self.do_downsample = do_downsample
        RecursivelyUpdatable.__init__(self, **kwargs)
        
    def save(self, filename):
        # want to convert the data to np arrays before storing, since tf tensors
        # are stateful objects that may hold references to a tf.Session or tf.Graph
        cleaned_fields = {field: np.array(value) for field, value in self.items()}
                
        out_data = {
            "dimension": self._dimension,
            "standard_domains": self._standard_domains,
            "fields": cleaned_fields
        }
        with open(filename, 'wb') as out_file:
            pickle.dump(out_data, out_file, pickle.HIGHEST_PROTOCOL)
            
    def _update(self):
        # do the downsampling
        if self.do_downsample:
            sample_indices = tf.random.uniform(
                (self._sample_count,),
                maxval=self.sampling_domain_size,
                dtype=tf.int32
            )
            for field, item in self._full_fields.items():
                self[field] = tf.gather(item, sample_indices, axis=0)
        else:
            self._fields = self._full_fields
        
        # do the perturbations    
        if self._dimension == 2:
            dimensions = ('x', 'y')
        else:
            dimensions = ('x', 'y', 'z')
            
        if self._start_perturbation is not None:
            for dim, ptb in zip(dimensions, self._start_perturbation):
                self[f"{dim}_start"] += tf.random.normal(
                    (self._sample_count,),
                    stddev=ptb,
                    dtype=tf.float64
                )
        if self._end_perturbation is not None:
            for dim, ptb in zip(dimensions, self._end_perturbation):
                self[f"{dim}_end"] += tf.random.normal(
                    (self._sample_count,),
                    stddev=ptb,
                    dtype=tf.float64
                )
                
    def from_samples(self, samples):
        # samples is a list of dicts, convert it to a dict of lists, in preparation for
        # concatenation.
        dict_of_lists = {}
        for sample in samples:
            for field, data in sample.items():
                try:
                    dict_of_lists[field].append(data)
                except(KeyError):
                    dict_of_lists[field] = [data]
                    
        # concatenate everything
        self._full_fields = {}
        for field, data_list in dict_of_lists.items():
            self._full_fields[field] = tf.concat(data_list, axis=0)
        self.update()
        
    def clear(self):
        self._full_fields = {}
        self._fields = self._full_fields
        
    def _generate_update_handles(self):
        return []

    @property
    def sampling_domain_size(self):
        return self._sampling_domain_size
        
    @property
    def standard_domains(self):
        return self._standard_domains
        
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
    def items(self):
        return self._fields.items
        
    @property
    def sample_count(self):
        return self._sample_count
        
    @sample_count.setter
    def sample_count(self, val):
        tf.debugging.assert_greater(
            val, 0, "PrecompiledSource: sample_count must be > 0."
        )
        tf.debugging.assert_integer(
            val, "PrecompiledSource: sample_count must be of integer type."
        )
        self._sample_count = val
        
    @property
    def start_perturbation(self):
        return self._start_perturbation
        
    @start_perturbation.setter
    def start_perturbation(self, val):
        if val is not None:
            if self._dimension == 2:
                shape = (2,)
            else:
                shape = (3,)
            try:
                val = tf.broadcast_to(val, shape)
            except:
                raise ValueError(
                    "PrecompiledSource: start_perturbation must be None, scalar, or"
                    "must have one entry per dimension."
                )
                
        self._start_perturbation = val
        
    @property
    def end_perturbation(self):
        return self._end_perturbation
        
    @end_perturbation.setter
    def end_perturbation(self, val):
        if val is not None:
            if self._dimension == 2:
                shape = (2,)
            else:
                shape = (3,)
            try:
                val = tf.broadcast_to(val, shape)
            except:
                raise ValueError(
                    "PrecompiledSource: end_perturbation must be None, scalar, or"
                    "must have one entry per dimension."
                )
                
        self._end_perturbation = val





