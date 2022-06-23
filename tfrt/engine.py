"""
The class that will hold the ray tracer, and the data organization tools it uses.
"""

import math

from abc import ABC, abstractmethod
import tensorflow as tf

from tfrt.update import RecursivelyUpdatable
import tfrt.operation as op
import tfrt.geometry as geometry

OPTICAL = 0
STOP = 1
TARGET = 2

SEGMENT_GEO_SIG = {"x_start", "y_start", "x_end", "y_end"}
ARC_GEO_SIG = {"x_center", "y_center", "angle_start", "angle_end", "radius"}
SOURCE_3D_SIG = {"x_start", "y_start", "z_start", "x_end", "y_end", "z_end"}
TRIANGLE_GEO_SIG = {"xp", "yp", "zp", "x1", "y1", "z1", "x2", "y2", "z2", "norm"}

PI = tf.constant(math.pi, dtype=tf.float64)

#=======================================================================================

class ReadOnlySet:
    """
    Object that acts like a ray or boundary set, but is read only.  Used to hold
    the amalgamated objects in an optical system
    """
    def __init__(self, fields):
        self._fields = fields
    
    def __getitem__(self, key):
        try:
            return self._fields[key]
        except(KeyError) as e:
            raise KeyError(f"key {key} not in the signature of this set.") from e
            
    def __bool__(self):
        return bool(self._fields)
    
    @property    
    def keys(self):
        return self._fields.keys
            
# --------------------------------------------------------------------------------------

def amalgamate(stuff, signature=None):
    """
    Join a list of indexable stuff that has a signature into a single dictionary of the
    same signature.
    
    If signature is specified, will only amalgamate those fields.  Otherwise, will
    only join fields that are common to every item in the list.
    """
    # clear empty sets out of stuff, these can be just ignored.
    processed_stuff = [item for item in stuff if bool(item)]
    
    if bool(processed_stuff):
        if not bool(signature):
            for item in processed_stuff:
                item_sig = item.keys()
                    
                if bool(signature):
                    signature = signature & item_sig
                else:
                    signature = item_sig
                    
        return {
            field: tf.concat([item[field] for item in processed_stuff], 0)
            for field in signature
        }
    else:
        return {}

# --------------------------------------------------------------------------------------
        
def recursive_dict_key_print(dict_in, spacer=""):
    """
    This is mostly a utility for testing the results produced by process_projection.
    
    This function takes a dict as input, and prints its keys, each on a line.  If the
    value for that key is something that has a shape, it prints it.  If the value is
    a dict, the function indents and recursively calls itself on that dict.  Otherwise
    it just prints the key.  Parameter spacer tells how much indenting to do, and should
    be left to the default value whenever this is called by the user, unless you really
    want to print something else in front of each line.
    """
    if type(dict_in) is not dict:
        return
    next_spacer = spacer + "    "
    for key, value in dict_in.items():
        try:
            print(spacer, f"{key} : {value.shape}")
        except(AttributeError):
            print(spacer, key)
        recursive_dict_key_print(value, next_spacer)

# --------------------------------------------------------------------------------------
        
def annotation_helper(parent, field, value, valid_shape_field, dtype=tf.float64):
    """
    This function will help keep an object annotated with a specific field.  This can
    for instance be used to apply materials to a boundary.
    
    This function doesn't just set a field, it adds an update handler that re-calculates
    the value of the field whenever the parent object is updated.
    
    Parameters
    ----------
    parent : A tfrt.RecursivelyUpdatable
        Most commonly will be a boundary or source.  This is the object that the new field
        will be added to.
    field : string
        The name of the new field that will be added to parent.  Doesn't actually 
        technically have to be a string - it could be anything that can be used as a key
        in a dict - but it is really recommended that this only ever be a string.
    value :
        This can be a callable that takes two arguments, a tf.TensorShape, and a tf.dtype 
        and returns a single value, which will be whatever is inserted into the field.
        Or value can be non-callable, in which case it is broadcast to the desired shape
        and cast to dtype.
    valid_shape_field : string
        This is the key of some valid field already present in parent.  The value
        in this field is used to determine what shape to give value.  Like field this
        doesn't technically have to be a string, but is strongly recommended to be.
    dtype : tf.DType, optional
        The data type that value should have.  Value will be cast to this dtype if it
        isn't a callable.  If value is callable, it is the responsibility of the callable
        to honor this parameter.
    """
    if callable(value):
        def f():
            shape = tf.shape(parent[valid_shape_field])
            parent[field] = value(shape, dtype)
    else:
        def f():
            shape = tf.shape(parent[valid_shape_field])
            parent[field] = tf.broadcast_to(tf.cast(value, dtype), shape)
    parent.post_update_handles.append(f)

# ======================================================================================

class OpticalSystemBase(RecursivelyUpdatable, ABC):
    """
    Class that holds all sources and boundaries, and manages annotations.
    
    Boundaries and sources can be added to this object by calling the appropriate 
    function, which take a list of boundary or source objects.  Will check that the
    dimensionality matches, and will automatically manage the update handles.
    Manually manipulating the update handles will be a little more difficult for this
    class, since we need to be able to turn on and off various optical features on
    the fly, so self.update_handles will be rebuilt every time this is done, by 
    default.  Given this, I think the ideal way to handle this is to let this class 
    use its default update handling behavior, and turn on and off updating for 
    individual optical components if updates are not necessary for that component.
    
    intersect_epsilion : float
        A small value to avoid divide by zero.
    size_epsilion : float
        If a ray is aimed directly at the end of a surface, it may or may not calculate
        an intersection with that surface, based on rounding errors.  This value
        makes surfaces slightly larger, to try to help catch this condition.  But
        if you don't want this behavior, set this to zero, or something else.  It could
        even be negative to exclude rays that strike very close to the edge of a surface.
    ray_start_epsilion : float
        A small value that helps avoid detecting rays that start on a boundary.  Rays will
        ignore boundaries they intersect less than this far from the start of the ray, to
        avoid intersecting with the boundary they originate from.  This number should be
        smaller than the distance between any surfaces.
    """
    def __init__(
        self,
        manual_update_management=False,
        intersect_epsilion=1e-10,
        size_epsilion=1e-10,
        ray_start_epsilion=1e-10,
        **kwargs
    ):
        self._sources = []
        self._read_only_sources = None
        self.source_handles = []
        self._amalgamated_sources = {}
        self.manual_update_management=manual_update_management
        self.materials = []
        self.intersect_epsilion = intersect_epsilion
        self.size_epsilion = size_epsilion
        self.ray_start_epsilion = ray_start_epsilion
        self.projection_results = {}
        super().__init__(**kwargs)
    
    @abstractmethod    
    def refresh_update_handles(self):
        raise NotImplementedError
        
    @abstractmethod
    def dimension(self):
        raise NotImplementedError
        
    @property
    def sources(self):
        if not bool(self._read_only_sources):
            self._read_only_sources = ReadOnlySet(self._amalgamated_sources)
        return self._read_only_sources
        
    @sources.setter
    def sources(self, new):
        self.source_handles = []
        for each in new:
            assert each.dimension == self.dimension
            self.source_handles.append(each.update)
        self._sources = new
        self.refresh_update_handles()
            
    @property        
    def materials(self):
        return self._materials
        
    @materials.setter
    def materials(self, val):
        assert type(val) is list
        self._materials = val
        
    @abstractmethod
    def intersect(self, rays):
        """
        Intersect the given rays with the system's merged boundaries.
        
        This function returns for dicts: active, finished, stopped, dead (each itself a 
        dict), split by the type of boundary intersected with and whose data is gathered 
        into pairs based on the intersections.
        
        Dead contains rays that did not intersect, and so will only have a single element,
        rays.  The others will have the fields rays, boundaries, and norms.  
        
        Returned ray sets will have all of the input ray's data, but the endpoint of the 
        ray will be moved to lie on the boundary where it terminates.
        
        Boundaries will have all of the boundary data, except for the geometrical data, 
        for either an optical or technical boundary, depending on the type of the 
        boundary.  The geometric data must be stripped so that arc and segment
        intersections can be combined in the 2D case.
        
        Norms will contain the norm of the surface at the intersection.  In the 2D case, 
        it will be a tensor that encodes the angle of the surface.  In the 3D case...
        
        """
        raise NotImplementedError

# -------------------------------------------------------------------------------------

class OpticalSystem2D(OpticalSystemBase):
    def __init__(self, **kwargs):
        self._optical_segments = []
        self.optical_segment_handles = []
        self._amalgamated_optical_segments = {}
        self._optical_arcs = []
        self.optical_arc_handles = []
        self._amalgamated_optical_arcs = {}
        self._stop_segments = []
        self.stop_segment_handles = []
        self._amalgamated_stop_segments = {}
        self._stop_arcs = []
        self.stop_arc_handles = []
        self._amalgamated_stop_arcs = {}
        self._target_segments = []
        self.target_segment_handles = []
        self._amalgamated_target_segments = {}
        self._target_arcs = []
        self.target_arc_handles = []
        self._amalgamated_target_arcs = {}
        self._materials = {}
        
        self.clear_read_only()
        
        super().__init__(**kwargs)
        
    def clear_read_only(self):
        self._read_only_sources = None
        self._read_only_optical_segments = None
        self._read_only_optical_arcs = None
        self._read_only_stop_segments = None
        self._read_only_stop_arcs = None
        self._read_only_target_segments = None
        self._read_only_target_arcs = None
        
    def refresh_update_handles(self):
        if not self.manual_update_management:
            self._generate_update_handles()
    
    def _generate_update_handles(self):
        self.update_handles = self.source_handles + self.optical_segment_handles +\
            self.optical_arc_handles + self.stop_segment_handles + \
            self.stop_arc_handles + self.target_segment_handles + \
            self.target_arc_handles
    
    @property    
    def dimension(self):
        return 2
        
    def _update(self):
        if bool(self._sources):
            self._amalgamated_sources = amalgamate(self._sources)
        if bool(self._optical_segments):
            self._amalgamated_optical_segments = amalgamate(self._optical_segments)
        if bool(self._optical_arcs):
            self._amalgamated_optical_arcs = amalgamate(self._optical_arcs)
        if bool(self._stop_segments):
            self._amalgamated_stop_segments = amalgamate(self._stop_segments)
        if bool(self._stop_arcs):
            self._amalgamated_stop_arcs = amalgamate(self._stop_arcs)
        if bool(self._target_segments):
            self._amalgamated_target_segments = amalgamate(self._target_segments)
        if bool(self._target_arcs):
            self._amalgamated_target_arcs = amalgamate(self._target_arcs)
            
        self._merge_boundaries()
            
        self.clear_read_only()
        
    @property
    def optical_segments(self):
        if not bool(self._read_only_optical_segments):
            self._read_only_optical_segments = ReadOnlySet(
                self._amalgamated_optical_segments
            )
        return self._read_only_optical_segments
        
    @optical_segments.setter
    def optical_segments(self, new):
        self.optical_segment_handles = []
        for each in new:
            assert each.dimension == self.dimension
            self.optical_segment_handles.append(each.update)
        self._optical_segments = new
        self.refresh_update_handles()
        
    @property
    def optical_arcs(self):
        if not bool(self._read_only_optical_arcs):
            self._read_only_optical_arcs = ReadOnlySet(
                self._amalgamated_optical_arcs
            )
        return self._read_only_optical_arcs
        
    @optical_arcs.setter
    def optical_arcs(self, new):
        self.optical_arc_handles = []
        for each in new:
            assert each.dimension == self.dimension
            self.optical_arc_handles.append(each.update)
        self._optical_arcs = new
        self.refresh_update_handles()
        
    @property
    def stop_segments(self):
        if not bool(self._read_only_stop_segments):
            self._read_only_stop_segments = ReadOnlySet(
                self._amalgamated_stop_segments
            )
        return self._read_only_stop_segments
        
    @stop_segments.setter
    def stop_segments(self, new):
        self.stop_segment_handles = []
        for each in new:
            assert each.dimension == self.dimension
            self.stop_segment_handles.append(each.update)
        self._stop_segments = new
        self.refresh_update_handles()
        
    @property
    def stop_arcs(self):
        if not bool(self._read_only_stop_arcs):
            self._read_only_stop_arcs = ReadOnlySet(
                self._amalgamated_stop_arcs
            )
        return self._read_only_stop_arcs
        
    @stop_arcs.setter
    def stop_arcs(self, new):
        self.stop_arc_handles = []
        for each in new:
            assert each.dimension == self.dimension
            self.stop_arc_handles.append(each.update)
        self._stop_arcs = new
        self.refresh_update_handles()
        
    @property
    def target_segments(self):
        if not bool(self._read_only_target_segments):
            self._read_only_target_segments = ReadOnlySet(
                self._amalgamated_target_segments
            )
        return self._read_only_target_segments
        
    @target_segments.setter
    def target_segments(self, new):
        self.target_segment_handles = []
        for each in new:
            assert each.dimension == self.dimension
            self.target_segment_handles.append(each.update)
        self._target_segments = new
        self.refresh_update_handles()
        
    @property
    def target_arcs(self):
        if not bool(self._read_only_target_arcs):
            self._read_only_target_arcs = ReadOnlySet(
                self._amalgamated_target_arcs
            )
        return self._read_only_target_arcs
        
    @target_arcs.setter
    def target_arcs(self, new):
        self.target_arc_handles = []
        for each in new:
            assert each.dimension == self.dimension
            self.target_arc_handles.append(each.update)
        self._target_arcs = new
        self.refresh_update_handles()
        
    def _merge_boundaries(self):
        """
        Labels and merges the boundaries (targets, stops, and optical ones) in preparation
        for tracing.  Returns the two data sets, the segments and arcs, that hold all
        of the geometric data needed for projection, but without any other data needed
        for computing reactions, since we can't elegantly merge optical and technical
        surfaces, because they may have different signatures.  So those sets will have
        to be gathered to conform to intersections after projection.
        """
        
        optical_seg = self._amalgamated_optical_segments
        if bool(optical_seg):
            optical_seg["catagory"] = OPTICAL * tf.ones_like(
                optical_seg["x_start"],
                dtype=tf.int64
            )
            self._optical_seg_count = tf.shape(
                optical_seg["x_start"],
                out_type=tf.int64
            )[0]
        else:
            self._optical_seg_count = 0
            
        stop_seg = self._amalgamated_stop_segments
        if bool(stop_seg):
            stop_seg["catagory"] = STOP * tf.ones_like(
                stop_seg["x_start"],
                dtype=tf.int64
            )
            self._stop_seg_count = tf.shape(
                stop_seg["x_start"],
                out_type=tf.int64
            )[0]
        else:
            self._stop_seg_count = 0
            
        target_seg = self._amalgamated_target_segments
        if bool(target_seg):
            target_seg["catagory"] = TARGET * tf.ones_like(
                target_seg["x_start"],
                dtype=tf.int64
            )
            self._target_seg_count = tf.shape(
                target_seg["x_start"],
                out_type=tf.int64
            )[0]
        else:
            self._target_seg_count = 0
            
        self._merged_segments = amalgamate(
            [optical_seg, stop_seg, target_seg], 
            SEGMENT_GEO_SIG | {"catagory"}
        )
        
        optical_arc = self._amalgamated_optical_arcs
        if bool(optical_arc):
            optical_arc["catagory"] = OPTICAL * tf.ones_like(
                optical_arc["x_center"],
                dtype=tf.int64
            )
            self._optical_arc_count = tf.shape(
                optical_arc["x_center"],
                out_type=tf.int64
            )[0]
        else:
            self._optical_arc_count = 0
            
        stop_arc = self._amalgamated_stop_arcs
        if bool(stop_arc):
            stop_arc["catagory"] = STOP * tf.ones_like(
                stop_arc["x_center"],
                dtype=tf.int64
            )
            self._stop_arc_count = tf.shape(
                stop_arc["x_center"],
                out_type=tf.int64
            )[0]
        else:
            self._stop_arc_count = 0
            
        target_arc = self._amalgamated_target_arcs
        if bool(target_arc):
            target_arc["catagory"] = TARGET * tf.ones_like(
                target_arc["x_center"],
                dtype=tf.int64
            )
            self._target_arc_count = tf.shape(
                target_arc["x_center"],
                out_type=tf.int64
            )[0]
        else:
            self._target_arc_count = 0
            
        self._merged_arcs = amalgamate(
            [optical_arc, stop_arc, target_arc], 
            ARC_GEO_SIG | {"catagory"}
        )
        
    def intersect(self, rays):
        """
        Intersect a set of rays with all of the system's surfaces.
        
        Returns:
            Each return is a dict that contains various fields.  The value in 
            each field is 1-D and has as many elements as there are rays in the input ray 
            set.  Elements that correspond to non-intersected rays are garbage, and should
            be filtered out, the element in the 'valid' field will be false wherever the 
            data is garbage.
        segment_intersections :
            Contains data about the intersections between rays and segments, but not
            data about the segments themselves.
            Fields:
            "x", "y" :
                The location of the intersection.
            "valid" :
                True where the ray intersected a segment.
            "ray_u" : 
                The parameter along the ray.
            "segment_u" :
                The parameter along the segment.
            "gather_ray", "gather_segment"
                Indices that can be used to gather ray or segment data out of the system's
                merged sets with tf.gather.  Contains entries for invalid intersections, 
                so should also be masked with tf.boolean_mask and "valid".
            "norm" :
                The norm of the boundary at the point of intersection.
        arc_intersections :
            Same as above but for the arcs.
            Fields: "x", "y", "valid", "ray_u", "arc_u", "gather_ray", "gather_arc",
            "norm"
            
            
        """
        has_segments = bool(self._merged_segments)
        has_arcs = bool(self._merged_arcs)
        
        seg = {}
        arc = {}
        
        if has_segments:
            # do segment intersection
            seg["x"], seg["y"], seg["valid"], seg["ray_u"], seg["segment_u"], \
                seg["gather_ray"], seg["gather_segment"] = self._segment_intersection(
                    rays["x_start"],
                    rays["y_start"],
                    rays["x_end"],
                    rays["y_end"],
                    self._merged_segments["x_start"],
                    self._merged_segments["y_start"],
                    self._merged_segments["x_end"],
                    self._merged_segments["y_end"],
                    self.intersect_epsilion,
                    self.size_epsilion,
                    self.ray_start_epsilion
                )
            seg["norm"] = tf.gather(
                tf.atan2(
                    self._merged_segments["y_end"] - self._merged_segments["y_start"],
                    self._merged_segments["x_end"] - self._merged_segments["x_start"]
                ) + PI/2.0,
                seg["gather_segment"]
            )
            
        if has_arcs:
            # do arc intersection
            arc["x"], arc["y"], arc["valid"], arc["ray_u"], arc["arc_u"], \
                arc["gather_ray"], arc["gather_arc"] = self._arc_intersection(
                    rays["x_start"],
                    rays["y_start"],
                    rays["x_end"],
                    rays["y_end"],
                    self._merged_arcs["x_center"],
                    self._merged_arcs["y_center"],
                    self._merged_arcs["angle_start"],
                    self._merged_arcs["angle_end"],
                    self._merged_arcs["radius"],
                    self.intersect_epsilion,
                    self.size_epsilion,
                    self.ray_start_epsilion
                )
            arc["norm"] = self._get_arc_norm(
                self._merged_arcs["radius"], arc["arc_u"], arc["gather_arc"]
            )
             
        if has_segments and has_arcs:
            # has arcs and segments, so we need to chooose between segment and arc 
            # intersections.
            seg["valid"], arc["valid"] = self._seg_or_arc(
                seg["ray_u"], arc["ray_u"], seg["valid"], arc["valid"]
            )
                
        return seg, arc

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.bool),
            tf.TensorSpec(shape=(None,), dtype=tf.bool)
        ]
    )"""
    @staticmethod
    def _seg_or_arc(seg_u, arc_u, seg_valid, arc_valid):
        """all_u = tf.concat([seg_u, arc_u], axis=0)
        inf = 2 * tf.reduce_max(all_u) * tf.ones_like(seg_u)
        seg_u = tf.where(seg_valid, seg_u, inf)
        arc_u = tf.where(arc_valid, arc_u, inf)
        
        choose_segment = tf.less(seg_u, arc_u)
        seg_valid = tf.logical_and(choose_segment, seg_valid)
        arc_valid = tf.logical_and(
            tf.logical_not(choose_segment),
            arc_valid
        )"""
        
        """
        # here is another possibly faster way to do this
        has_both = tf.logical_and(seg_valid, arc_valid)
        seg_less = tf.less(seg_u, arc_u)
        # pick entries to remove from the valid tensors.
        remove_arc = tf.logical_and(has_both, seg_less)
        remove_seg = tf.logical_and(has_both, tf.logical_not(seg_less))
        seg_valid = tf.logical_and(seg_valid, tf.logical_not(remove_seg))
        arc_valid = tf.logical_and(arc_valid, tf.logical_not(remove_arc))"""
        
        
        # even faster than above?
        has_both = tf.logical_and(seg_valid, arc_valid)
        seg_less = tf.less(seg_u, arc_u)
        seg_valid = tf.where(has_both, seg_less, seg_valid)
        arc_valid = tf.where(has_both, tf.logical_not(seg_less), arc_valid)
        
        return seg_valid, arc_valid

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        ]
    )"""
    @staticmethod
    def _get_arc_norm(radius, arc_u, gather_ray):
        radius = tf.gather(radius, gather_ray)
        arc_norm = tf.where(tf.less(radius, 0), arc_u + PI, arc_u)
        return tf.math.mod(arc_norm + PI, 2*PI) - PI
    
    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64)
        ]
    )"""
    @staticmethod
    def _segment_intersection(
        rx1, ry1, rx2, ry2, sx1, sy1, sx2, sy2,
        intersect_epsilion,
        size_epsilion,
        ray_start_epsilion
    ):
        """
        Returns:
            The returns are all 1-D, and are value per ray (they have as many elements as
            rays were given to the system.  Where valid is false, the corresponding value
            in each of the other returns is garbage.  You can tf.boolean_mask the
            other outputs with valid to get only non-garbage results.
        x, y :
            The location of the intersection.
        valid :
            True when a ray found any valid intersection
        ray_u, seg_u :
            The parameter along the ray or segment to the intersection.  For rays, length 
            is in parameter-space, so it will only be the true distance if the ray started 
            at length 1.  But whatever units this is in, it should compare correctly to 
            the equivalent value from the arc intersection.
        gather_segment, gather_ray : 
            The index of the segment and ray per intersection.  These values can be used 
            to gather data out of the sets.  gather_ray is relative to the rays in the set 
            input to this function, not necessairly to system.sources.  gather_segment is 
            relative to system._merged_segments.  Since the optical segments are 
            listed first, if we mask out stopped / finished ray intersections, this should 
            map into system.optical_segments.
        """
        x, y, valid, ray_u, seg_u = geometry.line_intersect(
            rx1, ry1, rx2, ry2, sx1, sy1, sx2, sy2, intersect_epsilion
        )
        
        # prune invalid ray / segment intersections
        valid = tf.logical_and(valid, tf.greater_equal(seg_u, -size_epsilion))
        valid = tf.logical_and(valid, tf.less_equal(seg_u, 1 + size_epsilion))
        valid = tf.logical_and(valid, tf.greater_equal(ray_u, ray_start_epsilion))
        
        # fill ray_u with large values wherever the intersection is invalid
        inf = 2 * tf.reduce_max(ray_u) * tf.ones_like(ray_u)
        ray_u = tf.where(valid, ray_u, inf)
        
        # find the closest ray intersection
        closest_segment = tf.argmin(ray_u, axis=0)
        
        # gather / reduce variables down to a 1-D list that indexes intersected rays
        # right now everything is square, where the first index is for the segment, and 
        # the second is for the ray.
        valid = tf.reduce_any(valid, axis=0)
        ray_range = tf.range(tf.shape(rx1)[0], dtype=tf.int64)
        #gather_segment = tf.boolean_mask(closest_segment, valid)
        gather_segment = closest_segment
        #gather_ray = tf.boolean_mask(ray_range, valid)
        gather_ray = ray_range
        gather_both = tf.transpose(tf.stack([gather_segment, gather_ray]))
        
        x = tf.gather_nd(x, gather_both)
        y = tf.gather_nd(y, gather_both)
        ray_u = tf.gather_nd(ray_u, gather_both)
        seg_u = tf.gather_nd(seg_u, gather_both)
        
        return x, y, valid, ray_u, seg_u, gather_ray, gather_segment

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64)
        ]
    )"""  
    @staticmethod      
    def _arc_intersection(
        rx1, ry1, rx2, ry2, xc, yc, a1, a2, r,
        intersect_epsilion,
        size_epsilion,
        ray_start_epsilion
    ):
        """
        Returns:
        x, y :
            1-D, the location of the intersection.  Has been gathered, so only valid
            intersections are included.
        valid :
            1-D, takes the size of the rays, true when a ray found any valid intersection
        ray_u :
            1-D, same shape as valid, the length along the ray to the intersection.  
            Length is in parameter-space, so it will only be the true distance if the
            ray started at length 1.  But whatever units this is in, it should compare
            correctly to the equivalent value from the arc intersection.
        arc_u :
            The the arc polar parameter.  Is the angle at which the intersection occurs,
            and so can be converted to the surface norm once the 
        gather_arc, gather_ray : 
            1-D, The index of the segment and ray per intersection, only for valid 
            intersections.  These values can be used to gather data out of the sets.
            gather_ray is relative to the rays in the set input to this function, not
            necessairly to system.sources.  gather_arc is relative to 
            system._merged_arcs.  But since the optical arcs are listed first, if
            we mask out stopped / finished ray intersections, this should map into
            system.optical_arcs.
        """
        plus, minus = geometry.line_circle_intersect(
            rx1, ry1, rx2, ry2, xc, yc, r, intersect_epsilion
        )
        
        # prune invalid ray / segment intersections
        plus["valid"] = tf.logical_and(
            plus["valid"],
            tf.greater_equal(plus["u"], ray_start_epsilion)
        )
        minus["valid"] = tf.logical_and(
            minus["valid"],
            tf.greater_equal(minus["u"], ray_start_epsilion)
        )
        
        a1 = tf.reshape(a1, (-1, 1))
        a2 = tf.reshape(a2, (-1, 1))
        plus["valid"] = tf.logical_and(
            plus["valid"],
            geometry.angle_in_interval(
                plus["v"],
                a1,
                a2
            )
        )
        minus["valid"] = tf.logical_and(
            minus["valid"],
            geometry.angle_in_interval(
                minus["v"],
                a1,
                a2
            )
        )
        
        # fill u with large values wherever the intersection is invalid
        inf = 2 * tf.reduce_max(plus["u"]) * tf.ones_like(plus["u"])
        plus["u"] = tf.where(plus["valid"], plus["u"], inf)
        minus["u"] = tf.where(minus["valid"], minus["u"], inf)
        
        # choose between plus and minus cases.
        choose_minus = tf.less(minus["u"], plus["u"])
        minus["valid"] = tf.logical_and(minus["valid"], choose_minus)
        plus["valid"] = tf.logical_and(plus["valid"], tf.logical_not(choose_minus))
        
        valid = tf.logical_or(minus["valid"], plus["valid"])
        x = tf.where(choose_minus, minus["x"], plus["x"])
        y = tf.where(choose_minus, minus["y"], plus["y"])
        ray_u = tf.where(choose_minus, minus["u"], plus["u"])
        arc_u = tf.where(choose_minus, minus["v"], plus["v"])
        
        # find the closest ray intersection
        closest_arc = tf.argmin(ray_u, axis=0)
        
        # gather / reduce variables down to a 1-D list that indexes intersected rays
        # right now everything is square, where the first index is for the arc, and 
        # the second is for the ray.
        valid = tf.reduce_any(valid, axis=0)
        ray_range = tf.range(tf.shape(rx1)[0], dtype=tf.int64)
        #gather_arc = tf.boolean_mask(closest_arc, valid)
        gather_arc = closest_arc
        #gather_ray = tf.boolean_mask(ray_range, valid)
        gather_ray = ray_range
        gather_both = tf.transpose(tf.stack([gather_arc, gather_ray]))
        
        x = tf.gather_nd(x, gather_both)
        y = tf.gather_nd(y, gather_both)
        ray_u = tf.gather_nd(ray_u, gather_both)
        arc_u = tf.gather_nd(arc_u, gather_both)
        
        return x, y, valid, ray_u, arc_u, gather_ray, gather_arc


# -------------------------------------------------------------------------------------

class OpticalSystem3D(OpticalSystemBase):
    def __init__(self, **kwargs):
        self._optical = []
        self.optical_handles = []
        self._amalgamated_optical = {}
        self._stop = []
        self.stop_handles = []
        self._amalgamated_stop = {}
        self._target = []
        self.target_handles = []
        self._amalgamated_target = {}
        self._materials = {}
        
        self.clear_read_only()
        
        super().__init__(**kwargs)
        
    def clear_read_only(self):
        self._read_only_sources = None
        self._read_only_optical = None
        self._read_only_stop = None
        self._read_only_target = None
        
    def refresh_update_handles(self):
        if not self.manual_update_management:
            self._generate_update_handles()
    
    def _generate_update_handles(self):
        self.update_handles = self.source_handles + self.optical_handles + \
            self.stop_handles + self.target_handles
    
    @property    
    def dimension(self):
        return 3
        
    def _update(self):
        if bool(self._sources):
            self._amalgamated_sources = amalgamate(self._sources)
        if bool(self._optical):
            self._amalgamated_optical = amalgamate(self._optical)
        if bool(self._stop):
            self._amalgamated_stop = amalgamate(self._stop)
        if bool(self._target):
            self._amalgamated_target = amalgamate(self._target)
        
        self._merge_boundaries()
            
        self.clear_read_only()
        
    @property
    def optical(self):
        if not bool(self._read_only_optical):
            self._read_only_optical = ReadOnlySet(
                self._amalgamated_optical
            )
        return self._read_only_optical
        
    @optical.setter
    def optical(self, new):
        self.optical_handles = []
        for each in new:
            assert each.dimension == self.dimension
            self.optical_handles.append(each.update)
        self._optical = new
        self.refresh_update_handles()
        
    @property
    def stops(self):
        if not bool(self._read_only_stop):
            self._read_only_stop = ReadOnlySet(
                self._amalgamated_stop
            )
        return self._read_only_stop
        
    @stops.setter
    def stops(self, new):
        self.stop_handles = []
        for each in new:
            assert each.dimension == self.dimension
            self.stop_handles.append(each.update)
        self._stop = new
        self.refresh_update_handles()
        
    @property
    def targets(self):
        if not bool(self._read_only_target):
            self._read_only_target = ReadOnlySet(
                self._amalgamated_target
            )
        return self._read_only_target
        
    @targets.setter
    def targets(self, new):
        self.target_handles = []
        for each in new:
            assert each.dimension == self.dimension
            self.target_handles.append(each.update)
        self._target = new
        self.refresh_update_handles()
        
    def _merge_boundaries(self):
        """
        Labels and merges the boundaries (targets, stops, and optical ones) in preparation
        for tracing.
        """
        optical = self._amalgamated_optical
        if bool(optical):
            optical["catagory"] = OPTICAL * tf.ones_like(
                optical["xp"],
                dtype=tf.int64
            )
            self._optical_count = tf.shape(
                optical["xp"],
                out_type=tf.int64
            )[0]
        else:
            self._optical_count = 0
            
        stop = self._amalgamated_stop
        if bool(stop):
            stop["catagory"] = STOP * tf.ones_like(
                stop["xp"],
                dtype=tf.int64
            )
            self._stop_count = tf.shape(
                stop["xp"],
                out_type=tf.int64
            )[0]
        else:
            self._stop_count = 0
            
        target = self._amalgamated_target
        if bool(target):
            target["catagory"] = TARGET * tf.ones_like(
                target["xp"],
                dtype=tf.int64
            )
            self._target_count = tf.shape(
                target["xp"],
                out_type=tf.int64
            )[0]
        else:
            self._target_count = 0
        
        self._merged = amalgamate(
            [optical, stop, target], 
            TRIANGLE_GEO_SIG | {"catagory"}
        )
        
    def intersect(self, rays):
        """
        Intersect a set of rays with all of the system's surfaces.
        
        Returns
        -------
        segment_intersections : dict
            Contains data about the intersections between rays and segments, but not
            data about the segments themselves.
            Fields:
            "x", "y", "z" :
                The location of the intersection.
            "valid" :
                True where the ray intersected a segment.
            "ray_u" : 
                The parameter along the ray.
            "trig_u", "trig_v" :
                The parameters along the triangle.
            "gather_ray", "gather_trig"
                Indices that can be used to gather ray or triangle data out of the system's
                merged sets with tf.gather.  Contains entries for invalid intersections, 
                so should also be masked with tf.boolean_mask and "valid".
            "norm" :
                The norm of the boundary at the point of intersection.
            
            
        """        
        result = {}
        
        if bool(self._merged):
            result["x"], result["y"], result["z"], result["valid"], result["ray_u"], \
                result["trig_u"], result["trig_v"], result["gather_ray"], \
                result["gather_trig"] = self._intersection(
                    rays["x_start"],
                    rays["y_start"],
                    rays["z_start"],
                    rays["x_end"],
                    rays["y_end"],
                    rays["z_end"],
                    self._merged["xp"],
                    self._merged["yp"],
                    self._merged["zp"],
                    self._merged["x1"],
                    self._merged["y1"],
                    self._merged["z1"],
                    self._merged["x2"],
                    self._merged["y2"],
                    self._merged["z2"],
                    self.intersect_epsilion,
                    self.size_epsilion,
                    self.ray_start_epsilion
                )
                
            result["norm"] = tf.gather(
                self._merged["norm"],
                result["gather_trig"]
            )
               
        return result

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64)
        ]
    )"""   
    @staticmethod
    def _intersection(
        rx1, ry1, rz1, rx2, ry2, rz2, xp, yp, zp, x1, y1, z1, x2, y2, z2,
        intersect_epsilion,
        size_epsilion,
        ray_start_epsilion
    ):
        """
        Returns:
            The returns are all 1-D, and are value per ray (they have as many elements as
            rays were given to the system.  Where valid is false, the corresponding value
            in each of the other returns is garbage.  You can tf.boolean_mask the
            other outputs with valid to get only non-garbage results.
        x, y, z :
            The location of the intersection.
        valid :
            True when a ray found any valid intersection
        ray_u, trig_u, trig_v :
            The parameter along the ray or segment to the intersection.  For rays, length 
            is in parameter-space, so it will only be the true distance if the ray started 
            at length 1.  But whatever units this is in, it should compare correctly to 
            the equivalent value from the arc intersection.
        gather_ray, gather_trig : 
            The index of the segment and ray per intersection.  These values can be used 
            to gather data out of the sets.  gather_ray is relative to the rays in the set 
            input to this function, not necessairly to system.sources.  gather_segment is 
            relative to system._merged_segments.  Since the optical segments are 
            listed first, if we mask out stopped / finished ray intersections, this should 
            map into system.optical_segments.
        """        
        x, y, z, valid, ray_u, trig_u, trig_v = geometry.line_triangle_intersect(
            rx1, ry1, rz1, rx2, ry2, rz2, xp, yp, zp, x1, y1, z1, x2, y2, z2, 
            intersect_epsilion
        )
        
        # prune invalid ray / segment intersections
        valid = tf.logical_and(valid, tf.greater_equal(trig_u, -size_epsilion))
        valid = tf.logical_and(valid, tf.greater_equal(trig_v, -size_epsilion))
        valid = tf.logical_and(valid, tf.less_equal(trig_u + trig_v, 1 + size_epsilion))
        valid = tf.logical_and(valid, tf.greater_equal(ray_u, ray_start_epsilion))
        
        # fill ray_u with large values wherever the intersection is invalid
        inf = 2 * tf.reduce_max(ray_u) * tf.ones_like(ray_u)
        ray_u = tf.where(valid, ray_u, inf)
        
        # find the closest ray intersection
        closest_trig = tf.argmin(ray_u, axis=0)
        
        # gather / reduce variables down to a 1-D list that indexes intersected rays
        # right now everything is square, where the first index is for the triangle, and 
        # the second is for the ray.
        valid = tf.reduce_any(valid, axis=0)
        ray_range = tf.cast(tf.range(tf.shape(rx1)[0]), tf.int64)
        gather_triangle = closest_trig
        gather_ray = ray_range
        gather_both = tf.transpose(tf.stack([gather_triangle, gather_ray]))
        
        x = tf.gather_nd(x, gather_both)
        y = tf.gather_nd(y, gather_both)
        z = tf.gather_nd(z, gather_both)
        ray_u = tf.gather_nd(ray_u, gather_both)
        trig_u = tf.gather_nd(trig_u, gather_both)
        trig_v = tf.gather_nd(trig_v, gather_both)
        
        return x, y, z, valid, ray_u, trig_u, trig_v, gather_ray, gather_triangle

# =====================================================================================

class OpticalEngine:
    """
    Builds and holds the tracing / optimization model.
    
    Here is how I envision this class working:
    Step 1:
        Call the OpticalEngine constructor, and feed it the dimension and a list of 
        RayOperations that define what kind of inputs and outputs you want from the
        engine.  This is a list because order matters, but it would prefer to be an
        ordered set, since we do not want repetitions.  Will check for exclusive 
        RayOperations and throw an error if it finds any.
        
        The OpticalEngine will generate six public attributes:
            1: input_signature, the signature of all rays sets input to the sytem.
            2: output_signature, the signature of all ray sets output from the system.
            3: optical_signature, the signature of all optically active boundaries.
            4: target_signature, the signature of all targets.
            4: stop_signature, the signature of all stops.
            5: material_signature, the signature of items in the materials list
                
        These signatures do NOT contain the geometric fields, only extras.
        
    Step 2:
        Build the optical system, and connect it to the engine.
    Step 3 (optional):
        Call engine.annotate to add annotations to the system.  Necessary if the 
        operations require some kind of annotation which wasn't manually added
        to each source / boundary.
    Step 4:
        Call OpticalSystem.update to build everything.  This call is necessary to join
        each element in the various optical system sets into a single object that can be
        passed into the ray tracer.
    Step 5 (optional):
        Call OpticalSystem.validate_system to ensure that all of the parts in the optical
        system have the correct, populated signatures.  This is not necssary, but can help
        catch errors.
    ...
        
    Every RayOperation that will ever be used by the script should be declared in the 
    constructor, and the set signatures should always posess every field that will 
    ever be used by the script.  But since the value of a field in a set may be none, 
    ray operations can be turned on and off as needed to speed things up when they are
    not needed.
        
    """
    
    def __init__(
        self,
        dimension,
        operations,
        optical_system=None,
        compile_technical_intersections=False,
        compile_stopped_rays=False,
        compile_dead_rays=False,
        compile_finished_rays=True,
        compile_active_rays=True,
        dead_ray_length=None,
        compile_geometry_specific_result=False,
        new_ray_length=1.0,
        simple_ray_inheritance={"wavelength"}
    ):
        """
        Parameters
        dimension : int
            Must be exactly 2 or 3.
        operations : list
            A list of operations to perform on the generated rays.
        optical_system : an OpticalSystem, optional
            The system to attach to this engine.  May be None.
        compile_technical_intersections : bool, optional
            Defaults to False, in which case technical boundaries will not be gathered
            to reacting rays, and ray reactions with technical boundaries will not be
            processed.  This improves performance but might possibly prevent some 
            operations from working properly.  An operation will be documented if it
            requires this flag be set to True, but will not on its own be able to
            influence this flag, in which case it will simply raise an exception.
        compile_stopped_rays, compile_dead_rays : bool, optional
            Defaults to False, in which case these rays will not be compiled into a
            list and returned as part of the ray trace result.  Slight performance 
            improvement when off (False).  But for testing purposes you might want to
            visualize what happens with every ray, in which case you can turn this on.
        compile_finished_rays, compile_active_rays : bool, optional
            Same as above two, except it defaults to True.  Compile_finished_rays
            should be True if an optimizer is going to be used on the output.  Active rays
            are always gathered and passed to react, but the active ray history will only
            be maintained if the flag is True.
        dead_ray_length : float or None, optional
            Defaults to None, in which case dead rays will not have their length changed.
            Will only have effect if compile_dead_rays is True.  If not None, dead rays
            will have their length changed by this factor.  Might be useful to visualize
            dead rays leaving system far field.
        compile_geometry_specific_result : bool, optional
            Defaults to False.  If True, will generate separate results for each type
            of boundary (segment, arc, or triangle).  This might possibly be necessary for 
            some ray operation whose implementation differs for different types of 
            boundary.  An operation will be documented if it requires this flag be set to 
            True, but will not on its own be able to influence this flag, in which case it 
            will simply raise an exception.
        new_ray_length : float, optional
            The length of newly generated rays.  Not guarenteed that every operation will
            respect this, but given to those that want it.
        simple_ray_inheritance : set of string, optional
            A set of fields that each new ray will inherit from its most recent ancestor.
            
        """  
        if dimension == 2:
            self.process_projection = self.process_projection_2D
        elif dimension ==3:
            self.process_projection = self.process_projection_3D
        else:
            raise ValueError(
                f"RayEngine: dimension must be 2 or 3, but was given {dimension}."
            )
            
        self._dimension = dimension
        self._check_exclusions(operations)
        self._operations = operations
        self._optical_system = optical_system
        self.compile_technical_intersections = compile_technical_intersections
        self.compile_stopped_rays = compile_stopped_rays
        self.compile_dead_rays = compile_dead_rays
        self.compile_finished_rays = compile_finished_rays
        self.compile_active_rays = compile_active_rays
        self.dead_ray_length = dead_ray_length
        self.compile_geometry_specific_result = compile_geometry_specific_result
        self.new_ray_length = new_ray_length
        
        self.clear_ray_history()
        
        self.last_projection_result = {}
        
        self.input_signature = set()
        self.output_signature = set()
        self.optical_signature = set()
        self.stop_signature = set()
        self.target_signature = set()
        self.material_signature = set()
        self.simple_ray_inheritance = simple_ray_inheritance
        for op in operations:
            self.input_signature = self.input_signature | op.input_signature
            self.output_signature = self.output_signature | op.output_signature
            self.optical_signature = self.optical_signature | op.optical_signature
            self.stop_signature = self.stop_signature | op.stop_signature
            self.target_signature = self.target_signature | op.target_signature
            self.material_signature = self.material_signature | op.material_signature
            self.simple_ray_inheritance = self.simple_ray_inheritance | \
                op.simple_ray_inheritance
                
    def add_inheritable_field(self, fields):
        """
        Adds a set of strings to the engine's simple ray inheritance set.  Ray data fields
        here will automatically pass from parent to child when a new ray is created.
        This process happens before postprocess, and can be used to pass something like 
        a source's rank through to the finished rays.
        """
        if type(fields) is str:
            fields = set(fields)
        self.simple_ray_inheritance = self.simple_ray_inheritance | fields
        
    def _check_exclusions(self, operations):
        exclusions = set()
        used_operations = set()
        for op in operations:
            used_operations.add(op.__class__)
            exclusions = exclusions | op.exclusions
        exclusion_matches = used_operations & exclusions
        if bool(exclusion_matches):
            raise RuntimeError(
                f"RayEngine: discovered exclusive operations: {exclusion_matches}"
            )
        self.operations = operations
        
    def update(self):
        try:
            self.optical_system.update()
        except(AttributeError):
            pass
        
    def annotate(self, op_list=None):
        """
        Add annotations to the sources.  By default will run the annotation (if it is 
        defined) for each operation being used by this engine.  If passed a list of 
        operations to parameter op_list, will only run the annotation functions of those 
        operations.
        """
        if bool(self.optical_system):
            if op_list is None:
                op_list = self._operations
            for op in op_list:
                op.annotate(self)
        else:
            print("No optical system found, so annotating nothing.")
            
    @property
    def dimension(self):
        return self._dimension
        
    @property
    def new_ray_length(self):
        return self._new_ray_length
        
    @new_ray_length.setter
    def new_ray_length(self, val):
        self._new_ray_length = tf.constant(val, dtype=tf.float64)
            
    @property
    def optical_system(self):
        return self._optical_system
        
    @property
    def active_rays(self):
        return ReadOnlySet(amalgamate(self._active_rays))
            
    @property
    def finished_rays(self):
        return ReadOnlySet(amalgamate(self._finished_rays))
            
    @property
    def dead_rays(self):
        return ReadOnlySet(amalgamate(self._dead_rays))
            
    @property
    def stopped_rays(self):
        return ReadOnlySet(amalgamate(self._stopped_rays))
            
    @property
    def all_rays(self):
        return ReadOnlySet(amalgamate(
            self._active_rays + self._finished_rays + self._dead_rays + self._stopped_rays
        ))
        
    @property
    def unfinished_rays(self):
        return self._unfinished_rays
        
    @optical_system.setter
    def optical_system(self, val):
        if val is not None:
            if val.dimension != self.dimension:
                raise ValueError(
                    f"OpticalEngine: attempted to set an optical system with "
                    f"dimension {val.dimension}, but this engine is set to dimension "
                    f"{self.dimension}"
                )
        self._optical_system = val
            
    def validate_system(self):
        """
        Checks that all components of the optical system have the correct signatures
        to run the ray operations.
        """
        if bool(self.optical_system):
            for material in self.optical_system.materials:
                sig = set(material.keys())
                required = self.material_signature
                if not (sig >= required):
                    raise RuntimeError(
                        f"Optical engine failed materials signature check.  "
                        f"System signature is {sig} but needed {required}"
                    )
        
            if self.dimension == 2:
                if bool(self.optical_system._amalgamated_sources):
                    sig = set(self.optical_system._amalgamated_sources.keys())
                    required = SEGMENT_GEO_SIG | self.input_signature
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed sources signature check.  System " 
                            f"signature is {sig}, but needed {required}."
                        )
                if bool(self.optical_system._amalgamated_optical_segments):
                    sig = set(self.optical_system._amalgamated_optical_segments.keys())
                    required = SEGMENT_GEO_SIG | self.optical_signature
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed optical segments signature check.  "
                            f"System signature is {sig}, but needed {required}."
                        )
                if bool(self.optical_system._amalgamated_optical_arcs):
                    sig = set(self.optical_system._amalgamated_optical_arcs.keys())
                    required = ARC_GEO_SIG | self.optical_signature
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed optical arcs signature check.  "
                            f"System signature is {sig}, but needed {required}."
                        )
                if bool(self.optical_system._amalgamated_stop_segments):
                    sig = set(self.optical_system._amalgamated_stop_segments.keys())
                    required = SEGMENT_GEO_SIG | self.stop_signature
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed stop segments signature check.  "
                            f"System signature is {sig}, but needed {required}."
                        )
                if bool(self.optical_system._amalgamated_stop_arcs):
                    sig = set(self.optical_system._amalgamated_stop_arcs.keys())
                    required = ARC_GEO_SIG | self.stop_signature
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed stop arcs signature check.  "
                            f"System signature is {sig}, but needed {required}."
                        )
                if bool(self.optical_system._amalgamated_target_segments):
                    sig = set(self.optical_system._amalgamated_target_segments.keys())
                    required = SEGMENT_GEO_SIG | self.target_signature
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed target segments signature check.  "
                            f"System signature is {sig}, but needed {required}."
                        )
                if bool(self.optical_system._amalgamated_target_arcs):
                    sig = set(self.optical_system._amalgamated_target_arcs.keys())
                    required = ARC_GEO_SIG | self.target_signature
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed target arcs signature check.  "
                            f"System signature is {sig}, but needed {required}."
                        )
            else: # dimension == 3
                if bool(self.optical_system._amalgamated_sources):
                    sig = set(self.optical_system._amalgamated_sources.keys())
                    required = SOURCE_3D_SIG | self.input_signature
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed sources signature check.  System " 
                            f"signature is {sig}, but needed {required}."
                        )
                if bool(self.optical_system._amalgamated_optical):
                    sig = set(self.optical_system._amalgamated_optical.keys())
                    required = TRIANGLE_GEO_SIG | self.optical_signature
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed optical segments signature check.  "
                            f"System signature is {sig}, but needed {required}."
                        )
                if bool(self.optical_system._amalgamated_stop):
                    sig = set(self.optical_system._amalgamated_stop.keys())
                    required = TRIANGLE_GEO_SIG | self.stop_signature
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed stop segments signature check.  "
                            f"System signature is {sig}, but needed {required}."
                        )
                if bool(self.optical_system._amalgamated_target):
                    sig = set(self.optical_system._amalgamated_target.keys())
                    required = TRIANGLE_GEO_SIG | self.target_signature
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed target segments signature check.  "
                            f"System signature is {sig}, but needed {required}."
                        )
        else:
            print("No optical system found, so validating nothing.")
            
    def validate_output(self):
        """
        Checks that the output from ray tracing has the correct signature.
        """
        if self.dimension == 2:
            required = SEGMENT_GEO_SIG | self.output_signature
            for rays in [
                self.active_rays,
                self.finished_rays,
                self.stopped_rays,
                self.dead_rays
            ]:
                if bool(rays):
                    sig = set(rays.keys())
                    if not (sig >= required):
                        raise RuntimeError(
                            f"Optical engine failed output signature check.  System " 
                            f"signature is {sig}, but needed {required}."
                        )
            
    def process_projection_2D(self, input_rays):
        """
        Project input_rays into the optical system, and process the results.
        
        Please note that this function will change input_rays (by projecting them to the
        boundary), so this function should not be applied to a source, but rather to 
        a copy of a source.
        
        This function will sort and gather the rays input to it into four sets:
        active_rays, dead_rays, stopped_rays, and finished_rays.  These sets will returned 
        and will also be concatenated to the class attributes of the same name, storing 
        the full ray history, but only if the corresponding compile flag is set.  
        Active_rays are always compiled, but may not be concatenated to the history.
        
        The profile for a result (the dict at current_pass_result) is:
        rays : a dict containing...
            active : rays that interacted with an optical surface
            finished : rays that hit a target surface
            stopped : rays that hit a stop
            dead : rays that hit nothing
        optical : the boundary data (containing the optical signature but not the 
            geometrics) for the optical surfaces.
        target : the boundary data (containing the target signature but not the 
            geometrics) for the target surfaces.
        stop : the boundary data (containing the stop signature but not the 
            geometrics) for the stop surfaces.
            
        If the compile_geometry_specific_result flag is set, the result 
        (current_pass_result) will contain all of the above and also two copies of itself
        called 'segment' and 'arc' which contain data specific to that boundary type.
        This is for any operations that require geometric data about the boundary.  The
        geometric data for arcs and segments cannot be combined.
        
        But most operations won't need this, so most of the time 
        compile_geometry_specific_result will be false and there will only be one copy
        of the result data.
        """
        if not bool(self.optical_system):
            return {}
        if self.dimension != 2:
            raise RuntimeError("Should not call 2D function on 3D system")
            
        result = {}
        active_rays = {}
        input_ray_fields = input_rays.keys()
        
        has_segments = bool(self.optical_system._merged_segments)
        has_arcs = bool(self.optical_system._merged_arcs)
        
        seg_proj, arc_proj = self.optical_system.intersect(input_rays)
        
        # compile dead rays
        if self.compile_dead_rays:
            dead_rays = {}
            if has_segments and has_arcs:
                is_dead = tf.logical_not(
                    tf.logical_or(seg_proj["valid"], arc_proj["valid"])
                )
            elif has_segments:
                is_dead = tf.logical_not(seg_proj["valid"])
            elif has_arcs:
                is_dead = tf.logical_not(arc_proj["valid"])
            for field in input_ray_fields:
                dead_rays[field] = tf.boolean_mask(
                    input_rays[field],
                    is_dead
                )
            if bool(self.dead_ray_length):
                # change the length of the dead rays
                dead_rays["x_end"] = dead_rays["x_start"] + \
                    self.dead_ray_length * (dead_rays["x_end"] - 
                    dead_rays["x_start"])
                dead_rays["y_end"] = dead_rays["y_start"] + \
                    self.dead_ray_length * (dead_rays["y_end"] - 
                    dead_rays["y_start"])
                    
            self._dead_rays.append(dead_rays)
        
        if has_segments: 
             # update the input_rays with the projections
            input_rays["x_end"] = tf.where(
                seg_proj["valid"], seg_proj["x"], input_rays["x_end"]
            )
            input_rays["y_end"] = tf.where(
                seg_proj["valid"], seg_proj["y"], input_rays["y_end"]
            )
            
            # gather the boundary types
            seg_boundary_type = tf.gather(
                self.optical_system._merged_segments["catagory"], 
                seg_proj["gather_segment"]
            )
            
            # compile active_rays
            active_seg_rays = {}
            is_active_seg = tf.logical_and(
                seg_proj["valid"],
                tf.equal(seg_boundary_type, OPTICAL)
            )
            for field in input_ray_fields:
                active_seg_rays[field] = tf.boolean_mask(
                    input_rays[field],
                    is_active_seg
                )
            if self.compile_active_rays:
                self._active_rays.append(active_seg_rays)
            
            # compile finished_rays
            if self.compile_finished_rays:
                finished_seg_rays = {}
                is_finished_seg = tf.logical_and(
                    seg_proj["valid"],
                    tf.equal(seg_boundary_type, TARGET)
                )
                for field in input_ray_fields:
                    finished_seg_rays[field] = tf.boolean_mask(
                        input_rays[field],
                        is_finished_seg
                    )
                self._finished_rays.append(finished_seg_rays)
                        
            # compile stopped_rays
            if self.compile_stopped_rays:
                stopped_seg_rays = {}
                is_stopped_seg = tf.logical_and(
                    seg_proj["valid"],
                    tf.equal(seg_boundary_type, STOP)
                )
                for field in input_ray_fields:
                    stopped_seg_rays[field] = tf.boolean_mask(
                        input_rays[field],
                        is_stopped_seg
                    )
                self._stopped_rays.append(stopped_seg_rays)
                        
            # compile optical boundaries
            gather_optical_segment = tf.boolean_mask(
                seg_proj["gather_segment"],
                is_active_seg
            )
            optical_segments = {
                field: tf.gather(
                    self.optical_system._amalgamated_optical_segments[field],
                    gather_optical_segment
                )
                for field in self.optical_system._amalgamated_optical_segments.keys()
            }
            optical_seg_norm = tf.boolean_mask(
                seg_proj["norm"], is_active_seg
            )
            try:
                if optical_seg_norm.shape[0] > 0:
                    optical_segments["norm"] = optical_seg_norm
            except(TypeError):
                optical_segments["norm"] = tf.zeros((0,))
            
            # compile techinical boundaries
            # This step is only performed if the both flags are set.
            if self.compile_technical_intersections and self.compile_stopped_rays:
                gather_stop_segment = tf.boolean_mask(
                    (
                        seg_proj["gather_segment"] - 
                        self.optical_system._optical_seg_count
                    ),
                    is_stopped_seg
                )
                stop_segments = {
                    field: tf.gather(
                        self.optical_system._amalgamated_stop_segments[field],
                        gather_stop_segment
                    )
                    for field in self.optical_system._amalgamated_stop_segments.keys()
                }
                stop_seg_norm = tf.boolean_mask(
                    seg_proj["norm"], is_stopped_seg
                )
                try:
                    if stop_seg_norm.shape[0] > 0:
                        stop_segments["norm"] = stop_seg_norm
                except(TypeError):
                    stop_segments["norm"] = tf.zeros((0,)) 
                
            if (self.compile_technical_intersections and 
                self.compile_finished_rays
            ):
                gather_target_segment = tf.boolean_mask(
                    (
                        seg_proj["gather_segment"] - 
                        self.optical_system._optical_seg_count -
                        self.optical_system._stop_seg_count
                    ),
                    is_finished_seg
                )
                target_segments = {
                    field: tf.gather(
                        self.optical_system._amalgamated_target_segments[field],
                        gather_target_segment
                    )
                    for field in \
                        self.optical_system._amalgamated_target_segments.keys()
                }
                finished_seg_norm = tf.boolean_mask(
                    seg_proj["norm"], is_finished_seg
                )
                try:
                    if finished_seg_norm.shape[0] > 0:
                        target_segments["norm"] = finished_seg_norm
                except(TypeError):
                    target_segments["norm"] = tf.zeros((0,)) 
                
            # compile the segment stuff into the segment result
            if self.compile_geometry_specific_result:
                result["segment"] = {
                    "rays": {"active": active_seg_rays},
                    "optical": optical_segments
                }
                if self.compile_finished_rays:
                    result["segment"]["rays"]["finished"] = \
                        finished_seg_rays
                    if self.compile_technical_intersections:
                        result["segment"]["target"] = \
                            target_segments
                if self.compile_stopped_rays:
                    result["segment"]["rays"]["stopped"] = \
                        stopped_seg_rays
                    if self.compile_technical_intersections:
                        result["segment"]["stop"] = \
                            stop_segments
                if self.compile_dead_rays:
                    result["segment"]["rays"]["dead"] = \
                        dead_rays
            
        if has_arcs:
            # update the input_rays with the projections
            input_rays["x_end"] = tf.where(
                arc_proj["valid"], arc_proj["x"], input_rays["x_end"]
            )
            input_rays["y_end"] = tf.where(
                arc_proj["valid"], arc_proj["y"], input_rays["y_end"]
            )
            
            # gather the boundary types
            arc_boundary_type = tf.gather(
                self.optical_system._merged_arcs["catagory"],
                arc_proj["gather_arc"]
            )
            
            # compile active_rays
            active_arc_rays = {}
            is_active_arc = tf.logical_and(
                arc_proj["valid"],
                tf.equal(arc_boundary_type, OPTICAL)
            )
            for field in input_ray_fields:
                active_arc_rays[field] = tf.boolean_mask(
                    input_rays[field],
                    is_active_arc
                )
            if self.compile_active_rays:
                self._active_rays.append(active_arc_rays)

            # compile finished_rays
            if self.compile_finished_rays:
                finished_arc_rays = {}
                is_finished_arc = tf.logical_and(
                    arc_proj["valid"],
                    tf.equal(arc_boundary_type, TARGET)
                )
                for field in input_ray_fields:
                    finished_arc_rays[field] = tf.boolean_mask(
                        input_rays[field],
                        is_finished_arc
                    )
                self._finished_rays.append(finished_arc_rays)
                    
            # compile stopped_rays
            if self.compile_stopped_rays:
                stopped_arc_rays = {}
                is_stopped_arc = tf.logical_and(
                    arc_proj["valid"],
                    tf.equal(arc_boundary_type, STOP)
                )
                for field in input_ray_fields:
                    stopped_arc_rays[field] = tf.boolean_mask(
                        input_rays[field],
                        is_stopped_arc
                    )
                self._stopped_rays.append(stopped_arc_rays)
                           
            # compile optical boundaries
            gather_optical_arc = tf.boolean_mask(
                arc_proj["gather_arc"],
                is_active_arc
            )
            optical_arcs = {
                field: tf.gather(
                    self.optical_system._amalgamated_optical_arcs[field],
                    gather_optical_arc
                )
                for field in self.optical_system._amalgamated_optical_arcs.keys()
            }
            optical_arc_norm = tf.boolean_mask(
                arc_proj["norm"], is_active_arc
            )
            try:
                if optical_arc_norm.shape[0] > 0:
                    optical_arcs["norm"] = optical_arc_norm
            except(TypeError):
                optical_arcs["norm"] = tf.zeros((0,)) 
        
            # compile techinical boundaries
            # This step is only performed if the both flags are set.
            if self.compile_technical_intersections and self.compile_stopped_rays:
                gather_stop_arc = tf.boolean_mask(
                    (
                        arc_proj["gather_arc"] - 
                        self.optical_system._optical_arc_count
                    ),
                    is_stopped_arc
                )
                stop_arcs = {
                    field: tf.gather(
                        self.optical_system._amalgamated_stop_arcs[field],
                        gather_stop_arc
                    )
                    for field in self.optical_system._amalgamated_stop_arcs.keys()
                }
                stop_arc_norm = tf.boolean_mask(
                    arc_proj["norm"], is_stopped_arc
                )
                try:
                    if stop_arc_norm.shape[0] > 0:
                        stop_arcs["norm"] = stop_arc_norm
                except(TypeError):
                    stop_arcs["norm"] = tf.zeros((0,)) 
                
            if (self.compile_technical_intersections and 
                self.compile_finished_rays
            ):
                gather_target_arc = tf.boolean_mask(
                    (
                        arc_proj["gather_arc"] - 
                        self.optical_system._optical_arc_count -
                        self.optical_system._stop_arc_count
                    ),
                    is_finished_arc
                )
                target_arcs = {
                    field: tf.gather(
                        self.optical_system._amalgamated_target_arcs[field],
                        gather_target_arc
                    )
                    for field in self.optical_system._amalgamated_target_arcs.keys()
                }
                finished_arc_norm = tf.boolean_mask(
                    arc_proj["norm"], is_finished_arc
                )
                try:
                    if finished_arc_norm.shape[0] > 0:
                        target_arcs["norm"] = finished_arc_norm
                except(TypeError):
                    target_arcs["norm"] = tf.zeros((0,)) 
                
            # compile the arc stuff into the arc result
            if self.compile_geometry_specific_result:
                result["arc"] = {
                    "rays": {"active": active_arc_rays},
                    "optical": optical_arcs
                }
                if self.compile_finished_rays:
                    result["arc"]["rays"]["finished"] = \
                        finished_arc_rays
                    if self.compile_technical_intersections:
                        result["arc"]["target"] = \
                            target_arcs
                if self.compile_stopped_rays:
                    result["arc"]["rays"]["stopped"] = \
                        stopped_arc_rays
                    if self.compile_technical_intersections:
                        result["arc"]["stop"] = \
                            stop_arcs
                if self.compile_dead_rays:
                    result["arc"]["rays"]["dead"] = \
                        dead_rays
        
        # compile into the unified result
        if has_segments and not has_arcs:
            # has only segments
            result["rays"] = {"active": active_seg_rays}
            result["optical"] = optical_segments
            if self.compile_finished_rays:
                result["rays"]["finished"] = finished_seg_rays
                if self.compile_technical_intersections:
                    result["target"] = target_segments
            if self.compile_stopped_rays:
                result["rays"]["stopped"] = stopped_seg_rays
                if self.compile_technical_intersections:
                    result["stop"] = stop_segments
            
        elif has_arcs and not has_segments:
            # has only arcs
            result["rays"] = {"active": active_arc_rays}
            result["optical"] = optical_arcs
            if self.compile_finished_rays:
                result["rays"]["finished"] = finished_arc_rays
                if self.compile_technical_intersections:
                    result["target"] = target_arcs
            if self.compile_stopped_rays:
                result["rays"]["stopped"] = stopped_arc_rays
                if self.compile_technical_intersections:
                    result["stop"] = stop_arcs
        else:
            # has both, so the whole arc and segment thing has to be concatenated
            ray_sig = active_seg_rays.keys()
            result["rays"] = {
                "active": amalgamate(
                    [active_seg_rays, active_arc_rays]
                )
            }
            result["optical"] = amalgamate(
                [optical_arcs, optical_segments]
            )
            if self.compile_finished_rays:
                result["rays"]["finished"] = amalgamate(
                    [finished_seg_rays, finished_arc_rays]
                )
                if self.compile_technical_intersections:
                    result["target"] = amalgamate(
                    [target_arcs, target_segments]
                )
            if self.compile_stopped_rays:
                result["rays"]["stopped"] = amalgamate(
                    [stopped_seg_rays, stopped_arc_rays]
                )
                if self.compile_technical_intersections:
                    result["stop"] = amalgamate(
                    [stop_arcs, stop_segments]
                )
                
        if self.compile_dead_rays:
            result["rays"]["dead"] = dead_rays
        
        return result
            
    def process_projection_3D(self, input_rays):
        """
        Project input_rays into the optical system, and process the results.
        
        Please note that this function will change input_rays (by projecting them to the
        boundary), so this function should not be applied to a source, but rather to 
        a copy of a source.
        
        This function will sort and gather the rays input to it into four sets:
        active_rays, dead_rays, stopped_rays, and finished_rays.  These sets will returned 
        and will also be concatenated to the class attributes of the same name, storing 
        the full ray history, but only if the corresponding compile flag is set.  
        Active_rays are always compiled, but may not be concatenated to the history.
        
        The profile for a result (the dict at current_pass_result) is:
        rays : a dict containing...
            active : rays that interacted with an optical surface
            finished : rays that hit a target surface
            stopped : rays that hit a stop
            dead : rays that hit nothing
        optical : the optical surfaces.
        target : the target surfaces.
        stop : the stop surfaces.
            
        Unlike the 2D case where arcs and segments cannot be fully merged, the surface
        dicts will contain all of the data for the surfaces.  There are no geometry 
        specific results because there is only one geometry.
        """
        if not bool(self.optical_system):
            return {}
        if self.dimension != 3:
            raise RuntimeError("Should not call 3D function on 2D system")
            
        result = {"rays": {}}
        active_rays = {}
        input_ray_fields = input_rays.keys()
        
        intersect_result = self.optical_system.intersect(input_rays)
        
        # compile dead rays
        if self.compile_dead_rays:
            dead_rays = {}
            is_dead = tf.logical_not(intersect_result["valid"])
            for field in input_ray_fields:
                dead_rays[field] = tf.boolean_mask(
                    input_rays[field],
                    is_dead
                )
            if bool(self.dead_ray_length):
                # change the length of the dead rays
                dead_rays["x_end"] = dead_rays["x_start"] + \
                    self.dead_ray_length * (dead_rays["x_end"] - 
                    dead_rays["x_start"])
                dead_rays["y_end"] = dead_rays["y_start"] + \
                    self.dead_ray_length * (dead_rays["y_end"] - 
                    dead_rays["y_start"])
                dead_rays["z_end"] = dead_rays["z_start"] + \
                    self.dead_ray_length * (dead_rays["z_end"] - 
                    dead_rays["z_start"])
                    
            self._dead_rays.append(dead_rays)
            result["rays"]["dead"] = dead_rays

        # update the input_rays with the projections
        input_rays["x_end"] = tf.where(
            intersect_result["valid"], intersect_result["x"], input_rays["x_end"]
        )
        input_rays["y_end"] = tf.where(
            intersect_result["valid"], intersect_result["y"], input_rays["y_end"]
        )
        input_rays["z_end"] = tf.where(
            intersect_result["valid"], intersect_result["z"], input_rays["z_end"]
        )
        
        # gather the boundary types
        boundary_type = tf.gather(
            self.optical_system._merged["catagory"], 
            intersect_result["gather_trig"]
        )
        
        # compile active_rays
        active_rays = {}
        is_active = tf.logical_and(
            intersect_result["valid"],
            tf.equal(boundary_type, OPTICAL)
        )
        for field in input_ray_fields:
            active_rays[field] = tf.boolean_mask(
                input_rays[field],
                is_active
            )
        if self.compile_active_rays:
            self._active_rays.append(active_rays)
        result["rays"]["active"] = active_rays
        
        # compile finished_rays
        if self.compile_finished_rays:
            finished_rays = {}
            is_finished = tf.logical_and(
                intersect_result["valid"],
                tf.equal(boundary_type, TARGET)
            )
            for field in input_ray_fields:
                finished_rays[field] = tf.boolean_mask(
                    input_rays[field],
                    is_finished
                )
            self._finished_rays.append(finished_rays)
            result["rays"]["finished"] = finished_rays
                    
        # compile stopped_rays
        if self.compile_stopped_rays:
            stopped_rays = {}
            is_stopped = tf.logical_and(
                intersect_result["valid"],
                tf.equal(boundary_type, STOP)
            )
            for field in input_ray_fields:
                stopped_rays[field] = tf.boolean_mask(
                    input_rays[field],
                    is_stopped
                )
            self._stopped_rays.append(stopped_rays)
            result["rays"]["stopped"] = stopped_rays
                    
        # compile optical boundaries
        gather_optical = tf.boolean_mask(
            intersect_result["gather_trig"],
            is_active
        )
        optical = {
            field: tf.gather(
                self.optical_system._amalgamated_optical[field],
                gather_optical
            )
            for field in self.optical_system._amalgamated_optical.keys()
        }
        optical_norm = tf.boolean_mask(
            intersect_result["norm"], is_active
        )
        try:
            if optical_norm.shape[0] > 0:
                optical["norm"] = optical_norm
        except(TypeError):
            optical["norm"] = tf.zeros((0,))
        result["optical"] = optical
        
        # compile techinical boundaries
        # This step is only performed if the both flags are set.
        if self.compile_technical_intersections and self.compile_stopped_rays:
            gather_stop = tf.boolean_mask(
                (
                    intersect_result["gather_trig"] - 
                    self.optical_system._optical_count
                ),
                is_stopped
            )
            stop = {
                field: tf.gather(
                    self.optical_system._amalgamated_stop[field],
                    gather_stop
                )
                for field in self.optical_system._amalgamated_stop.keys()
            }
            stop_norm = tf.boolean_mask(
                intersect_result["norm"], is_stopped
            )
            try:
                if stop_norm.shape[0] > 0:
                    stop["norm"] = stop_norm
            except(TypeError):
                stop["norm"] = tf.zeros((0,))
            result["stop"] = stop
            
        if (self.compile_technical_intersections and 
            self.compile_finished_rays
        ):
            gather_target = tf.boolean_mask(
                (
                    intersect_result["gather_trig"] - 
                    self.optical_system._optical_count -
                    self.optical_system._stop_count
                ),
                is_finished
            )
            target = {
                field: tf.gather(
                    self.optical_system._amalgamated_target[field],
                    gather_target
                )
                for field in \
                    self.optical_system._amalgamated_target.keys()
            }
            finished_norm = tf.boolean_mask(
                intersect_result["norm"], is_finished
            )
            try:
                if finished_norm.shape[0] > 0:
                    target["norm"] = finished_norm
            except(TypeError):
                target["norm"] = tf.zeros((0,))
            result["target"] = target
            
        return result
        
    def single_pass(self, input_rays):
        if not bool(self.optical_system):
            return {}
    
        self.last_projection_result = self.process_projection(input_rays)
        
        # need to cull results that appear present but are actually empty 
        # (number of rays is zero)
        try:
            domain = self.last_projection_result["rays"]
            types = set(domain.keys())
            for t in types:
                if domain[t]["x_start"].shape[0] == 0:
                    domain.pop(t)
        except(KeyError):
            pass
        try:
            domain = self.last_projection_result["arc"]["rays"]
            types = set(domain.keys())
            for t in types:
                if domain[t]["x_start"].shape[0] == 0:
                    domain.pop(t)
        except(KeyError):
            pass
        try:
            domain = self.last_projection_result["seg"]["rays"]
            types = set(domain.keys())
            for t in types:
                if domain[t]["x_start"].shape[0] == 0:
                    domain.pop(t)
        except(KeyError):
            pass
        
        # Feed the result and self into each operation's preprocess.  Preprocess will 
        # typically edit values in the result, but doesn't return anything.
        for op in self._operations:
            op.preprocess(self, self.last_projection_result)
            
        # Now run the main function on each operation, which returns a dict of ray data 
        # for each new ray generated from the projection result.  It is the operation's
        # responsibility to generate and interpret this data.  We need to append it to the
        # new_rays dict only if the operation generated any data.
        new_ray_dict = {}
        for op in self._operations:
            op_result = op.main(self, self.last_projection_result)
            if bool(op_result):
                new_ray_dict[op] = op_result
                
        # Do the simple ray inheritance.
        for op_entry in new_ray_dict.values():
            if bool(op_entry):
                for entry_type, entry in op_entry.items():
                    for sig in self.simple_ray_inheritance:
                        if entry_type == "active":
                            entry["rays"][sig] = \
                                self.last_projection_result["rays"]["active"][sig]
                        elif entry_type == "finished":
                            entry["rays"][sig] = \
                                self.last_projection_result["rays"]["finished"][sig]
                        elif entry_type == "stopped":
                            entry["rays"][sig] = \
                                self.last_projection_result["rays"]["stopped"][sig]
                        elif entry_type == "dead":
                            entry["rays"][sig] = \
                                self.last_projection_result["rays"]["dead"][sig]
                        elif entry_type == "active_seg":
                            entry["rays"][sig] = \
                                self.last_projection_result["seg"]["rays"]["active"][sig]
                        elif entry_type == "finished_seg":
                            entry["rays"][sig] = \
                                self.last_projection_result["seg"]["rays"]["finished"][sig]
                        elif entry_type == "stopped_seg":
                            entry["rays"][sig] = \
                                self.last_projection_result["seg"]["rays"]["stopped"][sig]
                        elif entry_type == "dead_seg":
                            entry["rays"][sig] = \
                                self.last_projection_result["seg"]["rays"]["dead"][sig]
                        elif entry_type == "active_arc":
                            entry["rays"][sig] = \
                                self.last_projection_result["arc"]["rays"]["active"][sig]
                        elif entry_type == "finished_arc":
                            entry["rays"][sig] = \
                                self.last_projection_result["arc"]["rays"]["finished"][sig]
                        elif entry_type == "stopped_arc":
                            entry["rays"][sig] = \
                                self.last_projection_result["arc"]["rays"]["stopped"][sig]
                        elif entry_type == "dead_arc":
                            entry["rays"][sig] = \
                                self.last_projection_result["arc"]["rays"]["dead"][sig]

        # Now run each operation's postprocess, which will interpret and possibly change 
        # the data in new_ray_dict.
        for op in self._operations:
            op.postprocess(self, self.last_projection_result, new_ray_dict)
        
        new_ray_list = []    
        # Now we have to interpret and amalgamate the ray data in new_ray_dict.  
        # new_ray_list will be a list of ray data dicts pulled out of the 
        # new_ray_dict, and masked.
        for op_entry in new_ray_dict.values():
            if bool(op_entry):
                for geo_entry in op_entry.values():
                    raw_ray_set = geo_entry["rays"]
                    valid = geo_entry["valid"]
                    new_ray_list.append({
                        key: tf.boolean_mask(field, valid) \
                        for key, field in raw_ray_set.items()
                    })
                    
        return amalgamate(new_ray_list)
            
    def clear_ray_history(self):
        self._active_rays = []
        self._finished_rays = []
        self._stopped_rays = []
        self._dead_rays = []
        self._unfinished_rays = {}
        
    def ray_trace(self, max_iterations=25):
        """
        Trace the optical system.
        
        max_trace_iterations : int, optional
            The maximum number of passes that the tracer will execute on a system, before
            stopping.  Will stop early if there are no more active rays left to trace.):
        """
        if not bool(self.optical_system):
            return
        
        self.clear_ray_history()    
        starting_rays = self.optical_system._amalgamated_sources.copy()
        for i in range(max_iterations):
            result = self.single_pass(starting_rays)
            
            if bool(result):
                starting_rays = result
            else:
                break
        
    
    
    
    
    
    
    
    
    
    
    
