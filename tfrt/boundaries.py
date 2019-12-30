"""
Classes that represent optical boundaries.
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from tfrt.engine import amalgamate
from tfrt.update import RecursivelyUpdatable

# =====================================================================================

class BoundaryBase(RecursivelyUpdatable):
    """
    Base class for boundaries.
    
    Requires sub classes to implement _update, _generate_update_handles.
    """
    
    def __init__(
        self,
        name=None,
        mat_in=None, 
        mat_out=None,
        **kwargs
    ):
        self._name = name
        self._fields = {}
        if mat_in is not None and mat_out is not None:
            self.optically_active = True
            self.mat_in = mat_in
            self.mat_out = mat_out
        else:
            self.optically_active = False
        super().__init__(**kwargs)
        
    @property
    def name(self):
        return self._name
    
    @property
    @abstractmethod
    def dimension(self):
        raise NotImplementedError
    
    @property   
    def keys(self):
        return self._fields.keys
        
    def __getitem__(self, key):
        return self._fields[key]

    def __setitem__(self, key, item):
        self._fields[key] = item
        
# -------------------------------------------------------------------------------------
        
class ManualSegmentBoundary(BoundaryBase):
    """
    Class that builds a boundary directly from a set of points.
    
    If update_function is specified, it must be a callable that takes no argument and
    returns four values, x_start, y_start, x_end, y_end.  Defaults to None, and if left
    as None, calling update on this class will do nothing.
    
    segment_splitter is a convenient function for converting from the segment data
    format to the one needed by update_function.
    
    Data can be fed to this class either by indexing its fields after instantiation
    (ManualBoundary["x_start"] = my_points) or by calling feed_segments, which takes
    a list of 4-tuples, each of which is (x_start, y_start, x_end, y_end).
    
    Can add new fields at any time simply by indexing the new field.
    """
    def __init__(self, update_function=None, **kwargs):
        self.update_function = update_function
        super().__init__(**kwargs)
        
    def _generate_update_handles(self):
        return []
        
    @property
    def dimension(self):
        return 2
            
    def _update(self):
        if self.update_function is not None:
            self["x_start"], self["y_start"], self["x_end"], self["y_end"] = \
                self.update_function()
                
    def feed_segments(self, segments):
        self["x_start"], self["y_start"], self["x_end"], self["y_end"] = \
            self.segment_splitter(segments)
    
    @staticmethod    
    def segment_splitter(segments):
        x_start, y_start, x_end, y_end = tf.unstack(segments, axis=1)
        return tf.reshape(x_start, (-1,)), tf.reshape(y_start, (-1,)), \
            tf.reshape(x_end, (-1,)), tf.reshape(y_end, (-1,))
            
# =====================================================================================

class ParametricSegmentBoundary(BoundaryBase):
    """
    A single parametric surface.
    
    This class generates a single open curved surface, approximated as a series of
    line segments.  Takes two sets of base points (distribution.BasePointDistribution-
    Base) that must sample exactly the same number of points.  I prefer using aperature
    point distributions, though beams would work as well.  These points are matched
    1:1 and define a series of vectors pointing from one to the other.  The
    actual vertices of the line segments will lie along these vectors, and the 
    parameter tells how far along the vector the segment lies.  Parameters can take any
    value, though they might want to be constrained.  If the parameter is tf.zeros(),
    then the surface will pass through every point in zero_distribution.  If the 
    parameter is tf.ones(), then the surface will pass through every point in 
    ones_distribution.  The surface will have one less segment than the base point 
    distributions have points.
    
    The class contains an attribute: parameter, which is a tf.Variable that stores
    the shape of the surface.  This variable can be the target of a TF optimizer.
    I am not sure what will happen to the previous value of the variable if 
    the number of segments changes, so I consider a variable number of segments to be
    sketchy, but nonetheless I will allow it; pass validate_shape=False to allow this
    behavior.
    
    I will also allow the user to construct their own variable and plug it into the
    boundary.
    
    Always check the norm of the surface (can be done with drawing.SegmentDrawer).  The 
    norm should be defined as... If you take a vector from the distribution start point to 
    the distribution end point, and rotate it 90 degrees counterclockwise, that should be 
    the direction the norm points.  If the norm is the wrong way, reverse the order of the 
    distribution points, or you can simply toggle the value of flip_norm, to flip the norm 
    around.
    """
    def __init__(
        self,
        zero_distribution,
        one_distribution,
        flip_norm=False,
        initial_parameters=0.0,
        validate_shape=True,
        parameters=None,
        **kwargs
     ):
        self._zero_distribution = zero_distribution
        self._one_distribution = one_distribution
        self.flip_norm = flip_norm
        
        if parameters is None:
            initial_parameters = tf.cast(
                tf.broadcast_to(
                    initial_parameters,
                    tf.shape(zero_distribution.base_points_x)
                ),
                tf.float64
            )
            self.parameters = tf.Variable(
                initial_parameters,
                validate_shape=validate_shape,
                dtype=tf.float64
            )
        else:
            self.parameters = parameters
        super().__init__(**kwargs)
        
    def _generate_update_handles(self):
        return [self._zero_distribution.update, self._one_distribution.update]
        
    def _update(self):
        self["x_start"], self["y_start"], self["x_end"], self["y_end"], = \
            self._update_internal(
                self._zero_distribution.base_points_x,
                self._zero_distribution.base_points_y,
                self._one_distribution.base_points_x,
                self._one_distribution.base_points_y,
                self.parameters,
                self.flip_norm
            )
        if self.optically_active:
            shape = tf.shape(self["x_start"])
            self["mat_in"] = tf.cast(tf.broadcast_to(self.mat_in, shape), tf.int64)
            self["mat_out"] = tf.cast(tf.broadcast_to(self.mat_out, shape), tf.int64)
            
    @staticmethod
    @tf.function(input_signature=
    [
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.bool)
    ])
    def _update_internal(x_zero, y_zero, x_one, y_one, parameter, flip_norm):
        x_start = x_zero + parameter * (x_one - x_zero)
        y_start = y_zero + parameter * (y_one - y_zero)
        if flip_norm:
            return x_start[1:], y_start[1:], x_start[:-1], y_start[:-1]
        else:
            return x_start[:-1], y_start[:-1], x_start[1:], y_start[1:]
            
    @property
    def dimension(self):
        return 2
        
    @property
    def zero_distribution(self):
        return self._zero_distribution
        
    @property
    def one_distribution(self):
        return self._one_distribution
        
# =====================================================================================

class Constraint(ABC):
    """
    A class that handles a constraint between two surfaces.
    
    This class can be used to impose a constraint on any two objects that expose the
    attribute 'parameters' (like a parametric boundary).  This class will generate a 
    function that can be added an RecursivelyUpdatable's update_handles that will perform
    the work of enforcing the constraint.
    
    Constraint sub classes should process various parameters specific to the 
    subclass inside the subclass constructor.
    
    The design intent for this class assumes that it will most often be used with multi- 
    boundaries.  In this circumstance, the user can use the default values for parent
    in the constructor, which will result in the constraint being relative to either the 
    previous surface in the multi-boundary or to the zero in parameter space.  But I also
    want the user to have flexibility about which surface to constrain relative to when 
    using a multi-boundary.  And I also want this class to be usable manually, outside of
    the context of a multi-boundary.
    
    The target object, the object being constrained, will be fed to the constraint's make 
    method, which must be called to generate the actual update function.  Because of this,
    a single constraint object could actually be used to impose multiple identical 
    constraints on different pairs of objects.
    
    The parent object, the object that the constraint is relative to, will be specified in
    the constructor, but must also be fed to make().  There are several modes the 
    constraint can be in which will determine how this parameter is interpreted:  
    1) If parent is the string 'prev', which is the default option, then the parent passed
        to make() must be a list of surfaces, and target must be an integer that specifies
        the index of the constrained surface within this list of surfaces.  The parent that
        will be used will either be the surface a index target-1, or zero if target==0.  
        This should be the choice most commonly used when using this constraint with a 
        multi-boundary.
    2) If parent is the string 'literal', then the parameters target and 
        parent are interpreted by make() as literals - the surfaces.  This will usually be 
        what you want if you are using the constraint manually (outside of a multi-
        boundary).  This option is not compatable with a multi-boundary.
    3) If parent is the string 'zero', will use a value of zero for the parent's parameters.
        and make() will ignore the parameter passed to it.  Target will be interpreted as 
        literal.  You will still have to pass something to the parameter in make(), since 
        parent is not an optional parameter, but you may pass None.  Since the constraint 
        will automatically choose this option if 'prev' is used and the target is the first 
        surface, this option will only rarely need to be invoked manually by the user.
    4) If parent is an integer, then the parent passed to make() must be a list of 
        surfaces, and target must be an integer that specifies the index of the constrained 
        surface within this list of surfaces.  Parent must be >= 0.
    """
    def __init__(self, parent="prev"):
        if type(parent) is int:
            if parent < 0:
                raise ValueError("Constraint: integer parent must be >= 0")
        elif type(parent) is str:
            if parent not in {"zero", "prev"}:
                raise ValueError("Constraint: string parent must be 'zero' or 'prev'.")
        self._parent = parent
        
    @property
    def parent(self):
        return self._parent
    
    def interpret_params(self, target, parent):
        """
        Subclasses should usually call this inside make to interpret the parameters passed
        to make.  But this isn't strictly necessary.
        """
        if self._parent == "literal":
            return target.parameters, parent.parameters
        if self._parent == "zero":
            return target.parameters, tf.zeros_like(
                target.parameters,
                dtype=target.parameters.dtype
            )
        if self._parent == "prev":
            if target == 0:
                new_target = parent[target].parameters
                return new_target, tf.zeros_like(new_target, dtype=new_target.dtype)
            else:
                new_target = parent[target].parameters
                return new_target, parent[target-1].parameters
        # self._parent is an int > 0
        return parent[target].parameters, parent[self._parent].parameters
            
    
    @abstractmethod
    def make(self, target, parent):
        raise NotImplementedError

# --------------------------------------------------------------------------------------
        
class NoConstraint(Constraint):
    """
    Constraint that does nothing, in case you don't want one but still have to specify
    for the multi_boundary.
    """
    def make(self, target, parent):
        return lambda: None

# --------------------------------------------------------------------------------------

class PointConstraint(Constraint):
    """
    Ensures that the distance between two specific points on two surfaces is fixed.
    
    By distance I mean the distance in parameter space, not the actual Euclidian distance.
    It will equal the Euclidian distance as long as the surfaces' zero and one points
    are 1 unit apart, and target_vertex == parent_vertex.
    
    This constraint is not guarenteed to prevent the constrained surface from clipping 
    the parent surface.
    
    self.distance : float
        The distance between the specified vertices.  May be positive or negative
    target_vertex : int
        The index of the vertex on the constrained surface to use.
    parent : int, a boundary, or None; optional
        See the constraint base class for details.
    parent_vertex : int or None, optional
        The index of the vertex on the parent surface to use.  If none, will be the same
        as target_vertex.
    """
    def __init__(self, distance, target_vertex, parent_vertex=None, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance
        self.target_vertex = target_vertex
        self.parent_vertex = parent_vertex or target_vertex
    
    def make(self, target, parent):
        def handler(target=target, parent=parent):
            target, parent = self.interpret_params(target, parent)
            diff = parent[self.parent_vertex] - target[self.target_vertex] + self.distance
            diff = tf.broadcast_to(diff, target.shape)
            target.assign_add(diff)
            
        return handler

# --------------------------------------------------------------------------------------
        
class ThicknessConstraint(Constraint):
    """
    Ensures that the thickness of a layer is fixed.
    
    Will keep the minimum or maximum distance between any points on the surfaces fixed.
    
    By distance I mean the distance in parameter space, not the actual Euclidian distance.
    It will equal the Euclidian distance as long as the multi_boundary's zero and one points
    are 1 unit apart, when measured perpendicular to the lines connecting each zero and 
    one point.
    
    On min mode, should always keep the constrained surface from clipping the parent
    surface.  This is not guarenteed when on max mode.
     
    self.distance : float
        The distance between the surfaces.  May be positive or negative
    mode : string
        Must be either 'min' or 'max'.  If min, the minimum distance between the surfaces
        will be fixed.  If max, will use the maximum distance instead.
    parent : int or None, optional
        The index of the surface this constraint is relative to, within the multi_boundary.
        Must be < len(multi_boundary.surfaces).  Defaults to None, in which case
        the previous surface is used.  May be -1, in which case the constraint is made
        with respect to the multi_boundary zero point.  This can, for instance, be used to 
        fix a point on the first surface.
    """
    def __init__(self, distance, mode, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance
        self.mode = mode
    
    @property
    def mode(self):
        return self._mode
        
    @mode.setter
    def mode(self, val):
        if val not in {"min", "max"}:
            raise ValueError("ThicknessConstraint: mode must be either 'min' or 'max'.")
        else:
            self._mode = val
            if val == "min":
                self._reduce = tf.reduce_max
            else:
                self._reduce = tf.reduce_min
    
    def make(self, target, parent):
        def handler(target=target, parent=parent):
            target, parent = self.interpret_params(target, parent)
            diff = self._reduce(parent - target) + self.distance
            diff = tf.broadcast_to(diff, target.shape)
            target.assign_add(diff)
            
        return handler

# =====================================================================================
        
class ParametricMultiSegmentBoundary(BoundaryBase):
    """
    Multiple segment surfaces that share a common set of base points.
    
    This class is intended to be used to generate several boundaries that are tightly 
    related to each other, like a single fixed optical component.  For disparate optical
    components, it might make more sense to simply use multiple ParametricSegmentBoundary,
    or even multiple ParametricMultiSegmentBoundary.    
    
    This class uses ParametricSegmentBoundary to construct each individual layer the optic,
    so it inherits a lot of features from that class.  One difference is that flip_norm is 
    now a required parameter to the constructor, since all the surfaces share the same base
    points, and you probably don't want them all facing the same way.
    
    The reason that this class uses a shared set of base points is because that makes it 
    really easy to apply constraints between the surfaces.  If the distance between each
    zero point and each corresponding one point is 1.0, then the constraints will take
    the physical interpretation as being the distance between the surfaces.
    
    This class can be indexed, has fields and a signature like other boundaries.  It can
    also be updated.  This is useful for something like feeding this optic to a drawer.  
    But for connecting this object to the OpticalSystem, I recommend using the attribute
    surfaces, which returns a list to each individual ParametricSegmentBoundary.  It
    can be concatenated with any other segment boundary lists and fed to 
    OpticalSystem.optical_segments.  
    
    Each ParametricSegmentBoundary will be given a callable that will enforce the 
    constraint.  This is the only way that constraints can be applied (automatically), so 
    don't throw them away if you prune the update tree.
    
    You can get access to the variable that holds the parameters of the nth surface
    via ParametricMultiSegmentBoundary.surfaces[n].parameters
    """
    def __init__(
        self,
        zero_distribution,
        one_distribution,
        constraints, 
        flip_norm,
        initial_parameters=0.0,
        validate_shape=True,
        parameters=None,
        mat_in=None, 
        mat_out=None,
        **kwargs
     ):
        """
        zero_distribution : BasePointDistributionBase
            The locations of the first surface's vertices when parameter = 0
        one_distribution : BasePointDistributionBase
            The locations of the first surface's vertices when parameter = 1
        constraints : list Constraint
            A list that defines the constraints between surfaces.  The size of this
            list determines the number of surfaces generated, and every other parameter, if
            it is a list, must have the same length as this one.
        flip_norm : list of bool
            Flips the norm of each surface.
        n_in : list
            The value to be stored for the refractive index inside the surface, for each
            surface.
        n_out : list
            The value to be stored for the refractive index outside the surface, for each
            surface.
        n_mode 
        """
        self._zero_distribution = zero_distribution
        self._one_distribution = one_distribution
        try:
            self._surface_count = len(constraints)
        except(TypeError) as e:
            raise ValueError(
                "ParametricMultiSegmentBoundary: constraints must be iterable."
            ) from e
        
        self.flip_norm = flip_norm
        try:
            if len(self.flip_norm) != self._surface_count:
                raise ValueError(
                    "ParametricMultiSegmentBoundary: constraints and flip_norm must have " 
                    "the same size."
                )
        except(TypeError) as e:
            raise ValueError(
                "ParametricMultiSegmentBoundary: flip_norm must be iterable."
            ) from e
        
        try:
            if len(initial_parameters) != self._surface_count:
                raise ValueError(
                    "ParametricMultiSegmentBoundary: constraints and initial_parameters "
                    "must have the same size."
                )
        except(TypeError):
            initial_parameters = [initial_parameters] * self._surface_count
        self.initial_parameters = initial_parameters
        
        try:
            if len(validate_shape) != self._surface_count:
                raise ValueError(
                    "ParametricMultiSegmentBoundary: constraints and validate_shape "
                    "must have the same size."
                )
        except(TypeError):
            validate_shape = [validate_shape] * self._surface_count
        self.validate_shape = validate_shape
        
        if parameters is None:
            parameter_size = tf.shape(zero_distribution.base_points_x)
            self.parameters = [
                tf.Variable(
                    tf.cast(
                        tf.broadcast_to(
                            initial_parameter,
                            parameter_size
                        ),
                        tf.float64,
                    ),
                    validate_shape=validate_shape,
                    dtype=tf.float64
                ) 
                for initial_parameter, validate_shape in zip(
                    self.initial_parameters,
                    self.validate_shape
                )
            ]
        else:
            self.parameters = parameters
            try:
                if len(self.parameters) != self._surface_count:
                    raise ValueError(
                        "ParametricMultiSegmentBoundary: constraints and parameters "
                        "must have the same size."
                    )
            except(TypeError) as e:
                raise TypeError(
                    "ParametricMultiSegmentBoundary: parameters must be None or iterable."
                ) from e
                
        if mat_in is not None and mat_out is not None:
            self.optically_active = True
            
            try:
                if len(mat_in) != self._surface_count:
                    raise ValueError(
                        "ParametricMultiSegmentBoundary: constraints and mat_in "
                        "must have the same size."
                    )
            except(TypeError):
                mat_in = [mat_in] * self._surface_count      
            self.mat_in = mat_in
            
            try:
                if len(mat_out) != self._surface_count:
                    raise ValueError(
                        "ParametricMultiSegmentBoundary: constraints and mat_out "
                        "must have the same size."
                    )
            except(TypeError):
                mat_out = [mat_out] * self._surface_count      
            self.mat_out = mat_out
        else:
            self.mat_in = [None] * self._surface_count 
            self.mat_out = [None] * self._surface_count
            self.optically_active = False
        
        self.surfaces = [
            ParametricSegmentBoundary(
                zero_distribution,
                one_distribution,
                flip_norm=flip_norm,
                parameters=parameter,
                mat_in=mat_in,
                mat_out=mat_out,
                **kwargs
            )
            for flip_norm, parameter, mat_in, mat_out in zip(
                self.flip_norm,
                self.parameters,
                self.mat_in,
                self.mat_out
            )
        ]
        
        self.constraints = constraints
        for i in range(self.surface_count):
            surface = self.surfaces[i]
            constraint = self.constraints[i]
            # If the constraint explicitly attached to zero, then we need the target
            # to be literal, instead of an index.
            if constraint.parent != "zero":
                surface.update_handles.append(constraint.make(i, self.surfaces))
            else:
                surface.update_handles.append(constraint.make(self.surfaces[i], None))
                
        super().__init__(**kwargs)
        
    def _update(self):
        self._fields = amalgamate(self.surfaces)
        
    def _generate_update_handles(self):
        return [self._zero_distribution.update, self._one_distribution.update] + \
            [surface.update for surface in self.surfaces]
            
    @property
    def dimension(self):
        return 2
            
    @property
    def surface_count(self):
        return self._surface_count

# =====================================================================================
        
class ManualArcBoundary(BoundaryBase):
    """
    Class that builds a boundary directly from data.
    
    If update_function is specified, it must be a callable that takes no argument and
    returns five values, x_center, y_center, angle_start, angle_end, radius.  Defaults to 
    None, and if left as None, calling update on this class will do nothing.
    
    Can add new fields at any time simply by indexing the new field.
    """
    def __init__(self, update_function=None, **kwargs):
        self.update_function = update_function
        super().__init__(**kwargs)
        
    def _generate_update_handles(self):
        return []
        
    @property
    def dimension(self):
        return 2
            
    def _update(self):
        if self.update_function is not None:
            self["x_center"], self["y_center"], self["angle_start"], self["angle_end"], \
                self["radius"] = self.update_function()  
        
        
        
        
        
        
        
        
        
        
        
        
        
