"""
Classes that represent optical boundaries.
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
import pyvista as pv

from tfrt.engine import amalgamate
from tfrt.update import RecursivelyUpdatable
import tfrt.mesh_tools as mt

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
    
    A parametric multi surface will automatically handle the constraints.  If you need to
    add a constraint manually, here is how.
        
    my_parametric_surface.update_handles.append(
        my_constraint.make(my_parametric_surface, my_target, my_parent)
    )
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
        
# --------------------------------------------------------------------------------------

class ClipConstraint:
    """
    Clips the parameters of a surface to lie within certain values.  Unlike the other
    constraints, this one only accepts absolute values.
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        
    def make(self, target, _):
        def handler(target=target):
            target.parameters.assign(tf.clip_by_value(
                target.parameters,
                self.lower,
                self.upper
            ))
        return handler

# =========================================================================================

class VectorGeneratorBase(ABC):
    """
    Base class for vector generators.
    
    Vector generators can take whatever parameters they need, and should contain their
    parameters and settings as read/write public attributes.  But the only thing the must
    do is define a method generate, which takes a tf.float64 tensor of shape (N, 3) and 
    returns a tensor of the same shape and type that represents a 3-vector field that
    can be attached to the set of input points to parametrize the surface.  Strongly
    recommended that subclasses ensure that the return vector field is normalized.
    """
    @abstractmethod
    def generate(self, zero):
        raise NotImplementedError
        
    @staticmethod
    def normalize(val):
        return tf.linalg.normalize(val, axis=1)[0]
        
# -----------------------------------------------------------------------------------------

class SecondSurfaceVG(VectorGeneratorBase):
    """
    Vector generator that uses a second surface to define its vectors.
    
    The second surface may be a string, in which case it is interpreted as a filename
    and read into a pyvista PolyData.  Or it can be a pyvista PolyData.  Or it can be
    a tensor of shape (None, 3).
    
    Whatever the origin of surface it must have exactly the same number of vertices as
    the zero points that will be fed to it, and in order to work correctly the vertices
    must be in the same order.  No idea how to help ensure this, or to reorder the
    vertices of a mesh.  The best way to use this class is probably to use two copies of the
    same mesh, with some transformation applied to the second.
    """
    def __init__(self, surface):
        self.surface = surface
        
    def generate(self, zero):
        return self.normalize(self.points - zero)
    
    @property
    def surface(self):
        return self._surface
       
    @surface.setter 
    def surface(self, val):
        # if val is a string, interpret it as a file name, and convert it to a mesh
        if type(val) is str:
            val = pv.read(surface)
            
        try:
            # set points from val.points (works if val is a pyvista mesh)
            self.points = val.points
            self._surface = val
        except(AttributeError):
            # if that fails, val is a tensor, so use it directly
            self.points = val
            self._surface = None
            
    @property
    def points(self):
        return self._points
        
    @points.setter
    def points(self, val):
        self._points = tf.cast(val, tf.float64)
        
# -----------------------------------------------------------------------------------------

class FromPointVG(VectorGeneratorBase):
    """
    Vector generator that projects from a single 3-D point.
    """
    def __init__(self, point):
        self.point = point
        
    def generate(self, zero):
        return self.normalize(zero - self.point)
        
    @property
    def point(self):
        return self._point
        
    @point.setter
    def point(self, val):
        self._point = tf.cast(val, tf.float64)

# -----------------------------------------------------------------------------------------
        
class FromVectorVG(VectorGeneratorBase):
    """
    Vector generator that just copies an input vector.
    
    The input may be either a single 3-vector, or an (N, 3) tensor which is a list of
    vectors.  If a list of vectors, it is the responsibility of the user to ensure
    that they broadcast to the shape of the points.
    """
    def __init__(self, vector):
        self.vector = vector
        
    def generate(self, zero):
        return self.normalize(tf.broadcast_to(self.vector, tf.shape(zero)))
        
    @property
    def vector(self):
        return self._vector
        
    @vector.setter
    def vector(self, val):
        self._vector = tf.cast(val, tf.float64)
        
# -----------------------------------------------------------------------------------------
        
class FromAxisVG(VectorGeneratorBase):
    """
    Vector generator that projects perpendicular to a line, through points.
    
    The input will always be two 3-vectors that determine the axis/line that is the
    projection origin.  The first argument is always a point on the line.  The second
    argument is a required keyword argument that is either a second point, or a vector. 
    """
    def __init__(self, first, **kwargs):
        self.axis_point = tf.cast(first, tf.float64)
        
        # interpret the second parameter
        try:
            axis_vector = kwargs["point"] - self.axis_point
        except(KeyError):
            try:
                axis_vector = kwargs["direction"]
            except(KeyError) as e:
                raise ValueError(
                    "FromAxisVG: Must provide a kwarg 'point' or 'direction' to define "
                    "the axis."
                ) from e
        axis_vector = tf.reshape(axis_vector, (1, 3))
        self.axis_vector = self.normalize(tf.cast(axis_vector, tf.float64))
        
    def generate(self, zero):
        axis_vector = tf.broadcast_to(self.axis_vector, tf.shape(zero))
        d = tf.reduce_sum((zero - self.axis_point) * axis_vector, axis=1)
        closest_point = self.axis_point + axis_vector * tf.reshape(d, (-1, 1))
        
        return self.normalize(zero - closest_point)

# =========================================================================================

class BoundaryBase(RecursivelyUpdatable):
    """
    Base class for boundaries.

    Requires sub classes to implement _update, _generate_update_handles.
    
    Parameter material is a dict whose keys should be added to the boundary's fields.
    """
    
    def __init__(self, name=None, material_dict={}, **kwargs):
        self._name = name
        self._fields = {} 
        self.material_dict = material_dict
        super().__init__(**kwargs)
        self.update_materials()
        
    @property
    def name(self):
        return self._name
    
    @property
    @abstractmethod
    def dimension(self):
        raise NotImplementedError
        
    @abstractmethod
    def update_materials(self):
        """
        Sets the fields from the data in self.material_dict.  This function is a convenience
        that automatically broadcasts scalar material values.  If the value for a key isn't
        a scalar, then it is up to the user to ensure that it is compatible with the 
        boundary's geometric data (meaning that it's first dimension is equal to the number
        of boundaries.)
        """
        raise NotImplementedError
        
    def _update_materials(self, broadcast_shape):
        for field, value in self.material_dict.items():
            if tf.rank(value) < 1:
                self[field] = tf.broadcast_to(value, broadcast_shape)
            else:
                self[field] = value
    
    @property   
    def keys(self):
        return self._fields.keys
        
    def __getitem__(self, key):
        return self._fields[key]

    def __setitem__(self, key, item):
        self._fields[key] = item
        
# =========================================================================================

class ArcBoundaryBase(BoundaryBase):
    """
    Base class for arcs.
    """
    @property
    def dimension(self):
        return 2
        
    def update_materials(self):
        try:
            self._update_materials(self, tf.shape(self["x_center"]))
        except(KeyError):
            pass

# -----------------------------------------------------------------------------------------

class ManualArcBoundary(ArcBoundaryBase):
    """
    Class that builds a boundary directly from data.
    
    If update_function is specified, it must be a callable that takes no argument and
    returns five values, x_center, y_center, angle_start, angle_end, radius.  Defaults to 
    None, and if left as None, calling update on this class will do nothing.
    
    Can add new fields at any time simply by indexing the new field.
    """
        
    def _generate_update_handles(self):
        return []
            
    def _update(self):
        pass
                
# =========================================================================================

class SegmentBoundaryBase(BoundaryBase):
    """
    Base class for segments.
    """
    @property
    def dimension(self):
        return 2
        
    def update_materials(self):
        try:
            self._update_materials(self, tf.shape(self["x_start"]))
        except(KeyError):
            pass

# ------------------------------------------------------------------------------------------
       
class ManualSegmentBoundary(SegmentBoundaryBase):
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
    def _generate_update_handles(self):
        return []
                
    def feed_segments(self, segments):
        self["x_start"], self["y_start"], self["x_end"], self["y_end"] = \
            self.segment_splitter(segments)
            
    def _update(self):
        pass
    
    @staticmethod    
    def segment_splitter(segments):
        x_start, y_start, x_end, y_end = tf.unstack(tf.cast(segments, tf.float64), axis=1)
        return tf.reshape(x_start, (-1,)), tf.reshape(y_start, (-1,)), \
            tf.reshape(x_end, (-1,)), tf.reshape(y_end, (-1,))
            
# -----------------------------------------------------------------------------------------

class ParametricSegmentBoundary(SegmentBoundaryBase):
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
                    (tf.shape(zero_distribution.points)[0],)
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
                self._zero_distribution.points,
                self._one_distribution.points,
                self.parameters,
                self.flip_norm
            )
            
    
    """@tf.function(input_signature=
    [
        tf.TensorSpec(shape=(None, 2), dtype=tf.float64),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.bool)
    ])"""
    @staticmethod
    def _update_internal(zero, one, parameter, flip_norm):
        points = zero + tf.reshape(parameter, (-1, 1)) * (one - zero)
        if flip_norm:
            return points[1:, 0], points[1:, 1], points[:-1, 0], points[:-1, 1]
        else:
            return points[:-1, 0], points[:-1, 1], points[1:, 0], points[1:, 1]
        
    @property
    def zero_distribution(self):
        return self._zero_distribution
        
    @property
    def one_distribution(self):
        return self._one_distribution

# -----------------------------------------------------------------------------------------
        
class ParametricMultiSegmentBoundary(SegmentBoundaryBase):
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
        material_list=[],
        **kwargs
     ):
        """
        zero_distribution : BasePointDistributionBase
            The locations of the surfaces' vertices when parameter = 0
        one_distribution : BasePointDistributionBase
            The locations of the surfaces' vertices when parameter = 1
        constraints : list Constraint
            A list that defines the constraints between surfaces.  The size of this
            list determines the number of surfaces generated, and every other parameter, if
            it is a list, must have the same length as this one.
        flip_norm : list of bool
            Flips the norm of each surface.
        initial_parameters : list of tensor, optional
            The initial value of the parameter for each surface.
        validate_shape : list of bool, optional
            Whether variable assignment done on the parameters of each surface should 
            perform shape checking.
        parameters : list of tf.Variable, optional
            The variable to use for each surface.
        material_list : list of dictionaries, optional
            List of material data dictionaries to pass to each surface.
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
        initial_parameters = initial_parameters
        
        try:
            if len(validate_shape) != self._surface_count:
                raise ValueError(
                    "ParametricMultiSegmentBoundary: constraints and validate_shape "
                    "must have the same size."
                )
        except(TypeError):
            validate_shape = [validate_shape] * self._surface_count
        validate_shape = validate_shape
        
        if parameters is None:
            parameter_size = tf.shape(zero_distribution.points)[0]
            parameters = [
                tf.Variable(
                    tf.cast(
                        tf.broadcast_to(
                            initial_parameter,
                            (parameter_size,)
                        ),
                        tf.float64,
                    ),
                    validate_shape=validate_shape,
                    dtype=tf.float64
                ) 
                for initial_parameter, validate_shape in zip(
                    initial_parameters,
                    validate_shape
                )
            ]
        else:
            parameters = parameters
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
                
        if len(material_list) == 0:
            material_list = [{}] * self._surface_count
        else:
            try:
                if len(material_list) != self._surface_count:
                    raise ValueError(
                        "ParametricMultiSegmentBoundary: constraints and material_list "
                        "must have the same size."
                    )
            except(TypeError) as e:
                raise TypeError(
                    "ParametricMultiSegmentBoundary: material_list must be iterable."
                ) from e
        
        self.surfaces = [
            ParametricSegmentBoundary(
                zero_distribution,
                one_distribution,
                flip_norm=flip_norm,
                parameters=parameter,
                material_dict=material,
                **kwargs
            )
            for flip_norm, parameter, material in zip(
                self.flip_norm,
                self.parameters,
                material_list
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
    def surface_count(self):
        return self._surface_count
        
    @property
    def parameters(self):
        return [surface.parameter for surface in self.surfaces]
        
# =========================================================================================

class TriangleBoundaryBase(BoundaryBase):
    """
    Base class for triangle boundaries.
    
    The constructor can populate the internal pyvista mesh either from an already existing
    mesh or by loading one from a file that pyvista can read.  Can accept .stl files.  If 
    file_name is specified, will load the mesh from that file, and the value of the mesh 
    parameter will be ignored.  If file_name is None, will use the value of the mesh 
    parameter instead.  If both are None then the internal mesh will not be initialized.
    """
    def __init__(self, file_name=None, mesh=None, vertex_update_map=None, **kwargs):
        if file_name:
            self._mesh = pv.read(file_name)
        elif mesh is not None:
            self._mesh = mesh
        else:
            self._mesh = None
        self.vertex_update_map = vertex_update_map
        super().__init__(**kwargs)
        
    @property
    def dimension(self):
        return 3
        
    def update_materials(self):
        try:
            self._update_materials(tf.shape(self["xp"]))
        except(KeyError):
            pass
        
    @property
    def mesh(self):
        return self._mesh
        
    @property
    def vertices(self):
        return self._vertices
        
    @property
    def faces(self):
        return self._faces
        
    def save(self, filename, **kwargs):
        if self._mesh:
            self._mesh.save(filename, **kwargs)
        
    def update_vertices_from_mesh(self):
        """
        Updates the vertices and faces from self._mesh
        """
        if self._mesh is not None:
            self._vertices = tf.cast(self._mesh.points, dtype=tf.float64)
            try:
                # could fail if the mesh does not consist only of triangles
                self._faces = tf.reshape(self._mesh.faces, (-1, 4))
            except tf.errors.InvalidArgumentError as e:
                raise ValueError(
                    "TriangleBoundary: mesh must consist entirely of triangles."
                ) from e
        
    def update_fields_from_vertices(self):
        """
        Updates the fields from self._vertices.
        """
        if self._faces is not None and self._vertices is not None:
            _, first_index, second_index, third_index = tf.unstack(self._faces, axis=1)
            first_points = tf.gather(self._vertices, first_index)
            second_points = tf.gather(self._vertices, second_index)
            third_points = tf.gather(self._vertices, third_index)
            
            if self.vertex_update_map is not None:
                first_updatable, second_updatable, third_updatable = \
                    tf.unstack(self.vertex_update_map, axis=1)
                    
                first_updatable = tf.reshape(first_updatable, (-1, 1))
                second_updatable = tf.reshape(second_updatable, (-1, 1))
                third_updatable = tf.reshape(third_updatable, (-1, 1))
                    
                first_points = tf.where(
                    first_updatable, first_points, tf.stop_gradient(first_points))
                second_points = tf.where(
                    second_updatable, second_points, tf.stop_gradient(second_points))
                third_points = tf.where(
                    third_updatable, third_points, tf.stop_gradient(third_points))
            
            self["xp"], self["yp"], self["zp"] = tf.unstack(first_points, axis=1)
            self["x1"], self["y1"], self["z1"] = tf.unstack(second_points, axis=1)
            self["x2"], self["y2"], self["z2"] = tf.unstack(third_points, axis=1)
            self["norm"] = tf.linalg.normalize(
                tf.linalg.cross(
                    second_points - first_points,
                    third_points - second_points
                ),
                axis=1)[0]
        
    def update_from_mesh(self):
        """
        Updates the fields from the mesh.
        """
        if self._mesh is not None:
            self.update_vertices_from_mesh()
            self.update_fields_from_vertices()
        
    def update_mesh_from_vertices(self):
        """
        Updates the mesh from self._vertices
        """
        if self._vertices is not None:
            self._mesh.points = self._vertices.numpy()
            
# -------------------------------------------------------------------------------------
     
class ManualTriangleBoundary(TriangleBoundaryBase):
    """
    Class that builds a boundary directly from a set of points.
    
    This is the class to use if you want a static boundary.  Pass a file_name to the
    constructor to load the mesh data from the file.  Uses pyvista to do this, so it can
    take whatever pyvista.read() can take.  Can take STL files.
    
    All 3D surfaces must be composed only from triangles.  
    
    If update_function is specified, it must be a callable that takes no argument and
    returns nine values, xp, yp, zp, x1, y1, z1, x2, y2, z2.  Defaults to None, and if left
    as None, calling update on this class will do nothing.
    
    Can add new fields at any time simply by indexing the new field.
    """
        
    def _generate_update_handles(self):
        return []
            
    def _update(self):
        self.update_from_mesh()

# -----------------------------------------------------------------------------------------
        
class ParametricTriangleBoundary(TriangleBoundaryBase):
    """
    A single parametric surface.
    
    This class generates a single open curved surface, approximated as a mesh of
    triangles.  Takes a mesh, either generated with pyvista or loaded from a file that
    defines the surface when the parameters are all zeros, and a vector generator that
    somehow generates a set of vectors, one for each vertex in the mesh.  The
    actual vertices of the surface will lie along the line defined by the zero points and 
    the vectors, and the parameter tells how far along the vector the segment lies.  
    Parameters can take any value, though they might want to be constrained.
    
    This class will not automatically update if its zero_points mesh is changed, it
    creates its own mesh that is a copy of zero_points.  But it also by default does not
    automatically update the points stored in this mesh.  The true point data used by this
    class is stored in self.vertices a tf tensor, for speed when being used with an 
    optimizer.  If you want to display the surface, or for any other reason need the
    pyvista mesh of the surface, you have to explicitly call 
    self.update_mesh_from_vertices() OR set auto_update_mesh to True, in which case
    the mesh will be automatically updated every time update is called.
    
    The class contains an attribute: parameter, which is a tf.Variable that stores
    the shape of the surface.  This variable can be the target of a TF optimizer.
    I am not sure what will happen to the previous value of the variable if 
    the number of segments changes, so I consider a variable number of segments to be
    sketchy, but nonetheless I will allow it; pass validate_shape=False to allow this
    behavior.
    
    I will also allow the user to construct their own variable and plug it into the
    boundary.
    
    Always check the norm of the surface (can be done with drawing.TriangleDrawer).  If 
    the norm is the wrong way, you can toggle the value of flip_norm, to flip the 
    norm around.  In this class, flip_norm is a method rather than an attribute, which
    toggles the direction of the norm, and should not be frequently needed.
    """
    def __init__(
        self,
        zero_points,
        vector_generator,
        flip_norm=False,
        initial_parameters=0.0,
        validate_shape=True,
        parameters=None,
        auto_update_mesh=False,
        vertex_update_map=None,
        **kwargs
     ):
        # if zero points is a string, interpret it as a filename.
        if type(zero_points) is str:
            zero_points = pv.read(zero_points)
        else:
            zero_points = zero_points.copy()
            
        # do what needs to be done to flip the norm of the surface around
        if flip_norm:
            zero_points = self._flip_norm(zero_points)
            if vertex_update_map is not None:
                vertex_update_map = np.take(vertex_update_map, [2, 1, 0], axis=1)
        self._zero_points = tf.cast(zero_points.points, tf.float64)
        self.vertex_update_map = vertex_update_map
        
        # build the mesh, as a copy of zero_points, and for a single time extract
        # the vertices and faces from this mesh.  This operation will be banned from
        # here on, and the mesh will be a slave of the vertices.
        self._mesh = pv.PolyData(zero_points.copy())
        super().update_vertices_from_mesh()
        
        self.vector_generator = vector_generator
        self.auto_update_mesh = auto_update_mesh
        self.reparametrize(self._zero_points)
        
        # set up the parameters
        if parameters is None:
            initial_parameters = tf.cast(
                tf.broadcast_to(
                    initial_parameters,
                    (tf.shape(self._zero_points)[0],)
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
        
        # we need to explicitly call BoundaryBase's constructor, not TriangleBoundaryBase
        # because the latter will overwrite self._mesh
        BoundaryBase.__init__(self, **kwargs)
        if not self.auto_update_mesh:
            self.update_mesh_from_vertices()
        
    def _generate_update_handles(self):
        return []
        
    def _update(self):
        self._vertices = self._update_internal(
            self._zero_points,
            self._vectors,
            self.parameters
        )
        if self.auto_update_mesh:
            self.update_mesh_from_vertices()
        self.update_fields_from_vertices()

    """@tf.function(input_signature=
    [
        tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
        tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64)
    ])"""
    @staticmethod
    def _update_internal(zero, vectors, parameter):
        return zero + tf.reshape(parameter, (-1, 1)) * vectors
        
    def reparametrize(self, zero_points):
        self._vectors = self.vector_generator.generate(self._zero_points)

    @property
    def zero_points(self):
        return self._zero_points_mesh
        
    @property    
    def vectors(self):
        return self._vectors
        
    def _flip_norm(self, mesh):
        # swap points to reverse the face
        faces = np.reshape(mesh.faces, (-1, 4))
        faces = np.take(faces, [0, 3, 2, 1], axis=1)
        mesh.faces = np.reshape(faces, (-1,))
        return mesh
    
    def update_vertices_from_mesh(self):
        raise RuntimeError(
            "ParametricTriangleBoundary: update_vertices_from_mesh is disabled for "    
            "parametric boundaries."
        )
    def update_from_mesh(self):
        raise RuntimeError(
            f"ParametricTriangleBoundary: update_from_mesh is disabled for parametric "
            "boundaries."
        )

# -----------------------------------------------------------------------------------------

class MasterSlaveParametricTriangleBoundary(ParametricTriangleBoundary):
    """
    Parametric Triangle Boundary where some parameters are shared between multiple vertices.

    In a situation where you want to enforce some kind of symmetry on the optic, such as
    linear symmetry, but need to ray trace the whole thing, this class allows a smaller set
    of parameters to control a larger set of vertices.

    Use is identical to ParametricTriangleBoundary, except two additional functions must
    be specified: filter_masters and attach_slaves.
    """

    def __init__(self, filter_masters, attach_slaves, *args, initial_parameters=0.0, **kwargs):
        """

        Parameters
        ----------
        filter_masters : function or array of indices
            Determines which indices are considered as master indices, that will
            be given a parameter.  This can be either a callable, in which case it
            must be a function that determines whether any given vertex is a master or
            slave.  It's calling format must be:

            Parameters
            ----------
            vertices : 2d array
                The coordinates of all vertices in the boundary.

            Returns
            -------
            A 1D boolean array whose length is equal to the size of the first dimension of
            vertices and contains a list of which vertices (indices) are considered to be
            masters.

            Or filter masters may be a 1D iterable of indices of vertices to consider
            masters (i.e. the output from the above defined function)

        attach_slaves : function
            This function determines which slaves should be attached to a given master.
            It's calling format must be:

            Parameters
            ----------
            vertices : 2d array
                The coordinates of all vertices in the boundary.
            master : int
                The index into the vertices of the master of whose slaves to find.
            available_slaves : set of ints
                A set of indices of slave vertices that have not yet been claimed by
                a master.

            Returns
            -------
            A set of int indices into the vertices that indicates which slaves to
            attach to this master.
        """
        super().__init__(*args, **kwargs)

        # determine indices of all masters and slaves.
        # masters is a list of which vertices are designated as masters, and the elements
        # are indices into the original list, which has the same number of elements as
        # there are vertices.
        # master index is a dictionary that maps those indices down to the list of
        # masters alone.
        if callable(filter_masters):
            masters = filter_masters(self._vertices)
        else:
            masters = filter_masters
        master_index = {masters[i]: i for i in range(len(masters))}
        unclaimed_slaves = set(range(self._vertices.shape[0])) - set(masters)

        slave_masters = {}
        for master in masters:
            # master is an index into vertices, this line links slaves to the
            # index of their master in the list of all vertices
            slaves = attach_slaves(self._vertices, master, unclaimed_slaves)
            unclaimed_slaves -= set(slaves)

            for slave in slaves:
                # Update each slave's reference to the index of it's master
                # within just the list of masters.
                slave_masters[slave] = master_index[master]

        # This gather will expand the parameters to the size of the vertices
        self._gather = tf.constant([
            master_index[i] if i in masters else slave_masters[i] for i in range(self._vertices.shape[0])
        ])

        # Overwrite the old value of parameters
        initial_parameters = tf.cast(
            tf.broadcast_to(
                initial_parameters,
                (len(masters),)
            ),
            tf.float64
        )
        self.parameters = tf.Variable(
            initial_parameters,
            dtype=tf.float64
        )

        self._update()

        # I have over-polymorphismed this class, and am having difficulties getting
        # the correct parts to set the materials dicts called in the correct order.
        for field, value in kwargs["material_dict"].items():
            self[field] = tf.broadcast_to(value, self["xp"].shape)


    def _update(self):
        try:
            params = tf.reshape(tf.gather(self.parameters, self._gather), (-1, 1))
            self._vertices = self._zero_points + params * self.vectors
            if self.auto_update_mesh:
                self.update_mesh_from_vertices()
            self.update_fields_from_vertices()
        except(AttributeError):
            # This is a lazy way of ignoring the error that is thrown when super's
            # constructor is called and subsequently calls this function before
            # self._gather has been defined.
            pass

# -----------------------------------------------------------------------------------------
        
class ParametricMultiTriangleBoundary(TriangleBoundaryBase):
    """
    Multiple triangle surfaces that share a common set of base points.
    
    This class is intended to be used to generate several boundaries that are tightly 
    related to each other, like a single fixed optical component.  For disparate optical
    components, it might make more sense to simply use multiple ParametricSegmentBoundary,
    or even multiple ParametricMultiSegmentBoundary.    
    
    This class uses ParametricTriangleBoundary to construct each individual layer the optic,
    so it inherits a lot of features from that class.  One difference is that flip_norm is 
    now a required parameter to the constructor, since all the surfaces share the same base
    points, and you probably don't want them all facing the same way.
    
    The reason that this class uses a shared set of base points is because that makes it 
    really easy to apply constraints between the surfaces.  The constraints will take
    the physical interpretation as being the distance between the surfaces, as long as
    the vector generator is normalized.
    
    This class can be indexed, has fields and a signature like other boundaries.  It can
    also be updated.  This is useful for something like feeding this optic to a drawer.  
    But for connecting this object to the OpticalSystem, I recommend using the attribute
    surfaces, which returns a list to each individual ParametricTriangleBoundary.  It
    can be concatenated with any other triangle boundary lists and fed to 
    OpticalSystem3D.optical.  
    
    Each ParametricTriangleBoundary will be given a callable that will enforce the 
    constraint.  This is the only way that constraints can be applied (automatically), so 
    don't throw them away if you prune the update tree.
    
    You can get access to the variable that holds the parameters of the nth surface
    via ParametricMultiTriangleBoundary.surfaces[n].parameters, or via self.parameters.
    """
    def __init__(
        self,
        zero_points,
        vector_generator,
        constraints, 
        flip_norm,
        initial_parameters=0.0,
        validate_shape=True,
        parameters=None,
        material_list=[],
        **kwargs
     ):
        # if zero points is a string, interpret it as a filename.
        if type(zero_points) is str:
            zero_points = pv.read(zero_points)
        self.zero_points = zero_points
        self.vector_generator = vector_generator
        
        # process and validate the list parameters.
        try:
            self._surface_count = len(constraints)
        except(TypeError) as e:
            raise ValueError(
                "ParametricMultiTriangleBoundary: constraints must be iterable."
            ) from e
            
        try:
            if len(flip_norm) != self._surface_count:
                raise ValueError(
                    "ParametricMultiTriangleBoundary: constraints and flip_norm must have " 
                    "the same size."
                )
        except(TypeError) as e:
            raise ValueError(
                "ParametricMultiTriangleBoundary: flip_norm must be iterable."
            ) from e
        
        try:
            if len(initial_parameters) != self._surface_count:
                raise ValueError(
                    "ParametricMultiTriangleBoundary: constraints and initial_parameters "
                    "must have the same size."
                )
        except(TypeError):
            initial_parameters = [initial_parameters] * self._surface_count
        
        try:
            if len(validate_shape) != self._surface_count:
                raise ValueError(
                    "ParametricMultiTriangleBoundary: constraints and validate_shape "
                    "must have the same size."
                )
        except(TypeError):
            validate_shape = [validate_shape] * self._surface_count
        
        if parameters is None:
            parameter_size = tf.shape(zero_points.points)[0]
            parameters = [
                tf.Variable(
                    tf.cast(
                        tf.broadcast_to(
                            initial_parameter,
                            (parameter_size,)
                        ),
                        tf.float64,
                    ),
                    validate_shape=validate_shape,
                    dtype=tf.float64
                )
                for initial_parameter, validate_shape in zip(
                    initial_parameters,
                    validate_shape
                )
            ]
        else:
            parameters = parameters
            try:
                if len(parameters) != self._surface_count:
                    raise ValueError(
                        "ParametricMultiTriangleBoundary: constraints and parameters "
                        "must have the same size."
                    )
            except(TypeError) as e:
                raise TypeError(
                    "ParametricMultiTriangleBoundary: parameters must be None or iterable."
                ) from e
                
        if len(material_list) == 0:
            material_list = [{}] * self._surface_count
        else:
            try:
                if len(material_list) != self._surface_count:
                    raise ValueError(
                        "ParametricMultiTriangleBoundary: constraints and material_list "
                        "must have the same size."
                    )
            except(TypeError) as e:
                raise TypeError(
                    "ParametricMultiTriangleBoundary: material_list must be iterable."
                ) from e
        
        self._surfaces = [
            ParametricTriangleBoundary(
                self.zero_points,
                self.vector_generator,
                flip_norm=flip_norm,
                parameters=parameter,
                material_dict=material,
                **kwargs
            )
            for flip_norm, parameter, material in zip(
                flip_norm,
                parameters,
                material_list
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
                
        RecursivelyUpdatable.__init__(self, **kwargs)
        
    def _update(self):
        self._fields = amalgamate(self._surfaces)
        
    def _generate_update_handles(self):
        return [surface.update for surface in self._surfaces]
            
    @property
    def surface_count(self):
        return self._surface_count  
        
    @property
    def surfaces(self):
        return self._surfaces 
        
    @property
    def parameters(self):
        return [surface.parameters for surface in self._surfaces]
        
# -----------------------------------------------------------------------------------------
        
class ParametricCylindricalGuide(TriangleBoundaryBase):
    """
    A single closed parametric surface.
    
    This class generates a closed cylinder-like surface whose length is fixed but whose
    width can can change along its axis.  It can be either rotationally symmetric, or not.
    This is intended for optics that are more like linear light guides than planar optics.
    
    This surface will be constrained such so that at least one vertex will always be at
    mimumum_radius away from the axis.  Other constraints might be added later if this
    does not work well.
    
    This class will generate its own mesh and parameters, and will use only a single 
    constraint: the mimimum thickness of the optic, since it is intended to be more of a
    high-level helper class compared to the more general parametric boundaries defined above.
    
    If the optic is built rotationally symmetric, parameters will have z_res elements, one
    for each layer of triangles, and the parameters will encode the radius of the cylinder 
    at points along the axis.  If the optic is build non-rotationally symmetric, parameters
    will have z_res * theta_res elements, one for each vertex in the surface (except the
    end caps), and the value of each parameter encodes the distance between each vertex and 
    the axis minus the minimum thickness of the optic.
    
    The constructor for this class will run mesh_tools.mesh_parametrization_tools to generate
    an accumulator and vertex update map for the mesh.  The accumulator is available through
    a class attribute, and the property vertex_update_map will control whether the map is
    used or disabled.
    
    Parameters
    ----------
    start : float 3-vector
        The point on the axis where the cylinder will start.
    end : float 3-vector
        The point on the axis where the cylinder will end.
    minimum_radius : float
        The radius of the cylinder.  A radius of zero is permissible in case that is how
        you want to parametrize things.
    theta_res : int, optional
        The number of points to place around the diameter of the cylinder.  Defaults to 6.
    z_res : int, optional
        The number of points to place along the axis of the cylinder.  Defaults to 8.
    start_cap : bool, optional
        Defaults to True, in which case triangles are generated to close the start of the 
        cylinder.
    end_cap : bool, optional
        Defaults to True, in which case triangles are generated to close the end of the 
        cylinder.
    use_twist : bool, optional
        Defaults to True, in which case the triangles will be twisted around the axis each
        layer.  Does not work well for small angular resolution, but could possibly work 
        better at high angular resolutions.
    epsilion : float, optional
        A small value to compare to to detect zero length vectors.  Defaults to 1e-6.
        If the distance between end and start is too small, you may need to reduce the size
        of epsilion.
    rotationally_symmetric : bool, optional
        If True, will constrain the optic to be rotationally symmetric.  In this case
        parameters will have shape (z_res,).  If False, parameters will have shape 
        (z_res*theta_res,).
    initial_parameters : float tensor, optional
        Initial value for the parameters.  Must be broadcastable to shape of the parameters.
        May be overridden by initial_taper.
    initial_taper : float 2-tuple, or None
        Defaults to None, in which case initial_parameters are used instead.  If not None,
        will ignore initial_parameters and initialize the optic as a tapered cylinder (or
        a truncated cone).  The first value is the radius at the start, and the second the
        radius at the end.  One of these values should be equal to minimum radius, since
        the constraint will be applied after applying these initial conditions, but this
        isn't strictly necessary.
    auto_update_mesh : bool, optional
        Defaults to False.  If true, will automatically update the mesh from the vertices
        every time update() is called.  This is disabled by default because it wastes time
        if you don't need to redraw the surface after a step.
    use_vertex_update_map : bool, optional
        If True, uses the vertex_update_map.  Can be changed on the fly.
        
    Public Read/Write Attributes
    ----------------------------
    use_vertex_update_map
    auto_update_mesh
    
    Public Read-Only Attributes
    ---------------------------
    accumulator : rank 2 float tensor
        An accumulator matrix for this surface, which can be used by the optimizer to help
        smooth out updates to the surface.
    rotationally_symmetric
    mesh
    
    """
    def __init__(
        self,
        start,
        end,
        minimum_radius,
        theta_res=6,
        z_res=8,
        start_cap=True,
        end_cap=True,
        rotationally_symmetric=False,
        initial_parameters=0.0,
        initial_taper=None,
        auto_update_mesh=False,
        use_vertex_update_map=True,
        **kwargs
     ):
        # build the mesh and for a single time extract the vertices and faces from this 
        # mesh.  This operation will be banned from here on, and the mesh will be a slave
        # of the vertices.
        self._mesh = mt.cylindrical_mesh(
            start,
            end,
            radius=minimum_radius,
            theta_res=theta_res,
            z_res=z_res,
            end_cap=end_cap,
            start_cap=start_cap,
            **kwargs
        )
        self._zero_points_mesh = self._mesh.copy()
        self._zero_points = tf.cast(self._mesh.points, tf.float64)
        self._start_cap = start_cap
        self._end_cap = end_cap
        self._update_vertices_from_mesh()
        
        # perform the mesh tricks.
        self._vertex_update_map, self._accumulator = mt.mesh_parametrization_tools(
            self._mesh,
            mt.get_closest_point(self._mesh, start)
        )
        
        self.vector_generator = FromAxisVG(start, point=end)
        self.auto_update_mesh = auto_update_mesh
        self.reparametrize(self._zero_points)
        self.use_vertex_update_map = use_vertex_update_map
        
        # set up the parameters
        self._rotationally_symmetric = rotationally_symmetric
        self._cap_pad = tf.cast([[start_cap, end_cap]], dtype=tf.int64)
        
        if self._rotationally_symmetric:
            parameter_size = (z_res,)
        else:
            parameter_size = (z_res * theta_res,)
        self._param_repeater = [theta_res] * z_res
            
        if initial_taper:
            # ignore initial parameters and generate a taper instead
            try:
                taper_start = initial_taper[0]
                taper_end = initial_taper[1]
            except(IndexError, TypeError) as e:
                raise ValueError(
                    "ParametricCylindricalGuide: initial_taper must be None or a 2-tuple."
                ) from e
            initial_parameters = tf.cast(
                tf.linspace(taper_start, taper_end, z_res),
                tf.float64
            )
            if not self._rotationally_symmetric:
                initial_parameters = tf.repeat(initial_parameters, self._param_repeater)
        else:
            # use initial parameters
            initial_parameters = tf.cast(
                tf.broadcast_to(
                    initial_parameters,
                    parameter_size
                ),
                tf.float64
            )
        self.parameters = tf.Variable(
            initial_parameters,
            dtype=tf.float64
        )
        
        # we need to explicitly call BoundaryBase's constructor, not TriangleBoundaryBase
        # because the latter will overwrite self._mesh
        BoundaryBase.__init__(self, **kwargs)
        if not self.auto_update_mesh:
            self.update_mesh_from_vertices()
        
    def _generate_update_handles(self):
        return []
        
    def _update(self):
        self._constraint()
        parameters = self.parameters
        if self._rotationally_symmetric:
            parameters = tf.repeat(parameters, self._param_repeater)
        parameters = tf.reshape(tf.pad(parameters, self._cap_pad), (-1, 1))
    
        self._vertices = self._zero_points + parameters * self._vectors
        self._prune_vertices()  
        if self.auto_update_mesh:
            self.update_mesh_from_vertices()
        self.update_fields_from_vertices()
        
    def _constraint(self):
        self.parameters.assign_sub(tf.fill(
            self.parameters.shape,
            tf.reduce_min(self.parameters)
        ))
        
    def reparametrize(self, zero_points):
        self._vectors = self.vector_generator.generate(self._zero_points)

    @property
    def mesh(self):
        return self._mesh
        
    @property
    def zero_points(self):
        return self._zero_points_mesh
        
    @property    
    def vectors(self):
        return self._vectors

    def update_vertices_from_mesh(self):
        raise RuntimeError(
            "ParametricTriangleBoundary: update_vertices_from_mesh is disabled for "    
            "parametric boundaries."
        )
    def update_from_mesh(self):
        raise RuntimeError(
            f"ParametricTriangleBoundary: update_from_mesh is disabled for parametric "
            "boundaries."
        )
        
    @property
    def use_vertex_update_map(self):
        return self._use_vertex_update_map
        
    @use_vertex_update_map.setter
    def use_vertex_update_map(self, val):
        self._use_vertex_update_map = val
        if val:
            self.vertex_update_map = self._vertex_update_map
            
    @property
    def accumulator(self):
        return self._accumulator
        
    @property
    def rotationally_symmetric(self):
        return self._rotationally_symmetric
        
    def update_mesh_from_vertices(self):
        """
        Updates the mesh from self._vertices.
        
        Have to re-implement this to handle the non-updating end caps.
        """
        if self._vertices is not None:
            if self._start_cap:
                if self._end_cap:
                    # start cap, end cap
                    self._mesh.points[1:-1] = self._vertices.numpy()
                else:
                    # start cap, no end cap
                    self._mesh.points[1:] = self._vertices.numpy()
            else:
                if self._end_cap:
                    # no start cap, end cap
                    self._mesh.points[:-1] = self._vertices.numpy()
                else:
                    # no start cap, no end cap
                    self._mesh.points = self._vertices.numpy()
                    
    def _update_vertices_from_mesh(self):
        """
        Updates the vertices and faces from self._mesh.
        
        Have to re-implement this because of the damn end caps, which is starting to
        feel like it was a mistake...
        """
        if self._mesh is not None:
            self._vertices = tf.cast(self._mesh.points, dtype=tf.float64)
            self._prune_vertices()
            try:
                # could fail if the mesh does not consist only of triangles
                self._faces = tf.reshape(self._mesh.faces, (-1, 4))
            except tf.errors.InvalidArgumentError as e:
                raise ValueError(
                    "ManualTriangleBoundary: mesh must consist entirely of triangles."
                ) from e
                
    def _prune_vertices(self):
        if self._start_cap:
            if self._end_cap:
                # start cap, end cap
                self._vertices = self._vertices[1:-1]
            else:
                # start cap, no end cap
                self._vertices = self._vertices[1:]
        else:
            if self._end_cap:
                # no start cap, end cap
                self._vertices = self._vertices[:-1]
            else:
                # no start cap, no end cap
                self._vertices = self._vertices
        

        
        
        
        
        
        
        
        
        
        
        
