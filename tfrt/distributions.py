"""
Classes that define various distributions of points or angles.
"""


import math
import itertools
from abc import ABC, abstractmethod
from tfrt.update import RecursivelyUpdatable

import tensorflow as tf
import numpy as np

PI = tf.constant(math.pi, dtype=tf.float64)
COUNTER = itertools.count(0)

class AngularDistributionBase(ABC):
    """
    Abstract implementation of an angular distribution, that defines the bare
    minimum interface that all angular distributions should implement.
    
    Public read-only attributes
    -----------------
    angles : tf.float64 tensor of shape (None,)
        The angles held by the distribution.
    ranks : tf.float64 tensor of shape (None,) or None
        A number that that describes where each angle is relative to the whole
        distribution, and can be useful for defining the target destination for each
        ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
    
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the angles to
        change if you change, say the min_angle of the distribution.  This operation 
        can be called recursively by higher tier consumers of the angular 
        distribution (like a source), but updating a distribution will NOT 
        recursively update any lower tier object the distribution may be made from, 
        since these are expected to be at the bottom of the stack.
        
        Random ops will have their random distribution re-sampled every time they
        are updated.

    """

    def __init__(self, min_angle, max_angle, sample_count, name=None):
        """
        Angular distribution constructors do not need to follow the pattern set by
        the constructor defined here, but most will, so this implementation is
        provided for convenience.  All angular distribution constructors should 
        accept a name keyword argument, which is used to define a namespace in which 
        all tf ops constructed by the class are placed.

        """
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.sample_count = sample_count
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"

        self.update()

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ]
    )"""
    @staticmethod
    def _angle_limit_validation(
        max_angle, min_angle, sample_count, lower_limit, upper_limit, name
    ):
        tf.debugging.assert_greater_equal(
            max_angle, min_angle, message=f"{name}: max_angle must be >= min_angle."
        )
        tf.debugging.assert_greater_equal(
            min_angle,
            lower_limit,
            message=f"{name}: min_angle must be >= {lower_limit}.",
        )
        tf.debugging.assert_less_equal(
            max_angle,
            upper_limit,
            message=f"{name}: max_angle must be <= {upper_limit}.",
        )
        tf.debugging.assert_positive(
            sample_count, message=f"{name}: sample_count must be > 0."
        )
        return

    def angle_limit_validation(self, lower_limit, upper_limit):
        """
        Convenience for sub classes that require angle limits, to help with range
        checking.

        """
        self._angle_limit_validation(
            self.max_angle,
            self.min_angle,
            self.sample_count,
            lower_limit,
            upper_limit,
            self._name,
        )

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ]
    )"""
    @staticmethod
    def _update_ranks(angles, min_angle, max_angle):
        """
        Convenience function for generating the angle ranks in most cases.  If this 
        pattern does not work, or if you generate ranks in the process of building 
        the distribution, do not use it.
        
        This function should be called once and returns an optimized function that
        itself calculates and returns the ranks.

        """
        return angles / tf.cast(
            tf.reduce_max(tf.abs(tf.stack([min_angle, max_angle]))), tf.float64
        )

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def angles(self):
        return self._angles

    @property
    def ranks(self):
        return self._ranks


class ManualAngularDistribution(AngularDistributionBase, RecursivelyUpdatable):
    """
    Package any set of custom angles into a format compatible with a source.
    
    This class nothing except cast the inputs to tf.float64 and return an object that
    is compatible with the sources defined in this module.  This is the quick and easy
    way to get a custom angular distribution, but if you find yourself using a 
    particular pattern often it might be worthwhile to write your own angular 
    distribution class that inherets from AngularDistributionBase.
    
    Parameters
    ----------
    angles : tf.float64 tensor of shape (None,)
    ranks : tf.float64 tensor of shape (None,)
    name : string, optional
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public read-only attributes
    ---------------------------
    name : string
        The name of the distribution.
        
    Public read-write attributes
    ----------------------------
    angles : tf.float64 tensor of shape (None,)
        The angles held by the distribution.
    ranks : tf.float64 tensor of shape (None,) or None
        A number that that describes where each angle is relative to the whole
        distribution, and can be useful for defining the target destination for each
        ray.  Will be none if ranks were not defined for this distribution.
    
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution, if update_function was 
        set.  May be called by sources that consume this distribution.
        

    """

    def __init__(self, angles, ranks=None, name=None, frozen=True, **kwargs):
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self.angles = angles
        self.ranks = ranks
        RecursivelyUpdatable.__init__(self, **kwargs)
        self.frozen=frozen
        
    def _generate_update_handles(self):
        return []

    def update(self):
        self._update()

    def _update(self):
        pass
            
    @property
    def angles(self):
        return self._angles
        
    @angles.setter
    def angles(self, val):
        self._angles = val

    @property
    def ranks(self):
        return self._ranks
        
    @ranks.setter
    def ranks(self, val):
        self._ranks = val
        

class StaticUniformAngularDistribution(AngularDistributionBase):
    """
    A set of angles that are uniformally distributed between two extrema.
    
    For this distribution, rank will be normalized so that the most extreme angle
    generated by the distribution (farthest from the center of the distribution)
    will have |rank| == 1.
    
    Parameters
    ----------
    min_angle : tf.float64 tensor of shape (None,)
        The minimum angle to include in the distribution.  Must be inside {-PI, PI}
    max_angle : tf.float64 tensor of shape (None,)
        The maximum angle to include in the distribution.  Must be inside {-PI, PI}
    sample_count : scalar tf.int64 tensor
        The number of angles to return in the distribution.
    name : string, optional
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public read-only attributes
    -----------------
    angles : tf.float64 tensor of shape (None,)
        The angles held by the distribution.
    ranks : tf.float64 tensor of shape (None,) or None
        A number that that describes where each angle is relative to the whole
        distribution, and can be useful for defining the target destination for each
        ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
        
    Public read-write attributes
    ----------------------------
    min_angle : tf.float64 tensor of shape (None,)
        The miniumum angle to include in the distribution.  Interpreted as relative 
        to the central angle of the source.
    max_angle : tf.float64 tensor of shape (None,)
        The maxiumum angle to include in the distribution.  Interpreted as relative 
        to the central angle of the source.
    sample_count : scalar tf.int64 tensor
        The number of angles to include in the distribution.
    
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the angles 
        to change if you change, say the min_angle of the distribution.  This 
        operation can be called recursively by higher tier consumers of the angular 
        distribution (like a source), but updating a distribution will NOT 
        recursively update any lower tier object the distribution may be made from, 
        since these are expected to be at the bottom of the stack.
        
    """

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )"""
    @staticmethod
    def _update(min_angle, max_angle, sample_count):
        angles = tf.linspace(min_angle, max_angle, sample_count)
        angles = tf.cast(angles, tf.float64)
        return angles

    def update(self):
        self.angle_limit_validation(-PI, PI)
        self._angles = self._update(self.min_angle, self.max_angle, self.sample_count)
        self._ranks = self._update_ranks(self.angles, self.min_angle, self.max_angle)


class RandomUniformAngularDistribution(AngularDistributionBase):
    """
    A set of angles that are randomly uniformally sampled between two extrema.
    
    For this distribution, rank will be normalized so that the most extreme angle
    generated by the distribution (farthest from the center of the distribution)
    will have |rank| == 1.
    
    Parameters
    ----------
    min_angle : scalar tf.float64 tensor
        The minimum angle to include in the distribution.  Must be inside {-PI, PI}
    max_angle : scalar tf.float64 tensor
        The maximum angle to include in the distribution.  Must be inside {-PI, PI}
    sample_count : scalar tf.int64 tensor
        The number of angles to return in the distribution.
    name : string, optional
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public read-only attributes
    -----------------
    angles : tf.float64 tensor of shape (None,)
        The angles held by the distribution.
    ranks : tf.float64 tensor of shape (None,) or None
        A number that that describes where each angle is relative to the whole
        distribution, and can be useful for defining the target destination for each
        ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
        
    Public read-write attributes
    ----------------------------
    min_angle : scalar tf.float64 tensor
        The miniumum angle to include in the distribution.  Interpreted as relative 
        to the central angle of the source.
    max_angle : scalar tf.float64 tensor
        The maxiumum angle to include in the distribution.  Interpreted as relative 
        to the central angle of the source.
    sample_count : scalar tf.float64 tensor
        The number of angles to include in the distribution.
        
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the angles 
        to change if you change, say the min_angle of the distribution.  This 
        operation can be called recursively by higher tier consumers of the angular 
        distribution (like a source), but updating a distribution will NOT 
        recursively update any lower tier object the distribution may be made from, 
        since these are expected to be at the bottom of the stack.
        
        Random distributions have their random distribution re-sampled each time 
        update is called.
        
    """

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )"""
    @staticmethod
    def _update(min_angle, max_angle, sample_count):
        angles = tf.random.uniform(
            (sample_count,), minval=min_angle, maxval=max_angle, dtype=tf.float64
        )
        return angles

    def update(self):
        self.angle_limit_validation(-PI, PI)
        self._angles = self._update(self.min_angle, self.max_angle, self.sample_count)
        self._ranks = self._update_ranks(self.angles, self.min_angle, self.max_angle)


class StaticLambertianAngularDistribution(AngularDistributionBase):
    """
    A set of angles spanning two extrema whose distribution follows a Lambertian
    (cosine) distribution around zero.
    
    For this source, rank will be the sine of the angle, so that the maximum rank
    will have magnitude sin(max(abs(angle_limits))).  Rank will be distributed
    uniformally.
    
    Parameters
    ----------
    min_angle : scalar tf.float64 tensor
        The minimum angle to include in the distribution.  Must be inside {-PI/2, PI/2}
    max_angle : scalar tf.float64 tensor
        The maximum angle to include in the distribution.  Must be inside {-PI/2, PI/2}
    sample_count : scalar tf.int64 tensor
        The number of angles to return in the distribution.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public read-only attributes
    -----------------
    angles : tf.float64 tensor of shape (None,)
        The angles held by the distribution.
    ranks : tf.float64 tensor of shape (None,) or None
        A number that that describes where each angle is relative to the whole
        distribution, and can be useful for defining the target destination for each
        ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
        
    Public read-write attributes
    ----------------------------
    min_angle : tf.float64 tensor of shape (None,)
        The miniumum angle to include in the distribution.  Interpreted as relative to the 
        central angle of the source.
    max_angle : tf.float64 tensor of shape (None,)
        The maxiumum angle to include in the distribution.  Interpreted as relative to the 
        central angle of the source.
    sample_count : tf.float64 scalar
        The number of angles to include in the distribution.
        
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the angles to 
        change if you change, say the min_angle of the distribution.  This operation can
        be called recursively by higher tier consumers of the angular distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
    """

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )"""
    @staticmethod
    def _update(min_angle, max_angle, sample_count):
        lower_rank_cutoff = tf.sin(min_angle)
        upper_rank_cutoff = tf.sin(max_angle)
        ranks = tf.cast(
            tf.linspace(lower_rank_cutoff, upper_rank_cutoff, sample_count), tf.float64
        )
        angles = tf.asin(ranks)
        return angles, ranks

    def update(self):
        self.angle_limit_validation(-PI / 2.0, PI / 2.0)
        self._angles, self._ranks = self._update(
            self.min_angle, self.max_angle, self.sample_count
        )


class RandomLambertianAngularDistribution(AngularDistributionBase):
    """
    A set of angles randomly sampling from a Lambertian (cosine) distribution around 
    zero and between two extrema.
    
    For this source, rank will be the sine of the angle, so that the maximum rank
    will have magnitude sin(max(abs(angle_limits))).  Rank will be distributed
    uniformally.
    
    Parameters
    ----------
    min_angle : scalar tf.float64 tensor
        The minimum allowed angle that can be included in the distribution.  Must be 
        inside {-PI/2, PI/2}
    max_angle : scalar tf.float64 tensor
        The maximum allowed angle that can be included in the distribution.  Must be 
        inside {-PI/2, PI/2}
    sample_count : scalar tf.int64 tensor
        The number of angles to return in the distribution.
    name : string, fixed, optional, fixed
        The name of the distribution.
        
    Public read-only attributes
    -----------------
    angles : tf.float64 tensor of shape (None,)
        The angles held by the distribution.
    ranks : tf.float64 tensor of shape (None,) or None
        A number that that describes where each angle is relative to the whole
        distribution, and can be useful for defining the target destination for each
        ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
        
    Public read-write attributes
    ----------------------------
    min_angle : scalar tf.float64 tensor
        The miniumum angle to include in the distribution.  Interpreted as relative to the 
        central angle of the source.
    max_angle : scalar tf.float64 tensor
        The maxiumum angle to include in the distribution.  Interpreted as relative to the 
        central angle of the source.
    sample_count : scalar tf.int64 tensor
        The number of angles to include in the distribution.
        
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the angles to 
        change if you change, say the min_angle of the distribution.  This operation can
        be called recursively by higher tier consumers of the angular distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
        Random distributions have their random distribution re-sampled each time update
        is called.
        
    """

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )"""
    @staticmethod
    def _update(min_angle, max_angle, sample_count):
        lower_rank_cutoff = tf.sin(min_angle)
        upper_rank_cutoff = tf.sin(max_angle)
        ranks = tf.random.uniform(
            (sample_count,), lower_rank_cutoff, upper_rank_cutoff, dtype=tf.float64
        )
        angles = tf.asin(ranks)
        return angles, ranks

    def update(self):
        self.angle_limit_validation(-PI / 2.0, PI / 2.0)
        self._angles, self._ranks = self._update(
            self.min_angle, self.max_angle, self.sample_count
        )


# ====================================================================================


class BasePointDistributionBase(ABC):
    """
    Abstract implementation of a base point distribution, which defines the bare
    minimum interface that all base point distributions should implement.
    
    Public read-only attributes
    ---------------------------
    base_points : 2-tuple of tf.float64 tensor of shape (None,)
        The x and y coordinates of the base points represented by the distribution
    ranks : tf.float64 tensor of shape (None,) or None
        A number that that describes where each base point is relative to the 
        whole distribution, and can be useful for defining the target destination for 
        each ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
    
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the points to 
        change if you change one of the parameters.  This operation can be called 
        recursively by higher tier consumers of the base point distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
        Random ops will have their random distribution re-sampled every time they
        are updated.

    """

    def __init__(self, name=None):
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self._base_points_x = None
        self._base_points_y = None
        self._ranks = None

        self.update()

    @abstractmethod
    def update(self):
        raise NotImplementedError

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ]
    )"""
    @staticmethod
    def _sample_count_validation(sample_count, name):
        tf.debugging.assert_positive(
            sample_count, message=f"{name}: sample_count must be > 0."
        )
        return

    def sample_count_validation(self):
        """
        Convenience for checking that the sample count is valid:
        Cast to int64, check that it is scalar, and ensure it is positive.
        """
        self._sample_count_validation(self.sample_count, self._name)

    @property
    def name(self):
        return self._name

    @property
    def points(self):
        return self._points

    @property
    def ranks(self):
        return self._ranks


class ManualBasePointDistribution(BasePointDistributionBase, RecursivelyUpdatable):
    """
    Package any set of custom base points into a format compatible with a source.
    
    This class is different than other distributions, in that this class inherits from
    update.RecursivelyUpdatable, which enables update handlers to be set up for this class.
    It will have to be done manually however.
    
    This class does nothing except cast the inputs to tf.float64 and return an object that
    is compatible with the sources defined in this module.  This is the quick and easy
    way to get a custom base point distribution, but if you find yourself using a 
    particular pattern often it might be worthwhile to write your own base point 
    distribution class that inherets from one of the abstract base point distribution
    classes.
    
    Parameters
    ----------
    dimension : int scalar
        Must be either 2 or 3.  The dimension of the distribution.
    points : tf.float64 tensor of shape (None, dimension), optional
    ranks : tf.float64 tensor, optional
    name : string, optional
        The name of the distribution
    from_mesh : None or pyvista.PolyData, optional
        Defaults to None.  If non-none, this is a PolyData object whose points will
        be used passed to self.points.  In this case, the 'points' parameter should be left
        as None.  This will be hooked into update() without any additional user 
        interaction (update() will pull the vertex data every time it is called).  But 
        this will NOT fill ranks, you will have to write your own handler to calculate 
        them if needed.  (Append some callable to self.update_handles or 
        self.post_update_handles, see tfrt.update).
        
    Public read-only attributes
    ---------------------------
    name : string
        The name of the distribution.
        
    Public read-write attributes
    ----------------------------
    points : tf.float64 tensor of shape (None, dimension)
    ranks : tf.float64 tensor of shape (None,) or None
    from_mesh : None or pyvista.PolyData
        Same as the parameter of the same name in the constructor.
        
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution, if update_function was set.
        May be called by sources that consume this distribution.

    """

    def __init__(
        self,
        dimension,
        points=None,
        ranks=None,
        name=None,
        from_mesh=None,
        frozen=True,
        **kwargs
    ):
        self._set_dimension(dimension)
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self.points = points
        self.ranks = ranks
        self.from_mesh = from_mesh
        RecursivelyUpdatable.__init__(self, **kwargs)
        self.frozen=frozen
        
    def _generate_update_handles(self):
        return []

    def update(self):
        self._update()

    def _update(self):
        if bool(self.from_mesh):
            self.points = tf.cast(self.from_mesh.points, dtype=tf.float64)
            
    def _set_dimension(self, dimension):
        if dimension not in {2, 3}:
            raise ValueError("ManualBasePointDistribution: dimension must be 2 or 3")
        else:
            self._dimension = dimension
            
    @property
    def dimension(self):
        return self._dimension
        
    @dimension.setter
    def dimension(self, val):
        self._set_dimension(val)
    
    @property
    def points(self):
        return self._points
        
    @points.setter
    def points(self, val):
        if val is None:
            self._points = tf.zeros((0, self._dimension), dtype=tf.float64)
        else:
            self._points = val

    @property
    def ranks(self):
        return self._ranks
        
    @ranks.setter
    def ranks(self, val):
        if val is None:
            self._ranks = tf.zeros_like(self._points, dtype=tf.float64)
        else:
            self._ranks = val


class BeamPointBase(BasePointDistributionBase):
    """
    Base points that can be used to define a beam.
    
    Base class for beam-type base point distributions, which are defined by a
    central_angle and two distances that denote the width of the beam perpendicular
    to the angle.  The coordinates returned by beams will be relative to the origin
    (so they should be interpreted as relative coordinates).  The base points will
    lie along a line perpendicular to central_angle, and will extend outward by the
    distances given to the constructor.  Positive distances correspond to points
    CCW to the central_angle, and negative distances correspond to points CW to the
    central_angle.

    The central_angle may be fed to the constructor.  By default it will take the
    value 0.0, and it should usually be kept to this value since the source that consumes
    this distribution will set this value.

    The rank generated by this distribution will have its zero at the relative origin
    of the beam, and will have magnitude 1 for the point(s) farthest from the origin.
    
    Parameters
    ----------
    beam_start : scalar tf.float64 tensor
        The distance from the center of the beam to its CCW edge.
    beam_end : scalar tf.float64 tensor
        The distance from the center of the beam to its CW edge.
    sample_count  : scalar tf.int64 tensor
        The number of points sampled by the beam.
    name : string, optional
        The name of the distribution.
    central_angle : scalar tf.float64 tensor, optional
        The angle to which the beam is perpendicular.
    
    Public read-only attributes
    ---------------------------
    base_points : 2-tuple of tf.float64 tensor of shape (None,)
        The x and y coordinates of the base points represented by the distribution
    ranks : tf.float64 tensor of shape (None,) or None
        A number that that describes where each base point is relative to the 
        whole distribution, and can be useful for defining the target destination for 
        each ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
        
    Public read-write attributes
    ----------------------------
    beam_start : scalar tf.float64 tensor
        The distance from the center of the beam to its CCW edge.
    beam_end : scalar tf.float64 tensor
        The distance from the center of the beam to its CW edge.
    sample_count  : scalar tf.int64 tensor
        The number of points sampled by the beam.
    central_angle : scalar tf.float64 tensor
        The angle to which the beam is perpendicular.
    
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the points to 
        change if you change one of the parameters.  This operation can be called 
        recursively by higher tier consumers of the base point distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
        Random ops will have their random distribution re-sampled every time they
        are updated.
    """

    def __init__(
        self, beam_start, beam_end, sample_count, name=None, central_angle=0.0
    ):
        self.beam_start = beam_start
        self.beam_end = beam_end
        self.sample_count = sample_count
        self.central_angle = central_angle

        super().__init__(name=name)

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ]
    )"""
    @staticmethod
    def _parametrize_beam(beam_start, beam_end, central_angle, sample_count, name):
        """
        Utility that interprets beam_start, beam_end, and
        central_angle.  Calculates the endpoints of the beam and the limits on the
        beam rank.  Beam rank can then be interpreted as a parameter that generates
        beam points on a line from the origin to endpoint.

        """
        tf.debugging.assert_less_equal(
            beam_start, beam_end, message=f"{name}: beam_start must be < beam_end."
        )
        tf.debugging.assert_greater(
            sample_count,
            tf.constant(0, dtype=tf.int64),
            message=f"{name}: sample_count must be > 0.",
        )

        rank_scale = tf.reduce_max(tf.abs(tf.stack([beam_start, beam_end])))
        start_rank = beam_start / rank_scale
        end_rank = beam_end / rank_scale

        endpoint_x = beam_start / tf.abs(start_rank) * tf.cos(central_angle - PI / 2.0)
        endpoint_y = beam_start / tf.abs(start_rank) * tf.sin(central_angle - PI / 2.0)

        return start_rank, end_rank, tf.stack([endpoint_x, endpoint_y]), rank_scale

    def parametrize_beam(self):
        self._start_rank, self._end_rank, self._endpoint, self._rank_scale = \
            self._parametrize_beam(
                self.beam_start,
                self.beam_end,
                self.central_angle,
                self.sample_count,
                self._name,
            )

    @abstractmethod
    def _update_ranks(start_rank, end_rank, sample_count):
        raise NotImplementedError

    def update_ranks(self):
        self._ranks = self._update_ranks(
            self._start_rank, self._end_rank, self.sample_count
        )

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(2,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
        ]
    )"""
    @staticmethod
    def _update(endpoint, ranks):
        """
        Utility that takes the rank and the endponts and constructs the actual
        base points.
        """
        base_points = tf.reshape(endpoint, (1, 2)) * tf.reshape(ranks, (-1, 1))

        return base_points

    def update(self):
        self.parametrize_beam()
        self.update_ranks()
        self._points = self._update(self._endpoint, self._ranks)


class StaticUniformBeam(BeamPointBase):
    """
    A set of base points uniformally spread across the width of a beam.
        
    Parameters
    ----------
    beam_start : scalar tf.float64 tensor
        The distance from the center of the beam to its CCW edge.
    beam_end : scalar tf.float64 tensor
        The distance from the center of the beam to its CW edge.
    sample_count  : scalar tf.int64 tensor
        The number of points sampled by the beam.
    name : string, optional
        The name of the distribution.
    central_angle : scalar tf.float64 tensor, optional
        The angle to which the beam is perpendicular.
    
    Public read-only attributes
    ---------------------------
    base_points : 2-tuple of tf.float64 tensor of shape (None,)
        The x and y coordinates of the base points represented by the distribution
    ranks : tf.float64 tensor of shape (None,) or None
        A number that that describes where each base point is relative to the 
        whole distribution, and can be useful for defining the target destination for 
        each ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
        
    Public read-write attributes
    ----------------------------
    beam_start : scalar tf.float64 tensor
        The distance from the center of the beam to its CCW edge.
    beam_end : scalar tf.float64 tensor
        The distance from the center of the beam to its CW edge.
    sample_count  : scalar tf.int64 tensor
        The number of points sampled by the beam.
    central_angle : scalar tf.float64 tensor
        The angle to which the beam is perpendicular.
    
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the points to 
        change if you change one of the parameters.  This operation can be called 
        recursively by higher tier consumers of the base point distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
    """

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )"""
    @staticmethod
    def _update_ranks(min_rank, max_rank, sample_count):
        return tf.linspace(min_rank, max_rank, sample_count)


class RandomUniformBeam(BeamPointBase):
    """
    A set of base points uniformally randomly sampled across the width of a beam.
        
    Parameters
    ----------
    beam_start : scalar tf.float64 tensor
        The distance from the center of the beam to its CCW edge.
    beam_end : scalar tf.float64 tensor
        The distance from the center of the beam to its CW edge.
    sample_count  : scalar tf.int64 tensor
        The number of points sampled by the beam.
    name : string, optional
        The name of the distribution.
    central_angle : scalar tf.float64 tensor, optional
        The angle to which the beam is perpendicular.
    
    Public read-only attributes
    ---------------------------
    base_points : 2-tuple of tf.float64 tensor of shape (None,)
        The x and y coordinates of the base points represented by the distribution
    ranks : tf.float64 tensor of shape (None,) or None
        A number that that describes where each base point is relative to the 
        whole distribution, and can be useful for defining the target destination for 
        each ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
        
    Public read-write attributes
    ----------------------------
    beam_start : scalar tf.float64 tensor
        The distance from the center of the beam to its CCW edge.
    beam_end : scalar tf.float64 tensor
        The distance from the center of the beam to its CW edge.
    sample_count  : scalar tf.int64 tensor
        The number of points sampled by the beam.
    central_angle : scalar tf.float64 tensor
        The angle to which the beam is perpendicular.
    
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the points to 
        change if you change one of the parameters.  This operation can be called 
        recursively by higher tier consumers of the base point distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
        Random ops will have their random distribution re-sampled every time they
        are updated.
        
    """

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )"""
    @staticmethod
    def _update_ranks(min_rank, max_rank, sample_count):
        return tf.random.uniform(
            tf.reshape(sample_count, (1,)), min_rank, max_rank, dtype=tf.float64
        )


class AperaturePointBase(BasePointDistributionBase):
    """
    Point distribution spanning two end points, for filling an aperature.
    
    Base class for aperature-type base point distributions, which are defined by two
    2-D points that define the edges of the aperature.  This class generates base
    points along the line segment bounded by the end points.  Convenient when you want
    to specify absolute positions for the edges of the source, rather than relative
    ones.

    A point with rank zero will be located at the start point, and rank will increase
    along the line till it reaches one at the end point.
    
    Parameters
    ----------
    start_point : 2-tuple of scalar tf.float64 tensor
        The first edge of the aperature.
    end_point : 2-tuple of scalar tf.float64 tensor
        The second edge of the aperature.
    name : string
        The name of the distribution.
    sample_count : scalar tf.int64 tensor
        The number of points sampled by the beam.
    
    Public read-only attributes
    ---------------------------
    base_points : 2-tuple of tf.float64 tensor of shape (None,)
        The x and y coordinates of the base points represented by the distribution
    ranks : tf.float64 tensor of shape (none,) or None
        A number that that describes where each base point is relative to the 
        whole distribution, and can be useful for defining the target destination for 
        each ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
    
    Public read-write attributes
    ----------------------------
    start_point : 2-tuple of scalar tf.float64 tensor
        The first edge of the aperature.
    end_point : 2-tuple of scalar tf.float64 tensor
        The second edge of the aperature.
    sample_count : scalar tf.int64 tensor
        The number of points sampled by the beam.
        
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the points to 
        change if you change one of the parameters.  This operation can be called 
        recursively by higher tier consumers of the base point distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
        Random ops will have their random distribution re-sampled every time they
        are updated.

    """

    def __init__(self, start_point, end_point, sample_count, name=None):
        self.start_point = start_point
        self.end_point = end_point
        self.sample_count = sample_count

        super().__init__(name=name)

    @abstractmethod
    def _update_ranks(sample_count):
        """
        This abstract method will generate the point ranks, according to some
        distribution.

        """
        raise NotImplementedError

    def update_ranks(self):
        self._ranks = self._update_ranks(self.sample_count)

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(2,), dtype=tf.float64),
            tf.TensorSpec(shape=(2,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ]
    )"""
    @staticmethod
    def _update(start_point, end_point, ranks, name):
        
        start_point = tf.reshape(start_point, (1, 2))
        end_point = tf.reshape(end_point, (1, 2))
        ranks = tf.reshape(ranks, (-1, 1))
        
        points = start_point + ranks * (end_point - start_point)

        return points

    def update(self):
        self.sample_count_validation()
        self.update_ranks()
        self._points = self._update(
            self.start_point, self.end_point, self._ranks, self._name
        )


class StaticUniformAperaturePoints(AperaturePointBase):
    """
    A set of base points uniformally spaced between two end points.
        
    Parameters
    ----------
    start_point : 2-tuple of scalar tf.float64 tensor
        The first edge of the aperature.
    end_point : 2-tuple of scalar tf.float64 tensor
        The second edge of the aperature.
    sample_count : scalar tf.int64 tensor
        The number of points sampled by the beam.
    name : string, optional
        The name of the distribution.
    
    Public read-only attributes
    ---------------------------
    base_points : 2-tuple of tf.float64 tensor of shape (None,)
        The x and y coordinates of the base points represented by the distribution
    ranks : tf.float64 tensor of shape (none,) or None
        A number that that describes where each base point is relative to the 
        whole distribution, and can be useful for defining the target destination for 
        each ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
    
    Public read-write attributes
    ----------------------------
    start_point : 2-tuple of scalar tf.float64 tensor
        The first edge of the aperature.
    end_point : 2-tuple of scalar tf.float64 tensor
        The second edge of the aperature.
    sample_count : scalar tf.int64 tensor
        The number of points sampled by the beam.
        
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the points to 
        change if you change one of the parameters.  This operation can be called 
        recursively by higher tier consumers of the base point distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
    """

    """@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int64)])"""
    @staticmethod
    def _update_ranks(sample_count):
        return tf.cast(tf.linspace(0.0, 1.0, sample_count), tf.float64)


class RandomUniformAperaturePoints(AperaturePointBase):
    """
    A set of base points uniformally randomly sampled between two end points.
        
    Parameters
    ----------
    start_point : 2-tuple of scalar tf.float64 tensor
        The first edge of the aperature.
    end_point : 2-tuple of scalar tf.float64 tensor
        The second edge of the aperature.
    name : string
        The name of the distribution.
    sample_count : scalar tf.int64 tensor
        The number of points sampled by the beam.
    
    Public read-only attributes
    ---------------------------
    base_points : 2-tuple of tf.float64 tensor of shape (None,)
        The x and y coordinates of the base points represented by the distribution
    ranks : tf.float64 tensor of shape (none,) or None
        A number that that describes where each base point is relative to the 
        whole distribution, and can be useful for defining the target destination for 
        each ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the distribution.
    
    Public read-write attributes
    ----------------------------
    start_point : 2-tuple of scalar tf.float64 tensor
        The first edge of the aperature.
    end_point : 2-tuple of scalar tf.float64 tensor
        The second edge of the aperature.
    sample_count : scalar tf.int64 tensor
        The number of points sampled by the beam.
        
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the points to 
        change if you change one of the parameters.  This operation can be called 
        recursively by higher tier consumers of the base point distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
        Random ops will have their random distribution re-sampled every time they
        are updated.
        
    """

    """@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int64)])"""
    @staticmethod
    def _update_ranks(sample_count):
        return tf.random_uniform(
            tf.reshape(sample_count, (1,)), 0.0, 1.0, dtype=tf.float64
        )
        

# =========================================================================================

class SquareBase(BasePointDistributionBase):
    """
    Base class for square distributions.
    
    Generates a grid of x,y points, centered on zero.
    
    Will generate a set of x_res * y*res points randomly but uniformally distributed in a 
    rectangle.  The rank will be 2D, and will be equal to the points, but normalized.  The 
    rank will always be square even if the distribution is rectangular, by which I mean 
    that the rank will go between +/- 1 along the longest side, and between +/- short 
    length/long length along the short side.
    
    Parameters
    ----------
    x_size : scalar float
        The center-to-edge distance of the distribution in the x direction.
    x_res : scalar int
        The number of points to use across the x direction.
    y_size : scalar float, optional
        The center-to-edge distance of the distribution in the x direction.  Defaults to
        the same value as x_size
    y_res : scalar int, optional
        The number of points to use across the y direction.  Defaults to the same value
        as x_res
        
    Public Attributes
    -----------------
    x_res, y_res, x_size, y_size are all read-writable after instantiation.
    points : float tensor of shape (x_res*y*res, 2)
        The points held by the distribution.
    ranks : float tensor of shape (x_res*y*res, 2)
        The points, but normalized.
        
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the points to 
        change if you change one of the parameters.  This operation can be called 
        recursively by higher tier consumers of the base point distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
        Random ops will have their random distribution re-sampled every time they
        are updated.
    
    """
    def __init__(self, x_size, x_res, y_size=None, y_res=None, **kwargs):
        self.x_size = x_size
        self.x_res = x_res
        self.y_size = y_size or x_size
        self.y_res = y_res or x_res
        super().__init__(**kwargs)
        
    @property
    def x_size(self):
        return self._x_size
        
    @x_size.setter
    def x_size(self, val):
        val = tf.cast(val, tf.float64)
        tf.debugging.assert_greater(
            val,
            tf.constant(0, dtype=tf.float64),
            "SquareDistribution: x_size must be > 0."
        )
        self._x_size = val
        
    @property
    def y_size(self):
        return self._y_size
        
    @y_size.setter
    def y_size(self, val):
        val = tf.cast(val, tf.float64)
        tf.debugging.assert_greater(
            val,
            tf.constant(0, dtype=tf.float64),
            "SquareDistribution: y_size must be > 0."
        )
        self._y_size = val
        
    @property
    def x_res(self):
        return self._x_res
        
    @x_res.setter
    def x_res(self, val):
        tf.debugging.assert_greater(val, 0, "SquareDistribution: x_res must be > 0.")
        tf.debugging.assert_integer(
            val, "SquareDistribution: x_res must be of integer type."
        )
        self._x_res = val
        
    @property
    def y_res(self):
        return self._y_res
        
    @y_res.setter
    def y_res(self, val):
        tf.debugging.assert_greater(val, 0, "SquareDistribution: x_res must be > 0.")
        tf.debugging.assert_integer(
            val, "SquareDistribution: y_res must be of integer type."
        )
        self._y_res = val
        
    @abstractmethod
    def _make_points(self):
        raise NotImplementedError
        
    @property
    def points(self):
        return self._points
        
    def update(self):
        self._make_points()
        self._ranks = self._points / tf.reduce_max([self._x_size, self._y_size])

# -----------------------------------------------------------------------------------------

class StaticUniformSquare(SquareBase):
    """
    Points uniformally distributed on a grid inside a square.  See SquareBase for details.
    """
    def _make_points(self):
        x = tf.linspace(-self._x_size, self._x_size, self._x_res)
        y = tf.linspace(-self._y_size, self._y_size, self._y_res)
        x, y = tf.meshgrid(x, y)
        x = tf.reshape(x, (-1,))
        y = tf.reshape(y, (-1,))
        self._points = tf.stack([x, y], axis=1)

# -----------------------------------------------------------------------------------------
    
class RandomUniformSquare(SquareBase):
    """
    Points randomly uniformally spread inside a square.  See SquareBase for details.
    """
    def _make_points(self):
        x = tf.random.uniform(
            (self._x_res*self._y_res,),
            minval=-self._x_size,
            maxval=self._x_size,
            dtype=tf.float64
        )
        y = tf.random.uniform(
            (self._x_res*self._y_res,),
            minval=-self._y_size,
            maxval=self._y_size,
            dtype=tf.float64
        )
        self._points = tf.stack([x, y], axis=1)

# =========================================================================================

class CircleBase(ABC):
    """
    Base class for circular distributions.
    
    Generates a 2D set of sample_count points evenly spaced inside a circle of given radius
    centered around the origin.  This class gives its points in both polar and cartesian 
    coordinates.  This class defines two sets of ranks, polar and cartesian are just a 
    normalized version of the points (theta in [0, 2PI] and r in [0, 1], or x, y in [-1, 
    1]).
    
    Uses a Golden Spiral to generate the points.  This method isn't perfect, but it is
    fast and pretty good.  Be warned that none of the points will be generated at
    rational places.  No point will be generated at precisely r=0 or r=radius, but a point
    will be placed within 1/(2*sample_count) of there.
    Useful refrence: https://stackoverflow.com/questions/9600801/
        evenly-distributing-n-points-on-a-sphere/44164075#44164075
    
    Parameters
    ----------
    sample_count : scalar int
        The number of points to include in the distribution.  Must be an odd number 
        squared if you seek to use the squared ranks.
    radius : scalar float, optional
        The radius of the point generated farthest from the origin.  Defaults to 1.
        
    Public Attributes
    -----------------
    sample_count and radius are public read-write attributes.
    points : float tensor of shape (sample_count, 2)
        The points (in cartesian coordinates).
    polar_points : a 2-tuple of float tensor of shape (sample_count,)
        The points in polar coordinates.
    ranks : float tensor of shape (sample_count, 2)
        The points (in cartesian coordinates) but normalized so that the values are in
        [0, 1].
    polar_ranks : a 2-tuple of float tensor of shape (sample_count,)
        The points (in polar coordinates) but normalized so that the values 
        r are in [0, 1] and theta in [0, 2PI].
        
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the points to 
        change if you change one of the parameters.  This operation can be called 
        recursively by higher tier consumers of the base point distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
        Random ops will have their random distribution re-sampled every time they
        are updated.
    """
    def __init__(self, sample_count, radius=1.0):
        self.sample_count = sample_count
        self.radius = radius
    
    @property
    def sample_count(self):
        return self._sample_count
        
    @sample_count.setter
    def sample_count(self, val):
        tf.debugging.assert_greater(
            val, 0, "CircleDistribution: sample_count must be > 0."
        )
        tf.debugging.assert_integer(
            val, "CircleDistribution: sample_count must be of integer type."
        )
        self._sample_count = val
        
    @property
    def radius(self):
        return self._radius
        
    @radius.setter
    def radius(self, val):
        val = tf.cast(val, tf.float64)
        tf.debugging.assert_greater(
            val,
            tf.constant(0, dtype=tf.float64), 
            "CircleDistribution: radius must be > 0.")
        self._radius = val
    
    @property
    def points(self):
        return self._radius * tf.stack(
            [self._r * tf.cos(self._theta), self._r * tf.sin(self._theta)],
            axis=1
        )
        
    @property
    def polar_points(self):
        return (self._radius * self._r, tf.math.floormod(self._theta, 2*PI))
        
    @property    
    def ranks(self):
        return tf.stack(
            [self._r * tf.cos(self._theta), self._r * tf.sin(self._theta)],
            axis=1
        )
        
    @property
    def polar_ranks(self):
        return (self._r, tf.math.floormod(self._theta, 2*PI))

# -----------------------------------------------------------------------------------------

class StaticUniformCircle(CircleBase):
    """
    Points uniformaly and non-randomly spread inside a circle.  See CircleBase for details.
    """
    def update(self):
        indices = tf.range(self._sample_count, dtype=tf.float64) + .5
        self._r = tf.sqrt(indices / self._sample_count)
        self._theta = PI * (1 + 5**0.5) * indices
    
# -----------------------------------------------------------------------------------------

class RandomUniformCircle(CircleBase):
    """
    Points randomly uniformaly spread inside a circle.  See CircleBase for details.
    """
    def update(self):
        self._r = tf.sqrt(tf.random.uniform((self._sample_count,), dtype=tf.float64))
        self._theta = 2 * PI * tf.random.uniform((self._sample_count,), dtype=tf.float64)
    
# =========================================================================================

class SphereBase(ABC):
    """
    Base class for spherical distributions.
    
    Generates a 3D set of sample_count points spread over the surface of a sphere.  The 
    sphere will always be symmetric in the yz plane, but you can control its extent along
    the x-axis.  Points will always start on the x-axis and will continue until they make 
    an angle of angular_size with the x_axis, so angular_size is the opening angle of the
    distribution relative to the origin, facing toward the x-axis.  The sphere is always
    centered at the origin
    
    The points generated are a 3-vector in cartesian coordinates.  The ranks generated are
    a 2-tuple in spherical coordinates, ignoring the r coordinate.  The first is the
    azimuthal angle and the second the polar angle.  The pole points along the x-axis.
    
    Uses a Golden Spiral to generate the points.  This method isn't perfect, but it is
    fast and pretty good.  Be warned that none of the points will be generated at
    rational places.  No point will be generated at precisely r=0 or r=radius, but a point
    will be placed within 1/(2*sample_count) of there.
    Useful refrence: https://stackoverflow.com/questions/9600801/
        evenly-distributing-n-points-on-a-sphere/44164075#44164075
    
    Parameters
    ----------
    angular_size : scalar float
        The opening angle of the distribution.  Must be in (0, PI].
    sample_count : scalar int
        How many points to generate in the distribution.  Must be > 0.
    radius : scalar float, optional
        The radius of the sphere.  Defaults to 1.
        
    Public attributes
    -----------------
    angular_size, sample_count, and radius are all public read-write attributes.
        
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution.  This allows the points to 
        change if you change one of the parameters.  This operation can be called 
        recursively by higher tier consumers of the base point distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
        Random ops will have their random distribution re-sampled every time they
        are updated.
    """
    def __init__(self, angular_size, sample_count, radius=1):
        self.angular_size = angular_size
        self.sample_count = sample_count
        self.radius = radius
        
        self.update()
        
    @property
    def sample_count(self):
        return self._sample_count
        
    @sample_count.setter
    def sample_count(self, val):
        tf.debugging.assert_greater(
            val, 0, "SphericalDistribution: sample_count must be > 0."
        )
        tf.debugging.assert_integer(
            val, "SphericalDistribution: sample_count must be of integer type."
        )
        self._sample_count = val
        
    @property
    def radius(self):
        return self._radius
        
    @radius.setter
    def radius(self, val):
        val = tf.cast(val, tf.float64)
        tf.debugging.assert_greater(
            val,
            tf.constant(0, dtype=tf.float64), 
            "SphereicalDistribution: radius must be > 0.")
        self._radius = val
        
    @property
    def angular_size(self):
        return self._angular_size
        
    @angular_size.setter
    def angular_size(self, val):
        val = tf.cast(val, tf.float64)
        tf.debugging.assert_greater(
            val,
            tf.constant(0, dtype=tf.float64), 
            "SphericalDistribution: angular_size must be > 0.")
        tf.debugging.assert_less_equal(
            val,
            tf.constant(PI/2.0, dtype=tf.float64), 
            "SphericalDistribution: angular_size must be <= PI/2.")
        self._angular_size = val
        
    @property
    def points(self):
        return self._radius * tf.stack(
            [
                tf.cos(self._phi),
                tf.sin(self._phi) * tf.cos(self._theta),
                tf.sin(self._phi) * tf.sin(self._theta)],
            axis=1
        )
        
    @property
    def ranks(self):
        return tf.stack([self._phi, tf.math.floormod(self._theta, 2*PI)], axis=1)
        
    @abstractmethod
    def update(self):
        raise NotImplementedError
        
# -----------------------------------------------------------------------------------------

class StaticUniformSphere(SphereBase):
    """
    Points uniformally and non-randomly spread on the surface of a sphere.  See SphereBase
    for details.
    """
    def update(self):
        indices = tf.range(self._sample_count, dtype=tf.float64) + .5
        phi = tf.linspace(
            1, 
            tf.cos(self._angular_size),
            self._sample_count
        )
        self._phi = tf.acos(phi)
        self._theta = PI * (1 + 5**0.5) * indices

# -----------------------------------------------------------------------------------------
        
class RandomUniformSphere(SphereBase):
    """
    Points randomly uniformally spread on the surface of a sphere.  See SphereBase
    for details.
    """
    def update(self):
        phi = tf.random.uniform(
            (self._sample_count,),
            minval=tf.cos(self._angular_size),
            maxval=1,
            dtype=tf.float64
        )
        self._phi = tf.acos(phi)
        self._theta = PI * (1 + 5**0.5) * tf.random.uniform(
            (self._sample_count,), dtype=tf.float64
        )

# -----------------------------------------------------------------------------------------

class StaticLambertianSphere(SphereBase):
    """
    Points non-randomly spread on the surface of a sphere with a Lambertian distribution.  
    See SphereBase for details.
    
    Lambertian distributions can accept an angular size of at most PI/2.
    
    Inverse CDF comes from integrating the density cos(phi)sin(phi)dphi dtheta, where the 
    cosine comes from the Lambertian, and the rest is the standard spherical area element.
    Interestingly / obious in hindsight but not in foresight: The lambertian spherical
    distribution is equivalent to the uniform circular distribution when linearly projected
    along the x-axis into the sphere (the y, z coordinates do not change).  Or, when viewed
    from far away along the x-axis, the spherical lambertian distribution looks exactly
    like a uniform circular distribution.
    """
        
    def update(self):
        indices = tf.range(self._sample_count, dtype=tf.float64) + .5
        phi = tf.linspace(
            1, 
            tf.cos(self._angular_size)**2,
            self._sample_count
        )
        self._phi = tf.acos(tf.sqrt(phi))
        self._theta = PI * (1 + 5**0.5) * indices

# -----------------------------------------------------------------------------------------
        
class RandomLambertianSphere(SphereBase):
    """
    Points randomly spread on the surface of a sphere with a Lambertian distribution.  
    See SphereBase for details.
    
    Lambertian distributions can accept an angular size of at most PI/2.
    
    Inverse CDF comes from integrating the density cos(phi)sin(phi)dphi dtheta, where the 
    cosine comes from the Lambertian, and the rest is the standard spherical area element.
    Interestingly / obious in hindsight but not in foresight: The lambertian spherical
    distribution is equivalent to the uniform circular distribution when linearly projected
    along the x-axis into the sphere (the y, z coordinates do not change).  Or, when viewed
    from far away along the x-axis, the spherical lambertian distribution looks exactly
    like a uniform circular distribution.
    """
    
    def update(self):
        phi = tf.random.uniform(
            (self._sample_count,),
            minval=tf.cos(self._angular_size)**2,
            maxval=1,
            dtype=tf.float64
        )
        self._phi = tf.acos(tf.sqrt(phi))
        self._theta = PI * (1 + 5**0.5) * tf.random.uniform(
            (self._sample_count,),
            dtype=tf.float64
        )




