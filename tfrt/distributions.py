"""
Classes that define various distributions of points or angles.
"""


import math
import itertools
from abc import ABC, abstractmethod
from tfrt.update import RecursivelyUpdatable

import tensorflow as tf
import numpy as np

PI = math.pi
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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ]
    )
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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ]
    )
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


class ManualAngularDistribution(AngularDistributionBase):
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
    update_function : callable
        A function that takes zero arguments and returns two values, the updated
        vales of angles, and of ranks.  May also be none, in which case no update
        function will be called.  You can use this function to define how the manual
        distribution is updated.  Or you can perform the update yourself, since
        angle and ranks are read-write attributes for manual distributions.    
        
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
    update_function : callable
        Same as the parameter of the same name in the constructor.
    
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution, if update_function was 
        set.  May be called by sources that consume this distribution.
        

    """

    def __init__(self, angles, ranks=None, name=None, update_function=None):
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self.angles = angles
        self.ranks = ranks
        self.update_function = update_function

    def update(self):
        if self.update_function is not None:
            self.angles, self.ranks = self.update_function()
            
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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )
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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )
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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )
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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )
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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ]
    )
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
    def base_points_x(self):
        return self._base_points_x

    @property
    def base_points_y(self):
        return self._base_points_y

    @property
    def ranks(self):
        return self._ranks


class ManualBasePointDistribution(BasePointDistributionBase):
    """
    Package any set of custom base points into a format compatible with a source.
    
    This class nothing except cast the inputs to tf.float64 and return an object that
    is compatible with the sources defined in this module.  This is the quick and easy
    way to get a custom base point distribution, but if you find yourself using a 
    particular pattern often it might be worthwhile to write your own base point 
    distribution class that inherets from one of the abstract base point distribution
    classes.
    
    Parameters
    ----------
    x_points : tf.float64 tensor of shape (None,)
    y_points : tf.float64 tensor of shape (None,)
    ranks : tf.float64 tensor of shape (None,), optional
    name : string, optional
        The name of the distribution
    update_function : callable
        A function that takes zero arguments and returns two values, the updated
        vales of angles, and of ranks.  May also be none, in which case no update
        function will be called.  You can use this function to define how the manual
        distribution is updated.  Or you can perform the update yourself, since
        angle and ranks are read-write attributes for manual distributions.    
        
    Public read-only attributes
    ---------------------------
    name : string
        The name of the distribution
        
    Public read-write attributes
    ----------------------------
    base_points_x : tf.float64 tensor of shape (None,)
    base_points_y : tf.float64 tensor of shape (None,)
    ranks : tf.float64 tensor of shape (None,) or None
    update_function : callable
        Same as the parameter of the same name in the constructor.
        
    Public members
    --------------
    update()
        Recalculates the values stored by the distribution, if update_function was set.
        May be called by sources that consume this distribution.

    """

    def __init__(self, x_points, y_points, ranks=None, name=None, update_function=None):
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self.base_points_x = x_points
        self.base_points_y = y_points
        self.ranks = ranks
        self.update_function = update_function

    def update(self):
        if self.update_function is not None:
            self.angles, self.ranks = self.update_function()
            
    @property
    def base_points_x(self):
        return self._base_points_x
        
    @base_points_x.setter
    def base_points_x(self, val):
        self._base_points_x = val

    @property
    def base_points_y(self):
        return self._base_points_y
        
    @base_points_y.setter
    def base_points_y(self, val):
        self._base_points_y = val

    @property
    def ranks(self):
        return self._ranks
        
    @ranks.setter
    def ranks(self, val):
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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ]
    )
    def _parametrize_beam(beam_start, beam_end, central_angle, sample_count, name):
        """
        Utility that interprets beam_start, beam_end, and
        central_angle.  Calculates the endpoints of the beam and the limits on the
        beam rank.  Beam rank can then be interpreted as a parameter that generates
        beam points on a line from the origin to endpoint.

        """
        tf.debugging.assert_less(
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

        return start_rank, end_rank, endpoint_x, endpoint_y, rank_scale

    def parametrize_beam(self):
        self._start_rank, self._end_rank, self._endpoint_x, self._endpoint_y, \
            self._rank_scale = self._parametrize_beam(
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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
        ]
    )
    def _update(endpoint_x, endpoint_y, ranks):
        """
        Utility that takes the rank and the endponts and constructs the actual
        base points.
        """
        base_points_x = endpoint_x * ranks
        base_points_y = endpoint_y * ranks

        return base_points_x, base_points_y

    def update(self):
        self.parametrize_beam()
        self.update_ranks()
        self._base_points_x, self._base_points_y = self._update(
            self._endpoint_x, self._endpoint_y, self._ranks
        )


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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )
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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ]
    )
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

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(2), dtype=tf.float64),
            tf.TensorSpec(shape=(2), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ]
    )
    def _update(start_point, end_point, ranks, name):
        start_x = start_point[0]
        start_y = start_point[1]
        end_x = end_point[0]
        end_y = end_point[1]
        base_points_x = start_x + ranks * (end_x - start_x)
        base_points_y = start_y + ranks * (end_y - start_y)

        return base_points_x, base_points_y

    def update(self):
        self.sample_count_validation()
        self.update_ranks()
        self._base_points_x, self._base_points_y = self._update(
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

    @staticmethod
    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int64)])
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

    @staticmethod
    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int64)])
    def _update_ranks(sample_count):
        return tf.random_uniform(
            tf.reshape(sample_count, (1,)), 0.0, 1.0, dtype=tf.float64
        )
