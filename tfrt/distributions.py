"""
Classes that define various distributions of points or angles.

TODO: convert all 'distributions' to 'point cloud's, and let distributions be...
not sets of base points but callables that can map a uniform 'seed' onto a specific 
density function.
"""

import pickle
import math
import itertools
from abc import ABC, abstractmethod

import tensorflow_graphics.geometry.transformation.quaternion as quaternion
import tensorflow as tf
import numpy as np
import imageio
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d

from tfrt.update import RecursivelyUpdatable

PI = tf.constant(math.pi, dtype=tf.float64)
COUNTER = itertools.count(0)
X_AXIS = tf.constant((1, 0, 0), dtype=tf.float64)

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
        max_angle = tf.cast(max_angle, tf.float64)
        min_angle = tf.cast(min_angle, tf.float64)
        sample_count = tf.cast(sample_count, tf.int64)
        lower_limit = tf.cast(lower_limit, tf.float64)
        upper_limit = tf.cast(upper_limit, tf.float64)
        
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
        min_angle = tf.cast(min_angle, tf.float64)
        max_angle = tf.cast(max_angle, tf.float64)
        sample_count = tf.cast(sample_count, tf.int64)
        
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


class BasePointDistributionBase(RecursivelyUpdatable):
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

    def __init__(self, **kwargs):
        self._points = None
        self._ranks = None
        RecursivelyUpdatable.__init__(self, **kwargs)

    @abstractmethod
    def _update(self):
        raise NotImplementedError

    """@tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ]
    )"""
    @staticmethod
    def _sample_count_validation(sample_count):
        tf.debugging.assert_positive(
            sample_count, message=f"BasePointDistribution: sample_count must be > 0."
        )
        return

    def sample_count_validation(self):
        """
        Convenience for checking that the sample count is valid:
        Cast to int64, check that it is scalar, and ensure it is positive.
        """
        self._sample_count_validation(self.sample_count)

    @property
    def points(self):
        return self._points

    @property
    def ranks(self):
        return self._ranks
        
    def _generate_update_handles(self):
        return []


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
        self, beam_start, beam_end, sample_count, central_angle=0.0, **kwargs
    ):
        self.beam_start = beam_start
        self.beam_end = beam_end
        self.sample_count = sample_count
        self.central_angle = central_angle

        super().__init__(**kwargs)

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.int64)
        ]
    )
    def _parametrize_beam(beam_start, beam_end, central_angle, sample_count):
        """
        Utility that interprets beam_start, beam_end, and
        central_angle.  Calculates the endpoints of the beam and the limits on the
        beam rank.  Beam rank can then be interpreted as a parameter that generates
        beam points on a line from the origin to endpoint.

        """
        tf.debugging.assert_less_equal(
            beam_start, beam_end, message=f"BeamPointBase: beam_start must be < beam_end."
        )
        tf.debugging.assert_greater(
            sample_count,
            tf.constant(0, dtype=tf.int64),
            message=f"BeamPointBase: sample_count must be > 0.",
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
                self.sample_count
            )

    @abstractmethod
    def _update_ranks(start_rank, end_rank, sample_count):
        raise NotImplementedError

    def update_ranks(self):
        self._ranks = self._update_ranks(
            self._start_rank, self._end_rank, self.sample_count
        )

    def _update(self):
        """
        Utility that takes the rank and the endponts and constructs the actual
        base points.
        """
        self.parametrize_beam()
        self.update_ranks()
        self._points = tf.reshape(self._endpoint, (1, 2)) * tf.reshape(self._ranks, (-1, 1))

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

    def __init__(self, start_point, end_point, sample_count, **kwargs):
        self.start_point = start_point
        self.end_point = end_point
        self.sample_count = sample_count

        super().__init__(**kwargs)

    @abstractmethod
    def _update_ranks(sample_count):
        """
        This abstract method will generate the point ranks, according to some
        distribution.

        """
        raise NotImplementedError

    def update_ranks(self):
        self._ranks = self._update_ranks(self.sample_count)

    def _update(self):
        self.sample_count_validation()
        self.update_ranks()
        self._ranks = tf.reshape(self._ranks, (-1, 1))
        
        self._points = self._start_point + self._ranks * (
            self._end_point - self._start_point
        )
        
    @property
    def start_point(self):
        return self._start_point
        
    @start_point.setter
    def start_point(self, val):
        val = tf.reshape(val, (1, 2))
        self._start_point = tf.cast(val, tf.float64)
        
    @property
    def end_point(self):
        return self._end_point
        
    @end_point.setter
    def end_point(self, val):
        val = tf.reshape(val, (1, 2))
        self._end_point = tf.cast(val, tf.float64)


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
        
    def _update(self):
        self._make_points()
        self._ranks = self._points / tf.reduce_max([self._x_size, self._y_size])
        
    def _generate_update_handles(self):
        return []

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

class ThetaMod:
    @property
    def theta_start(self):
        return self._theta_start
        
    @theta_start.setter
    def theta_start(self, val):
        val = tf.cast(val, tf.float64)
        try:
            tf.debugging.assert_less(
                val,
                self._theta_end, 
                "CircleDistribution: theta_start must be < theta_end.")
        except(AttributeError):
            # if _theta_end hasn't been set yet, there is nothing to compare to
            pass
        self._theta_start = val
        self._set_theta_mod()
        
        
    @property
    def theta_end(self):
        return self._theta_end
        
    @theta_end.setter
    def theta_end(self, val):
        val = tf.cast(val, tf.float64)
        try:
            tf.debugging.assert_greater(
                val,
                self._theta_start, 
                "CircleDistribution: theta_start must be < theta_end.")
        except(AttributeError):
            # if _theta_start hasn't been set yet, there is nothing to compare to
            pass
        self._theta_end = val
        self._set_theta_mod()
        
    def _set_theta_mod(self):
        try:
            if self._theta_start == 0 and self._theta_end == 2*PI:
                self._theta_mod = lambda x: x
            else:
                self._theta_mod = self._theta_mod_impl
        except(AttributeError):
            # this is called by the setters, but this means that the first time it is
            # called, both variables haven't yet been set, so we should do nothing.
            pass
            
    def _theta_mod_impl(self, theta):
        return theta % (self._theta_end - self._theta_start) + self._theta_start

# -----------------------------------------------------------------------------------------

class CircleBase(ThetaMod, RecursivelyUpdatable):
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
    def __init__(self, sample_count, radius=1.0, theta_start=0, theta_end=2*PI, **kwargs):
        self.sample_count = sample_count
        self.radius = radius
        self.theta_start = theta_start
        self.theta_end = theta_end
        RecursivelyUpdatable.__init__(self, **kwargs)
        
    @abstractmethod
    def _update(self):
        raise NotImplementedError
    
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
        return self._points
        
    @property
    def polar_points(self):
        return tf.stack(
            [self._radius * self._r, tf.math.floormod(self._theta, 2*PI)],
            axis=1
        )
        
    @property
    def ranks(self):
        return tf.stack(
            [self._r * tf.cos(self._theta), self._r * tf.sin(self._theta)],
            axis=1
        )
        
    @property
    def polar_ranks(self):
        return tf.stack(
            [self._r, tf.math.floormod(self._theta, 2*PI)],
            axis=1
        )
    
    def _generate_update_handles(self):
        return []

# -----------------------------------------------------------------------------------------

class StaticUniformCircle(CircleBase):
    """
    Points uniformaly and non-randomly spread inside a circle.  See CircleBase for details.
    """
    def _update(self):
        indices = tf.range(self._sample_count, dtype=tf.float64) + .5
        self._r = tf.sqrt(indices / self._sample_count)
        self._theta = PI * (1 + 5**0.5) * indices
        self._theta = self._theta_mod(self._theta)
        self._points = self._radius * tf.stack(
            [self._r * tf.cos(self._theta), self._r * tf.sin(self._theta)],
            axis=1
        )
    
# -----------------------------------------------------------------------------------------

class RandomUniformCircle(CircleBase):
    """
    Points randomly uniformaly spread inside a circle.  See CircleBase for details.
    """
    def _update(self):
        self._r = tf.sqrt(tf.random.uniform((self._sample_count,), dtype=tf.float64))
        self._theta = 2 * PI * tf.random.uniform((self._sample_count,), dtype=tf.float64)
        self._theta = self._theta_mod(self._theta)
        self._points = self._radius * tf.stack(
            [self._r * tf.cos(self._theta), self._r * tf.sin(self._theta)],
            axis=1
        )
    
# =========================================================================================

class SphereBase(ThetaMod, RecursivelyUpdatable):
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
    def __init__(
        self,
        angular_size,
        sample_count, 
        radius=1, 
        theta_start=0, 
        theta_end=2*PI,
        **kwargs
    ):
        self.angular_size = angular_size
        self.sample_count = sample_count
        self.radius = radius
        self.theta_start = theta_start
        self.theta_end = theta_end
        RecursivelyUpdatable.__init__(self, **kwargs)
        
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
        return self._points
        
    @property
    def ranks(self):
        return tf.stack([self._phi, tf.math.floormod(self._theta, 2*PI)], axis=1)
        
    @abstractmethod
    def _update(self):
        raise NotImplementedError
        
    def _generate_update_handles(self):
        return []
        
# -----------------------------------------------------------------------------------------

class StaticUniformSphere(SphereBase):
    """
    Points uniformally and non-randomly spread on the surface of a sphere.  See SphereBase
    for details.
    """
    def _update(self):
        indices = tf.range(self._sample_count, dtype=tf.float64) + .5
        phi = tf.linspace(
            1, 
            tf.cos(self._angular_size),
            self._sample_count
        )
        self._phi = tf.acos(phi)
        self._theta = PI * (1 + 5**0.5) * indices
        self._theta = self._theta_mod(self._theta)
        self._points = self._radius * tf.stack(
            [
                tf.cos(self._phi),
                tf.sin(self._phi) * tf.cos(self._theta),
                tf.sin(self._phi) * tf.sin(self._theta)],
            axis=1
        )

# -----------------------------------------------------------------------------------------
        
class RandomUniformSphere(SphereBase):
    """
    Points randomly uniformally spread on the surface of a sphere.  See SphereBase
    for details.
    """
    def _update(self):
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
        self._theta = self._theta_mod(self._theta)
        self._points = self._radius * tf.stack(
            [
                tf.cos(self._phi),
                tf.sin(self._phi) * tf.cos(self._theta),
                tf.sin(self._phi) * tf.sin(self._theta)],
            axis=1
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
        
    def _update(self):
        indices = tf.range(self._sample_count, dtype=tf.float64) + .5
        phi = tf.linspace(
            1, 
            tf.cos(self._angular_size)**2,
            self._sample_count
        )
        self._phi = tf.acos(tf.sqrt(phi))
        self._theta = PI * (1 + 5**0.5) * indices
        self._theta = self._theta_mod(self._theta)
        self._points = self._radius * tf.stack(
            [
                tf.cos(self._phi),
                tf.sin(self._phi) * tf.cos(self._theta),
                tf.sin(self._phi) * tf.sin(self._theta)],
            axis=1
        )

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
    
    def _update(self):
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
        self._theta = self._theta_mod(self._theta)
        self._points = self._radius * tf.stack(
            [
                tf.cos(self._phi),
                tf.sin(self._phi) * tf.cos(self._theta),
                tf.sin(self._phi) * tf.sin(self._theta)],
            axis=1
        )

# =========================================================================================

class BasePointTransformation():
    """
    Adds a transformation operation to a base point distribution.
    
    Base point distributions are usually sets of points that represent relative position,
    and the source takes care of translating them.  But AperatureSource doesn't apply
    any transformation to its points, and I realized it will sometimes be useful to be
    able to use these base point distributions for that purpose.  Hence this class.
    
    This class modifies a base point distribution, it isn't itself a base point 
    distribution.  It works by adding a post update operation to some other distribution,
    that only modifies the distribution's points attribute.  It isn't possible to apply
    the same transformation to more than one distribution, nor to modify which distribution
    the transformation applies to.  But if the transformation object itself is deleted,
    it will no longer be applied to the distribution.  The transformation object itself
    stores values for the rotation and translation to apply, which are both read-write.
    
    Designed to be used in 3D, rotations and translations are both performed in 3D.
    Scale is performed first, followed by rotation, and then translation.  While one 
    translation object can be used to perform all three of these operations, for
    complicated operations it might make more sense to use multiple instances of this
    class to apply successive transformations.  The order in which these translations are
    constructed (and applied to the base) should be preserved.
    
    """
    
    def __init__(self, base, rotation=None, translation=None, scale=None):
        """
        Base is the base point distribution this instance will wrap.
        Translation is applied before rotation.  If rotation or translation are None, than 
        that operation will not be applied.  Both operations are relative to the origin,
        and other base point distributions natively generate around the origin, so this 
        class should transform in an intuitive fashion.
        If both rotation and translation are None, this class will only have the effect of
        converting its base point distribution to 3D.  In converting to 3D, this class will
        always assume that the base points in base are on the y-z plane.
        
        There are several options for the rotation argument, based on its shape.
        Scalar : Angle around the x-axis.
        3-vector : Euler angle representation; the rotations around the x, y, and z axes.
        4-vector : A pre-computed quaternion.
        """
        self._base = base
        self.rotation = rotation
        self.translation = translation
        self.scale = scale
        self._base.post_update_handles.append(self._apply_transformation)
        
    def __del__(self):
        try:
            self._base.post_update_handles.remove(self._apply_transformation)
        except(ValueError):
            pass
        
    def _apply_transformation(self):
        base_points = self._base._points
        bp_shape = tf.shape(base_points)
        if tf.equal(bp_shape[1], 2):
            base_points = tf.pad(base_points, ((0, 0), (1, 0)))
        
        if self._scale is not None:
            base_points *= self._scale
            
        if self._rotation is not None:
            base_points = quaternion.rotate(base_points, self._rotation)
            
        if self._translation is not None:
            base_points += self._translation
    
        self._base._points = base_points
        
    @property
    def rotation(self):
        return self._rotation
        
    @rotation.setter
    def rotation(self, val):
        if val is not None:
            # the value set to self._rotation should be a quaternion.
            val = tf.cast(val, tf.float64)
            if val.shape == ():
                val = quaternion.from_euler((val, 0.0, 0.0))
            elif val.shape == (3,):
                val = quaternion.from_euler(val)
            elif val.shape == (4,):
                pass
            else:
                raise ValueError(
                    "BasePointTransformation: rotation must be scalar, 3D, or a quaternion."
                )
        self._rotation = val 
                   
        
    @property
    def translation(self):
        return self._translation
        
    @translation.setter
    def translation(self, val):
        if val is not None:
            val = tf.cast(val, tf.float64)
        self._translation = val
        
    @property
    def scale(self):
        return self._scale
        
    @scale.setter
    def scale(self, val):
        if val is not None:
            val = tf.cast(val, tf.float64)
        self._scale = val

# =========================================================================================

class ArbitraryDistribution:
    """
    Generate a 2D point cloud from an arbitrary distribution.
    
    This class can be called like a function, to generate a point cloud sampled from the
    given distribution.  The signature for calling is:
    
    Parameters
    ----------
    x, y : 1D float array
        Uniformally sampled x and y points.  Their shape must be exactly the same.
        
    Returns
    -------
    x, y : 1D float array
        The inputs, but transformed so that they now sample the given distribution.  Shape 
        will be the same as the inputs.
    """
    def __init__(self, density_function, evaluation_limits):
        """
        Parameters
        ----------
        density_function :
            There are three options for this argument:
            1) It can be a 2D float numpy array.
            2) It can be an analytic function to sample points from.  It should take two 
            arguments, x and y, (which can be numpy arrays) and returns a positive float 
            value of the same shape as the arguments that describes the probability density 
            of the points at x, y.
            3) It can be a string, in which case it is interpreted as the filename of an
            image file that can be read with imageio.  The image will be used as the density.
        evaluation_limits : tuple
            This is a nested tuple that defines the domain and resolution of the density
            function.  
            If density_function is a string (image file) or np array, this defines
            the domain and must be ((x_min, x_max), (y_min, y_max)).
            If density_function is a callable, this defines the domain and the resolution of
            the grid on which the density function is evaluated.  In this case it must be
            ((x_min, x_max, x_count), (y_min, y_max, y_count))
        """
        # interpret the parameters and calcuate the density
        if type(density_function) is str:
            # it's a filename, so load the file as an image.
            self._x_min, self._x_max = evaluation_limits[0]
            self._y_min, self._y_max = evaluation_limits[1]
            
            density = np.array(imageio.imread(density_function, as_gray=True))
            self._x_count, self._y_count = density.shape
        elif callable(density_function):
            # it's a function, so evaluate it on a grid.
            self._x_min, self._x_max, self._x_count = evaluation_limits[0]
            self._y_min, self._y_max, self._y_count = evaluation_limits[1] 
            
            evaluation_x = np.linspace(self._x_min, self._x_max, self._x_count)
            evaluation_y = np.linspace(self._y_min, self._y_max, self._y_count)
            grid_x, grid_y = np.meshgrid(evaluation_x, evaluation_y)
            
            density = density_function(grid_x, grid_y)
        else:
            # it's already an array, so check that it has the correct rank.
            try:
                density_function = np.array(density_function)
                density_shape = density_function.shape
            except(AttributeError):
                raise ValueError(
                    "PointCloudSampler: density function must be a str, a callable, or a "
                    "numpy array"
                )
            if density_shape != (2,):
                raise ValueError("PointCloudSampler: density function must be 2D.")
            self._x_min, self._x_max = evaluation_limits[0]
            self._y_min, self._y_max = evaluation_limits[1]
            
            density = density_function
            self._x_count, self._y_count = density.shape
        
        if np.any(density < 0):
            raise ValueError(
                "PointCloudSampler: density function must be non-negative on the whole "
                "evaluation grid."
            )
        
        # pad the density function, since we want our cumsums to start from zero.
        density = np.pad(density, ((1, 0), (1, 0)), mode="constant", constant_values=0)
        column_sums = np.cumsum(density, axis=0)
        row_sum = np.cumsum(column_sums[-1])
        column_sums_array = column_sums[:,1:] # now we can remove the pad column
        
        # scale the sums to match the ranges of the evaluation grid (since these will become
        # the dependent variable once we invert).
        row_sum = self._rescale(row_sum, self._x_min, self._x_max)
        column_sums = [
            self._rescale(column_sums_array[:,i], self._y_min, self._y_max)
            for i in range(self._x_count)
        ]
        
        """
        with np.printoptions(precision=3, suppress=True):
            print("density")
            print(density)
            print("column sums:")
            print(column_sums)
            print("row sum:")
            print(row_sum)
        """
        
        # Invert the cumsums (just swap x and y) and interpolate to generate the qunatile 
        # function.  We need new x and y coordinate lists with one extra element, since the 
        # cumsum adds a zero at the start.
        interpolate_x = np.linspace(self._x_min, self._x_max, self._x_count + 1)
        interpolate_y = np.linspace(self._y_min, self._y_max, self._y_count + 1)
        self._x_quantile = interp1d(row_sum, interpolate_x)
        self._y_quantiles = [
            interp1d(column_sum, interpolate_y)
            for column_sum in column_sums
        ]
    
    @staticmethod
    def _rescale(n, n_min, n_max):
        amax = np.amax(n)
        if amax <= 0:
            raise ValueError(
                "PointCloudSampler: Discovered a slice where the density was zero, which "
                "causes problems, because the quantile function would have to have infinite "
                "slope.  Either restrict the evaluation range, or add a very small constant "
                "to the density function."
            )
        return n * (n_max - n_min) / amax + n_min
        
    def __call__(self, x, y):
        x_out = self._x_quantile(x)
        
        # select which y quantile curve to use
        y_curve = (x_out - self._x_min) * self._x_count / (self._x_max - self._x_min)
        y_curve = np.floor(y_curve).astype("int")       
        
        # the below line works but is slow
        #y_out_slow = np.array([
        #    self._y_quantiles[y_curve[i]](y[i]) for i in range(y.shape[0])
        #])
        
        # this does the same thing, but is faster!   
        y_range = np.arange(y.shape[0])
        y_out = np.zeros_like(y)
        for i in range(self._y_count):
            mask = y_curve == i
            y_out[y_range[mask]] = self._y_quantiles[i](y[mask])
            
        return x_out, y_out

# -----------------------------------------------------------------------------------------

class AribitraryBasePoints(BasePointDistributionBase):
    """
    Base points that come from a completely arbitrary distribution.
    
    This class wraps an ArbitraryDistribution (defined above) with the things needed
    for a base point distribution.  The most important feature of this class is that it can 
    accept two ArbitraryDistribution the first being the base points where the rays start, 
    and the second being the optimization goal, as encoded in the ranks.  These two 
    distributions are automatically daisy-chained, thereby replacing the functionality of 
    ImageBasePoints and transform map defined below.  It is both more flexible and MASSIVELY 
    more performant than those options.  The two distributions are interpolated, meaning
    that they don't even need the same evaluation resolution or domain.
    
    There are two minor limitations.  The first is that the ArbitraryDistributions themselves 
    cannot be modified once instantiated, but you can generate a new one on the fly and 
    attach it to a class instance (by simply setting the property).  The second is that
    the ArbitraryDistributions doesn't handle regular, gridded points well, so this class
    is only able to be used in random mode.  By default the class re-rolls its random
    input distribution every time update is called, but this behavior can be suppressed
    with the reroll parameter.
    
    The ArbitraryDistributions are written in numpy, not TF, so you will not be able to
    get a gradient through them, but why would you ever even need that?  They are still
    pretty performant though.
    
    Parameters
    ----------
    base_point_distribution : ArbitraryDistribution
        The distribution used for the ray starts.  See the definition for 
        ArbitraryDistribution above for the parameters to its constructor.
    sample_count : scalar int
        The number of base points to generate.  Totally independant from anything in the
        ArbitraryDistributions.
    rank_distribution : ArbitraryDistribution, optional
        Defaults to None, in which case there are no ranks associated with this base point
        distribution.  But it may be second ArbitraryDistribution, which will automatically
        be daisy-chained to the first one, which is an excellent way to generate the
        optimization goal.
    auto_reroll : bool, optional
        Defaults to True, in which case a new set of random numbers is generated each time
        update is called.  If False, will only re-randomize the points the first time
        update is called or whenever the method reroll is called.
    preserve_etendue : bool, optional
        Defaults to True.  If True, will call the method enforce_etendue, which you can read
        about under its method documentation.
    etendue_origin : 2-tuple of floats, optional.
        The x,y coordinates of the point to be used as the center of the distribution, when
        using the preserve_etendue feature.  See the documentation for the method 
        enforce_etendue().
        
        
    Public Attributes
    -----------------
    base_point_distribution, sample_count, rank_distribution, and auto_reroll as described
    above are all public read/write attributes.
    
    rank_rescale : scalar float.
        This value is multiplied with the ranks after computation, to rescale them.  This
        parameter is set by the the method enforce_etendue, which you can read about in
        tht method documentation.  In general you shouldn't have to touch this, but it is
        here if you need it.
    
    Public Methods
    --------------
    reroll()
    enforce_etendue()
    
    """
    def __init__(
        self,
        base_point_distribution,
        sample_count,
        rank_distribution=None,
        auto_reroll=True,
        conserve_etendue=True,
        etendue_origin=(0, 0),
        **kwargs
    ):
        self.sample_count = sample_count
        self.base_point_distribution = base_point_distribution
        self.rank_distribution = rank_distribution
        self.auto_reroll = auto_reroll
        self.rank_scale_factor = 1
        super().__init__(**kwargs)
        if conserve_etendue:
            self.enforce_etendue(etendue_origin)
        
    @property
    def sample_count(self):
        return self._sample_count
        
    @sample_count.setter
    def sample_count(self, val):
        tf.debugging.assert_greater(
            val, 0, "AribitraryBasePoints: sample_count must be > 0."
        )
        tf.debugging.assert_integer(
            val, "AribitraryBasePoints: sample_count must be of integer type."
        )
        self._sample_count = val
        
    def reroll(self):
        """
        Generates a new set of input random numbers, which re-randomizes the base points.
        Only necessary if auto_reroll is set to false.
        """
        self._base_x = tf.random.uniform(
            (self._sample_count,),
            self.base_point_distribution._x_min,
            self.base_point_distribution._x_max
        )
        self._base_y = tf.random.uniform(
            (self._sample_count,),
            self.base_point_distribution._y_min,
            self.base_point_distribution._y_max
        )        
        
    def _update(self):
        if self.auto_reroll or self._ranks is None:
            self.reroll()
            
        self._points = tf.stack(
            self.base_point_distribution(self._base_x, self._base_y),
            axis=1
        )
        self._ranks = self.rank_scale_factor * tf.stack(
            self.rank_distribution(self._base_x, self._base_y),
            axis=1
        )
        
    def enforce_etendue(self, origin=(0, 0)):
        """
        Helper function to ensure the optimization goal conforms to conservation of
        etendue.
        
        An optical system conserves etendue, or at least, etendue must never decrease.  But
        the optimization goal you define might not conform to this constraint, in which case
        it will be impossible to achieve and the result will be poor.  It is a bit tricky
        ahead of time to determine how to preserve this condition when working with non-
        uniform distributions, which is why this helper function exists.  This function 
        calculates and sets the attribute rank_rescale, which is multiplied with the ranks
        to rescale them, which will hopefully cause the optimization goal to comply with
        conservation of etendue.  This function is called by default in the class constructor,
        and will rescale the ranks unless the parameter preserve_etendue is set to false
        in the constructor.
        
        The current implementation of this function uses an extremely simplified calculation:
        it simply uses the average distance of each point in the input and goal distributions 
        to the center of the distribution is the same.  This may or may not be sufficient
        to get a good result.
        
        Parameters
        ----------
        origin : 2-tuple of floats, optional.
            The x,y coordinates of the point to be used as the center of the distribution.
        """
        base_etendue = tf.reduce_mean(tf.norm(self._points - origin, axis=1))
        ranks_etendue = tf.reduce_mean(tf.norm(self._ranks - origin, axis=1))
        self.rank_scale_factor = base_etendue / ranks_etendue
        self._ranks *= self.rank_scale_factor
                
# =========================================================================================

def transform_map_old(fixed, mutable, origin=None, furthest_first=True):
    """
    Returns mutable with the the order of the elements rearranged to find the 'most 
    efficient' mapping from fixed to mutable.
    
    Fixed and mutable should both be distributions of points.  They should have the exact
    same shape.  This function finds the proper order to place the points in mutable so
    that they correlate to the points in fixed with the smallest possible distance
    between the two sets.  This function is for mapping between two distributions, for
    the purpose of defining an optimization goal.
    
    This function will start with the points in fixed farthest from the origin.  If the
    origin parameter is None, zero will be used instead.
    
    This is a slow function, and so should probably be precompiled.
    """
    
    # I have decided I want to write this function in np rather than TF.
    fixed = np.array(fixed)
    mutable = np.array(mutable)
    
    if fixed.shape != mutable.shape:
        raise ValueError("transform_map: both inputs must have exactly the same shape.")
        
    if origin is None:
        origin = np.zeros(fixed.shape[1])
    elif origin.shape[0] != fixed.shape[1]:
        raise ValueError("transform_map: origin must have the same dimension as fixed.")

    # determine the order to iterate through points.  Takes points in fixed in order of 
    # decreasing distance from origin
    distance = np.linalg.norm(fixed - origin, axis=1)
    if furthest_first:
        fixed_indices = np.argsort(distance)[::-1]
    else:
        fixed_indices = np.argsort(distance)
    
    rearanged = np.zeros_like(mutable)
    used_mutable = np.zeros(mutable.shape[0], dtype=bool)
    
    for fixed_index in fixed_indices:
        # compute distance between the selected fixed and every mutable
        distance = np.linalg.norm(fixed[fixed_index] - mutable, axis=1)
        
        # filter out mutables that have already been used
        max_distance = 2*np.amax(distance)
        distance[used_mutable] = max_distance
        
        # find just the closest mutable
        closest_mutable = np.argmin(distance)
        
        # update things for the next iteration
        used_mutable[closest_mutable] = True
        rearanged[fixed_index] = mutable[closest_mutable]
    return rearanged
    
def transform_map(fixed, mutable):
    """
    Returns mutable with the the order of the elements rearranged to find the 'most 
    efficient' mapping from fixed to mutable.
    
    Fixed and mutable should both be distributions of points.  They should have the exact
    same shape.  This function finds the proper order to place the points in mutable so
    that they correlate to the points in fixed with the smallest possible distance
    between the two sets.  This function is for mapping between two distributions, for
    the purpose of defining an optimization goal.
    
    This is a slow function, and so should probably be precompiled.
    """
    
    # I have decided I want to write this function in np rather than TF.
    fixed = np.array(fixed)
    mutable = np.array(mutable)

    # check that the shape of each is exactly the same
    if fixed.shape != mutable.shape:
        raise ValueError("transform_map: both inputs must have exactly the same shape.")
        
    # fixed and mutable should have shape (n, d).  Generate an nxn matrix that holds the 
    # distance between each point.  distance[n, m] should be the distance between fixed[n] and
    # mutable [m]
    
    point_count = fixed.shape[0]
    index_grid = np.mgrid[0:point_count, 0:point_count]

    diff = fixed[index_grid[0]] - mutable[index_grid[1]]
    distance = np.linalg.norm(diff, axis=2)

    # use the distance matrix as a cost matrix and use the Hungarian method implemented by 
    # scipy to solve the problem
    fixed_indices, mutable_indices = linear_sum_assignment(distance)
    
    # we want to rearrange mutable, but keep the order of fixed.  So we want to re-order 
    # mutable by the set of mutable_indices generated by the above function call, which 
    # themselves are ordered by fixed_indices.  During testing, I found that fixed_indices 
    # maintained its original order, but I don't want to rely on that.  Not sure if this is 
    # correct...
    
    return mutable[mutable_indices[fixed_indices]]
        
# =========================================================================================

class ImageBasePoints(BasePointDistributionBase):
    """
    Base class for base points generated from an image file.
    
    Generates a grid of x,y points, centered on zero.  Points are sampled randomly based
    on the greyscale value of each pixel.
    
    The image needs to be a greyscale image that has previously been thresholded in an
    image manipulation program to have 10-20 greyscale levels.
    """
    def __init__(self, filename, x_size, y_size=None, **kwargs):
        self.x_size = x_size
        self.y_size = y_size or x_size
        
        # load the image into an np array
        raw_image = np.array(imageio.imread(filename, as_gray=True))
        self._x_res, self._y_res = raw_image.shape
        
        # grey values are automatically 8-bit, but the image should have already been
        # thresholded (externally in an image manipulation program) so we need to convert
        # the values down so that they start at zero and count up in increments of one.
        unique, unique_indices = np.unique(raw_image, return_inverse=True)
        self._grey_levels = len(unique)
        grey_values = np.arange(self._grey_levels)
        processed_image = grey_values[unique_indices]
        
        # np.unique flattens its array by default.
        self._image = np.reshape(processed_image, (self._x_res, self._y_res))
        
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
            "ImageBasePoints: x_size must be > 0."
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
            "ImageBasePoints: y_size must be > 0."
        )
        self._y_size = val
        
    @property
    def grey_levels(self):
        return self._grey_levels
        
    @property
    def x_res(self):
        return self._x_res
        
    @property
    def y_res(self):
        return self._y_res
        
    @property
    def points(self):
        return self._points
        
    def _update(self):
        points_list = []
        x_edges = np.linspace(-self._x_size/2, self._x_size/2, self._x_res + 1)
        y_edges = np.linspace(-self._y_size/2, self._y_size/2, self._y_res + 1)
        
        for x in range(self._x_res):
            for y in range(self._y_res):
                x_pts = np.random.uniform(x_edges[x], x_edges[x+1], self._image[x, y])
                y_pts = np.random.uniform(y_edges[y], y_edges[y+1], self._image[x, y])
                points_list.append(np.stack([x_pts, y_pts], axis=1))
        self._points = np.concatenate(points_list, axis=0)
        
    def _generate_update_handles(self):
        return []
        
    @property
    def ranks(self):
        return self._ranks
        
    @ranks.setter
    def ranks(self, val):
        self._ranks = val
        
# =========================================================================================

class PrecompiledBasePoints(RecursivelyUpdatable):
    """
    A set of base points that have been precompiled and stored in a file, for performance 
    reasons.
    
    This class is mostly intended to provide a performance boost when using transform_map.
    """
    def __init__(
        self,
        arg,
        sample_count=100,
        do_downsample=True,
        perturbation=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        arg :
            The thing to initialze the base points from, with several different options
            depending on the type of object passed to this parameter:
            
            str : arg is interpreted as a filename, and the points are loaded from this
                file.
            None :
                An empty set of points is generated.
            default : arg is interpreted as another base point object, and the points are
                copied from this object.
        sample_count : int, optional
            Only has an effect if do_downsample is True.  The number of points to generate
            for this set each time it is updated.  Defaults to 100
        do_downsample : bool, optional
            If true, will select sample_count points randomly out of all the points stored
            in the set.  Random selection is performed with replacement.  This number
            may be larger than the actual number of points stored in the set, but will
            usually be smaller.  Defaults to True.
        perturbation : float tensor, optional
            Defaults to None, in which case each update will generate points matching the
            internally stored points with no perturbation.  But if it is not None, it must 
            be broadcastable to the number of dimensions, and will add a random value to 
            each point, sampled from a normal distribution with a standard deviation equal 
            to the value of each element of this parameter.  Useful for generating a 
            slightly randomized set of points from a larger set of non-random ones.  For 
            example, if the points are 3D, and perturbation is (.1, .1, 0), then the x and 
            y coordinates of each point will be moved, but the z coordinate will remain 
            unchanged.  May just be a scalar, to perturb each dimension equally.
        """
        if type(arg) is str:
            # interpret the first arg as a filename
            with open(arg, 'rb') as in_file:
                in_data = pickle.load(in_file)
                self._full_points = in_data["points"]
                self._full_ranks = in_data["ranks"]
        else:
            # interpret arg as another base point distribution
            try:
                self._full_points = arg.points
            except(AttributeError):
                self._full_points = None
            try:
                self._full_ranks = arg.ranks
            except(AttributeError):
                self._full_ranks = None
            
        self.perturbation = perturbation
        try:
            self._sampling_domain_size = tf.shape(self._full_points)[0]
        except:
            self._sampling_domain_size = 0
        self.sample_count = sample_count
        self.do_downsample = do_downsample
    
        RecursivelyUpdatable.__init__(self, **kwargs)
        
    def save(self, filename):
        # want to convert the data to np arrays before storing, since tf tensors
        # are stateful objects that may hold references to a tf.Session or tf.Graph
        if self._full_points is None:
            points = None
        else:
            points = np.array(self._full_points)
        if self._full_ranks is None:
            ranks = None
        else:
            ranks = np.array(self._full_ranks)
            
        out_data = {"points": points, "ranks": ranks}
        with open(filename, 'wb') as out_file:
            pickle.dump(out_data, out_file, pickle.HIGHEST_PROTOCOL)
            
    def _update(self):
        # do the downsampling
        if self.do_downsample:
            sample_indices = tf.random.uniform(
                (self.sample_count,),
                maxval=self.sampling_domain_size,
                dtype=tf.int32
            )
            if self._full_points is not None:
                self._points = tf.gather(self._full_points, sample_indices, axis=0)
            if self._full_ranks is not None:
                self._ranks = tf.gather(self._full_ranks, sample_indices, axis=0)
        else:
            self._points = self._full_points
            self._ranks = self._full_ranks
            
        # do the perturbations
        if self._perturbation is not None and self._points is not None:
            shape = tf.shape(self._points)[0]
            perturbation = [
                tf.random.normal(
                    (shape,),
                    stddev=dev,
                    dtype=tf.float64
                )
                for dev in self._perturbation.numpy()
            ]
            self._points += tf.stack(perturbation, axis=1)
            
    def clear(self):
        self._full_points = None
        self._full_ranks = None
        self._points = None
        self._ranks = None
        
    def _generate_update_handles(self):
        return []
        
    @property
    def sampling_domain_size(self):
        return self._sampling_domain_size
        
    @property
    def points(self):
        return self._points
        
    @points.setter
    def points(self, val):
        self._full_points = val
        
    @property
    def ranks(self):
        return self._ranks
        
    @ranks.setter
    def ranks(self, val):
        self._full_ranks = val
        
    @property
    def full_points(self):
        return self._full_points
        
    @property
    def full_ranks(self):
        return self._full_ranks
        
    @property
    def perturbation(self):
        return self._perturbation
        
    @perturbation.setter
    def perturbation(self, val):
        if val is not None:
            try:
                shape = tf.shape(self._full_points)[1]
                val = tf.broadcast_to(val, (shape,))
            except:
                raise ValueError(
                    "PrecompiledBasePoints: perturbation must be None, scalar, or "
                    "must have one entry per dimension of the points."
                )
        self._perturbation = val
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    


