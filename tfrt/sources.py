"""
Classes to help make source light ray sets that can be fed to the ray tracer.

The ray tracer requires as one of its inputs a tensor that encodes the light input to 
the system.  This tensor must be rank 2, whose first dimension can be of any size and 
indexes each ray, and whose second dimension must be of size >= 5, and whose first 
five elements are xStart, yStart, xEnd, yEnd, and wavelength.  Rays are defined as 
line segments between the two points, Start and End.  But they are interpreted as 
semi-infinite rays whose origin is Start and whose angle is set by End.  The length 
of a ray does not matter.  The ray tracer will add a few extra elements to each ray 
in its output that help track where the ray originated from and what surfaces it 
interacted with.  Ray sets with these extra elements can be fed to a second ray 
tracer without issue, the extra information will simply be ignored.  This is why the 
second dimension of a ray set is defined as having size >= 5, rather than size == 5.

Since TensorFlow will automatically convert some python objects into tensors (so 
called tensor-like objects), there are multiple valid ways to generate starting ray 
sets for the ray tracer.  One issue is that TF by default interprets python floats as 
32-bit floats, and the ray tracer requires its inputs to be 64-bit floats, so if you 
are building your ray sets with python logic it may be necessary to cast them.  A 
common pattern that I have used in the past to generate ray sets was to use a python 
list comprehension to build a nested list, and then cast it to a numpy array.  
Example:

startingRays = np.array(
    [
        [
            0.0,
            y,
            -1.0,
            y,
            wavelength
        ] for y in yValues for wavelength in wavelengths
    ],
    dtype=np.float64
)

This object can be fed directly to the ray tracer.  It is also possible to give the 
ray tracer a placeholder and feed rays at runtime.

Though these ways of building a rayset are valid, this module has been provided to 
make ray set generation more easy and convenient.  The sources defined in this module 
automatically package the ray set, the data used to build it, and ray ranks, which 
can help define target locations when using an optimizer.  Most of the parameters in 
the class constructors in this module can accept tensors as inputs, to simplify 
building parameterized ray sets.  Some of these parameters (like source center) can 
even be used as optimization targets to, for instance, figure out at what location a 
ray has to strike the entrance aperture of an optic in order to pass through a 
specific location inside the optic.  Any parameter that can accept a tensor (and 
therefore can be parametric) is labeled as parametric in the constructor's 
documentation.  Any parameter which can only accept a python object will be labeled 
as fixed, and the behavior controlled by that parameter cannot be adjusted after 
object instantiation.

There are three types of object defined in this module: Angular distributions, which 
describe the angle of rays in a ray set; base point distributions, which describe a 
point along the ray; and sources which describe ray sets and are the ultimate goal of 
this module.  Different sources accept different numbers of angular and base point 
distributions as parameters.

There are three types of distributions: manual, static, and random.  

Manual distributions are simply utilities that package user-defined data into the 
format used by sources.  This is a convenience for users who want to use a source 
partially defined by this module and partially defined by themselves (like the list 
comprehension and cast to numpy array example provided above).  Manual distributions 
do not do any kind of error checking on the data fed to them!

Static distributions may be parametric, but they will yield the exact same ray set 
each time they are evaluated by a TF session (unless their parameters change).

Random distributions are built out of TF random ops, and so will yield different rays 
each time they are evaluated, such as during optimization.  Be warned that if you 
desire to visualize starting rays during individual steps of optimization, if you are 
using a random distribution you will need to extract the ray set in the same 
session.run call where you run the optimization op, or else your visualization will 
not display rays that were actually used during optimization.

Sources may be built either dense (the default) or not.  If a source is dense, it 
will generate a ray for each combination of elements in its distributions.  For 
instance, if you build a source and feed it 3 angles, 4 base points, and 5 
wavelengths, a dense source will yield 3*4*5 = 60 rays.  If you build a source as un-
dense, each distribution must have exactly the same number of elements, and the 
source will yield that many rays.  It is recommended that any source that uses a 
random distribution be un-dense (simply because that will make it more random) but 
this is not required.

The objects defined in this module each take an optional name parameter in their 
constructor.  The name is used to define a name scope under which all ops added to 
the TF graph by the object are placed.  The desired effect of this is to make the 
graph visualization via TensorBoard cleaner.  By default a source will group the ops 
used to build a distribution inside its own group (under its own name scope) so that 
the source will appear in its entirety as a single node in TensorBoard.  If you do 
not want this behavior, you can call the distribution.build method to place the ops 
that build the distribution inside whatever name scope is currently active when the 
build method is called.  The expectation is that each distribution will be used only 
once, for a single source.  It is possible to share a distribution between each 
source, but if this is done, by default the distribution's ops will be grouped inside 
the source node of the source which is built first, and it will appear on TensorBoard 
that later sources have a dependency on the first source (through the shared 
distribution.)  In this case, it is probably cleaner to build the distribution 
manually before either source, so that it appears as its own group.

Public Angular Distributions
----------------------------
ManualAngularDistribution
StaticUniformAngularDistribution
RandomUniformAngularDistribution
StaticLambertianAngularDistribution
RandomLambertianAngularDistribution

Public Base Point Distributions
-------------------------------
ManualBasePointDistribution
StaticUniformBeam
RandomUniformBeam
StaticUniformAperaturePoints
RandomUniformAperaturePoints

Sources
-------
Point Source
Angular Source
Aperature Source


"""
import math
import itertools
from abc import ABC, abstractmethod
from tfrt.update import RecursivelyUpdatable

import tensorflow as tf
import numpy as np

PI = math.pi
COUNTER = itertools.count(0)

# ====================================================================================


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
        change if you change, say the min_angle of the distribution.  This operation can
        be called recursively by higher tier consumers of the angular distribution 
        (like a source), but updating a distribution will NOT recursively update any 
        lower tier object the distribution may be made from, since these are expected 
        to be at the bottom of the stack.
        
        Random ops will have their random distribution re-sampled every time they
        are updated.

    """

    def __init__(self, min_angle, max_angle, sample_count, name=None):
        """
        Angular distribution constructors do not need to follow the pattern set by
        the constructor defined here, but most will, so this implementation is
        provided for convenience.  All angular distribution constructors should accept 
        a name keyword argument, which is used to define a namespace in which all tf 
        ops constructed by the class are placed.

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
        pattern does not work, or if you generate ranks in the process of building the 
        distribution, do not use it.
        
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
        Recalculates the values stored by the distribution, if update_function was set.
        May be called by sources that consume this distribution.
        

    """

    def __init__(self, angles, ranks=None, name=None, update_function=None):
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self.angles = angles
        self.ranks = ranks
        self.update_function = update_function

    def update(self):
        if self.update_function is not None:
            self.angles, self.ranks = self.update_function()
        


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
        The miniumum angle to include in the distribution.  Interpreted as relative to the 
        central angle of the source.
    max_angle : tf.float64 tensor of shape (None,)
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
        The miniumum angle to include in the distribution.  Interpreted as relative to the 
        central angle of the source.
    max_angle : scalar tf.float64 tensor
        The maxiumum angle to include in the distribution.  Interpreted as relative to the 
        central angle of the source.
    sample_count : scalar tf.float64 tensor
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


# ====================================================================================


class SourceBase(RecursivelyUpdatable, ABC):
    """
    Abstract implementation of a source, which defines the bare minimum interface
    that all sources should implement.
    
    Exposes its data in the signature format used by RaySets; so the object can be
    indexed like a dictionary.  Differences are that sources allow you to 
    redefine the names of fields in the signature, but that you are allowed to add 
    new fields.
    This can be used to, for instance, add ray intensity data to a source.  But be 
    warned that by default added data fields will not be updated (have an update 
    function called for them) unless you manually write an update function and hook 
    it to the sources via add_update_handle.  If you pass a set of strings that will 
    name any desired custom fields to the constructor, will automatically initialize 
    them.
    
    A source has a public method, update, which will cause the source to recalculate 
    all of its values.  By default, the update call will recursively call update on 
    the distribution objects the source consumes, and any extra handles added by 
    add_update_handle.  This behavior can be disabled by setting the constructor 
    argument recursively_update to False.  But be warned that doing this means any 
    added update handles will never be called automatically by this source.
    
    If update_handles is kept with the default value of None, then the source 
    constructor will populate it with the update method of the distributions consumed 
    by the source.  But if you know that one consumed distribution will be static and 
    thus never need to be updated, but that the other will need to be updated, you 
    can feed this parameter a list that contains the distribution.update methods that 
    you want to respond to recursive updating.
    
    A dense source generates a set of rays that takes each combination of its 
    inputs.  An un-dense source requires its input all have the same size and 
    produces that many rays by matching data in its inputs 1:1.  Dense sources are  
    convenient for static sources (sources that use static distributions) since it 
    generates rays that span the entire range of specified inputs.  It is not
    recommended that sources that use random distributions choose to be dense, simply 
    because they will be less random.  But the source will work either way.
    
    Parameters
    ----------
    name : string, optional
        The name of the distribution.
    dense : bool, optional
        True if the source is dense.
    extra_fields : set, optional
        Extra fields to define in this source's signature.
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
    signature : set of strings
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
        extra_fields=set(),
        **kwargs
    ):
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self.dense = dense
        self._fields = {sig: None for sig in self._build_signature(extra_fields)}
        super().__init__(**kwargs)

    @abstractmethod
    def _build_signature(self, extra_fields):
        """
        define the signature for this source.
        """
        raise NotImplementedError

    @abstractmethod
    def _make_dense(self):
        """
        Inspect the parameter distributions, and use meshgrid and reshape to densify
        their values.  This function should set the internal copy of the values,
        which will used by the source and be passed out of the source as read-only
        properties.

        """
        raise NotImplementedError

    @abstractmethod
    def _make_undense(self):
        """
        Inspect the parameters and set the internal copy of the values, which will
        used by the source and be passed out of the source as read-only properties.
        Either _make_dense or _make_undense should always be called exactly once in
        generating the source, depending on whether dense is true or not.

        """
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    def __getitem__(self, key):
        return self._fields[key]

    def __setitem__(self, key, item):
        self._fields[key] = item

    @property
    def signature(self):
        return set(self._fields.keys())
        
    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value


class Source2D(SourceBase):
    @property
    def dimension(self):
        return 2

    def _build_signature(self, extra_fields):
        default = set({"x_start", "y_start", "x_end", "y_end", "wavelength"})
        return default | extra_fields


class PointSource(Source2D):
    """
    Rays eminating from a single point.
    
    Exposes its data in the signature format used by RaySets; so the object can be
    indexed like a dictionary.  Differences are that sources do not allow you to redefine
    the names of fields in the signature, but that you are allowed to add new fields.
    This can be used to, for instance, add ray intensity data to a source.  But be warned
    that by default added data fields will not be updated (have an update function called
    for them) unless you manually write an update function and hook it to the sources via
    add_update_handle.  If you pass a set of strings that will name any desired custom 
    fields to the constructor, will automatically initialize them.
    
    A source has a public method, update, which will cause the source to recalculate all
    of its values.  By default, the update call will recursively call update on the
    distribution objects the source consumes, and any extra handles added by 
    add_update_handle.  This behavior can be disabled by setting the constructor argument
    recursively_update to False.  But be warned that doing this means any added update
    handles will never be called automatically by this source.
    
    If update_handles is kept with the default value of None, then the source constructor
    will populate it with the update method of the distributions consumed by the source.
    But if you know that one consumed distribution will be static and thus never need
    to be updated, but that the other will need to be updated, you can feed this parameter
    a list that contains the distribution.update methods that you want to respond to
    recursive updating.
    
    A dense source generates a set of rays that takes each combination of its inputs.  An
    un-dense source requires its input all have the same size and produces that many rays 
    by matching data in its inputs 1:1.  Dense sources are  convenient for static sources 
    (sources that use static distributions) since it generates rays that span the entire 
    range of specified inputs.  It is not recommended that sources that use random 
    distributions choose to be dense, simply because they will be less random.  But the 
    source will work either way.
    
    Parameters
    ----------
    center : tf.float64 tensor of shape (2,)
        The x and y coordinates of the center of the source.
    central_angle : scalar tf.float64 tensor
        The angle which the center of the source faces.
    angular_distribution : a child of AngularDistributionBase
        The angular distribution that describes the angles of rays generated by the source.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of the rays generated by the source.
    name : string, optional
        The name of the distribution.
    dense : bool, optional
        True if the source is dense.
    start_on_center : bool, optional
        True if rays should start on the center point, for a diverging source.
        False if rays should end on the center point, for a converging source.
    ray_length : tf.float64 tensor broadcastable to (None,), optional
        The length of all generated rays.  Usually a scalar, but could possibly also be 1-D
        and have the exact length of the total number of rays generated by the source.
    extra_fields : set, optional
        Extra fields to define in this source's signature.
    update_handles : list, optional
        A list of functions that should be called when this source is update, before
        it updates itself.  Defaults to None, in which case a default is used that is
        simply the update function of the distributions this source consumes.  If anything
        other than None is passed to this parameter, then the default update handles
        will NOT be added to the handler list.  But if you know that one distribution
        needs to be updated and the other does not, you can pass this parameter a list
        containing just the update method of the distribution that needs to be updated, and
        that distribution will be updated by a call to the source's update method.
        These update methods will NOT be called if recursively_update is False.
    recursively_update : bool, optional
        If true, will call all methods in update_handles before updating itself, each
        time update is called.  Otherwise will only call the source's own update method.
        
    Public read-only attributes
    ---------------------------
    angular_distribution : a child of AngularDistributionBase
        The angular distribution that describes the angles of rays generated by the source.
    angles : tf.float64 tensor of shape (None,)
        The angles of each ray in the distribution.  Differs from the angles described
        by the angular_distribution if the source is dense, since the angles will be
        combined with the wavelengths.
    angle_ranks : tf.float64 tensor of shape (None,)
        The rank of the angle of each ray in the distribution.  Differs from the ranks 
        described by the angular_distribution if the source is dense, since the angles
        will becombined with the wavelengths.
    signature : set of strings
        The keys that can be used to index data out of this source.
    dimension: int
        Either 2 or 3, the dimension of the space in which the source is embedded.
    name : string
        The name of the distribution.
    
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
    recursively_update : bool
        If true, will call all methods in update_handles before updating itself, each
        time update is called.  Otherwise will only call the source's own update method.
    update_handles : list
        A list of functions that should be called when this source is update, before
        it updates itself.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of light generated by this source
        
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

    """

    def __init__(
        self,
        center,
        central_angle,
        angular_distribution,
        wavelengths,
        name=None,
        dense=True,
        start_on_center=True,
        ray_length=1.0,
        **kwargs,
    ):
        self.center = center
        self.central_angle = central_angle
        self._angular_distribution = angular_distribution
        self._wavelengths = wavelengths
        self.start_on_center = start_on_center
        self.ray_length = ray_length

        super().__init__(name, dense=dense, **kwargs)

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(2,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.bool),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        ]
    )
    def _source_update(
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

    def _update(self):
        if self.dense:
            if self._angular_distribution.ranks is not None:
                self._angles, self._processed_wavelengths, self._angle_ranks = \
                    self._make_dense(
                        self._angular_distribution.angles,
                        self._wavelengths,
                        self._angular_distribution.ranks,
                    )
            else:
                self._angles, self._processed_wavelengths, _ = self._make_dense(
                    self._angular_distribution.angles,
                    self._wavelengths,
                    tf.zeros_like(self._angular_distribution.angles),
                )
        else:
            self._angles, self._processed_wavelengths, self._angle_ranks = \
                self._make_undense(
                    self._angular_distribution.angles,
                    self._wavelengths,
                    self._angular_distribution.ranks,
                )
        self["x_start"], self["y_start"], self["x_end"], self["y_end"], \
            self["wavelength"] = \
            self._source_update(
            self.center,
            self.central_angle,
            self._processed_wavelengths,
            self._angles,
            self.start_on_center,
            self.ray_length,
        )

    def _generate_update_handles(self):
        return [self._angular_distribution.update]

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
        ]
    )
    def _make_dense(angles, wavelengths, angle_ranks):
        out_angles, out_wavelengths = tf.meshgrid(angles, wavelengths)
        out_angle_ranks, _ = tf.meshgrid(angle_ranks, wavelengths)

        out_angle_ranks = tf.reshape(out_angle_ranks, (-1,))
        out_angles = tf.reshape(out_angles, (-1,))
        out_wavelengths = tf.reshape(out_wavelengths, (-1,))

        return out_angles, out_wavelengths, out_angle_ranks

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
        ]
    )
    def _make_undense(angles, wavelengths, angle_ranks):
        angle_shape = tf.shape(angles)
        wavelength_shape = tf.shape(wavelengths)
        tf.assert_equal(
            angle_shape,
            wavelength_shape,
            message=f"For un dense source, need "
            f"exactly as many angles as wavelengths.",
        )

        return angles, wavelengths, angle_ranks

    @property
    def angles(self):
        return self._angles

    @property
    def angle_ranks(self):
        return self._angle_ranks

    @property
    def angular_distribution(self):
        return self._angular_distribution


class AngularSource(Source2D):
    """
    Rays eminating from multiple points.
    
    Exposes its data in the signature format used by RaySets; so the object can be
    indexed like a dictionary.  Differences are that sources do not allow you to redefine
    the names of fields in the signature, but that you are allowed to add new fields.
    This can be used to, for instance, add ray intensity data to a source.  But be warned
    that by default added data fields will not be updated (have an update function called
    for them) unless you manually write an update function and hook it to the sources via
    add_update_handle.  If you pass a set of strings that will name any desired custom 
    fields to the constructor, will automatically initialize them.
    
    A source has a public method, update, which will cause the source to recalculate all
    of its values.  By default, the update call will recursively call update on the
    distribution objects the source consumes, and any extra handles added by 
    add_update_handle.  This behavior can be disabled by setting the constructor argument
    recursively_update to False.  But be warned that doing this means any added update
    handles will never be called automatically by this source.
    
    If update_handles is kept with the default value of None, then the source constructor
    will populate it with the update method of the distributions consumed by the source.
    But if you know that one consumed distribution will be static and thus never need
    to be updated, but that the other will need to be updated, you can feed this parameter
    a list that contains the distribution.update methods that you want to respond to
    recursive updating.
    
    A dense source generates a set of rays that takes each combination of its inputs.  An
    un-dense source requires its input all have the same size and produces that many rays 
    by matching data in its inputs 1:1.  Dense sources are  convenient for static sources 
    (sources that use static distributions) since it generates rays that span the entire 
    range of specified inputs.  It is not recommended that sources that use random 
    distributions choose to be dense, simply because they will be less random.  But the 
    source will work either way.
    
    Parameters
    ----------
    center : tf.float64 tensor of shape (2,)
        The x and y coordinates of the center of the source.
    central_angle : scalar tf.float64 tensor
        The angle which the center of the source faces.
    angular_distribution : a child of AngularDistributionBase
        The angular distribution that describes the angles of rays generated by the source.
    base_point_distribution : a child of BasePointDistributionBase
        The base point distribution that determines where rays generated by the source
        start.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of the rays generated by the source.
    name : string, optional
        The name of the distribution.
    dense : bool, optional
        True if the source is dense.
    start_on_base : bool, optional
        True if rays should start on the base points, for a diverging source.
        False if rays should end on the base points, for a converging source.
    ray_length : tf.float64 tensor broadcastable to (None,), optional
        The length of all generated rays.  Usually a scalar, but could possibly also be 1-D
        and have the exact length of the total number of rays generated by the source.
    extra_fields : set, optional
        Extra fields to define in this source's signature.
    update_handles : list, optional
        A list of functions that should be called when this source is update, before
        it updates itself.  Defaults to None, in which case a default is used that is
        simply the update function of the distributions this source consumes.  If anything
        other than None is passed to this parameter, then the default update handles
        will NOT be added to the handler list.  But if you know that one distribution
        needs to be updated and the other does not, you can pass this parameter a list
        containing just the update method of the distribution that needs to be updated, and
        that distribution will be updated by a call to the source's update method.
        These update methods will NOT be called if recursively_update is False.
    recursively_update : bool, optional
        If true, will call all methods in update_handles before updating itself, each
        time update is called.  Otherwise will only call the source's own update method.
        
    Public read-only attributes
    ---------------------------
    angular_distribution : a child of AngularDistributionBase
        The angular distribution that describes the angles of rays generated by the source.
    angles : tf.float64 tensor of shape (None,)
        The angles of each ray in the distribution.  Differs from the angles described
        by the angular_distribution if the source is dense, since the angles will be
        combined with the wavelengths and base points.
    angle_ranks : tf.float64 tensor of shape (None,)
        The rank of the angle of each ray in the distribution.  Differs from the ranks 
        described by the angular_distribution if the source is dense, since the angles
        will becombined with the wavelengths and base points.
    base_point_distribution : a child of BasePointDistributionBase
        The base point distribution that determines where rays generated by the source
        start.
    base_points : a 2-tuple of tf.float64 tensor of shape (None,)
        The x and y coordinates of the fixed point of each ray in the distribution.  
        Differs from the base points described by the base_point_distribution if the 
        source is dense, since the base points will be combined with the wavelengths and 
        angles.
    base_point_ranks : tf.float64 tensor of shape (None,)
        The rank of the fixed point of each ray in the distribution.  Differs from the 
        ranks described by the base_point_distribution if the source is dense, since the 
        base points will be combined with the wavelengths and angles.
    signature : set of strings
        The keys that can be used to index data out of this source.
    dimension: int
        Either 2 or 3, the dimension of the space in which the source is embedded.
    name : string
        The name of the distribution.
    
    Public read-write attributes
    ----------------------------
    center : tf.float64 tensor of shape (2,)
        The x and y coordinates of the center of the source.
    central_angle : scalar tf.float64 tensor
        The angle which the center of the source faces.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of the rays generated by the source.
    start_on_base : bool
        True if rays should start on the base points, for a diverging source.
        False if rays should end on the base points, for a converging source.
    ray_length : tf.float64 tensor broadcastable to (None,), optional
        The length of all generated rays.
    dense : bool
        True if the source is dense.
    recursively_update : bool
        If true, will call all methods in update_handles before updating itself, each
        time update is called.  Otherwise will only call the source's own update method.
    update_handles : list
        A list of functions that should be called when this source is update, before
        it updates itself.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of light generated by this source
        
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
    """

    def __init__(
        self,
        center,
        central_angle,
        angular_distribution,
        base_point_distribution,
        wavelengths,
        name=None,
        dense=True,
        start_on_base=True,
        ray_length=1.0,
        **kwargs
    ):
        self.center = center
        self.central_angle = central_angle
        self._angular_distribution = angular_distribution
        self._base_point_distribution = base_point_distribution
        self._wavelengths = wavelengths
        self.start_on_base=start_on_base
        self.ray_length = ray_length
        
        super().__init__(name, dense=dense, **kwargs)

    @staticmethod
    @tf.function(
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
    )
    def _source_update(
        center,
        central_angle,
        wavelengths,
        angles,
        start_on_base,
        ray_length,
        base_points_x,
        base_points_y
    ):
        angles = angles + central_angle
        start_x = center[0] + base_points_x
        start_y = center[1] + base_points_y
        end_x = start_x + ray_length * tf.cos(angles)
        end_y = start_y + ray_length * tf.sin(angles)

        if start_on_base:
            return start_x, start_y, end_x, end_y, wavelengths
        else:
            return end_x, end_y, start_x, start_y, wavelengths
            
    def _update(self):
        try:
            self._base_point_distribution.central_angle = self.central_angle
            self._base_point_distribution.update()
        except AttributeError:
            pass
            
        if self.dense:
            if self._angular_distribution.ranks is not None:
                angle_ranks = self._angular_distribution.ranks
            else:
                angle_ranks = tf.zeros_like(self._angular_distribution.angles)
            if self._base_point_distribution.ranks is not None:
                base_point_ranks = self._base_point_distribution.ranks
            else:
                base_point_ranks = tf.zeros_like(
                    self._base_point_distribution.base_points_x
                )
                
            self._angles, self._base_points_x, self._base_points_y, \
                self._processed_wavelengths, self._angle_ranks, self._base_point_ranks = \
                self._make_dense(
                    self._angular_distribution.angles,
                    self._base_point_distribution.base_points_x,
                    self._base_point_distribution.base_points_y,
                    self._wavelengths,
                    self._angular_distribution.ranks,
                    self._base_point_distribution.ranks
                )
                
            if self._angular_distribution.ranks is None:
                self._angle_ranks = None
            if self._base_point_distribution.ranks is None:
                self._base_point_ranks = None
        else:
            self._angles, self._base_points_x, self._base_points_y, \
                self._processed_wavelengths, self._angle_ranks, self._base_point_ranks = \
                self._make_undense(
                    self._angular_distribution.angles,
                    self._base_point_distribution.base_points_x,
                    self._base_point_distribution.base_points_y,
                    self._wavelengths,
                    self._angular_distribution.ranks,
                    self._base_point_distribution.ranks
                )
        self["x_start"], self["y_start"], self["x_end"], self["y_end"], \
            self["wavelength"] = self._source_update(
                self.center,
                self.central_angle,
                self._processed_wavelengths,
                self._angles,
                self.start_on_base,
                self.ray_length,
                self._base_points_x,
                self._base_points_y
            )
        
    def _generate_update_handles(self):
        return [self._angular_distribution.update]

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64)
        ]
    )
    def _make_dense(
        angles,
        base_points_x,
        base_points_y,
        wavelengths,
        angle_ranks,
        base_point_ranks
    ):
        out_angles, out_wavelengths, out_base_x = tf.meshgrid(
            angles, wavelengths, base_points_x
        )
        _, _, out_base_y = tf.meshgrid(angles, wavelengths, base_points_y)
        
        out_angle_ranks, _, out_base_point_ranks = tf.meshgrid(
            angle_ranks, wavelengths, base_points_x
        )
        
        return (tf.reshape(out_angles, (-1,)),
            tf.reshape(out_base_x, (-1,)),
            tf.reshape(out_base_y, (-1,)),
            tf.reshape(out_wavelengths, (-1,)),
            tf.reshape(out_angle_ranks, (-1,)),
            tf.reshape(out_base_point_ranks, (-1,))
        )

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64)
        ]
    )
    def _make_undense(
        angles,
        base_points_x,
        base_points_y,
        wavelengths,
        angle_ranks,
        base_point_ranks
    ):
        angle_shape = tf.shape(angles)
        wavelength_shape = tf.shape(wavelengths)
        base_x_shape = tf.shape(base_points_x)
        base_y_shape = tf.shape(base_points_y)
        tf.assert_equal(
            angle_shape,
            wavelength_shape,
            message=f"For un dense source, need "
            f"exactly as many angles as wavelengths.",
        )
        tf.assert_equal(
            base_x_shape,
            wavelength_shape,
            message=f"For un dense source, need "
            f"exactly as many base points as wavelengths.",
        )
        tf.assert_equal(
            base_y_shape,
            wavelength_shape,
            message=f"For un dense source, need "
            f"exactly as many base points as wavelengths.",
        )
        return angles, base_points_x, base_points_y, wavelengths, angle_ranks, \
            base_point_ranks

    @property
    def angles(self):
        return self._angles

    @property
    def angle_ranks(self):
        return self._angle_ranks
        
    @property
    def angular_distribution(self):
        return self._angular_distribution

    @property
    def base_points(self):
        return self._base_points_x, self._base_points_y

    @property
    def base_point_ranks(self):
        return self._base_point_ranks

    @property
    def base_point_distribution(self):
        return self._base_point_distribution


class AperatureSource(Source2D):
    """
    A set of rays that span two sets of endpoints.
    
    This source does not use an angular distribution, and instead makes rays between
    two sets of points.  Useful if you know your input light is bounded by two
    apertures, and you don't want to calculate angles.

    This source will not attempt to feed central_angle to the base point distributions
    because there is no angle, and this source isn't expected to be used with
    beam distributions.  But if you feed your own central_angle to the distribution,
    you can still use a beam point distribution with this class.
    
    Exposes its data in the signature format used by RaySets; so the object can be
    indexed like a dictionary.  Differences are that sources do not allow you to redefine
    the names of fields in the signature, but that you are allowed to add new fields.
    This can be used to, for instance, add ray intensity data to a source.  But be warned
    that by default added data fields will not be updated (have an update function called
    for them) unless you manually write an update function and hook it to the sources via
    add_update_handle.  If you pass a set of strings that will name any desired custom 
    fields to the constructor, will automatically initialize them.
    
    A source has a public method, update, which will cause the source to recalculate all
    of its values.  By default, the update call will recursively call update on the
    distribution objects the source consumes, and any extra handles added by 
    add_update_handle.  This behavior can be disabled by setting the constructor argument
    recursively_update to False.  But be warned that doing this means any added update
    handles will never be called automatically by this source.
    
    If update_handles is kept with the default value of None, then the source constructor
    will populate it with the update method of the distributions consumed by the source.
    But if you know that one consumed distribution will be static and thus never need
    to be updated, but that the other will need to be updated, you can feed this parameter
    a list that contains the distribution.update methods that you want to respond to
    recursive updating.
    
    A dense source generates a set of rays that takes each combination of its inputs.  An
    un-dense source requires its input all have the same size and produces that many rays 
    by matching data in its inputs 1:1.  Dense sources are  convenient for static sources 
    (sources that use static distributions) since it generates rays that span the entire 
    range of specified inputs.  It is not recommended that sources that use random 
    distributions choose to be dense, simply because they will be less random.  But the 
    source will work either way.
    
    Parameters
    ----------
    center : tf.float64 tensor of shape (2,)
        The x and y coordinates of the center of the source.
    central_angle : scalar tf.float64 tensor
        The angle which the center of the source faces.
    angular_distribution : a child of AngularDistributionBase
        The angular distribution that describes the angles of rays generated by the source.
    base_point_distribution : a child of BasePointDistributionBase
        The base point distribution that determines where rays generated by the source
        start.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of the rays generated by the source.
    name : string, optional
        The name of the distribution.
    dense : bool, optional
        True if the source is dense.
    extra_fields : set, optional
        Extra fields to define in this source's signature.
    update_handles : list, optional
        A list of functions that should be called when this source is update, before
        it updates itself.  Defaults to None, in which case a default is used that is
        simply the update function of the distributions this source consumes.  If anything
        other than None is passed to this parameter, then the default update handles
        will NOT be added to the handler list.  But if you know that one distribution
        needs to be updated and the other does not, you can pass this parameter a list
        containing just the update method of the distribution that needs to be updated, and
        that distribution will be updated by a call to the source's update method.
        These update methods will NOT be called if recursively_update is False.
    recursively_update : bool, optional
        If true, will call all methods in update_handles before updating itself, each
        time update is called.  Otherwise will only call the source's own update method.
        
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
    start_point_ranks : tf.float64 tensor of shape (None,)
        The rank of the start point of each ray in the distribution.  Differs from the 
        ranks described by the start_point_distribution if the source is dense, since the 
        start points will be combined with the wavelengths and end points.
    end_point_distribution : a child of BasePointDistributionBase
        The base point distribution that determines where rays generated by the source
        end.
    end_points : a 2-tuple of tf.float64 tensor of shape (None,)
        The x and y coordinates of the end point of each ray in the distribution.  
        Differs from the base points described by the end_point_distribution if the 
        source is dense, since the end points will be combined with the wavelengths and 
        start points.
    end_point_ranks : tf.float64 tensor of shape (None,)
        The rank of the end point of each ray in the distribution.  Differs from the 
        ranks described by the end_point_distribution if the source is dense, since the 
        end points will be combined with the wavelengths and start points.
    signature : set of strings
        The keys that can be used to index data out of this source.
    dimension: int
        Either 2 or 3, the dimension of the space in which the source is embedded.
    name : string
        The name of the distribution.
    
    Public read-write attributes
    ----------------------------
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of the rays generated by the source.
    dense : bool
        True if the source is dense.
    recursively_update : bool
        If true, will call all methods in update_handles before updating itself, each
        time update is called.  Otherwise will only call the source's own update method.
    update_handles : list
        A list of functions that should be called when this source is update, before
        it updates itself.
    wavelengths : tf.float64 tensor of shape (None,)
        The wavelengths of light generated by this source
        
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

    """

    def __init__(
        self,
        start_point_distribution,
        end_point_distribution,
        wavelengths,
        name=None,
        dense=True,
        **kwargs
    ):
        self._start_point_distribution = start_point_distribution
        self._end_point_distribution = end_point_distribution
        self._wavelengths = wavelengths
        super().__init__(name, dense=dense, **kwargs)
        
    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64)
        ]
    )
    def _source_update(start_x, start_y, end_x, end_y, wavelengths):
        """
        Feels a little silly that I am doing this, but this will force the shape and type 
        specifications, and also adheres to how the other sources work.
        """
        return start_x, start_y, end_x, end_y, wavelengths
        
    def _update(self):
        if self._start_point_distribution.ranks is not None:
            start_ranks = self._start_point_distribution.ranks
        else:
            start_ranks = tf.zeros_like(
                self._start_point_distribution.base_points_x,
                dtype=tf.float64
            )
        if self._end_point_distribution.ranks is not None:
            end_ranks = self._end_point_distribution.ranks
        else:
            end_ranks = tf.zeros_like(
                self._end_point_distribution.base_points_x,
                dtype=tf.float64
            )
                
        if self.dense:
            self["x_start"], self["y_start"], self["x_end"], self["y_end"],\
                self["wavelength"], self._start_ranks,\
                self._end_ranks = self._make_dense(
                    self._start_point_distribution.base_points_x,
                    self._start_point_distribution.base_points_y,
                    self._end_point_distribution.base_points_x,
                    self._end_point_distribution.base_points_y,
                    self._wavelengths,
                    start_ranks,
                    end_ranks
                )
        else:
            self["x_start"], self["y_start"], self["x_end"], self["y_end"],\
                self["wavelength"], self._start_ranks,\
                self._end_ranks = self._make_undense(
                    self._start_point_distribution.base_points_x,
                    self._start_point_distribution.base_points_y,
                    self._end_point_distribution.base_points_x,
                    self._end_point_distribution.base_points_y,
                    self._wavelengths,
                    start_ranks,
                    end_ranks
                )
                
        if self._start_point_distribution.ranks is None:
            self._start_ranks = None
        if self._end_point_distribution.ranks is None:
            self._end_ranks = None
            
    def _generate_update_handles(self):
        return [
            self.start_point_distribution.update,
            self.end_point_distribution.update
        ]

    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64)
        ]
    )
    def _make_dense(start_x, start_y, end_x, end_y, wavelengths, start_ranks, end_ranks):
        out_start_x, out_end_x, out_wavelengths = tf.meshgrid(
            start_x, end_x, wavelengths
        )
        out_start_y, out_end_y, _ = tf.meshgrid(start_y, end_y, wavelengths)
        out_start_ranks, out_end_ranks, _ = tf.meshgrid(
            start_ranks, end_ranks, wavelengths
        )
        
        return tf.reshape(out_start_x, (-1,)), tf.reshape(out_start_y, (-1,)), \
            tf.reshape(out_end_x, (-1,)), tf.reshape(out_end_y, (-1,)), \
            tf.reshape(out_wavelengths, (-1,)), tf.reshape(out_start_ranks, (-1,)), \
            tf.reshape(out_end_ranks, (-1,))
            
    @staticmethod
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64),
            tf.TensorSpec(shape=(None,), dtype=tf.float64)
        ]
    )
    def _make_undense(start_x, start_y, end_x, end_y, wavelengths, start_ranks, end_ranks):
        wavelength_shape = tf.shape(wavelengths)
        start_x_shape = tf.shape(start_x)
        start_y_shape = tf.shape(start_y)
        end_x_shape = tf.shape(end_x)
        end_y_shape = tf.shape(end_y)
        start_ranks_shape = tf.shape(start_ranks)
        end_ranks_shape = tf.shape(end_ranks)
        tf.assert_equal(
            start_x_shape,
            wavelength_shape,
            message=f"For un dense source, need "
            f"exactly as many start_x points as wavelengths.",
        )
        tf.assert_equal(
            start_y_shape,
            wavelength_shape,
            message=f"For un dense source, need "
            f"exactly as many start_y points as wavelengths.",
        )
        tf.assert_equal(
            end_x_shape,
            wavelength_shape,
            message=f"For un dense source, need "
            f"exactly as many end_x points as wavelengths.",
        )
        tf.assert_equal(
            end_y_shape,
            wavelength_shape,
            message=f"For un dense source, need "
            f"exactly as many end_y points as wavelengths.",
        )
        tf.assert_equal(
            start_ranks_shape,
            wavelength_shape,
            message=f"For un dense source, need "
            f"exactly as many start_rank points as wavelengths.",
        )
        tf.assert_equal(
            end_ranks_shape,
            wavelength_shape,
            message=f"For un dense source, need "
            f"exactly as many end_rank points as wavelengths.",
        )
        
        return start_x, start_y, end_x, end_y, wavelengths, start_ranks, end_ranks

    @property
    def start_points(self):
        return self._start_x, self._start_y

    @property
    def start_point_ranks(self):
        return self._start_ranks

    @property
    def end_points(self):
        return self._end_x, self._end_y

    @property
    def end_point_ranks(self):
        return self._end_ranks

    @property
    def start_point_distribution(self):
        return self._start_point_distribution

    @property
    def end_point_distribution(self):
        return self._end_point_distribution
