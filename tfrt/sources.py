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

import tensorflow as tf
import numpy as np

PI = math.pi

# ====================================================================================

COUNTER = itertools.count(0)


class AngularDistributionBase(ABC):
    """
    Abstract implementation of an angular distribution, that defines the bare
    minimum interface that all angular distributions should implement.
    
    Public attributes
    -----------------
    angles : tf.float64 tensor of shape (none,)
        The angles held by the distribution.
    ranks : tf.float64 tensor of shape (none,) or None
        A number that that describes the where each angle is relative to the whole
        distribution, and can be useful for defining the target destination for each
        ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the namespace under which all ops will be placed
    
    Public members
    --------------
    build()
        Called to actually build the ops that generate this distribution.  Should
        only be called once, usually by the source that consumes this distribution.  
        Won't throw an error if it is called more than once, but it will add 
        unnecessary ops to the TF graph and may orphan ops built by previous calls to 
        build.
    needs_build()
        Returns true if the distribution needs to be built.

    """

    def __init__(self, min_angle, max_angle, sample_count, name=None):
        """
        Angular distribution constructors do not need to follow the pattern set by
        the constructor defined here, but most will, so this implementation is
        provided for convenience.  A constructor should not construct any tf ops, it
        should only store values for use by the class at a later time.  All angular
        distribution constructors should accept a name keyword argument, which is used
        to define a namespace in which all tf ops constructed by the class are placed.

        """
        self._min_angle = min_angle
        self._max_angle = max_angle
        self._sample_count = sample_count
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"

        self._angles = None
        self._ranks = None

    def _angle_limit_validation_ops(self, lower_limit, upper_limit):
        """
        Convenience for sub classes that require angle limits, to help with range
        checking.  The return is a list of assert ops which should be listed as a
        dependency for the ops in build; use:

        with tf.control_dependencies(self.angle_limit_validation_ops()):

        """
        lower_limit = tf.cast(lower_limit, tf.float64)
        upper_limit = tf.cast(upper_limit, tf.float64)

        self._max_angle = tf.cast(self._max_angle, tf.float64)
        self._max_angle = tf.ensure_shape(
            self._max_angle, (), name=f"{self._name}_max_angle_shape_check"
        )
        self._min_angle = tf.cast(self._min_angle, tf.float64)
        self._min_angle = tf.ensure_shape(
            self._min_angle, (), name=f"{self._name}_min_angle_shape_check"
        )
        self._sample_count = tf.cast(self._sample_count, tf.int64)
        self._sample_count = tf.ensure_shape(
            self._sample_count, (), name=f"{self._name}_sample_count_shape_check"
        )
        return [
            tf.assert_greater_equal(
                self._max_angle,
                self._min_angle,
                message=f"{self._name}: max_angle must be >= min_angle.",
            ),
            tf.assert_greater_equal(
                self._min_angle,
                lower_limit,
                message=f"{self._name}: min_angle must be >= {lower_limit}.",
            ),
            tf.assert_less_equal(
                self._max_angle,
                upper_limit,
                message=f"{self._name}: max_angle must be <= {upper_limit}.",
            ),
            tf.assert_positive(
                self._sample_count, message=f"{self._name}: sample_count must be > 0."
            ),
        ]

    def _build_ranks(self):
        """
        Convenience function for generating the angle ranks in most cases.  Must be
        called inside build, after self._angles is built, and requires that min/max
        angles are set.  If this pattern does not work, or if you generate ranks
        in the process of building the distribution, do not use it

        """
        self._ranks = self._angles / tf.cast(
            tf.reduce_max(tf.abs(tf.stack([self._min_angle, self._max_angle]))),
            tf.float64,
        )

    @abstractmethod
    def build(self):
        """
        All subclasses must override this method.  This method will use the values
        stored inside the class in the constructor to build the TF ops that
        generate the distribution.  If you are going to use the angle limit
        validation, the ops in here should list those as dependencies.

        This method is separated from the constructor for cleanliness of the tf graph:
        It will allow the ops generated by this class to be easily colocated with the
        other ops used to build a source.

        The build function should overwrite self._angles which represents the angular
        distribution built by the class.

        The build function may also overwrite self._ranks, if you want the subclass
        to use angle ranks.  Otherwise this may be skipped.

        The ops in this function should be build within a namescope unique to the
        instance.  Use:

        with tf.name_scope(self._name) as scope:

        """
        raise NotImplementedError

    @property
    def needs_build(self):
        return self._angles is None

    @property
    def angles(self):
        """
        If overwritten, this should not add any ops to the tf graph.
        Should be a 1-D tensor.

        """
        if self.needs_build:
            raise RuntimeError(
                f"{self._name}: attempted to access angles before calling build."
            )
        return self._angles

    @property
    def ranks(self):
        """
        If overwritten, this should not add any ops to the tf graph.
        Should be a 1-D tensor of the same length as angles, but may also be None.

        """
        return self._ranks

    @property
    def name(self):
        return self._name


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
    angles : tensor-like of floats with shape ~(None,), parametric
    ranks : tensor-like of floats with shape ~(None,), parametric, optional
    name : string, fixed, optional
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public interface
    -----------------
    See AngularDistributionBase for an explanation of the attributes and methods  
    shared by all angular distributions.

    """

    def __init__(self, angles, ranks=None, name=None):
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self._angles = angles
        self._ranks = ranks

    def build(self):
        with tf.name_scope(self._name) as scope:
            self._angles = tf.cast(self._angles, tf.float64)
            self._angles = tf.ensure_shape(
                self._angles, (None,), name=f"{self._name}_angle_shape_check"
            )
            if self._ranks is not None:
                self._ranks = tf.cast(self._ranks, tf.float64)
                self._ranks = tf.ensure_shape(
                    self._ranks, (None,), name=f"{self._name}_rank_shape_check"
                )


class StaticUniformAngularDistribution(AngularDistributionBase):
    """
    A set of angles that are uniformally distributed between two extrema.
    
    For this distribution, rank will be normalized so that the most extreme angle
    generated by the distribution (farthest from the center of the distribution)
    will have |rank| == 1.
    
    Parameters
    ----------
    min_angle : scalar float tensor-like, parametric
        The minimum angle to include in the distribution.
    max_angle : scalar float tensor-like, parametric
        The maximum angle to include in the distribution.
    sample_count : scalar int tensor-like, parametric
        The number of angles to return in the distribution.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public interface
    -----------------
    See AngularDistributionBase for an explanation of the attributes and methods  
    shared by all angular distributions.
        
    """

    def __init__(self, min_angle, max_angle, sample_count, name=None):
        super().__init__(min_angle, max_angle, sample_count, name=name)

    def build(self):
        with tf.name_scope(self._name) as scope:
            with tf.control_dependencies(self._angle_limit_validation_ops(-PI, PI)):
                self._angles = tf.linspace(
                    self._min_angle, self._max_angle, self._sample_count
                )
            self._angles = tf.cast(self._angles, tf.float64)
            self._build_ranks()


class RandomUniformAngularDistribution(AngularDistributionBase):
    """
    A set of angles that are randomly uniformally sampled between two extrema.
    
    For this distribution, rank will be normalized so that the most extreme angle
    generated by the distribution (farthest from the center of the distribution)
    will have |rank| == 1.
    
    Parameters
    ----------
    min_angle : scalar float tensor-like, parametric
        The minimum allowed angle that can be included in the distribution.
    max_angle : scalar float tensor-like, parametric
        The maximum allowed angle that can be included in the distribution.
    sample_count : scalar int tensor-like, parametric
        The number of angles to return in the distribution.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public interface
    -----------------
    See AngularDistributionBase for an explanation of the attributes and methods  
    shared by all angular distributions.
        
    """

    def __init__(self, min_angle, max_angle, sample_count, name=None):
        super().__init__(min_angle, max_angle, sample_count, name=name)

    def build(self):
        with tf.name_scope(self._name) as scope:
            with tf.control_dependencies(self._angle_limit_validation_ops(-PI, PI)):
                self._angles = tf.random_uniform(
                    (self._sample_count,),
                    minval=self._min_angle,
                    maxval=self._max_angle,
                    dtype=tf.float64,
                )
                self._build_ranks()


class StaticLambertianAngularDistribution(AngularDistributionBase):
    """
    A set of angles spanning two extrema whose distribution follows a Lambertian
    (cosine) distribution around zero.
    
    For this source, rank will be the sine of the angle, so that the maximum rank
    will have magnitude sin(max(abs(angle_limits))).  Rank will be distributed
    uniformally.
    
    Parameters
    ----------
    min_angle : scalar float tensor-like, parametric
        The minimum angle to include in the distribution.
    max_angle : scalar float tensor-like, parametric
        The maximum angle to include in the distribution.
    sample_count : scalar int tensor-like, parametric
        The number of angles to return in the distribution.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public interface
    -----------------
    See AngularDistributionBase for an explanation of the attributes and methods  
    shared by all angular distributions.
        
    """

    def __init__(self, min_angle, max_angle, sample_count, name=None):
        super().__init__(min_angle, max_angle, sample_count, name=name)

    def build(self):
        with tf.name_scope(self._name) as scope:
            with tf.control_dependencies(
                self._angle_limit_validation_ops(-PI / 2.0, PI / 2.0)
            ):
                lower_rank_cutoff = tf.sin(self._min_angle)
                upper_rank_cutoff = tf.sin(self._max_angle)
                self._ranks = tf.cast(
                    tf.linspace(
                        lower_rank_cutoff, upper_rank_cutoff, self._sample_count
                    ),
                    tf.float64,
                )
                self._angles = tf.asin(self._ranks)


class RandomLambertianAngularDistribution(AngularDistributionBase):
    """
    A set of angles randomly sampling from a Lambertian (cosine) distribution around 
    zero and between two extrema.
    
    For this source, rank will be the sine of the angle, so that the maximum rank
    will have magnitude sin(max(abs(angle_limits))).  Rank will be distributed
    uniformally.
    
    Parameters
    ----------
    min_angle : scalar float tensor-like, parametric
        The minimum allowed angle that can be included in the distribution.
    max_angle : scalar float tensor-like, parametric
        The maximum allowed angle that can be included in the distribution.
    sample_count : scalar int tensor-like, parametric
        The number of angles to return in the distribution.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public interface
    -----------------
    See AngularDistributionBase for an explanation of the attributes and methods  
    shared by all angular distributions.
        
    """

    def __init__(self, min_angle, max_angle, sample_count, name=None):
        super().__init__(min_angle, max_angle, sample_count, name=name)

    def build(self):
        with tf.name_scope(self._name) as scope:
            with tf.control_dependencies(
                self._angle_limit_validation_ops(-PI / 2.0, PI / 2.0)
            ):
                lower_rank_cutoff = tf.sin(self._min_angle)
                upper_rank_cutoff = tf.sin(self._max_angle)
                self._ranks = tf.random_uniform(
                    (self._sample_count,),
                    lower_rank_cutoff,
                    upper_rank_cutoff,
                    dtype=tf.float64,
                )
                self._angles = tf.asin(self._ranks)


# ====================================================================================


class BasePointDistributionBase(ABC):
    """
    Abstract implementation of a base point distribution, which defines the bare
    minimum interface that all base point distributions should implement.
    
    Public attributes
    -----------------
    base_points : 2-tuple of tf.float64 tensor of shape (none,)
        The x and y coordinates of the base points represented by the distribution
    ranks : tf.float64 tensor of shape (none,) or None
        A number that that describes where each base point is relative to the 
        whole distribution, and can be useful for defining the target destination for 
        each ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the namespace under which all ops will be placed
    
    Public members
    --------------
    build()
        Called to actually build the ops that generate this distribution.  Should
        only be called once, usually by the source that consumes this distribution.  
        Won't throw an error if it is called more than once, but it will add 
        unnecessary ops to the TF graph and may orphan ops built by previous calls to 
        build.
    needs_build()
        Returns true if the distribution needs to be built.

    """

    def __init__(self, name=None):
        """
        A constructor should not construct any tf ops, it should only store values
        for use by the class at a later time.  All base point distribution
        constructors should accept a name keyword argument, which is used
        to define a namespace in which all tf ops constructed by the class are placed.

        """
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self._base_points = None
        self._ranks = None

    @abstractmethod
    def build(self):
        """
        All subclasses must override this method.  This method will use the values
        stored inside the class in the constructor to build the TF ops that
        generate the distribution.  The ops in here should list any parameter
        validation ops as dependencies.

        This method is separated from the constructor for cleanliness of the tf graph:
        It will allow the ops generated by this class to be easily colocated with the
        other ops used to build a source.

        The build function should overwrite self._base_points which represents the
        set of base points built by the class.

        The build function may also overwrite self._ranks, if you want the subclass
        to use base point ranks.  Otherwise this may be skipped.

        The ops in this function should be build within a namescope unique to the
        instance.  Use:

        with tf.name_scope(self._name) as scope:

        """
        raise NotImplementedError

    def _validate_sample_count(self):
        """
        Convenience for checking that the sample count is valid:
        Cast to int64, check that it is scalar, and ensure it is positive.
        """
        self._sample_count = tf.cast(self._sample_count, tf.int64)
        self._sample_count = tf.ensure_shape(
            self._sample_count, (), name=f"{self._name}_sample_count_shape_check"
        )
        return [
            tf.assert_positive(
                self._sample_count, message=f"{self._name}: sample_count must be > 0."
            )
        ]

    @property
    def needs_build(self):
        return self._base_points is None

    @property
    def base_points(self):
        """
        If overwritten, this should not add any ops to the tf graph.
        Should be a tuple of 1-D tensors, (x_points, y_points).

        """
        if self.needs_build:
            raise RuntimeError(
                f"{self._name}: attempted to access "
                "base points before calling build."
            )
        return self._base_points

    @property
    def ranks(self):
        """
        If overwritten, this should not add any ops to the tf graph.
        Should be a 1-D tensor with the same length as the x and y points, but may
        also be None.

        """
        return self._ranks

    @property
    def name(self):
        return self._name


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
    x_points : tensor-like of floats with shape ~(None,), parametric
    y_points : tensor-like of floats with shape ~(None,), parametric
    ranks : tensor-like of floats with shape ~(None,), parametric, optional
    name : string, fixed, optional
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public interface
    -----------------
    See BasePointDistributionBase for an explanation of the attributes and methods  
    shared by all base point distributions.

    """

    def __init__(self, x_points, y_points, ranks=None, name=None):
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self._x_points = x_points
        self._y_points = y_points
        self._ranks = ranks

    def build(self):
        with tf.name_scope(self._name) as scope:
            self._x_points = tf.cast(self._x_points, tf.float64)
            self._x_points = tf.ensure_shape(
                self._x_points, (None,), name=f"{self._name}_x_points_shape_check"
            )
            self._y_points = tf.cast(self._y_points, tf.float64)
            self._y_points = tf.ensure_shape(
                self._y_points, (None,), name=f"{self._name}_y_points_shape_check"
            )
            self._base_points = (self._x_points, self._y_points)

            if self._ranks is not None:
                self._ranks = tf.cast(self._ranks, tf.float64)
                self._ranks = tf.ensure_shape(
                    self._ranks, (None,), name=f"{self._name}_ranks_shape_check"
                )


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
    value 0.0.  However, to ease use with sources, a single-use setter is included
    with the beam classes.  The intent is to allow the user to pass a
    BeamPointDistribution to a source constructor without needing to define the
    central_angle; instead the central_angle of the BeamPointDistribution will be
    set by the source.  This setter must be called before the distribution's build
    method is called, or else an exception will be raised.

    The rank generated by this distribution will have its zero at the relative origin
    of the beam, and will have magnitude 1 for the point(s) farthest from the origin.
    
    Public attributes
    -----------------
    base_points : 2-tuple of tf.float64 tensor of shape (none,)
        The x and y coordinates of the base points represented by the distribution
    ranks : tf.float64 tensor of shape (none,) or None
        A number that that describes where each base point is relative to the 
        whole distribution, and can be useful for defining the target destination for 
        each ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the namespace under which all ops will be placed
    central_angle : scalar tf.float64 tensor
    
    Public members
    --------------
    build()
        Called to actually build the ops that generate this distribution.  Should
        only be called once, usually by the source that consumes this distribution.  
        Won't throw an error if it is called more than once, but it will add 
        unnecessary ops to the TF graph and may orphan ops built by previous calls to 
        build.
    needs_build()
        Returns true if the distribution needs to be built.
    """

    def __init__(
        self, beam_start, beam_end, sample_count, name=None, central_angle=0.0
    ):
        super().__init__(name=name)
        self._beam_start = beam_start
        self._beam_end = beam_end
        self._sample_count = sample_count
        self._central_angle = central_angle

    @property
    def central_angle(self):
        return self._central_angle

    @central_angle.setter
    def central_angle(self, val):
        if self._base_points is None:
            self._central_angle = val
        else:
            raise RuntimeError(
                f"{self._name}: Too late to set central_angle!  "
                "Build has already been called."
            )

    def _parametrize_beam(self):
        """
        Utility to be called inside build that interprets beam_start, beam_end, and
        central_angle.  Calculates the endpoints of the beam and the limits on the
        beam rank.  Beam rank can then be interpreted as a parameter that generates
        beam points on a line from the origin to endpoint.

        """
        self._beam_start = tf.cast(self._beam_start, tf.float64)
        self._beam_start = tf.ensure_shape(
            self._beam_start, (), name=f"{self._name}_beam_start_shape_check"
        )
        self._beam_end = tf.cast(self._beam_end, tf.float64)
        self._beam_end = tf.ensure_shape(
            self._beam_end, (), name=f"{self._name}_beam_end_shape_check"
        )
        self._central_angle = tf.cast(self._central_angle, tf.float64)
        self._central_angle = tf.ensure_shape(
            self._central_angle, (), name=f"{self._name}_central_angle_shape_check"
        )

        validate_endpoints = tf.assert_less_equal(
            self._beam_start,
            self._beam_end,
            message=f"{self._name}: beam_start must be <= beam_end.",
        )

        with tf.control_dependencies([validate_endpoints]):
            rank_scale = tf.reduce_max(
                tf.abs(tf.stack([self._beam_start, self._beam_end]))
            )
            self._max_rank = self._beam_start / rank_scale
            self._min_rank = self._beam_end / rank_scale

            self._endpoint_x = self._beam_start * tf.cos(self._central_angle + PI / 2.0)
            self._endpoint_y = self._beam_start * tf.sin(self._central_angle + PI / 2.0)

    def _build_points(self):
        """
        Utility that takes the rank and the endponts and constructs the actual
        base points.
        """
        self._base_points = (
            self._endpoint_x * self._ranks / self._max_rank,
            self._endpoint_y * self._ranks / self._max_rank,
        )


class StaticUniformBeam(BeamPointBase):
    """
    A set of base points uniformally spread across the width of a beam.
        
    Parameters
    ----------
    beam_start : scalar float tensor-like, parametric
        The width of the lower half of the beam, relative to its center.
    beam_start : scalar float tensor-like, parametric
        The width of the upper half of the beam, relative to its center.  Must be >=
        beam_start.
    sample_count : scalar int tensor-like, parametric
        The number of angles to return in the distribution.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
    central_angle : scalar float tensor-like, optional, parametric
        The angle where the beam points, the points will be perpendicular to this 
        angle.  Though you may set it here for most use cases don't; it may be
        overridden by the source that consumes this distribution.
        
    Public interface
    -----------------
    See BeamPointBase for an explanation of the attributes and methods  
    shared by all beam distributions.
        
    """

    def build(self):
        with tf.name_scope(self._name) as scope:
            self._parametrize_beam()
            with tf.control_dependencies(self._validate_sample_count()):
                self._ranks = tf.linspace(
                    self._min_rank, self._max_rank, self._sample_count
                )
            self._build_points()


class RandomUniformBeam(BeamPointBase):
    """
    A set of base points uniformally randomly sampled across the width of a beam.
        
    Parameters
    ----------
    beam_start : scalar float tensor-like, parametric
        The width of the lower half of the beam, relative to its center.
    beam_start : scalar float tensor-like, parametric
        The width of the upper half of the beam, relative to its center.  Must be >=
        beam_start.
    sample_count : scalar int tensor-like, parametric
        The number of angles to return in the distribution.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
    central_angle : scalar float tensor-like, optional, parametric
        The angle where the beam points, the points will be perpendicular to this 
        angle.  Though you may set it here for most use cases don't; it may be
        overridden by the source that consumes this distribution.
        
    Public interface
    -----------------
    See BeamPointBase for an explanation of the attributes and methods  
    shared by all beam distributions.
        
    """

    def build(self):
        with tf.name_scope(self._name) as scope:
            self._parametrize_beam()
            with tf.control_dependencies(self._validate_sample_count()):
                self._ranks = tf.random_uniform(
                    tf.reshape(self._sample_count, (1,)),
                    self._min_rank,
                    self._max_rank,
                    dtype=tf.float64,
                )
            self._build_points()


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
    
    Public attributes
    -----------------
    base_points : 2-tuple of tf.float64 tensor of shape (none,)
        The x and y coordinates of the base points represented by the distribution
    ranks : tf.float64 tensor of shape (none,) or None
        A number that that describes where each base point is relative to the 
        whole distribution, and can be useful for defining the target destination for 
        each ray.  Will be none if ranks were not defined for this distribution.
    name : string
        The name of the namespace under which all ops will be placed
    
    Public members
    --------------
    build()
        Called to actually build the ops that generate this distribution.  Should
        only be called once, usually by the source that consumes this distribution.  
        Won't throw an error if it is called more than once, but it will add 
        unnecessary ops to the TF graph and may orphan ops built by previous calls to 
        build.
    needs_build()
        Returns true if the distribution needs to be built.

    """

    def __init__(self, start_point, end_point, sample_count, name=None):
        super().__init__(name=name)
        self._start_point = start_point
        self._end_point = end_point
        self._sample_count = sample_count

    @abstractmethod
    def _build_ranks(self):
        """
        This abstract method will generate the point ranks, according to some
        distribution.  It will be called inside build, and must set self._ranks

        """
        raise NotImplementedError

    def build(self):
        """
        This method may be overridden, though I think this implementation will cover
        most use cases for this class.

        """
        with tf.name_scope(self._name) as scope:
            self._start_point = tf.cast(self._start_point, tf.float64)
            self._start_point = tf.ensure_shape(
                self._start_point, (2,), name=f"{self._name}_start_point_shape_check"
            )
            self._end_point = tf.cast(self._end_point, tf.float64)
            self._end_point = tf.ensure_shape(
                self._end_point, (2,), name=f"{self._name}_end_point_shape_check"
            )

            start_x = self._start_point[0]
            start_y = self._start_point[1]
            end_x = self._end_point[0]
            end_y = self._end_point[1]

            with tf.control_dependencies(self._validate_sample_count()):
                self._build_ranks()
            self._base_points = (
                start_x + self._ranks * (end_x - start_x),
                start_y + self._ranks * (end_y - start_y),
            )


class StaticUniformAperaturePoints(AperaturePointBase):
    """
    A set of base points uniformally spaced between two end points.
        
    Parameters
    ----------
    start_point : float tensor-like of shape (2,), parametric
        The start point of the distribution, in absolute coordinates.
    end_point : float tensor-like of shape (2,), parametric
        The end point of the distribution, in absolute coordinates.
    sample_count : scalar int tensor-like, parametric
        The number of angles to return in the distribution.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public interface
    -----------------
    See AperaturePointBase for an explanation of the attributes and methods  
    shared by all aperature distributions.
        
    """

    def _build_ranks(self):
        self._ranks = tf.cast(tf.linspace(0.0, 1.0, self._sample_count), tf.float64)


class RandomUniformAperaturePoints(AperaturePointBase):
    """
    A set of base points uniformally randomly sampled between two end points.
        
    Parameters
    ----------
    start_point : float tensor-like of shape (2,), parametric
        The start point of the distribution, in absolute coordinates.
    end_point : float tensor-like of shape (2,), parametric
        The end point of the distribution, in absolute coordinates.
    sample_count : scalar int tensor-like, parametric
        The number of angles to return in the distribution.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
        
    Public interface
    -----------------
    See AperaturePointBase for an explanation of the attributes and methods  
    shared by all aperature distributions.
        
    """

    def _build_ranks(self):
        self._ranks = tf.random_uniform(
            tf.reshape(self._sample_count, (1,)), 0.0, 1.0, dtype=tf.float64
        )


# ====================================================================================


class SourceBase(ABC):
    """
    Abstract implementation of a source, which defines the bare minimum interface
    that all sources should implement.

    The standard way of using a source will be to create an instances of the needed
    parameter distributions before calling the source constructor, and letting the
    source call the build method on its parameters, because this will colocate the
    parameter ops inside the namespace defined by the source.  But if you do not want
    this behavior, or if you want to re-use some of the parameter distributions, you
    may build the parameters before feeding them to the source.  Sources will check
    whether their parameters have already been built, and will build them if not.

    Sources should build themselves (add ops to the graph) in their constructor.

    Sources should expose their distributions as read-only properties.  They should
    also expose the expanded (possibly densified) values of each of their parameters
    as read-only properties.  This is necessary because, if you plan to use one of the
    distribution ranks to specify target locations for an optimizer, you need to
    use the possibly densified version of the ranks.  So you should use the source's
    version of the ranks, not the distribution's version.

    """

    def __init__(self, name=None, dense=True):
        """
        If a source is dense, that means that rays will be generated with every
        combination of the values of each of the parameters.  This is convenient for
        static sources (sources that use static distributions) since it generates
        rays that span the entire range of specified inputs.  It is not recommended
        that sources that use random distributions choose to be dense, simply because
        they will be less random.  But the source will work either way.

        """
        self._name = name or f"{self.__class__.__name__}-{next(COUNTER)}"
        self._rays = None
        self._dense = dense

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
    def is_dense(self):
        return self._dense

    @property
    def rays(self):
        """If overridden, should not add any ops to the graph."""
        return self._rays

    @property
    def name(self):
        return self._name


class PointSource(SourceBase):
    """
    Rays eminating from a single point.
    
    Parameters
    ----------
    center : float tensor-like of shape (2,), parametric
        The point from which all rays eminate.
    central_angle : scalar float tensor-like, parametric
        The angle of the center of the angular distribution.  The angles in the
        angular distribution will be relative to this value.
    angular_distribution
        The angles used for the rays.  The angular distribution is interpreted as 
        angles relative to central_angle, but they can be interpreted as absolute 
        angles by setting central_angle to zero.
    wavelengths : 1-D float tensor-like, parametric
        Which wavelengths of light to use for the rays.  Units are nm.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
    dense : bool, fixed, optional
        If true, will take every combination of angle in the angular distribution and
        wavelength in wavelengths.  If false, will match the angles and wavelengths 
        1:1, which requires that angular_distribution and wavelengths have the same
        number of elements.
    start_on_center : bool, fixed, optional
        If true, rays will be oriented such that their start point is center and the
        rays propigate outward, like a typical point source.  If false, the rays are
        all flipped a half-turn, so that their endpoint lies on center to create a 
        converging ray set.  Useful for generating a source with a known angular size
        that illuminates only a single point on the optic at a time, but can be 
        scanned across the surface by making center a variable.
    ray_length : scalar float tensor-like, parametric, optional
        The length given to all the rays generated by this source.  Ray length doesn't
        really matter to the ray tracer, since rays are interpreted as semi-infinite,
        so this setting is mostly for the purposes of display.
        
    Public attributes
    -----------------
    rays : tf.float64 tensor of shape (None, 5)
        The rays representing this source, the first dimension indexes each ray, and 
        the second is (x_start, y_start, x_end, y_end, wavelength).
    name : string
        The name given to this source
    is_dense : bool
        Whether the source was built as dense or not
    angles : 1-D tf.float64 tensor
        The absolute angle of each ray.  Will be expanded if the source is dense.  
        Will differ from the angles in the distribution if central_angle is not zero.
    angle_ranks : 1-D float64 tensor
        The angular rank of each ray.  Will be expanded if the source is dense.  
    wavelengths : 1-D float64 tensor
        The wavelength of each ray.  Will be expanded if the source is dense.
    angular_distribution
        A handle to the angular distribution used to create this source.  It won't be
        expanded if the source is dense.

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
    ):
        super().__init__(name, dense)
        with tf.name_scope(self._name) as scope:
            self._center = tf.cast(center, tf.float64)
            self._center = tf.ensure_shape(
                self._center, (2,), name=f"{self._name}_center_shape_check"
            )
            self._central_angle = tf.cast(central_angle, tf.float64)
            self._central_angle = tf.ensure_shape(
                self._central_angle, (), name=f"{self._name}_central_angle_shape_check"
            )
            self._wavelengths = tf.cast(wavelengths, tf.float64)
            self._wavelengths = tf.ensure_shape(
                self._wavelengths, (None,), name=f"{self._name}_wavelengths_shape_check"
            )

            self._angular_distribution = angular_distribution
            if self._angular_distribution.needs_build:
                self._angular_distribution.build()

            self._angles = tf.cast(self._angular_distribution.angles, tf.float64)
            self._angles = tf.ensure_shape(
                self._angles, (None,), name=f"{self._name}_angles_shape_check"
            )
            if self._angular_distribution.ranks is not None:
                self._angle_ranks = tf.cast(
                    self._angular_distribution.ranks, tf.float64
                )
                self._angle_ranks = tf.ensure_shape(
                    self._angle_ranks,
                    (None,),
                    name=f"{self._name}_angle_ranks_shape_check",
                )

            if self._dense:
                self._make_dense()
            else:
                self._make_undense()

            ray_count = tf.shape(self._angles)
            self._angles = self._angles + self._central_angle
            start_x = tf.broadcast_to(self._center[0], ray_count)
            start_y = tf.broadcast_to(self._center[1], ray_count)
            end_x = start_x + ray_length * tf.cos(self._angles)
            end_y = start_y + ray_length * tf.sin(self._angles)

            if start_on_center:
                self._rays = tf.stack(
                    [start_x, start_y, end_x, end_y, self._wavelengths], axis=1
                )
            else:
                self._rays = tf.stack(
                    [end_x, end_y, start_x, start_y, self._wavelengths], axis=1
                )

    def _make_dense(self):
        angles, wavelengths = tf.meshgrid(self._angles, self._wavelengths)
        if self._angle_ranks is not None:
            angle_ranks, _ = tf.meshgrid(self._angle_ranks, self._wavelengths)

            self._angle_ranks = tf.reshape(angle_ranks, (-1,))
        self._angles = tf.reshape(angles, (-1,))
        self._wavelengths = tf.reshape(wavelengths, (-1,))

    def _make_undense(self):
        angle_shape = tf.shape(self._angles)
        wavelength_shape = tf.shape(self._wavelengths)
        validation_ops = [
            tf.assert_equal(
                angle_shape,
                wavelength_shape,
                message=f"{self.__class__.__name__}: For un dense source, need "
                f"exactly as many angles as wavelengths.",
            )
        ]
        with tf.control_dependencies(validation_ops):
            self._angles = tf.identity(self._angles)

    @property
    def angles(self):
        return self._angles

    @property
    def angle_ranks(self):
        return self._angle_ranks

    @property
    def wavelengths(self):
        return self._wavelengths

    @property
    def angular_distribution(self):
        return self._angular_distribution


class AngularSource(SourceBase):
    """
    Rays eminating from multiple points.
    
    This source can be used to generate a beam if given a beam point distribution, or 
    it can act like light filling an entrance aperture if given an aperature point 
    distribution.  The light may have an angular spread, to represent a source of
    nonzero size.
    
    The source will attempt to feed central_angle to the base_point_distribution, in
    case it is a beam-type distribution, but if the distribution does not accept a
    central_angle, it will do nothing.  However, setting the central_angle requires
    that the base_point_distribution not already be built.  It won't fail in this
    case, but it will do nothing silently.  If you need to feed an already built
    base_point_distribution that requires a central_angle, you will need to manually
    feed that central_angle to the base_point_distribution's constructor.
    
    Parameters
    ----------
    center : float tensor-like of shape (2,), parametric
        The rays will originate from the sum of the base points and center.  So if a 
        beam base point distribution is fed (whose points are interpreted as relative)
        then center will set the center of the beam, as expected.  But an aperature
        base point distribution is interpreted as a set of absolute points, so in this
        case you should set center to [0, 0].  If you use a non-zero center with an
        aperature base point distribution (or any other base point distribution that
        uses absolute points) then the value of center will have the effect of adding
        an offset to the base points.
    central_angle : scalar float tensor-like, parametric
        The angle of the center of the angular distribution.  The angles in the
        angular distribution will be relative to this value.
    angular_distribution
        The angles used for the rays.  The angular distribution is interpreted as 
        angles relative to central_angle, but they can be interpreted as absolute 
        angles by setting central_angle to zero.
    base_point_distribution
        Points that will be used for one of the endpoints of the rays.  Values are 
        interpreted as relative to center.
    wavelengths : 1-D float tensor-like, parametric
        Which wavelengths of light to use for the rays.  Units are nm.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
    dense : bool, fixed, optional
        If true, will take every combination of angle in the angular distribution, 
        base point in the base point distribution, and wavelength in wavelengths.  If 
        false, will match one of each 1:1:1, which requires that all three have the
        same number of elements.
    start_on_center : bool, fixed, optional
        If true, rays will be oriented such that their start points lie on the base 
        ponits and the rays propigate outward.  If false, the rays are all flipped a 
        half-turn, so that their endpoint lies on center to create a converging ray 
        set.  In my opinion, using false gives the better display result when using 
        an aperature source.  Also useful for generating a source with a known 
        angular size that illuminates only a small portion on the optic at a time, 
        but can be scanned across the surface by making center a variable.
    ray_length : scalar float tensor-like, parametric, optional
        The length given to all the rays generated by this source.  Ray length doesn't
        really matter to the ray tracer, since rays are interpreted as semi-infinite,
        so this setting is mostly for the purposes of display.
        
    Public attributes
    -----------------
    rays : tf.float64 tensor of shape (None, 5)
        The rays representing this source, the first dimension indexes each ray, and 
        the second is (x_start, y_start, x_end, y_end, wavelength).
    name : string
        The name given to this source
    is_dense : bool
        Whether the source was built as dense or not
    angles : 1-D tf.float64 tensor
        The absolute angle of each ray.  Will be expanded if the source is dense.  
        Will differ from the angles in the distribution if central_angle is not zero.
    angle_ranks : 1-D float64 tensor
        The angular rank of each ray.  Will be expanded if the source is dense.  
    base_points : 2-tuple of 1-D float64 tensor
        The x and y coordinates of the base points.  Will be expanded if the source 
        is dense.  
    base_point_ranks : 1-D float64 tensor
        The base point rank of each ray.  Will be expanded if the source is dense.
    wavelengths : 1-D float64 tensor
        The wavelength of each ray.  Will be expanded if the source is dense.
    angular_distribution
        A handle to the angular distribution used to create this source.  It won't be
        expanded if the source is dense.
    base_point_distribution
        A handle to the base point distribution used to create this source.  It won't 
        be expanded if the source is dense.
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
    ):
        super().__init__(name, dense)
        with tf.name_scope(self._name) as scope:
            self._center = tf.cast(center, tf.float64)
            self._center = tf.ensure_shape(
                self._center, (2,), name=f"{self._name}_center_shape_check"
            )
            self._central_angle = tf.cast(central_angle, tf.float64)
            self._central_angle = tf.ensure_shape(
                self._central_angle, (), name=f"{self._name}_central_angle_shape_check"
            )
            self._wavelengths = tf.cast(wavelengths, tf.float64)
            self._wavelengths = tf.ensure_shape(
                self._wavelengths, (None,), name=f"{self._name}_wavelengths_shape_check"
            )

            self._angular_distribution = angular_distribution
            if self._angular_distribution.needs_build:
                self._angular_distribution.build()
            self._base_point_distribution = base_point_distribution
            try:
                self._base_point_distribution.central_angle = self._central_angle
            except (RuntimeError, AttributeError):
                # Two exceptions may be raised: RuntimeError, if the distribution has
                # already been built, and AttributeError, if the distribution doesn't
                # accept a central_angle.  In either case, do nothing.
                # Actually, printing a warning or something might be appropriate, but
                # I am not sure.
                pass
            if self._base_point_distribution.needs_build:
                self._base_point_distribution.build()

            self._angles = tf.cast(self._angular_distribution.angles, tf.float64)
            self._angles = tf.ensure_shape(
                self._angles, (None,), name=f"{self._name}_angles_shape_check"
            )
            if self._angular_distribution.ranks is not None:
                self._angle_ranks = tf.cast(
                    self._angular_distribution.ranks, tf.float64
                )
                self._angle_ranks = tf.ensure_shape(
                    self._angle_ranks,
                    (None,),
                    name=f"{self._name}_angle_ranks_shape_check",
                )
            self._base_points_x, self._base_points_y = (
                self.base_point_distribution.base_points
            )
            self._base_points_x = tf.cast(self._base_points_x, tf.float64)
            self._base_points_x = tf.ensure_shape(
                self._base_points_x,
                (None,),
                name=f"{self._name}_base_points_x_shape_check",
            )
            self._base_points_y = tf.cast(self._base_points_y, tf.float64)
            self._base_points_y = tf.ensure_shape(
                self._base_points_y,
                (None,),
                name=f"{self._name}_base_points_y_shape_check",
            )
            if self._base_point_distribution.ranks is not None:
                self._base_point_ranks = tf.cast(
                    self._base_point_distribution.ranks, tf.float64
                )
                self._base_point_ranks = tf.ensure_shape(
                    self._base_point_ranks,
                    (None,),
                    name=f"{self._name}_base_point_ranks_shape_check",
                )

            if self._dense:
                self._make_dense()
            else:
                self._make_undense()

            self._angles = self._angles + self._central_angle
            start_x = self._center[0] + self._base_points_x
            start_y = self._center[1] + self._base_points_y
            end_x = start_x + ray_length * tf.cos(self._angles)
            end_y = start_y + ray_length * tf.sin(self._angles)

            if start_on_base:
                self._rays = tf.stack(
                    [start_x, start_y, end_x, end_y, self._wavelengths], axis=1
                )
            else:
                self._rays = tf.stack(
                    [end_x, end_y, start_x, start_y, self._wavelengths], axis=1
                )

    def _make_dense(self):
        angles, wavelengths, base_x = tf.meshgrid(
            self._angles, self._wavelengths, self._base_points_x
        )
        _, _, base_y = tf.meshgrid(self._angles, self._wavelengths, self._base_points_y)
        if self._angle_ranks is not None:
            angle_ranks, _, _ = tf.meshgrid(
                self._angle_ranks, self._wavelengths, self._base_points_x
            )
            self._angle_ranks = tf.reshape(angle_ranks, (-1,))
        if self._base_point_ranks is not None:
            _, _, base_point_ranks = tf.meshgrid(
                self._angle_ranks, self._wavelengths, self._base_point_ranks
            )
            self._base_point_ranks = tf.reshape(base_point_ranks, (-1,))

        self._angles = tf.reshape(angles, (-1,))
        self._wavelengths = tf.reshape(wavelengths, (-1,))
        self._base_points_x = tf.reshape(base_x, (-1,))
        self._base_points_y = tf.reshape(base_y, (-1,))

    def _make_undense(self):
        angle_shape = tf.shape(self._angles)
        wavelength_shape = tf.shape(self._wavelengths)
        base_x_shape = tf.shape(self._base_points_x)
        base_y_shape = tf.shape(self._base_points_y)
        validation_ops = [
            tf.assert_equal(
                angle_shape,
                wavelength_shape,
                message=f"{self.__class__.__name__}: For un dense source, need "
                f"exactly as many angles as wavelengths.",
            ),
            tf.assert_equal(
                base_x_shape,
                wavelength_shape,
                message=f"{self.__class__.__name__}: For un dense source, need "
                f"exactly as many base points as wavelengths.",
            ),
            tf.assert_equal(
                base_y_shape,
                wavelength_shape,
                message=f"{self.__class__.__name__}: For un dense source, need "
                f"exactly as many base points as wavelengths.",
            ),
        ]
        with tf.control_dependencies(validation_ops):
            self._angles = tf.identity(self._angles)

    @property
    def angles(self):
        return self._angles

    @property
    def angle_ranks(self):
        return self._angle_ranks

    @property
    def wavelengths(self):
        return self._wavelengths

    @property
    def base_points(self):
        return self._base_points_x, self._base_points_y

    @property
    def base_point_ranks(self):
        return self._base_point_ranks

    @property
    def angular_distribution(self):
        return self._angular_distribution

    @property
    def base_point_distribution(self):
        return self._base_point_distribution


class AperatureSource(SourceBase):
    """
    A set of rays that span two sets of endpoints.
    
    This source does not use an angular distribution, and instead makes rays between
    two sets of points.  Useful if you know your input light is bounded by two
    apertures, and you don't want to calculate angles.

    This source will not attempt to feed central_angle to the base point distributions
    because there is no angle, and this source isn't expected to be used with
    beam distributions.  But if you feed your own central_angle to the distribution,
    you can still use a beam point distribution with this class.
    
    Parameters
    ----------
    start_point_distribution
        Points that will be used for the start point of the rays.
    end_point_distribution
        Points that will be used for the end point of the rays.
    wavelengths : 1-D float tensor-like, parametric
        Which wavelengths of light to use for the rays.  Units are nm.
    name : string, fixed, optional, fixed
        The name of the namespace under which the cast and ensure shape nodes will be 
        placed in the TF graph.
    dense : bool, fixed, optional
        If true, will take every combination of start and end points.  If 
        false, will match one of each 1:1, which requires that both have the
        same number of elements.
        
    Public attributes
    -----------------
    rays : tf.float64 tensor of shape (None, 5)
        The rays representing this source, the first dimension indexes each ray, and 
        the second is (x_start, y_start, x_end, y_end, wavelength).
    name : string
        The name given to this source
    is_dense : bool
        Whether the source was built as dense or not
    start_points : 2-tuple of 1-D float64 tensor
        The x and y coordinates of the start points.  Will be expanded if the source 
        is dense.  
    start_point_ranks : 1-D float64 tensor
        The start point rank of each ray.  Will be expanded if the source is dense.
    end_points : 2-tuple of 1-D float64 tensor
        The x and y coordinates of the end points.  Will be expanded if the source 
        is dense.  
    end_point_ranks : 1-D float64 tensor
        The end point rank of each ray.  Will be expanded if the source is dense.
    wavelengths : 1-D float64 tensor
        The wavelength of each ray.  Will be expanded if the source is dense.
    start_point_distribution
        A handle to the base point distribution of the start points.  It won't 
        be expanded if the source is dense.
    end_point_distribution
        A handle to the base point distribution of the end points.  It won't 
        be expanded if the source is dense.

    """

    def __init__(
        self,
        start_point_distribution,
        end_point_distribution,
        wavelengths,
        name=None,
        dense=True,
    ):
        super().__init__(name, dense)
        with tf.name_scope(self._name) as scope:
            self._wavelengths = tf.cast(wavelengths, tf.float64)
            self._wavelengths = tf.ensure_shape(
                self._wavelengths, (None,), name=f"{self._name}_wavelengths_shape_check"
            )
            self._start_point_distribution = start_point_distribution
            if self._start_point_distribution.needs_build:
                self._start_point_distribution.build()
            self._end_point_distribution = end_point_distribution
            if self._end_point_distribution.needs_build:
                self._end_point_distribution.build()

            self._start_x, self._start_y = self.start_point_distribution.base_points
            self._end_x, self._end_y = self.end_point_distribution.base_points
            self._start_ranks = self._start_point_distribution.ranks
            self._end_ranks = self._end_point_distribution.ranks

            self._start_x = tf.ensure_shape(
                self._start_x, (None,), name=f"{self._name}_start_x_shape_check"
            )
            self._start_y = tf.ensure_shape(
                self._start_y, (None,), name=f"{self._name}_start_y_shape_check"
            )
            self._end_x = tf.ensure_shape(
                self._end_x, (None,), name=f"{self._name}_end_x_shape_check"
            )
            self._end_y = tf.ensure_shape(
                self._end_y, (None,), name=f"{self._name}_end_y_shape_check"
            )
            if self._start_ranks is not None:
                self._start_ranks = tf.ensure_shape(
                    self._start_ranks,
                    (None,),
                    name=f"{self._name}_start_ranks_shape_check",
                )
            if self._end_ranks is not None:
                self._end_ranks = tf.ensure_shape(
                    self._end_ranks, (None,), name=f"{self._name}_end_ranks_shape_check"
                )

            if self._dense:
                self._make_dense()
            else:
                self._make_undense()

            self._rays = tf.stack(
                [
                    self._start_x,
                    self._start_y,
                    self._end_x,
                    self._end_y,
                    self._wavelengths,
                ],
                axis=1,
            )

    def _make_dense(self):
        start_x, end_x, wavelengths = tf.meshgrid(
            self._start_x, self._end_x, self._wavelengths
        )
        start_y, end_y, _ = tf.meshgrid(self._start_y, self._end_y, self._wavelengths)
        if self._start_ranks is not None:
            start_ranks, _, _ = tf.meshgrid(
                self._start_ranks, self._end_x, self._wavelengths
            )
            self._start_ranks = tf.reshape(start_ranks, (-1,))
        if self._end_ranks is not None:
            end_ranks, _, _ = tf.meshgrid(
                self._end_ranks, self._end_x, self._wavelengths
            )
            self._end_ranks = tf.reshape(start_ranks, (-1,))

        self._wavelengths = tf.reshape(wavelengths, (-1,))
        self._start_x = tf.reshape(start_x, (-1,))
        self._start_y = tf.reshape(start_y, (-1,))
        self._end_x = tf.reshape(end_x, (-1,))
        self._end_y = tf.reshape(end_y, (-1,))

    def _make_undense(self):
        wavelength_shape = tf.shape(self._wavelengths)
        start_x_shape = tf.shape(self._start_x)
        start_y_shape = tf.shape(self._start_y)
        end_x_shape = tf.shape(self._end_x)
        end_y_shape = tf.shape(self._end_y)
        validation_ops = [
            tf.assert_equal(
                start_x_shape,
                wavelength_shape,
                message=f"{self.__class__.__name__}: For un dense source, need "
                f"exactly as many start_x points as wavelengths.",
            ),
            tf.assert_equal(
                start_y_shape,
                wavelength_shape,
                message=f"{self.__class__.__name__}: For un dense source, need "
                f"exactly as many start_y points as wavelengths.",
            ),
            tf.assert_equal(
                end_x_shape,
                wavelength_shape,
                message=f"{self.__class__.__name__}: For un dense source, need "
                f"exactly as many end_x points as wavelengths.",
            ),
            tf.assert_equal(
                end_y_shape,
                wavelength_shape,
                message=f"{self.__class__.__name__}: For un dense source, need "
                f"exactly as many end_y points as wavelengths.",
            ),
        ]
        with tf.control_dependencies(validation_ops):
            self._wavelengths = tf.identity(self._wavelengths)

    @property
    def wavelengths(self):
        return self._wavelengths

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
