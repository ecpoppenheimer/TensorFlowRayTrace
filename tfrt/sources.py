"""
Classes to help make source light raysets that can be fed to the ray tracer.

This module defines a class for making static, determinisic light sources, and another
similar one that takes random samples each time its output ops are run.  Both classes 
generate TF tensors, so they can be fed directly into the ray tracer, but will
have to be evaluated with a session.run call before being passed to a 
drawing.RayDrawer.  All parameters should be pythonic, not tensors.  Both classes are 
considered read-only: once created the parameters that define the light sources 
cannot be changed.  If you need to change the parameters during run time, create a 
new instance.

Since the static source is constant, no special precautions need be taken when using 
its outputs.  If staying within the TF graph, no special precautions need be taken.
However, if you need to access values generated by the random source outside of the
TF graph, be aware that each session.run call made on a TF random tensor changes its
state, so if you extract two different parts of the random source on two separate 
session.run calls the results will not correlate; they will be built from two 
different random samples, so make sure you extract all parts you need from the source
inside the same session.run call.

"""
import math

from scipy.special import erfinv
import tensorflow as tf
import numpy as np

PI = math.pi

class StaticSource:
    """
    Generate a static ray set.
    
    This class generates a source that is non-random and does not change.  The rayset
    and several helper tensors that isolate useful properties of the rayset are 
    accessible as public attributes.  All attributes of this class are tf.constant.
    
    This source can be used to create a point source, if beam_radius = 0, or a beam 
    source otherwise.  Both the beam intensity profile and the angular spread can be
    sampled from a choice of distributions.  The source can contain one or more 
    wavelengths.  This source produces a dense sampling over its parameter space.  If
    you specify sampling B points across the beam width, A angles, and W different
    wavelengths, the rayset will contain A*B*W rays.
    
    Parameters
    ----------
    center : Tensor or tensor-like of shape exactly (2,), and dtype float
        The (x, y) coordinates of the center of this source.  May input a tensor-like
        value for this parameter (like a list).  May also be any tf op, including
        a tf.variable, which enables parametric ray sources that can be moved or
        even used as the dependant variable of an optimizer.
    wavelengths : 1-D iterable of floats
        The wavelength(s) to sample from.  Units must be nm.
    angular_samples : int
        The number of angles to include.
    angular_facing : Tensor or tensor-like of shape exactly (1,), and dtype float
        The angle at which the center of the angular distribution of the rayset 
        points.  Like center, may be any tf op.
    angular_lower_cutoff : float, optional
        The minimum (most clockwise) angle relative to angular_facing that will be
        included.
    angular_upper_cutoff : float, optional
        The maximum (most counterclockwise) angle relative to angular_facing that 
        will be included.
    angular_distribution : str, optional
        Defaults to 'uniform', where all sampled angles have the same distance
        between them, but also accepts 'lambertian', where the distance between angles
        scales like the cosine of the difference between the angle and the facing.
    beam_samples : int, optional
        Number of samples across the diameter of the beam.  Must be greater than 1 to 
        generate a beam.  If beam_upper == beam_lower and beam_samples != 1, will 
        generate multiple redundant rays at the position of beam_lower.
    beam_lower : float, optional
        If beam_upper == beam_lower, or beam_samples == 1, generate a point source.
        Otherwise, will generate rays that start at various points along a line which
        is perpendicular to angular_facing and whose midpoint is center.  beam_upper 
        and beam_lower specify the position along this line between the center and 
        the farthest ray origin, and the units are the same as the scale at which the
        rest of the geometry is built in.  Positive values are oriented 
        counterclockwise relative to facing.
    beam_upper : float, optional
        See beam_lower.  Should be >= beam_lower
    beam_distribution : str, optional
        Defaults to 'uniform', where the beam density is uniform, but also accepts
        'gaussian', so that distance between samples across the beam follows a
        gaussian distribution.
    name : str, optional
        The name to use to define a scope in which to place the ops generated by
        this class.  Defaults to 'ray_source'.
    ray_length : float, optional
        The length of the rays to generate.
    rays_forward : bool, optional
        If true, the rays generate forward (in the direction of the norm).  If false,
        the rays generate backwards (opposite to the direction of the norm).  This is 
        useful for making a converging ray source, which in turn can be used to send 
        rays to only a single point on the optic surface.
        
    Public Attributes
    -----------------
    rays : tf.Tensor, shape (n, 5)
        The ray set generated by this source.
    angles : tf.Tensor, shape (n,)
        The angles of each ray generated by this source.
    angle_ranks : tf.Tensor, shape(n,)
        A mapping that maps all ray angles generated by this source onto [-1, 1].  
        Useful for determing how to build the output error when using this rayset
        with an optimizer.  Tells you where on the object this ray originated from, 
        which can tell you where in the imaging plane it should end up.  If the
        distribution is even about the center, the rank vector will span [-1, 1].
        If this distribution is uneven, the rank will span a shortened subset, but
        the largest magnitude rank will still always be +/- 1.
    beam_ranks : tf.Tensor, shape(n, )
        Like angle_ranks except for position along the beam.
        
    """
    
    def __init__(
        self,
        center,
        wavelengths,
        angular_samples,
        angular_facing=0.0,
        angular_lower_cutoff=-PI,
        angular_upper_cutoff=PI,
        angular_distribution="uniform",
        beam_lower=0.0,
        beam_upper=0.0,
        beam_samples=1,
        beam_distribution="uniform",
        name="ray_source",
        ray_length=1.0,
        rays_forward=True
        # TODO
        # rays_forward not yet implemented
    ):
        with tf.name_scope(name) as scope:
            _validate_angle_cutoffs(
                angular_distribution,
                angular_lower_cutoff,
                angular_upper_cutoff
            )
            if angular_distribution == "uniform":
                samplable_angles = np.linspace(
                    angular_lower_cutoff,
                    angular_upper_cutoff,
                    angular_samples
                )
            elif angular_distribution == "lambertian":
                samplable_angles = np.linspace(
                    np.sin(angular_lower_cutoff),
                    np.sin(angular_upper_cutoff),
                    angular_samples
                )
                samplable_angles = np.arcsin(samplable_angles)
            # should have already thrown an exception in _validate_angle_cutoffs if
            # neither of these conditions are met
            
            center_x = center[0]
            center_y = center[1]
            parametrized_beam = _parametrize_beam(
                center_x,
                center_y,
                angular_facing,
                beam_upper,
                beam_lower
            )
            
            if beam_distribution == "uniform":
                samplable_beam_ranks = np.linspace(
                    parametrized_beam["p_lower"],
                    parametrized_beam["p_upper"],
                    beam_samples
                )
            elif beam_distribution == "gaussian":
                # Need two extra samples here to capture the +/1 inf.  Will peel
                # the extras off after passing into the CDF
                samplable_beam_ranks = np.linspace(
                    parametrized_beam["p_lower"],
                    parametrized_beam["p_upper"],
                    beam_samples + 2
                )
                samplable_beam_ranks = erfinv(samplable_beam_ranks) / math.sqrt(2)
                samplable_beam_ranks = samplable_beam_ranks[1:-1]
                # TODO
                # Testing confirms that this produces a beam that is slightly too
                # large.  With 50 samples, the beam is 3% larger than desired.
                # I am uncomfortable enough with the math to be unsure what to do
                # about this.
            else:
                raise ValueError(
                    f"Invalid choice {beam_distribution} for beam distribution."
                    "Must be 'uniform' or 'gaussian'."
                )
            
            WAVELENGTH = 0
            ANGLE = 1
            BEAM_RANK = 2
            full_sample_set = np.array([
                [w, a, b] 
                for w in wavelengths 
                for a in samplable_angles
                for b in samplable_beam_ranks
            ])
            
            angles = full_sample_set[:,ANGLE] + angular_facing
            
            x_start = center_x + full_sample_set[:,BEAM_RANK] * (
                parametrized_beam["x_upper"] - center_x)
            y_start = center_y + full_sample_set[:,BEAM_RANK] * (
                parametrized_beam["y_upper"] - center_y)
            x_end = x_start + ray_length * tf.cos(angles)
            y_end = y_start + ray_length * tf.sin(angles)
            rays = tf.stack(
                [x_start, y_start, x_end, y_end, full_sample_set[:,WAVELENGTH]],
                axis=1
            )
            
            # Build the private versions of the public attributes
            self._rays = tf.cast(rays, dtype=tf.float64)
            self._angles = tf.cast(angles, dtype=tf.float64)
            self._angle_ranks = self._angles / tf.reduce_max(
                tf.abs(self._angles)
            )
            self._beam_ranks = tf.constant(full_sample_set[:,BEAM_RANK], 
                dtype=tf.float64
            )
            
    @property
    def rays(self):
        """ The actual ray set generated by this source. """
        return self._rays
        
    @property
    def angles(self):
        """ The angle of each ray in the source, in order. """
        return self._angles
        
    @property
    def angle_ranks(self):
        """
        Normalized copy of angles, telling relative location in the angular 
        distribution each ray originates from.  Values are always in [-1, 1].
        
        """
        return self._rays
        
    @property
    def beam_ranks(self):
        """
        Tells relative position along the source each ray originates from.  Values 
        are always in [-1, 1].
        
        """
        return self._rays

# ------------------------------------------------------------------------------------

def _validate_angle_cutoffs(
    angular_distribution,
    angular_lower_cutoff,
    angular_upper_cutoff
):
    if angular_lower_cutoff > angular_upper_cutoff:
        raise ValueError("Invalid source angular cutoff.  Must have lower < upper.")
        
    if angular_distribution == "uniform":
        if angular_upper_cutoff > PI or angular_lower_cutoff < -PI:
            raise ValueError(
                "Invalid source angular cutoff.  Uniform distribution limits must "
                "be within [-PI, PI]"
        )
    elif angular_distribution == "lambertian":
        if angular_upper_cutoff > PI/2.0 or angular_lower_cutoff < -PI/2.0:
            raise ValueError(
                "Invalid source angular cutoff.  Uniform distribution limits must "
                "be within [-PI/2, PI/2]"
        )
    else:
        raise ValueError(
            f"Invalid choice {angular_distribution} for angular distribution."
            "Must be 'uniform' or 'lambertian'."
        )
            
def _parametrize_beam(center_x, center_y, angular_facing, beam_upper, beam_lower):
    try:
        scale_factor = max(abs(beam_upper), abs(beam_lower))
        return {
            "p_lower": beam_lower/scale_factor,
            "p_upper": beam_upper/scale_factor,
            "x_upper": center_x + scale_factor * tf.cos(angular_facing + PI/2.0),
            "y_upper": center_y + scale_factor * tf.sin(angular_facing + PI/2.0)
        }
    except(ZeroDivisionError):
        return {
            "p_lower": 0.0,
            "p_upper": 0.0,
            "x_upper": center_x,
            "y_upper": center_y
        }
            
            



