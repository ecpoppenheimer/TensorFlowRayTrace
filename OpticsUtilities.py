"""
OpticsUtilities.py
Eric Poppenheimer
December 2018

This module implements some useful helper utilities that make working with TFRayTrace easier
"""

import tensorflow as tf
import math
import numpy as np

PI = tf.constant(math.pi, name="PI", dtype=tf.float64)

#==================================================================================================================

#Definitions of refractive index for different material, conveniently packaged. info from https://refractive.info
Materials = {
    
    "acrylic": lambda x: tf.sqrt(2.1778 + 6.1209e-3*x**2 - 1.5004e-3*x**4 + 2.3678e-2*x**-2
        - 4.2137e-3*x**-4 + 7.3417e-4*x**-6 - 4.5042e-5*x**-8),
    "polycarbonate": lambda x: tf.sqrt(1 + 1.4182*x**2/(x**2 - 0.021304)),
    "fused_silica": lambda x: tf.sqrt(1 + 0.6961663*x**2/(x**2 - 0.004679148) 
        + 0.4079426*x**2/(x**2 - 0.013512063) + 0.8974794*x**2/(x**2 - 97.934002538)),
    "soda_lime": lambda x: 1.5130 - 0.003169*x**2 + 0.003962*x**-2,
    "vacuum": lambda x: tf.ones_like(x),
    "reflective": lambda x: tf.zeros_like(x),
    "extreme": lambda x: 5.0*tf.ones_like(x),
    "flint": lambda x: tf.sqrt(1 + 1.34533359*x**2/(x**2-0.00997743871) + 0.209073176*x**2/(x**2-0.0470450767) 
        + 0.937357162*x**2/(x**2-111.886764)),
    "crown": lambda x: tf.sqrt(1 + 1.1273555*x**2/(x**2-0.00720341707) + 0.124412303*x**2/(x**2-0.0269835916) 
        + 0.827100531*x**2/(x**2-100.384588))
    } 
    
#==================================================================================================================    
 
""" With the way the mutable surface is currently parametrized, the gradient from a ray striking a surface will
    improve the orientation of that surface, but will make the adjacent surfaces worse, since if we tilt one of the
    line segments, the two adjacent segments will have to tilt the opposite way to keep the surface continuous.
    I am fixing this problem by multiplying the gradient by this triangle matrix, which has the effect of linking
    gradient contributions along a facet.  If a facet surface has S segments, the gradient to the nth segment will
    be added to all other segments m: n <= m <= S.  But gradient will never transfer between facets, they
    are all independant.
"""
def triangleMatrix(size, subsize=None):
    if subsize is None:
        subsize = size
    if size%subsize != 0:
        raise ValueError("triangleMatrix: size {} not divisible by subsize {}".format(size, subsize))
    return np.array([[1 if x<=y and x//subsize==y//subsize else 0 for x in range(size)] for y in range(size)],
        dtype=np.float64)
        
#-------------------------------------------------------------------------------------------------------------------

"""
    Build a parametrizable lens surface that can be fed into the ray tracer and optimized.  This is just a single
    surface, for a full lens, use this function twice.  Lenses are constrained to be along the optical axis y=0.  
    This function can be used to generate multiple lens surfaces in a single pass, if desired.  The shape of
    the first parameter, position, will be used to determine the number of lenses generated, and all other 
    parameters must be compatible to match with this shape.  All parameters passed to this function should be rank
    1 at most, except for maxRadius which should be a scalar.  
    
    position: A tensor or array.  The x coordinate where the surface intersects the x axis
    power: A tensor or array.  A parameter that controlls the curvature of the lens surface.  This parameter is the 
        horizontal distance between where the lens intersects the lines y=0 and y=aperatureRadius.  A positive 
        power pushes the x coordinate of where the lens crosses y=apperatureRadius in the opposite direction of the 
        norm.  So with a standard material configuration for the lens, positive power yields a converging lens, and 
        higher power yields more curvature, and thus a closer focal length.
    aperatureRadius: May be a tensor or array, but may also be a scalar, if the value should be the same for all
        surfaces.  The radius of the maximum extent of the lens
    facingLeft: A tensor or array.  A boolean that is true where the norm of this element should face left, and 
        false where it should face right.
    materialIn, materialOut: Tensors or arrays.  Indices that index materials inside and outside the lens surface 
        out of the materials tuple fed to the ray tracer.
    maxRadius: A scalar.  If power = 0, radius would be infinite, which will cause problems for things like finding 
        the angluar extent of the lens.  So its absolute value is capped at this value.  This can be a problem for 
        optimization, if the gradient learning step is too small to bridge the gap imposed by this constraint.
        
    Return: The first is the boundaryArc tensor that describes this lens system.  This can be plugged into the
        ray tracer directly
"""
def LensSurface(position, power, aperatureRadius, facingLeft, materialIn, materialOut, maxRadius=100.0):
    with tf.name_scope("buildLensSurfaces") as scope:
        # extend shape to be indexable, in the case that we were fed scalars and only seek to make a single surface
        shape = position.shape
        if len(shape) == 0:
            shape = [1]
        
        # extend aperatureRadius to be indexable
        aperatureRadius = tf.cast(aperatureRadius, tf.float64)
        if len(aperatureRadius.shape) == 0:
            aperatureRadius = tf.tile([aperatureRadius], [shape[0]])
            
        # cast everything, to be sure we are working with float64
        position = tf.cast(position, tf.float64)
        power = tf.cast(power, tf.float64)
        materialIn = tf.reshape(tf.cast(materialIn, tf.float64), shape)
        materialOut = tf.reshape(tf.cast(materialOut, tf.float64), shape)
        maxRadius = tf.cast(maxRadius, tf.float64)
        maxRadius = tf.tile([maxRadius], [shape[0]])
        
        # calculate the parameters of the arcs
        radius = (aperatureRadius**2+power**2)/(2*power)
        radius = tf.where(tf.equal(power, 0), maxRadius, radius)
        radius = tf.clip_by_value(radius, -maxRadius, maxRadius)
        
        angularExtent = tf.atan2(tf.sign(radius)*aperatureRadius, power-radius)

        xpos = tf.where(facingLeft, position+radius, position-radius)
        ypos = tf.zeros(shape, dtype=tf.float64)
        angleStart = tf.floormod(tf.where(facingLeft, angularExtent, angularExtent + PI), 2*PI)
        angleEnd = tf.floormod(tf.where(facingLeft, -angularExtent, -angularExtent + PI), 2*PI)
        #radius
        #materialIn
        #materialOut
        
        return tf.stack([xpos, ypos, angleStart, angleEnd, radius, materialIn, materialOut], axis=1)
        
#-------------------------------------------------------------------------------------------------------------------
        
"""
    Given three parameters like in the above function, this function returns the x coordinate of the end of the
    lenses generated by the above function.  This can be useful for building constraints between multiple lenses.
"""
def lensEndX(position, power, facingLeft):
    return tf.where(facingLeft, position+power, position-power)
    
#-------------------------------------------------------------------------------------------------------------------

"""
    Similar to LensSurface, but with a single extra parameter, spacing.  For a discussion of the other parameters, 
    see LensSurface.  There are two differences: 1) this function cannot make only a single lens surface, so all of
    the parameters need to be at least length 2.  2) Position is a scalar, and all other surfaces are relative to 
    this value. 
       
    spacing: A rank 1 tensor or list, which must have length one less than the length of the other parameters.  
        Spacing specifies the minimum distance between each optical element in sucession.
    mode: The meaning of position.  May be 'left', which indicates that the leftmost surface of the lens is at 
        x = position, or 'center', which indicates that position is the center of the lens
"""    
def LensWithThicknessConstraint(position, power, aperatureRadius, facingLeft, spacing, materialIn, materialOut, 
    maxRadius=1000.0, mode="left"):
    
    # validate the length of power
    position = tf.cast(position, tf.float64)
    if power.shape[0] < 2:
        raise ValueError("LensWithThicknessConstraint: length of power must be at least 2.")
    
    # validate the length of spacing
    spacing = tf.cast(spacing, tf.float64)
    if spacing.shape[0] != power.shape[0]-1:
        raise ValueError("LensWithThicknessConstraint: length of spacing must be one less than the other " 
            "parameters.")
            
    # extend aperatureRadius to be indexable
    aperatureRadius = tf.cast(aperatureRadius, tf.float64)
    if len(aperatureRadius.shape) == 0:
            aperatureRadius = tf.tile([aperatureRadius], [power.shape[0]])
    
    # make the first surface        
    surfaces = [LensSurface(
        position,
        power[0],
        aperatureRadius[0],
        facingLeft[0],
        materialIn[0],
        materialOut[0],
        maxRadius=maxRadius
        )]
    firstPosition = lastPosition = position
    
    # make the remaining surfaces relative to the first one
    for i in np.array(range(spacing.shape[0]))+1:
        thisPosition = lastPosition + spacing[i-1]
        lastEnd = lensEndX(lastPosition, power[i-1], facingLeft[i-1])
        thisEnd = lensEndX(thisPosition, power[i], facingLeft[i])
        thisPosition = tf.where(
            tf.less(thisEnd - lastEnd, spacing[i-1]),
            thisPosition + spacing[i-1] - (thisEnd - lastEnd),
            thisPosition)
        surfaces.append(LensSurface(
            thisPosition,
            power[i],
            aperatureRadius[i],
            facingLeft[i],
            materialIn[i],
            materialOut[i],
            maxRadius=maxRadius
            ))
        lastPosition = thisPosition
        
    # stitch the tensors together
    lens = tf.concat(surfaces, axis=0)
    if mode == "left":
        return lens
    elif mode == "center":
        # how much to adjust the xcenter of each lens
        shift = (lastPosition - firstPosition)/2.0 - position
    
        # construct an array to adjust the xcenter of each lens
        surfaceCount = lens.shape[0]
        adjustment = []
        for each in range(surfaceCount):
            adjustment.append([shift, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
        return lens - adjustment
            
    else:
        raise ValueError("LensWithThicknessConstraint: mode {} invalid.  Must be 'left' or 'center'.".format(mode))
        
#-----------------------------------------------------------------------------------------------------------------

def refractTo(theta1, n1, theta2, n2, permitTIR=True):
    # computes the norm of a boundary needed to refract light from theta1 to theta2.  This is a purely pythonic 
    # function; it does not with with tensors.
    
    PI = math.pi
    
    #omega is the amount of angular deflection needed
    omega = theta1-theta2
    # if omega is too large, we have measured the wrong way
    if omega > PI:
        omega -= 2*PI
    elif omega < -PI:
        omega += 2*PI
    
    if n1>n2:
        # this is an internal refraction
        nr = n2/n1
        criticalAngle = math.acos(nr)
        
        if abs(omega) >= 2*criticalAngle:
            # not possible
            return
        elif abs(omega) >= criticalAngle:
            # total internal reflection
            if permitTIR:
                if theta2-theta1 > 0:
                    return ((theta1 + theta2 - PI) / 2.0)%(2*PI)
                else:
                    return ((theta1 + theta2 + PI) / 2.0)%(2*PI)
            else:
                return
        else:
            return (theta1 + math.asin(nr*math.sin(omega)/math.sqrt(nr*nr-2*nr*math.cos(omega)+1)))%(2*PI)
    else:
        # this is an external refraction
        nr = n1/n2
        criticalAngle = math.acos(nr)
        
        if abs(omega) >= criticalAngle:
            # not possible
            return
        else:
            return (theta2 - PI - math.asin(nr*math.sin(omega)/math.sqrt(nr*nr-2*nr*math.cos(omega)+1)))%(2*PI)
            
#-----------------------------------------------------------------------------------------------------------------

def LambertianSource(center, normal, angleCount, wavelengths, length=1.0):
    # Generates a point source with a lambertian distribution.
    # center: A 2-tuple, the location all rays will originate from
    # normal: The direction where the distributions maximum will point
    # angleCount: The number of angles that rays will take in the distribution
    # wavelengths: Either a float or an iterable of floats, the wavelengths to use for the rays.  Each angle will have
    #   a ray of each wavelength.
    # return: A three-tuple.  
    #   First: An nx5 ndarray formatted as a TFRT rayset.  Will contain angleCount * len(wavelengths) rays.  
    #   Second: The rank (mapping from -1 to 1) of each ray based on where it appears in the distribution.
    #       Useful for mapping the input to the output distribution.
    #   Third: The angle at which each ray is cast.
    
    # Ensure wavelengths is iterable
    try:
        foo = wavelengths[0]
    except (TypeError):
        wavelengths = (wavelengths,)
    
    pointRank = np.linspace(-1.0, 1.0, angleCount)
    #theta = np.arcsin(np.sign(pointRank) * np.sqrt(np.abs(pointRank))) + normal
    #theta = np.arcsin(pointRank) + normal
    
    outputs = np.array([[
        center[0],
        center[1],
        center[0] + length*np.cos(np.arcsin(p) + normal),
        center[1] + length*np.sin(np.arcsin(p) + normal),
        w,
        p,
        np.arcsin(p) + normal
        ] for p in pointRank for w in wavelengths], dtype=np.float64)
        
    return (outputs[:,:5], outputs[:,5], outputs[:,6])
    
def RandomLambertianSource(center, normal, count, allowedWavelengths, sourceWidth=0.0, rayLength=1.0, cutoff=PI/2.0):
    # Generates random rays that follow a lambertian distribution.
    # center: A 2-tuple, the midpoint of the line from which all rays will originate.
    # normal: The direction where the distribution maximum will point.
    # count: The number of rays to return.
    # allowedWavelengths: Either a float or a list of floats, the valid wavelengths to use.  An element from this 
    #   list will be selected randomly for each ray.
    # sourceWidth: Rays can originate along a line of this length, which is centered on center and perpendicular to 
    #   normal.
    # rayLength: The generated rays will have this length.
    # cutoff: No rays that have an angle greater than this value relative to the norm will be returned.
    
    # Pick wavelengths for each ray
    try:
        wavelengthIndices = tf.random.uniform((count,), minval=0, maxval=len(allowedWavelengths), dtype=tf.int64)
        wavelengths = tf.gather(allowedWavelengths, wavelengthIndices)
    except (TypeError):
        wavelengths = tf.ones(count) * allowedWavelengths
    
    # Cast some of the parameters to float64
    wavelengths = tf.cast(wavelengths, tf.float64)    
    normal = tf.cast(normal, tf.float64)
    
    # Generate the starting points of each ray
    startParameter = tf.random_uniform((count,), minval=-1.0, maxval=1.0, dtype=tf.float64) 
    startX = center[0] + startParameter * sourceWidth/2.0 * tf.sin(normal)
    startY = center[1] - startParameter * sourceWidth/2.0 * tf.cos(normal)
    
    # Generate the angle of each ray
    cutoff = math.sin(cutoff)
    angle = tf.random.uniform((count,), minval=-cutoff, maxval=cutoff, dtype=tf.float64)
    angle = tf.asin(angle) + normal
    
    return tf.stack([
        startX,
        startY,
        startX + rayLength * tf.cos(angle),
        startY + rayLength * tf.sin(angle),
        wavelengths
        ], axis=1)
        
    

#-----------------------------------------------------------------------------------------------------------------
    
def segmentParameterSmoother(parameter, strength=1.0, filterSize=5, sigma=1.5):
    # Applies a gaussian blur to a set of parameters, useful for smoothing out rough segment lenses.
    #
    # parameter: the parameter to smooth.  Must be 1-D.
    # strength: a parameter between 0 and 1 that controls how strong the smoothing is.  1 -> parameter is updated to 
    #   take the value of the full blur, and 0 -> parameter is not updated
    # filterSize: The number size of the kernel, the number of elements to blur together with each pass.  Should be 
    #   odd.    
    # sigma: The standard deviation of the kernel
    #
    # return: An op that applies the blur operation to the parameter
    if filterSize%2 == 0:
        raise ValueError("segmentParameterSmoother: filter size {} must be odd.".format(filterSize))
    if len(parameter.shape) != 1:
        raise ValueError("segmentParameterSmoother: parameter rank must be 1.")
    
    def gauss(n,sigma):
        r = range(-int(n/2),int(n/2)+1)
        return [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]
    kernel = np.reshape(gauss(filterSize, sigma), (-1, 1, 1))
    
    # pad the parameter, to prevent pulling down at the edges
    padSize = (filterSize - 1)/2
    smoothedParameter = tf.pad(parameter, [[padSize, padSize]], mode="SYMMETRIC")
    
    smoothedParameter = tf.reshape(smoothedParameter, (1, -1, 1))
    smoothedParameter = tf.nn.conv1d(smoothedParameter, kernel, 1, "VALID")
    smoothedParameter = tf.reshape(smoothedParameter, (-1,))
    return tf.assign(parameter, smoothedParameter*strength + parameter*(1-strength))
      
#==================================================================================================================

