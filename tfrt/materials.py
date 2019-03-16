"""
Defines various materials for use with TFRT.

The functions defined in this module are material definitions for use with TFRT.  
Collect the materials you need into a list, and pass it into the materials parameter
of tfrt.rayTrace().  The list should contain callables, so do not call the functions
first.

A TFRT material is simply a callable that accepts a tensor of any shape that encodes
wavelength in nanometers, and returns a tensor of the same shape whose values are the
refractive index of the material at that wavelength.

The data used to construct these material functions was obtained from
https://refractive.info.  Please note that refractive.info uses um by default in its
refractive index definitions, but TFRT uses nm to store wavelength, so you may need
to convert units.

"""

import tensorflow as tf

# ------------------------------------------------------------------------------------

def build_constant_material(n):
    """
    Call to build a material with constant refractive index.
    
    This is a convenience function to help build a material that has a constant value
    for refractive index.  Unlike the other materials, you should call this function,
    with the constant value you want the refractive index to take, and this function
    will generate the callable that should be fed to the materials list fed to the
    ray tracer.
    
    Parameters
    ----------
    n : float
        The constant to use for refractive index for this material
        
    Returns
    -------
    The callable material definition that should be fed to the materials list of the 
    ray tracer.
    
    """
    return lambda x: n * tf.ones_like(x)

# ------------------------------------------------------------------------------------

def acrylic(x):
    return tf.sqrt(
        2.1778
        + 6.1209e-9 * x ** 2
        - 1.5004e-15 * x ** 4
        + 2.3678e4 * x ** -2
        - 4.2137e9 * x ** -4
        + 7.3417e14 * x ** -6
        - 4.5042e19 * x ** -8
    )
    
def crown_glass(x):
    return tf.sqrt(
        1
        + 1.1273555e0 * x ** 2 / (x ** 2 - 7.20341707e3)
        + 1.24412303e-1 * x ** 2 / (x ** 2 - 2.69835916e4)
        + 8.27100531e-1 * x ** 2 / (x ** 2 - 1.00384588e8)
    )
    
def flint_glass(x):
    return tf.sqrt(
        1
        + 1.34533359e0 * x ** 2 / (x ** 2 - 9.97743871e3)
        + 2.09073176e-1 * x ** 2 / (x ** 2 - 4.70450767e4)
        + 9.37357162e-1 * x ** 2 / (x ** 2 - 1.11886764e8)
    )
    
def fused_silica(x):
    return tf.sqrt(
        1
        + 6.961663e-1 * x ** 2 / (x ** 2 - 4.679148e3)
        + 4.079426e-1 * x ** 2 / (x ** 2 - 1.3512063e4)
        + 8.974794e-1 * x ** 2 / (x ** 2 - 9.7934002538e7)
    )
    
def polycarbonate(x):
    return tf.sqrt(1 + 1.4182e0 * x ** 2 / (x ** 2 - 2.1304e4))
    
def reflective(x):
    return tf.zeros_like(x)
    
def soda_lime(x):
    return 1.5130e0 - 3.169e-9 * x ** 2 + 3.962e3 * x ** -2
    
def vacuum(x):
    return tf.zeros_like(x)
