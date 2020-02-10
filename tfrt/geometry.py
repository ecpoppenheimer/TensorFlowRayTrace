"""
Various geometric routines used by the ray tracer, but general enough that they may be 
used by a user as well.

"""
from math import pi as PI

import tensorflow as tf

# =====================================================================================

"""@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.float64),
        
    ]
)"""
def line_intersect(x1s, y1s, x1e, y1e, x2s, y2s, x2e, y2e, epsilion):
    """
    Given a set of N lines and a second set of M lines, computes all NxM intersections
    between lines in one set with those in the other.
      
    This function is safe when given input lines that are parallel, in which case
    there is no intersection.  In this case, the corresponding value in each
    of the output tensors will be garbage, but the corresponding element in the 
    valid intersections return (second returned value) will be false, indicating 
    that this intersection was invalid and its values should be ignored.
    
    Please note that if given N and M inputs, the outputs will have shape (N, M).
        
    Parameters
    ----------
    first eight : 1-D tf.float64 tensor-like
        The endpoints of the first and second set of lines.
    epsilion : scalar tf.float64 tensor-like
        A very small value used to avoid divide by zero, which can happen if the
        lines are parallel.  If you are getting incorrect results because you have 
        lines that are extremely close but not actually parallel, you may need to
        reduce this value from its recommended of 1e-10, but if you make this too small
        it may incorrectly identify parallel lines as actually intersecting, or
        will throw a nan into your tensors, which will kill tf.gradients.  As of
        05/19 I haven't tested how small you can go with this parameter, so lowering
        it a lot may be perfectly fine.
        
    Returns
    -------
    Intersection coordinates:
        A tuple of two float tensors of shape [M, N] holding the x and y
        values of the intersections.
    Valid intersections:
        A bool tensor of shape [M, N] which is true wherever the intersection is valid.
        Intersections are valid when the lines are not parallel.
    U:
        A float tensor of shape [M, N] holding the parameter along each of the first
        lines where the intersection occurs.
    V:
        A float tensor of shape [M, N] holding the parameter along each of the second
        lines where the intersection occurs.
        
    """

    x1s, x2s = tf.meshgrid(x1s, x2s)
    y1s, y2s = tf.meshgrid(y1s, y2s)
    x1e, x2e = tf.meshgrid(x1e, x2e)
    y1e, y2e = tf.meshgrid(y1e, y2e)

    return raw_line_intersect(
        x1s, y1s, x1e, y1e, x2s, y2s, x2e, y2e, epsilion
    )

# -------------------------------------------------------------------------------------

"""@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.float64),
        
    ]
)"""
def raw_line_intersect(x1s, y1s, x1e, y1e, x2s, y2s, x2e, y2e, epsilion):
    """
    Low-level core math used to find intersections between two sets of lines.
    
    This function does no error/shape checking on its inputs, and so its inputs can
    be any shape, as long as they are all co-broadcastable.  This function treats its
    inputs as infinite lines, and is safe with tf.gradients in the case of parallel 
    lines in the input sets.
    
    Parameters
    ----------
    first eight : tf.float64 tensor-like
        The endpoints of the first and second set of lines.
    epsilion : scalar tf.float64 tensor-like
        A very small value used to avoid divide by zero, which can happen if the
        lines are parallel.  If you are getting incorrect results because you have 
        lines that are extremely close but not actually parallel, you may need to
        reduce this value from its recommended of 1e-10, but if you make this too small
        it may incorrectly identify parallel lines as actually intersecting, or
        will throw a nan into your tensors, which will kill tf.gradients.  As of
        05/19 I haven't tested how small you can go with this parameter, so lowering
        it a lot may be perfectly fine.
        
    Returns
    -------
    X:
        float tensor holding the x coordinates of the intersections.
    Y:
        float tensor holding the y coordinates of the intersections.
    Valid intersections:
        A bool tensor which is true wherever the intersection is valid.  
        Intersections are valid when the lines are not parallel.
    U:
        A float tensor holding the parameter along each of the first lines where the 
        intersection occurs.
    V:
        A float tensor holding the parameter along each of the second lines where the 
        intersection occurs.
    """

    x1 = x1e - x1s
    y1 = y1e - y1s
    x2 = x2e - x2s
    y2 = y2e - y2s
    denominator = x1 * y2 - y1 * x2

    # The following three lines make this code safe for use with tf.gradients
    # because we absolutely have to avoid nan.
    valid_intersection = tf.greater_equal(tf.abs(denominator), epsilion)
    safe_value = tf.ones_like(denominator)
    safe_denominator = tf.where(valid_intersection, denominator, safe_value)
    safe_denominator = 1.0 / safe_denominator

    u = tf.where(
        valid_intersection,
        (x2 * (y1s - y2s) - y2 * (x1s - x2s)) * safe_denominator,
        safe_value,
    )
    v = tf.where(
        valid_intersection,
        (y1 * (x2s - x1s) - x1 * (y2s - y1s)) * safe_denominator,
        safe_value,
    )
    x = x1s + u * x1
    y = y1s + u * y1

    # for testing purposes, compute the intersection with the second line's
    # parameter.
    # x = x2s + v * x2
    # y = y2s + v * y2

    return x, y, valid_intersection, u, v
    
# ====================================================================================

"""@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.float64)
    ]
)"""    
def line_triangle_intersect(
    rx1, ry1, rz1, rx2, ry2, rz2, xp, yp, zp, x1, y1, z1, x2, y2, z2, epsilion
):
    """
    Given a set of N lines and a second set of M triangles, computes all NxM intersections.
      
    This function is safe when given input lines that are parallel to the triangle, in 
    which case there is no intersection.  In this case, the corresponding value in each
    of the output tensors will be garbage, but the corresponding element in the 
    valid intersections return (second returned value) will be false, indicating 
    that this intersection was invalid and its values should be ignored.
    
    This function is NOT safe when given malformed triangles, whose edges p1 and 
    p2 are parallel.
    
    Please note that if given N and M inputs, the outputs will have shape (N, M).
        
    Parameters
    ----------
    first 15 : 1-D tf.float64 tensor-like
        The endpoints of the first set of lines, as well as the vertices of the triangle
    epsilion : scalar tf.float64 tensor-like
        A very small value used to avoid divide by zero, which can happen if the
        lines are parallel.  If you are getting incorrect results because you have 
        lines that are extremely close but not actually parallel, you may need to
        reduce this value from its recommended of 1e-10, but if you make this too small
        it may incorrectly identify parallel lines as actually intersecting, or
        will throw a nan into your tensors, which will kill tf.gradients.  As of
        05/19 I haven't tested how small you can go with this parameter, so lowering
        it a lot may be perfectly fine.
        
    Returns
    -------
    x, y, z:
        float tensors of shape [M, N] holding the coordinates of the intersections.
    Valid:
        A bool tensor of shape [M, N] which is true wherever the intersection is valid.
        Intersections are valid when the lines are not parallel.
    ray_u:
        A float tensor of shape [M, N] holding the parameter along each of the first
        lines where the intersection occurs.
    trig_u, trig_v:
        A float tensor of shape [M, N] holding the parameter along two of the sides of the
        triangle where the intersection occurs
        
    """

    rx1_m, xp_m = tf.meshgrid(rx1, xp)
    ry1_m, yp_m = tf.meshgrid(ry1, yp)
    rz1_m, zp_m = tf.meshgrid(rz1, zp)
    rx2_m, x1_m = tf.meshgrid(rx2, x1)
    ry2_m, y1_m = tf.meshgrid(ry2, y1)
    rz2_m, z1_m = tf.meshgrid(rz2, z1)
    _, x2_m = tf.meshgrid(rx2, x2)
    _, y2_m = tf.meshgrid(ry2, y2)
    _, z2_m = tf.meshgrid(rz2, z2)
    
    return raw_line_triangle_intersect(
        rx1_m, ry1_m, rz1_m, rx2_m, ry2_m, rz2_m, xp_m, yp_m, zp_m, x1_m, y1_m, z1_m,
        x2_m, y2_m, z2_m, epsilion
    )
    
# ------------------------------------------------------------------------------------

"""@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.float64)
    ]
)"""    
def raw_line_triangle_intersect(
    rx1, ry1, rz1, rx2, ry2, rz2, xp, yp, zp, x1, y1, z1, x2, y2, z2, epsilion
):
    """
    Low-level core math used to find intersections between lines and triangles.
    
    Like line_triangle_intersect, except its input can be any shape.  Used internally by
    line_triangle_intersect, which simply meshgrids its input and feeds them to this
    function.
    """
    # I computed the algbra on Wolfram Alpha, sorry for the variable names
    a = rx1-rx2
    b = x1-xp
    c = x2-xp
    d = ry1-ry2
    f = y1-yp
    g = y2-yp
    h = rz1-rz2
    k = z1-zp
    l = z2-zp
    
    q = rx1-xp
    r = ry1-yp
    s = rz1-zp
    
    denominator = a*g*k+b*d*l+c*f*h-a*f*l-b*g*h-c*d*k
    ray_u_numerator = b*l*r+c*f*s+g*k*q-b*g*s-c*k*r-f*l*q
    trig_u_numerator = a*g*s+c*h*r+d*l*q-a*l*r-c*d*s-g*h*q
    trig_v_numerator = a*k*r+b*d*s+f*h*q-a*f*s-b*h*r-d*k*q
    
    # safe division
    valid = tf.greater_equal(tf.abs(denominator), epsilion)
    safe_value = tf.ones_like(rx1)
    safe_denominator = tf.where(valid, denominator, safe_value)
    ray_u = ray_u_numerator / safe_denominator
    trig_u = trig_u_numerator / safe_denominator
    trig_v = trig_v_numerator / safe_denominator
    
    # use ray_u and the ray to compute the intersection point
    # these use - instead of + because the definitions for a,d,h are reversed, because
    # they swapped sides of the equation.
    x = rx1 - ray_u * a
    y = ry1 - ray_u * d
    z = rz1 - ray_u * h
    
    return x, y, z, valid, ray_u, trig_u, trig_v


# =====================================================================================

"""@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(None,), dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.float64),
        
    ]
)"""
def line_circle_intersect(xs, ys, xe, ye, xc, yc, r, epsilion):
    """
    Given a set of N lines and a second set of M circles, computes all 2*NxM
    intersections.
    
    Lines are specified by their endpoints, xstart, ystart, xend, yend].
    Circles must are specified by xcenter, ycenter, radius.
    
    Though the lines are formatted as line segments, this function will treat its 
    input as inifinte lines.  Circles of radius zero will cause failure.
      
    A line may intersect a circle zero, one, or two times.  This function can handle
    all cases.  It will return two intersections for every line, labeled plus and
    minus based on which branch of the quadratic was taken to calculate the solution.
    The accompanying valid_plus/valid_minus bool tensors report which of the returned
    solutions actually correspond to valid intersections.  In the case where a line
    intersects a circle less than twice, the elements in the output tensors 
    corresponding to that intersection will be filled with garbage values that should
    not be used.
    
    Please note that if given N lines and M circles, the expected 
    shape of the output is [M, N].
        
    Parameters
    ----------
    xs, ys, xe, ye : tf.float64 1-D tensor-like
        The coordinates of the endpoints of the lines
    xc, yc, r : tf.float64 1-D tensor-like
        The coordinates of the center of the circles, and the radius
    epsilion : tf.float64 scalar tensor-like
        A very small value used to avoid divide by zero.  Recommended = 1e-10
        
    Returns
    -------
    A 2-tuple whose first element holds the intersection information for the
    intersection that resulted from the plus branch of the quadratic, and whose second
    element holds the intersection information for the intersection that resulted
    from the minus branch of the quadratic.  Each element of the tuple is a dict
    that contains the following keys:
    
    x:
        A tf.float64 tensor of shape [M, N] that holds the x coordinate of the 
        intersection.
    y:
        A tf.float64 tensor of shape [M, N] that holds the y coordinate of the 
        intersection.
    valid:
        A bool tensor of shape [M, N] which is true wherever the intersection is
        valid.
    u:
        A float tensor of shape [M, N] holding the parameter along the line where the 
        intersection occurs.
    v:
        A float tensor of shape [M, N] holding the angle of the circle where the
        intersection occurs.  Angles will be in the range [-PI, PI], with zero 
        pointing palong the x axis.
        
    """

    xs, xc = tf.meshgrid(xs, xc)
    ys, yc = tf.meshgrid(ys, yc)
    xe, _ = tf.meshgrid(xe, r)
    ye, r = tf.meshgrid(ye, r)

    return raw_line_circle_intersect(xs, ys, xe, ye, xc, yc, r, epsilion)


# -------------------------------------------------------------------------------------

"""@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.float64),
        
    ]
)"""
def raw_line_circle_intersect(xs, ys, xe, ye, xc, yc, r, epsilion):
    """
    Low-level core math used to find intersections between lines and circles.
    
    This function does no error/shape checking on its inputs, and so its inputs can
    be any shape, as long as they are all co-broadcastable.  This function treats its
    inputs as infinite lines, and is safe with tf.gradients given all possible special
    cases.  Outputs will be the same shape as the inputs
    
    Parameters
    ----------
    xs, ys, xe, ye : tensor-like
        The coordinates of the endpoints of the lines
    xc, yc, r : tensor-like
        The coordinates of the center of the circles, and the radius
    epsilion : tensor-like
        A very small value used to avoid divide by zero.  Recommended = 1e-10
        
    Returns
    -------
    A 2-tuple whose first element holds the intersection information for the
    intersection that resulted from the plus branch of the quadratic, and whose second
    element holds the intersection information for the intersection that resulted
    from the minus branch of the quadratic.  Each element of the tuple is a dict
    that contains the following keys:
    
    x:
        A tensor that holds the x coordinate of the 
        intersection.
    y:
        A tensor that holds the y coordinate of the 
        intersection.
    valid:
        A bool tensor which is true wherever the intersection is valid.
    u:
        A tensor holding the parameter along the line where the 
        intersection occurs.
    v:
        A tensor holding the angle of the circle where the
        intersection occurs.  Angles will be in the range [-PI, PI], with zero 
        pointing palong the x axis.
        
    """

    # coordinate adjustment
    with tf.name_scope("1/r"):
        inverse_r = 1.0 / r
    with tf.name_scope("xr"):
        xr = (xs - xc) * inverse_r
    with tf.name_scope("yr"):
        yr = (ys - yc) * inverse_r
    with tf.name_scope("xd"):
        xd = (xe - xs) * inverse_r
    with tf.name_scope("yd"):
        yd = (ye - ys) * inverse_r

    # quadratic equation parts
    with tf.name_scope("a") as scope:
        a = xd * xd + yd * yd
    with tf.name_scope("b") as scope:
        b = 2.0 * xr * xd + 2.0 * yr * yd
    with tf.name_scope("c") as scope:
        c = xr * xr + yr * yr - 1.0
    with tf.name_scope("rad") as scope:
        rad = b * b - 4.0 * a * c

    # rad ~= 0 special case. => line is tangent to circle.
    # testing the intersection algorithm with tangent lines is inconsistent,
    # because while rad is always very small, it is sometimes negative (due to
    # rounding error).  So I will check if it is less than epsilion, and if so
    # set it to precisely zero, so that we can correctly detect the point of
    # tangency, and so that both solutions returned will be identical.
    rad = tf.where(
        tf.less(tf.abs(rad), epsilion, name="special_case_tangent"),
        tf.zeros_like(rad),
        rad,
    )

    # rad < 0 special case. => line does not intersect circle.
    safe_value = tf.ones_like(a, name="safe_value")
    radLess = tf.less(rad, 0, name="special_case_no_intersection")
    uminus_valid = uplus_valid = tf.logical_not(radLess)
    safe_rad = tf.where(radLess, safe_value, rad, name="safe_rad")
    safe_rad = tf.sqrt(safe_rad)
    with tf.name_scope("uminus") as scope:
        uminus = tf.where(radLess, safe_value, (-b - safe_rad))
    with tf.name_scope("uplus") as scope:
        uplus = tf.where(radLess, safe_value, (-b + safe_rad))

    # a = 0 special case. => line start point == line end point.
    azero = tf.less(tf.abs(a), epsilion, name="special_case_azero")
    safe_denominator = tf.where(azero, safe_value, 2 * a, "safe_denominator")
    safe_denominator = 1.0 / safe_denominator
    uminus_valid = tf.logical_and(uminus_valid, tf.logical_not(azero))
    uminus = tf.where(azero, safe_value, uminus * safe_denominator)
    uplus_valid = tf.logical_and(uplus_valid, tf.logical_not(azero))
    uplus = tf.where(azero, safe_value, uplus * safe_denominator)

    # build the remaining output values from the calculated parameter
    with tf.name_scope("xminus") as scope:
        xminus = xs + (xe - xs) * uminus
    with tf.name_scope("xplus") as scope:
        xplus = xs + (xe - xs) * uplus
    with tf.name_scope("yminus") as scope:
        yminus = ys + (ye - ys) * uminus
    with tf.name_scope("yplus") as scope:
        yplus = ys + (ye - ys) * uplus
    with tf.name_scope("vminus") as scope:
        vminus = tf.atan2(yminus - yc, xminus - xc)
    with tf.name_scope("vplus") as scope:
        vplus = tf.atan2(yplus - yc, xplus - xc)

    return (
        {
            "x": xplus,
            "y": yplus,
            "valid": uplus_valid,
            "u": uplus,
            "v": vplus
        },
        {
            "x": xminus,
            "y": yminus,
            "valid": uminus_valid,
            "u": uminus,
            "v": vminus,
        },
    )


# ====================================================================================

"""@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.float64),
        
    ]
)"""
def snells_law_2D(x_start, y_start, x_end, y_end, norm, n_in, n_out, new_ray_length):
    """
    Performs an optical ray reaction (Snell's law) in 2D.
    
    This function assumes that ray intersection and projection has already been 
    performed.  It matches the data in each of its parameters 1-to-1.
    
    n_in and n_out are the refractive indices of this optical reaction.  They should
    be floats, and have already been evaluated as a function of wavelength.  If n_in
    is zero, this function will compute a reflective reaction.  Otherwise it will
    compute a refractive reaction.  n_out should not be zero.
    
    This function will generate a new set of rays that are the results from the 
    optical reaction.  The output rays will have exactly the same shape as the input rays.
    
    Parameters
    ----------
    x_start, y_start, x_end, y_end : float tensor-like
        The endpoints of the rays to react
    norm : float tensor-like
        The angle of the surface norm, measured in absolute coordinates, as opposed to
        measured relative to the ray, as the angle of incidence.
    n_in : float tensor-like
        The refractive index of the material opposite to the norm.
    n_out : float tensor-like
        The refractive index of the material in which the norm is embedded.
    new_ray_length : float scalar
        The length of new rays generated by this function.
        
    Returns
    -------
    x_start, y_start, x_end, y_end : float tensor
        Same shape as the inputs, the newly created rays.
    
    """

    norm = tf.math.mod(norm, 2 * PI)
    ray_angle = tf.atan2(y_start - y_end, x_start - x_end)
    ray_angle = tf.math.mod(ray_angle, 2 * PI)
    theta1 = norm - ray_angle
    # ensure we measure the difference the correct way
    theta1 = tf.where(theta1 > PI, theta1 - (2 * PI), theta1)
    theta1 = tf.where(theta1 < -PI, theta1 + (2 * PI), theta1)

    # choose between internal and external reactions
    internal_mask = tf.greater_equal(
        tf.abs(theta1), PI / 2
    )
    shape = tf.shape(theta1)

    one = tf.ones_like(theta1)
    zero = tf.zeros_like(theta1)

    n_in = tf.broadcast_to(n_in, shape)
    n_in_is_safe = tf.not_equal(n_in, 0.0)
    n_in_safe = tf.where(n_in_is_safe, n_in, one)

    n_out_is_safe = tf.not_equal(n_out, 0.0)
    n_out = tf.broadcast_to(n_out, shape)
    n_out_safe = tf.where(n_out_is_safe, n_out, one)

    n1 = tf.where(n_out_is_safe, n_in_safe / n_out_safe, zero)
    n2 = tf.where(n_in_is_safe, n_out_safe / n_in_safe, zero)
    n = tf.where(internal_mask, n1, n2, name="n")

    norm = tf.where(
        internal_mask,
        tf.broadcast_to(norm, shape),
        tf.broadcast_to(norm + PI, shape)
    )
    theta1 = tf.where(
        internal_mask, theta1 + PI, theta1
    )

    theta2 = n * tf.sin(theta1)
    new_angle = tf.where(
        tf.logical_and(
            tf.less_equal(tf.abs(theta2), 1.0), tf.not_equal(n, 0.0)
        ),
        norm - tf.asin(theta2),
        norm + theta1 + PI
    )

    x_start = x_end
    y_start = y_end
    x_end = x_start + new_ray_length * tf.cos(new_angle)
    y_end = y_start + new_ray_length * tf.sin(new_angle)
    
    return x_start, y_start, x_end, y_end
    
# -----------------------------------------------------------------------------------

"""@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=(None, 3), dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=(), dtype=tf.float64),
        
    ]
)"""
def snells_law_3D(
    x_start, y_start, z_start, x_end, y_end, z_end, norm, n_in, n_out, new_ray_length
):
    """
    Performs an optical ray reaction (Snell's law) in 3D.
    
    This function assumes that ray intersection and projection has already been 
    performed.  It matches the data in each of its parameters 1-to-1.
    
    n_in and n_out are the refractive indices of this optical reaction.  They should
    be floats, and have already been evaluated as a function of wavelength.  If n_in
    is zero, this function will compute a reflective reaction.  Otherwise it will
    compute a refractive reaction.  n_out should not be zero.
    
    This function will generate a new set of rays that are the results from the 
    optical reaction.  The output rays will have 
    exactly the same shape as the input rays.
    
    This algorithm is vector based, unlike everything else in the program.  It may well
    be faster than the 2D implementation.  The algorithm was taken from
    https://www.staff.science.uu.nl/~kreve101/asci/GAraytracer.pdf.
    
    Parameters
    ----------
    x_start, y_start, z_start, x_end, y_end, z_end: 1D float tensor-like
        The endpoints of the rays to react
    norm : float nx3 tensor-like
        The surface norm.  Note that this parameter is a full vector rather than a 
        coordinate like all the others, so it is 2D instead of 1D
    n_in : float tensor-like
        The refractive index of the material opposite to the norm.
    n_out : float tensor-like
        The refractive index of the material in which the norm is embedded.
    new_ray_length : float scalar
        The length of new rays generated by this function.
        
    Returns
    -------
    x_start, y_start, z_start, x_end, y_end, z_end : float tensor
        Same shape as the inputs, the newly created rays.
    
    """
    
    # a vector representing the ray direction.
    u = tf.stack([x_end-x_start, y_end-y_start, z_end-z_start], axis=1)
    u = tf.math.l2_normalize(u, axis=1)
    
    # need to normalize the norm (it isn't guarenteed to already be normed)
    n = tf.math.l2_normalize(norm, axis=1)
    nu = tf.reduce_sum(n*u, axis=1, keepdims=True)
    
    # process the index of refraction
    internal_mask = tf.greater(nu, 0)
    one = tf.ones_like(n_in)
    zero = tf.zeros_like(n_in)

    n_in_is_safe = tf.not_equal(n_in, 0.0)
    n_in_safe = tf.where(n_in_is_safe, n_in, one)
    n_out_is_safe = tf.not_equal(n_out, 0.0)
    n_out_safe = tf.where(n_out_is_safe, n_out, one)
    
    n1 = tf.reshape(tf.where(n_out_is_safe, n_in_safe / n_out_safe, zero), (-1, 1))
    n2 = tf.reshape(tf.where(n_in_is_safe, n_out_safe / n_in_safe, zero), (-1, 1))
    eta = tf.where(internal_mask, n1, n2)
    nu_eta = eta * nu
    
    # compute the refracted vector
    radicand = 1 - eta*eta + nu_eta*nu_eta
    do_tir = tf.less(radicand, 0)
    safe_radicand = tf.where(do_tir, tf.ones_like(radicand), radicand)
    refract = (tf.sign(nu) * tf.sqrt(safe_radicand) - nu_eta) * n + eta*u
    
    # compute the reflected vector
    reflect = -2 * nu * n + u
    
    # choose refraction or reflection
    reflective_surface = tf.reshape(tf.equal(n_in, 0), (-1, 1))
    do_reflect = tf.logical_or(do_tir, reflective_surface)
    new_vector = tf.where(do_reflect, reflect, refract)
    
    new_endpoints = tf.stack([x_end, y_end, z_end], axis=1) + new_ray_length * new_vector
    x, y, z = tf.unstack(new_endpoints, num=3, axis=1)
    return x_end, y_end, z_end, x, y, z
    

# ====================================================================================

"""@tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64)
        
    ]
)"""
def angle_in_interval(angle, start, end):
    """
    Checks whether an angle lies inside a closed angular interval.
    
    This function is only guarenteed to work when all of its inputs are in the range
    [-PI, PI].  This is the range produced by atan2.  But if you can't guarentee this range
    you can map them to this interval by doing something like
    angle -> tf.math.mod(angle + PI, 2*PI) - PI
    
    Parameters
    ----------
    angle : tf.float64 tensor-like
        The angle to check.
    start : tf.float64 tensor-like
        The start of the interval.  Must be broadcastable to the same shape as angle.
    end : tf.float64 tensor-like
        The end of the interval.  Must be broadcastable to the same shape as angle.
        
    Returns
    -------
    A bool tensor of the same shape as the inputs that is True wherever each angle 
    lies inside the interval defined by start and end.
    """
    
    reduced_angle = angle - start
    reduced_angle = tf.where(
        tf.less(reduced_angle, 0.0),
        reduced_angle + 2 * PI,
        reduced_angle
    )
    reduced_end = end - start
    reduced_end = tf.where(
        tf.less(reduced_end, 0.0),
        reduced_end + 2 * PI,
        reduced_end
    )
    return tf.less_equal(reduced_angle, reduced_end)
