"""
Various geometric routines used by the ray tracer, but general enough that they may be 
used by a user as well.

"""
from math import pi as PI

import tensorflow as tf

# =====================================================================================

def line_intersect(first_lines, second_lines, name=None, epsilion=1e-10):
    """
    Given a set of N lines and a second set of M lines, computes all NxM intersections
    between lines in one set with those in the other.
    
    First and second lines are two sets of lines.  Both must be rank 2 tensors, whose 
    first dimenson indexes each line in the set, and whose second dimension must have 
    size 4 and represents [xstart, ystart, xend, yend], the x and y coordinates of two 
    points that define the line.  Though the lines are formatted as line segments,
    this function will treat its input as inifinte lines.
      
    This function is safe when given input lines that are parallel, in which case
    there is no intersection.  In this case, the corresponding value in each
    of the output tensors will be garbage, but the corresponding element in the 
    valid intersections return (second returned value) will be false, indicating 
    that this intersection was invalid and its values should be ignored.
    
    Please note that if given two inputs of shape [N, 4] and [M, 4], the expected 
    shape of the output is [M, N].
        
    Parameters
    ----------
    first_lines : Float tensor-like of shape [N, 4]
        The first set of lines
    first_lines : Float tensor-like of shape [M, 4]
        The second set of lines
    name : str, optional
        The name to use in creating a name scope for the ops created by this 
        function.  If not specified will use 'line_intersect'.
    epsilion : float, optional
        A very small value used to avoid divide by zero, which can happen if the
        lines are parallel.  If you are getting incorrect results because you have 
        lines that are extremely close but not actually parallel, you may need to
        reduce this value from its default of 1e-10, but if you make this too small
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
    with tf.name_scope(name, default_name="line_intersect"):
        first_lines = tf.ensure_shape(
            first_lines,
            (None, 4),
            name="Check_First_Lines_Shape"
        )
        second_lines = tf.ensure_shape(
            second_lines,
            (None, 4),
            name="Check_Second_Lines_Shape"
        )
        
        x1s, y1s, x1e, y1e = tf.unstack(first_lines, axis=-1)
        x2s, y2s, x2e, y2e = tf.unstack(second_lines, axis=-1)
        
        x1s, x2s = tf.meshgrid(x1s, x2s)
        y1s, y2s = tf.meshgrid(y1s, y2s)
        x1e, x2e = tf.meshgrid(x1e, x2e)
        y1e, y2e = tf.meshgrid(y1e, y2e)
        
        return raw_line_intersect(
            (x1s, y1s, x1e, y1e),
            (x2s, y2s, x2e, y2e),
            epsilion=epsilion
            )

# -------------------------------------------------------------------------------------

def line_intersect_1to1(first_lines, second_lines, name=None, epsilion=1e-10):
    """
    Finds the intersections of two sets of lines, matching them 1 to 1 instead of
    taking all n^2 combinations.
    
    First and second lines are two sets of lines.  The function will return the 
    coordinates and parameters where each line in first_lines intersects with its
    corresponding line in second_lines.  Both must be rank 2 tensors, whose first
    dimenson indexes each line in the set, and whose second dimension must have size
    4 and represents [xstart, ystart, xend, yend], the x and y coordinates of two 
    points that define the line.  Though the lines are formatted as line segments,
    this function will treat its input as inifinte lines.
    
    The line sets will be implicitly broadcast by this function, so either both sets
    must have the same number of lines (first dimension of each has the same size)
    or one (or both) of the sets must have size 1 in the first dimension, in which
    case that line will be compared to every line in the other set.  So the 
    following input shape options are valid for this function:
    [1, 4], [1, 4]: will find the intersection between ex1ctly two lines.
    [n, 4], [1, 4]: will find all n intersections between lines in the first set and
        the second line.
    [1, 4], [n, 4]: will find all n intersections between lines in the second set 
        and the first line.
    [n, 4], [n, 4]: will find all n intersections between the n pairs of lines.
    
    [n, 4], [m, 4] if (n!=m) is not valid, because there is no way to match the 
        two sets of lines.
        
    This function is safe when given input lines that are parallel, in which case
    there is no intersection.  In this case, the corresponding value in each
    of the output tensors will be garbage, but the corresponding element in the 
    valid intersections return (second returned value) will be false, indicating 
    that this intersection was invalid and its values should be ignored.
        
    Parameters
    ----------
    first_lines : Float tensor-like of shape [None, 4]
        The first set of lines
    first_lines : Float tensor-like of shape [None, 4]
        The second set of lines
    name : str, optional
        The name to use in creating a name scope for the ops created by this 
        function.  If not specified will use 'line_intersect_1to1'.
    epsilion : float, optional
        A very small value used to avoid divide by zero, which can happen if the
        lines are parallel.  If you are getting incorrect results because you have 
        lines that are extremely close but not actually parallel, you may need to
        reduce this value from its default of 1e-10, but if you make this too small
        it may incorrectly identify parallel lines as actually intersecting, or
        will throw a nan into your tensors, which will kill tf.gradients.  As of
        05/19 I haven't tested how small you can go with this parameter, so lowering
        it a lot may be perfectly fine.
        
    Returns
    -------
    Intersection coordinates:
        A tuple of two float tensors of shape [n] holding the x and y
        values of the intersections.
    Valid intersections:
        A bool tensor of shape [n] which is true wherever the intersection is valid.
        Intersections are valid when the lines are not parallel.
    U:
        A float tensor of shape [n] holding the parameter along each of the first
        lines where the intersection occurs.
    V:
        A float tensor of shape [n] holding the parameter along each of the second
        lines where the intersection occurs.
        
    """
    with tf.name_scope(name, default_name="line_intersect_1to1"):
        first_lines = tf.ensure_shape(
            first_lines,
            (None, 4),
            name="Check_First_Lines_Shape"
        )
        second_lines = tf.ensure_shape(
            second_lines,
            (None, 4),
            name="Check_Second_Lines_Shape"
        )
        return raw_line_intersect(
            tf.unstack(first_lines, axis=-1),
            tf.unstack(second_lines, axis=-1),
            epsilion=epsilion
            )

# -------------------------------------------------------------------------------------
        
def raw_line_intersect(first_coordinates, second_coordinates, epsilion=1e-10):
    """
    Low-level core math used to find intersections between two sets of lines.
    
    This function does no error/shape checking on its inputs, and so its inputs can
    be any shape, as long as they are all co-broadcastable.  This function treats its
    inputs as infinite lines, and is safe with tf.gradients in the case of parallel 
    lines in the input sets.
    
    Parameters
    ----------
    first_coordinates : 4-tuple of tensors
        Coordinates (xstart, ystart, xend, yend) of each line in the first set.
    first_lines : 4-tuple of tensors
        Coordinates (xstart, ystart, xend, yend) of each line in the second set.
    epsilion : float, optional
        A very small value used to avoid divide by zero, which can happen if the
        lines are parallel.  If you are getting incorrect results because you have 
        lines that are extremely close but not actually parallel, you may need to
        reduce this value from its default of 1e-10, but if you make this too small
        it may incorrectly identify parallel lines as actually intersecting, or
        will throw a nan into your tensors, which will kill tf.gradients.  As of
        05/19 I haven't tested how small you can go with this parameter, so lowering
        it a lot may be perfectly fine.
        
    Returns
    -------
    Intersection coordinates:
        A tuple of two float tensors holding the x and y values of the intersections.
    Valid intersections:
        A bool tensor which is true wherever the intersection is valid.  Intersections 
        are valid when the lines are not parallel.
    U:
        A float tensor holding the parameter along each of the first lines where the 
        intersection occurs.
    V:
        A float tensor holding the parameter along each of the second lines where the 
        intersection occurs.
    """
    x1s, y1s, x1e, y1e = first_coordinates
    x2s, y2s, x2e, y2e = second_coordinates
    
    x1 = (x1e-x1s)
    y1 = (y1e-y1s)
    x2 = (x2e-x2s)
    y2 = (y2e-y2s)
    denominator = x1 * y2 - y1 * x2
    
    # The following three lines make this code safe for use with tf.gradients
    # because we absolutely have to avoid nan.
    valid_intersection = tf.greater_equal(tf.abs(denominator), epsilion)
    safe_value = tf.ones_like(denominator)
    safe_denominator = tf.where(valid_intersection, denominator, safe_value)
    
    u = tf.where(
        valid_intersection, 
        (x2 * (y1s - y2s) - y2 * (x1s - x2s))/safe_denominator,
        safe_value
    )
    v = tf.where(
        valid_intersection, 
        (y1 * (y2s - y1s) - x1 * (x2s - x1s))/safe_denominator,
        safe_value
    )
    x = x1s + u * x1
    y = y1s + u * y1
    
    # for testing purposes, compute the intersection with the second line's 
    # parameter.
    #x = x2s + v * x2
    #y = y2s + v * y2
    
    return (x, y), valid_intersection, u, v
        
# =====================================================================================
   
def line_circle_intersect(lines, circles, name=None, epsilion=1e-10):
    """
    Given a set of N lines and a second set of M circles, computes all 2*NxM
    intersections.
    
    Lines must be a rank 2 tensor whose first dimenson indexes each line in the set
    and whose second dimension must have size 4 and represents [xstart, ystart, xend, 
    yend], the x and y coordinates of two points that define the line.  Circles must 
    be a rank 2 tensor whose first dimension indexes each circle and whose second
    dimension must have size 3 and represents [xcenter, ycenter, radius].
    
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
    
    Please note that if given two inputs of shape [N, 4] and [M, 3], the expected 
    shape of the output is [M, N].
        
    Parameters
    ----------
    lines : Float tensor-like of shape [N, 4]
    circles : Float tensor-like of shape [M, 3]
    name : str, optional
        The name to use in creating a name scope for the ops created by this 
        function.  If not specified will use 'line_circle_intersect'.
    epsilion : float, optional
        A very small value used to avoid divide by zero.
        
    Returns
    -------
    A 2-tuple whose first element holds the intersection information for the
    intersection that resulted from the plus branch of the quadratic, and whose second
    element holds the intersection information for the intersection that resulted
    from the minus branch of the quadratic.  Each element of the tuple is a dict
    that contains the following keys:
    
    coords:
        A tuple of two float tensors of shape [M, N] holding the x and y
        values of the intersection.
    valid:
        A bool tensor of shape [M, N] which is true wherever the intersection is
        valid.
    u:
        A float tensor of shape [M, N] holding the parameter along the line where the 
        intersection occurs.
    v:
        A float tensor of shape [M, N] holding the angle of the circle where the
        intersection occurs.  Angles are generated with atan2, so they are all in the
        interval [-PI, PI], with zero pointing along the x axis.
        
    """
    with tf.name_scope(name, default_name="line_circle_intersect_1_to_1"):
        lines = tf.ensure_shape(
            lines,
            (None, 4),
            name="Check_Lines_Shape"
        )
        circles = tf.ensure_shape(
            circles,
            (None, 3),
            name="Check_Circles_Shape"
        )
        
        xs, ys, xe, ye = tf.unstack(lines, axis=-1)
        xc, yc, r = tf.unstack(circles, axis=-1)
        
        xs, xc = tf.meshgrid(xs, xc)
        ys, yc = tf.meshgrid(ys, yc)
        xe, _ = tf.meshgrid(xe, r)
        ye, r = tf.meshgrid(ye, r)
        
        return raw_line_circle_intersect(
            (xs, ys, xe, ye),
            (xc, yc, r),
            epsilion=epsilion
            )

# -------------------------------------------------------------------------------------

def line_circle_intersect_1to1(lines, circles, name=None, epsilion=1e-10):
    """
    Finds the intersections between a set of lines and a set of circles, matching
    them 1-to-1 instead of comparing all n^2 combinations.
    
    Lines must be a rank 2 tensor whose first dimenson indexes each line in the set
    and whose second dimension must have size 4 and represents [xstart, ystart, xend, 
    yend], the x and y coordinates of two points that define the line.  Circles must 
    be a rank 2 tensor whose first dimension indexes each circle and whose second
    dimension must have size 3 and represents [xcenter, ycenter, radius].
    
    Though the lines are formatted as line segments, this function will treat its 
    input as inifinte lines.  Circles of radius zero will cause failure.
    
    The sets will be implicitly broadcast by this function, so either both sets
    must have the same number of elements (first dimension of each has the same size)
    or one (or both) of the sets must have size 1 in the first dimension, in which
    case that object will be compared to every object in the other set.  So the 
    following input shape options are valid for this function:
    [1, 4], [1, 3]: will find the intersection between one line and one circle.
    [n, 4], [1, 3]: will find all n intersections between a set of lines and one
        circle.
    [1, 4], [n, 3]: will find all n intersections between a set of circles and one
        line.
    [n, 4], [n, 3]: will find all n intersections between n lines and n circles.
    
    [n, 4], [m, 3] if (n!=m) is not valid, because there is no way to match the 
        two sets 1-to-1.
      
    A line may intersect a circle zero, one, or two times.  This function can handle
    all cases.  It will return two intersections for every line, labeled plus and
    minus based on which branch of the quadratic was taken to calculate the solution.
    The accompanying valid_plus/valid_minus bool tensors report which of the returned
    solutions actually correspond to valid intersections.  In the case where a line
    intersects a circle less than twice, the elements in the output tensors 
    corresponding to that intersection will be filled with garbage values that should
    not be used.
        
    Parameters
    ----------
    lines : Float tensor-like of shape [None, 4]
    circles : Float tensor-like of shape [None, 3]
    name : str, optional
        The name to use in creating a name scope for the ops created by this 
        function.  If not specified will use 'line_circle_intersect_1to1'.
    epsilion : float, optional
        A very small value used to avoid divide by zero.
        
    Returns
    -------
    A 2-tuple whose first element holds the intersection information for the
    intersection that resulted from the plus branch of the quadratic, and whose second
    element holds the intersection information for the intersection that resulted
    from the minus branch of the quadratic.  Each element of the tuple is a dict
    that contains the following keys:
    
    coords:
        A tuple of two float tensors of shape [n] holding the x and y
        values of the intersection.
    valid:
        A bool tensor of shape [n] which is true wherever the intersection is
        valid.
    u:
        A float tensor of shape [n] holding the parameter along the line where the 
        intersection occurs.
    v:
        A float tensor of shape [n] holding the angle of the circle where the
        intersection occurs.  Angles are generated with atan2, so they are all in the
        interval [-PI, PI], with zero pointing along the x axis.
        
    """
    with tf.name_scope(name, default_name="line_circle_intersect_1to1"):
        lines = tf.ensure_shape(
            lines,
            (None, 4),
            name="Check_Lines_Shape"
        )
        circles = tf.ensure_shape(
            circles,
            (None, 3),
            name="Check_Circles_Shape"
        )
        return raw_line_circle_intersect(
            tf.unstack(lines, axis=-1),
            tf.unstack(circles, axis=-1),
            epsilion=epsilion
            )

# -------------------------------------------------------------------------------------
        
def raw_line_circle_intersect(lines, circles, epsilion=1e-10):
    """
    Low-level core math used to find intersections between lines and circles.
    
    This function does no error/shape checking on its inputs, and so its inputs can
    be any shape, as long as they are all co-broadcastable.  This function treats its
    inputs as infinite lines, and is safe with tf.gradients given all possible special
    cases.  Outputs will be the same shape as the inputs
    
    Parameters
    ----------
    lines : 4-tuple of tensors
        Coordinates (xstart, ystart, xend, yend) of each line in the set.
    circles : 3-tuple of tensors
        Coordinates (xcenter, ycenter, radius) of each circle in the set.
    epsilion : float, optional
        A very small value used to avoid divide by zero.
        
    Returns
    -------
    A 2-tuple whose first element holds the intersection information for the
    intersection that resulted from the plus branch of the quadratic, and whose second
    element holds the intersection information for the intersection that resulted
    from the minus branch of the quadratic.  Each element of the tuple is a dict
    that contains the following keys:
    
    coords:
        A tuple of two float tensors holding the x and y
        values of the intersection.
    valid:
        A bool tensor which is true wherever the intersection is
        valid.
    u:
        A float tensor holding the parameter along the line where the 
        intersection occurs.
    v:
        A float tensor holding the angle of the circle where the
        intersection occurs.  Angles are generated with atan2, so they are all in the
        interval [-PI, PI], with zero pointing along the x axis.
        
    """
    xs, ys, xe, ye = lines
    xc, yc, r = circles
    
    # coordinate adjustment
    with tf.name_scope("xr"):
        xr = (xs - xc) / r
    with tf.name_scope("yr"):
        yr = (ys - yc) / r
    with tf.name_scope("xd"):
        xd = (xe - xs) / r
    with tf.name_scope("yd"):
        yd = (ye - ys) / r
    
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
    # testing the intersection algorithm with tangent lines is inconsistent, because
    # while rad is always very small, it is sometimes negative (due to rounding error).
    # So I will check if it is less than epsilion, and if so set it to precisely zero,
    # so that we can correctly detect the point of tangency, and so that both 
    # solutions returned will be identical.
    rad = tf.where(
        tf.less(tf.abs(rad), epsilion, name="special_case_tangent"),
        tf.zeros_like(rad),
        rad
    )
      
    # rad < 0 special case. => line does not intersect circle.
    safe_value = tf.ones_like(a, name="safe_value")
    radLess = tf.less(rad, 0, name="special_case_no_intersection")
    uminus_valid = uplus_valid = tf.logical_not(radLess)
    safe_rad = tf.where(radLess, safe_value, rad, name="safe_rad")
    with tf.name_scope("uminus") as scope:
        uminus = tf.where(radLess, safe_value, (-b - tf.sqrt(safe_rad)))
    with tf.name_scope("uplus") as scope:
        uplus = tf.where(radLess, safe_value, (-b + tf.sqrt(safe_rad)))

    # a = 0 special case. => line start point == line end point.
    azero = tf.less(tf.abs(a), epsilion, name="special_case_azero")
    safe_demoninator = tf.where(azero, safe_value, 2 * a, "safe_demoninator")
    uminus_valid = tf.logical_and(uminus_valid, tf.logical_not(azero))
    uminus = tf.where(azero, safe_value, uminus / safe_demoninator)
    uplus_valid = tf.logical_and(uplus_valid, tf.logical_not(azero))
    uplus = tf.where(azero, safe_value, uplus / safe_demoninator)
    
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
            "coords": (xplus, yplus),
            "valid": uplus_valid,
            "u": uplus,
            "v": vplus
        },
        {
            "coords": (xminus, yminus),
            "valid": uminus_valid,
            "u": uminus,
            "v": vminus
        }  
    )
    
# ====================================================================================
        
def ray_reaction(rays, norm, n_in, n_out, new_ray_length=1.0):
    """
    Performs an optical ray reaction (Snell's law).
    
    This function assumes that ray intersection and projection has already been 
    performed.  It matches the data in each of its parameters 1-to-1.
    
    n_in and n_out are the refractive indices of this optical reaction.  They should
    be floats, and have already been evaluated as a function of wavelength.  If n_in
    is zero, this function will compute a reflective reaction.  Otherwise it will
    compute a refractive reaction.  n_out should not be zero.
    
    Rays is a tensor holding ray data in its last dimension.  It last dimension must 
    have at least four elements, the first four of which are [xstart, ystart, xend, 
    yend].  Any extra elements in the last dimension will be ignored by this function.
    Rays must have at least rank 2, and should have one greater dimension than the 
    other parameters.  It can have any shape before the last dimension, which can index
    individual rays and may include batches.  Norm, n_in, and n_out must have the same
    shape as rays does except for rays' last dimension.  In other words, rays is
    unstacked along its last dimension, and norm, n_in, and n_out must be broadcastable
    with each of the unstacked tensors.
    
    This function will generate a new set of rays that are the results from the 
    optical reaction.  The generated rays will automatically inherit any additional 
    elements from the last dimension (like wavelength).  The output rays will have 
    exactly the same shape as the input rays.
    
    Parameters
    ----------
    rays : float tensor-like
        The rays to react.
    norm : float tensor-like
        The angle of the surface norm, measured in absolute coordinates, as opposed to
        measured relative to the ray, as the angle of incidence.
    n_in : float tensor-like
        The refractive index of the material opposite to the norm.
    n_out : float tensor-like
        The refractive index of the material in which the norm is embedded.
    new_ray_length : float scalar, optional
        The length of new rays generated by this function.
        
    Returns
    -------
    Float tensor of the same shape as rays, containing the resulting reacted rays.
    
    """        
    with tf.name_scope("ray_reactions") as scope:
        ray_data = tf.unstack(rays, axis=-1)
        xstart, ystart, xend, yend = ray_data[:4]

        with tf.name_scope("norm") as scope:
            norm = tf.mod(norm, 2*PI)
        with tf.name_scope("ray_angle") as scope:
            ray_angle = tf.atan2(ystart-yend, xstart-xend)
            ray_angle = tf.mod(ray_angle, 2*PI)
        with tf.name_scope("theta_1") as scope:
            theta1 = norm - ray_angle
            # ensure we measure the difference the correct way
            theta1 = tf.where(theta1 > PI, theta1 - (2 * PI), theta1)
            theta1 = tf.where(theta1 < -PI, theta1 + (2 * PI), theta1)
            
        # choose between internal and external reactions
        internal_mask = tf.greater_equal(
            tf.abs(theta1), PI / 2, name="internal_refractions_mask"
        )
        mask_shape = tf.shape(internal_mask)
        n = tf.where(
            internal_mask,
            tf.broadcast_to(n_in / n_out, mask_shape),
            tf.broadcast_to(n_out / n_in, mask_shape),
            name="n"
        )
        norm = tf.where(
            internal_mask,
            tf.broadcast_to(norm, mask_shape),
            tf.broadcast_to(norm + PI, mask_shape),
            "norm_internal_external_selector")
        theta1 = tf.where(
            internal_mask,
            tf.broadcast_to(theta1 + PI, mask_shape),
            tf.broadcast_to(theta1, mask_shape),
            "theta1_internal_external_selector"
        )
        
        with tf.name_scope("theta_2") as scope:
            theta2 = n * tf.sin(theta1)
            new_angle = tf.where(
                tf.less_equal(tf.abs(theta2), 1.0),
                norm - tf.asin(theta2),
                norm + theta1 + PI,
                name="TIR_selector",
            )
            
        with tf.name_scope("new_ray") as scope:
            xstart = xend
            ystart = yend
            xend = xstart + new_ray_length * tf.cos(new_angle)
            yend = ystart + new_ray_length * tf.sin(new_angle)
        return tf.stack(
            [xstart, ystart, xend, yend] + ray_data[4:],
            axis=1, 
            name="active_rays")
            
#with tf.Session() as session:
    
        
        
        
        
        
        
        
        
        
