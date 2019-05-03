"""
Various geometric routines used by the ray tracer, but general enough that they may be 
used by a user as well.

"""

import tensorflow as tf

def line_intersect(first_lines, second_lines, name=None, epsilion=1e-10):
    """
    Given a set of N lines and a second set of M lines, computes all NxM intersections
    between lines in one set with those in the other
    
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
    Output coordinates:
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
        
        x1s, y1s, x1e, y1e = tf.unstack(first_lines, axis=1)
        x2s, y2s, x2e, y2e = tf.unstack(second_lines, axis=1)
        
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

def line_intersect_1_to_1(first_lines, second_lines, name=None, epsilion=1e-10):
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
    Output coordinates:
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
            tf.unstack(first_lines, axis=1),
            tf.unstack(second_lines, axis=1),
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
    Output coordinates:
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
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
