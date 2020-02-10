from __future__ import print_function

"""
TFRayTrace.py
Eric Poppenheimer
November 2018

This module implements a simplified ray tracer in using TensorFlow.  It is differentiable (using tf.gradients) so 
that it can be used within an optimization algorithm.

Boundaries and rays fed to this module should be rank 2 tensors whose first axis indexes individual entities, and 
whose second axis contains data about the line segment.  The second axis should be at least length 4, and contains 
in order xstart, ystart, xend, yend.  The order of the points of the boundary matters: the norm of the boundary is 
computed as norm = tf.atan2(yend-ystart, xend-xstart) + PI/2.  Extra data might be included after the endpoints, 
depending on which function is being used.
"""

import tensorflow as tf
import math
import numpy as np

PI = tf.constant(math.pi, name="PI", dtype=tf.float64)

# ==================================================================================================================

"""
rayTraceSinglePass

This function is more complex, but more general, and can handle both refractions and reflections (though only 
reflections with reflectivity=1, since we are not tracking intensity right now).  And it can handle an arbitrary 
combination of boundary material refractive index.  The last two elements in each boundary are indices that index
a material out of the materials tuple

rays: A tensor of shape (..., 5) that defines the rays to be traced, where each element is (xstart, ystart, xend, 
    yend, wavelength).
boundarySegments: A tensor of shape (..., 6) that defines optical boundary line segments, where each element is
    (xstart, ystart, xend, yend, n_in_index, n_out_index)
boundaryArcs: A tensor of shape (..., 7) that defines optical boundary arcs, where each element is (xcenter, 
    ycenter, angleStart, angleEnd, radius, n_in_index, n_out_index)
targetSegments: May be None, if this feature is unwanted.  Otherwise, is a tensor of shape (..., 4) that defines 
    non-optically active boundary line segments which represent where rays should terminate.  This allows for 
    sorting rays that have reached their desired destination, or exited the optical system, and so should not be 
    traced further.  The elements of this tensor are like boundarySegments, but omitting the to material indices: 
    (xstart, ystart, xend, yend)
targetArcs: May be None, or a tensor of shape (..., 5) that defines non-optically active boundary arcs, similar to
    targetSegments.  Elements are (xcenter, ycenter, angleStart, angleEnd, radius)
materials: A tuple that defines the refractive index of the materials of the boundaries.  Materials must be a 
    tuple of callables that can accept a rank 1 tensor (the wavelength) and output a tensor of the same shape (the 
    refractive index).  If a boundary indexes two refractive indices that are both zero, then that boundary will 
    act as a reflector rather than a refractor.
    
returns: 
reactedRays: a tensor of rays that have reacted with the optical system (have been projected to the boundary they 
    reacted with) and are now finished with raytracing.
activeRays: a tensor of the new rays that were generated as a result of the ray reactions, and may be raytraced
    again
finishedRays: a tensor of rays that have stopped reacting.
deadRays: a tensor of rays that did not react at all, and so should not be raytraced further
"""


def rayTraceSinglePass(
    rays,
    boundarySegments,
    boundaryArcs,
    targetSegments,
    targetArcs,
    materials,
    epsilion=1e-6,
):
    if boundarySegments is not None:
        b_usingSegments = True
    else:
        b_usingSegments = False

    if boundaryArcs is not None:
        b_usingArcs = True
    else:
        b_usingArcs = False

    with tf.name_scope("rayTraceSingle") as scope:
        # rayRange is just a list of all ray indexes, useful for constructing index tensors to be used
        # with gather
        with tf.name_scope("rayRange") as scope:
            rayRange = tf.range(
                tf.shape(rays, out_type=tf.int64)[0], dtype=tf.int64, name="rayRange"
            )

        # join boundaries and targets, for the purposes of finding the closest intersection
        with tf.name_scope("segmentTargetJoining") as scope:
            if b_usingSegments:
                opticalSegmentCount = tf.cast(
                    tf.shape(boundarySegments)[0], dtype=tf.int64
                )
            else:
                opticalSegmentCount = 0

            if targetSegments is not None:
                targetSegments = tf.pad(targetSegments, [[0, 0], [0, 2]])
                if b_usingSegments:
                    boundarySegments = tf.concat(
                        (boundarySegments, targetSegments),
                        0,
                        name="joinedBoundarySegments",
                    )
                elif targetSegments.shape[0] != 0:
                    boundarySegments = targetSegments
                    b_usingSegments = True

        with tf.name_scope("arcTargetJoining") as scope:
            if b_usingArcs:
                opticalArcCount = tf.cast(tf.shape(boundaryArcs)[0], dtype=tf.int64)
            else:
                opticalArcCount = 0

            if targetArcs is not None:
                targetArcs = tf.pad(targetArcs, [[0, 0], [0, 2]])
                if b_usingArcs:
                    boundaryArcs = tf.concat(
                        (boundaryArcs, targetArcs), 0, name="joinedBoundaryArcs"
                    )
                elif targetArcs.shape[0] != 0:
                    boundaryArcs = targetArcs
                    b_usingArcs = True

        # slice the input rays into sections
        with tf.name_scope("inputRaySlicing") as scope:
            xstart = rays[:, 0]
            ystart = rays[:, 1]
            xend = rays[:, 2]
            yend = rays[:, 3]

        # intersect rays and boundary segments
        if b_usingSegments:
            with tf.name_scope("ray-SegmentIntersection") as scope:
                with tf.name_scope("variableMeshing") as scope:
                    xa1, xb1 = tf.meshgrid(xstart, boundarySegments[:, 0])
                    ya1, yb1 = tf.meshgrid(ystart, boundarySegments[:, 1])
                    xa2, xb2 = tf.meshgrid(xend, boundarySegments[:, 2])
                    ya2, yb2 = tf.meshgrid(yend, boundarySegments[:, 3])

                xa = xa2 - xa1
                ya = ya2 - ya1
                xb = xb2 - xb1
                yb = yb2 - yb1

                # v is the parameter of the intersection for B (bounds), and u is for A (rays).  inf values signify
                # that this pair of lines is parallel
                with tf.name_scope("raw_v_parameter") as scope:
                    denominator = xa * yb - ya * xb
                    validSegmentIntersection = tf.greater_equal(
                        tf.abs(denominator), epsilion
                    )
                    safe_value = tf.ones_like(denominator)
                    safe_denominator = tf.where(
                        validSegmentIntersection, denominator, safe_value
                    )

                    segmentV = tf.where(
                        validSegmentIntersection,
                        (ya * (xb1 - xa1) - xa * (yb1 - ya1)) / safe_denominator,
                        safe_value,
                    )
                with tf.name_scope("raw_u_parameter") as scope:
                    segmentU = tf.where(
                        validSegmentIntersection,
                        (xb * (ya1 - yb1) - yb * (xa1 - xb1)) / safe_denominator,
                        safe_value,
                    )

                # Since B encodes line segments, not infinite lines, purge all occurances in v which are <=0 or >=1
                # since these imply rays that did not actually strike the segment, only intersected with its
                # infinite continuation.
                # And since A encodes semi-infinite rays, purge all occurances in u which are <epsilion, since
                # these are intersections that occur before the ray source.  We need to compare to epsilion to take
                # account of rays that are starting on a boundary
                with tf.name_scope("selectClosestValidIntersection") as scope:
                    validSegmentIntersection = tf.logical_and(
                        validSegmentIntersection, tf.greater_equal(segmentV, -epsilion)
                    )
                    validSegmentIntersection = tf.logical_and(
                        validSegmentIntersection,
                        tf.less_equal(segmentV, 1.0 + epsilion),
                    )
                    validSegmentIntersection = tf.logical_and(
                        validSegmentIntersection, tf.greater_equal(segmentU, epsilion)
                    )

                # true where a ray intersection was actually found (since raySegmentIndices = 0 if the ray
                # intersects with boundary 0, or if there was no intersection
                with tf.name_scope("raySegmentMask") as scope:
                    raySegmentMask = tf.reduce_any(validSegmentIntersection, axis=0)

                # match segmentU to each ray
                with tf.name_scope("segmentU") as scope:
                    # raySegmentIndices tells us which ray intersects with which boundary.
                    # raySegmentIndices[n]=m => ray n intersects boundary segment m
                    inf = 2 * tf.reduce_max(segmentU) * safe_value
                    segmentU = tf.where(validSegmentIntersection, segmentU, inf)
                    raySegmentIndices = tf.argmin(
                        segmentU, axis=0, name="raySegmentIndices"
                    )

                    # intersectIndicesSquare is a set of indices that can be used with gather_nd to select
                    # positions out of the grid tensors
                    intersectIndicesSquare = tf.transpose(
                        tf.stack([raySegmentIndices, rayRange])
                    )

                    # the u parameter for ray intersections, after filtering and processing
                    segmentU = tf.gather_nd(
                        segmentU, intersectIndicesSquare, name="segmentU"
                    )

                    # package and pair the boundary segments with the rays that intersect with them
                    boundarySegments = tf.gather(
                        boundarySegments, raySegmentIndices, name="boundarySegments"
                    )

        # intersect rays and boundary arcs
        if b_usingArcs:
            with tf.name_scope("ray-ArcIntersection") as scope:
                with tf.name_scope("inputMeshgrids") as scope:
                    x1, xc = tf.meshgrid(xstart, boundaryArcs[:, 0])
                    y1, yc = tf.meshgrid(ystart, boundaryArcs[:, 1])
                    x2, thetaStart = tf.meshgrid(xend, boundaryArcs[:, 2])
                    y2, thetaEnd = tf.meshgrid(yend, boundaryArcs[:, 3])
                    y2, r = tf.meshgrid(tf.reshape(yend, [-1]), boundaryArcs[:, 4])
                    # the reshape in the above line shouldn't be necessary, but I was getting some really wierd
                    # bugs that went away whenever I tried to read the damn tensor, and this fixes it for some
                    # reason.

                # a, b, c here are parameters to a quadratic equation for u, so we have some special cases to deal
                # with
                # a = 0 => ray of length zero.  This should never happen, but if it does, should invalidate
                # the intersections
                # rad < 0 => ray does not intersect circle

                # ?????
                # c = 0 => ray starts on circle => u = 0, -b/c
                #                                  c = 0 => ray ends on circle??? My mind has changed on this
                with tf.name_scope("coordinateAdjusting") as scope:
                    xr = (x1 - xc) / r
                    yr = (y1 - yc) / r
                    xd = (x2 - x1) / r
                    yd = (y2 - y1) / r

                with tf.name_scope("quadraticEquationParts") as scope:
                    with tf.name_scope("a") as scope:
                        a = xd * xd + yd * yd
                    with tf.name_scope("b") as scope:
                        b = 2.0 * xr * xd + 2.0 * yr * yd
                    with tf.name_scope("c") as scope:
                        c = xr * xr + yr * yr - 1.0

                    with tf.name_scope("rad") as scope:
                        rad = b * b - 4.0 * a * c

                safe_value = tf.ones_like(a, name="safe_value")

                with tf.name_scope("raw_u_parameter") as scope:
                    # u will be the parameter of the intersections along the ray
                    # rad < 0 special case
                    with tf.name_scope("specialCase_complex") as scope:
                        radLess = tf.less(rad, 0)
                        uminus_valid = uplus_valid = tf.logical_not(radLess)
                        safe_rad = tf.where(radLess, safe_value, rad)

                        uminus = tf.where(radLess, safe_value, (-b - tf.sqrt(safe_rad)))
                        uplus = tf.where(radLess, safe_value, (-b + tf.sqrt(safe_rad)))

                    # a = 0 special case
                    with tf.name_scope("specialCase_azero") as scope:
                        azero = tf.less(tf.abs(a), epsilion)
                        safe_a = tf.where(azero, safe_value, 2 * a)

                        uminus_valid = tf.logical_and(
                            uminus_valid, tf.logical_not(azero)
                        )
                        uminus = tf.where(azero, safe_value, uminus / safe_a)

                        uplus_valid = tf.logical_and(uplus_valid, tf.logical_not(azero))
                        uplus = tf.where(azero, safe_value, uplus / safe_a)

                        """
                        czero = tf.less(tf.abs(c), epsilion)
                        safe_c = tf.where(czero, safe_value, c)
                        
                        uplus_valid = tf.logical_and(uplus_valid, tf.logical_not(czero))
                        b_over_c = tf.where(czero, safe_value, b/safe_c)
                        uplus = tf.where(azero, -b_over_c, uplus/safe_a)
                        #uplus = tf.where(azero, -b/c, uplus/safe_a)"""

                    # cut out all of the rays that have a u < epsilion parameter, since we only want reactions
                    # ahead of the ray
                    with tf.name_scope("cullNegativeU") as scope:
                        uminus_valid = tf.logical_and(
                            uminus_valid, tf.greater_equal(uminus, epsilion)
                        )
                        uplus_valid = tf.logical_and(
                            uplus_valid, tf.greater_equal(uplus, epsilion)
                        )

                with tf.name_scope("raw_v_parameter") as scope:
                    # determine the x,y coordinate of the intersections
                    with tf.name_scope("xminus") as scope:
                        xminus = x1 + (x2 - x1) * uminus
                    with tf.name_scope("xplus") as scope:
                        xplus = x1 + (x2 - x1) * uplus
                    with tf.name_scope("yminus") as scope:
                        yminus = y1 + (y2 - y1) * uminus
                    with tf.name_scope("yplus") as scope:
                        yplus = y1 + (y2 - y1) * uplus

                    # determine the angle along the arc (arc's parameter) where the intersection occurs
                    """ these atan2 calls seem to be fucking up the gradient.  So I have to do something
                    convoluted."""
                    """
                    finiteUMinus = tf.debugging.is_finite(uminus)
                    finiteUPlus = tf.debugging.is_finite(uplus)
                    
                    def safe_atan2(y, x, safe_mask):
                        with tf.name_scope("safe_atan") as scope:
                            safe_x = tf.where(safe_mask, x, tf.ones_like(x))
                            safe_y = tf.where(safe_mask, y, tf.ones_like(y))
                            return tf.where(safe_mask, tf.atan2(safe_y, safe_x), tf.zeros_like(safe_x))"""

                    vminus = tf.atan2(yminus - yc, xminus - xc)
                    # vminus = safe_atan2(yminus-yc, xminus-xc, finiteUMinus)
                    vminus = tf.floormod(vminus, 2 * PI)

                    vplus = tf.atan2(yplus - yc, xplus - xc)
                    # vplus = safe_atan2(yplus-yc, xplus-xc, finiteUPlus)
                    vplus = tf.floormod(vplus, 2 * PI)

                # Cut out all cases where v does not fall within the angular extent of the arc
                with tf.name_scope("selectValid_v") as scope:
                    # my angle in interval algorithm fails when the interval is full (0->2PI).  So making the
                    # following adjustment to thetaStart
                    thetaStart = thetaStart + epsilion
                    vminus_valid = tf.less_equal(
                        tf.floormod(vminus - thetaStart, 2 * PI),
                        tf.floormod(thetaEnd - thetaStart, 2 * PI),
                    )
                    uminus_valid = tf.logical_and(vminus_valid, uminus_valid)

                    vplus_valid = tf.less_equal(
                        tf.floormod(vplus - thetaStart, 2 * PI),
                        tf.floormod(thetaEnd - thetaStart, 2 * PI),
                    )
                    uplus_valid = tf.logical_and(vplus_valid, uplus_valid)

                # now we can finally select between the plus and minus cases
                # arcU = tf.where(tf.less(uminus, uplus), uminus, uplus, name="arcU")
                # arcV = tf.where(tf.less(uminus, uplus), vminus, vplus, name="arcV")
                with tf.name_scope("choosePlusOrMinus") as scope:
                    # We have been keeping track of valid and invalid intersections in the u+/-_valid tensors.  But
                    # now we need to compare the values in the u+/- tensors and prepare for the argmin call that
                    # finds only the closest intersections.  To do this we now need to fill the invalid values in
                    # each tensor with some value that is larger than any valid value.  Unfortunately we cannot
                    # use np.inf because that seems to mess with the gradient calculator.
                    inf = (
                        2
                        * safe_value
                        * tf.reduce_max([tf.reduce_max(uminus), tf.reduce_max(uplus)])
                    )
                    uminus = tf.where(uminus_valid, uminus, inf)
                    uplus = tf.where(uplus_valid, uplus, inf)

                    choose_uminus = tf.less(uminus, uplus)

                    uminus_valid = tf.logical_and(uminus_valid, choose_uminus)
                    uplus_valid = tf.logical_and(
                        uplus_valid, tf.logical_not(choose_uminus)
                    )

                    # rayArcMask will tell us which rays have found at least one valid arc intersection
                    rayArcMask = tf.logical_or(uminus_valid, uplus_valid)
                    rayArcMask = tf.reduce_any(rayArcMask, axis=0)

                    arcU = tf.where(choose_uminus, uminus, uplus)
                    arcV = tf.where(choose_uminus, vminus, vplus)

                """
                # true where a ray intersection was actually found
                with tf.name_scope("rayArcMask") as scope:
                    rayArcMask = tf.is_finite(arcU)
                    rayArcMask = tf.reduce_any(rayArcMask, axis=0)"""

                # match arcU to each ray
                with tf.name_scope("arcU_and_arcV") as scope:
                    # rayArcIndices tells us which ray intersects with which boundary.
                    # rayArcIndices[n]=m => ray n intersects boundary segment m
                    rayArcIndices = tf.argmin(arcU, axis=0, name="rayArcIndices")

                    # intersectIndicesSquare is a set of indices that can be used with gather_nd to select
                    # positions out of the grid tensors
                    intersectIndicesSquare = tf.transpose(
                        tf.stack([rayArcIndices, rayRange])
                    )

                    # the u parameter for ray intersections, after filtering and processing
                    arcU = tf.gather_nd(arcU, intersectIndicesSquare, name="arcU")
                    arcV = tf.gather_nd(arcV, intersectIndicesSquare, name="arcV")

                    # package and pair the boundary arcs with the rays that intersect with them
                    boundaryArcs = tf.gather(
                        boundaryArcs, rayArcIndices, name="boundaryArcs"
                    )

        # determine which rays are dead
        with tf.name_scope("deadRays") as scope:
            if b_usingSegments and b_usingArcs:
                deadRays = tf.boolean_mask(
                    rays,
                    tf.logical_not(tf.logical_or(rayArcMask, raySegmentMask)),
                    name="deadRays",
                )
            else:
                if b_usingSegments:
                    deadRays = tf.boolean_mask(
                        rays, tf.logical_not(raySegmentMask), name="deadRays"
                    )
                elif b_usingArcs:
                    deadRays = tf.boolean_mask(
                        rays, tf.logical_not(rayArcMask), name="deadRays"
                    )
                else:
                    raise RuntimeError(
                        "rayTraceSinglePass: no boundaries provided for raytracing"
                    )

        # select between segment and arc intersections
        with tf.name_scope("arc_segment_selection") as scope:
            if b_usingSegments and b_usingArcs:
                chooseSegment = tf.logical_and(
                    tf.less(segmentU, arcU), raySegmentMask, name="chooseSegment"
                )
                chooseSegment = tf.logical_or(
                    chooseSegment,
                    tf.logical_and(raySegmentMask, tf.logical_not(rayArcMask)),
                )

                chooseArc = tf.logical_and(
                    tf.logical_not(chooseSegment), rayArcMask, name="chooseArc"
                )
                chooseArc = tf.logical_or(
                    chooseArc,
                    tf.logical_and(rayArcMask, tf.logical_not(raySegmentMask)),
                )
            else:
                if b_usingSegments:
                    chooseSegment = raySegmentMask
                if b_usingArcs:
                    chooseArc = rayArcMask

        # project ALL rays into the boundaries.  Rays that do not intersect with any boundaries will also be
        # projected to zero length, but these will be filtered off later
        with tf.name_scope("rayProjection") as scope:
            if b_usingSegments:
                with tf.name_scope("segments") as scope:
                    xstart = rays[:, 0]
                    ystart = rays[:, 1]
                    xend = rays[:, 2]
                    yend = rays[:, 3]
                    xend = xstart + (xend - xstart) * segmentU
                    yend = ystart + (yend - ystart) * segmentU
                    reactedRays_Segment = tf.stack(
                        [xstart, ystart, xend, yend, rays[:, 4], rays[:, 5]], axis=1
                    )

            if b_usingArcs:
                with tf.name_scope("arcs") as scope:
                    xstart = rays[:, 0]
                    ystart = rays[:, 1]
                    xend = rays[:, 2]
                    yend = rays[:, 3]
                    xend = xstart + (xend - xstart) * arcU
                    yend = ystart + (yend - ystart) * arcU
                    reactedRays_Arc = tf.stack(
                        [xstart, ystart, xend, yend, rays[:, 4], rays[:, 5]], axis=1
                    )

        # determine which rays are finished
        with tf.name_scope("finishedRays") as scope:
            finishedRays = tf.zeros([0, 6], dtype=tf.float64)

            if b_usingSegments:
                finishedSegmentMask = tf.greater_equal(
                    raySegmentIndices, opticalSegmentCount, name="finishedSegmentMask"
                )
                fsMask = tf.logical_and(finishedSegmentMask, chooseSegment)
                finishedRays_Segment = tf.boolean_mask(reactedRays_Segment, fsMask)
                finishedRays = tf.cond(
                    tf.reduce_any(fsMask),
                    lambda: tf.concat([finishedRays, finishedRays_Segment], axis=0),
                    lambda: finishedRays,
                )

            if b_usingArcs:
                finishedArcMask = tf.greater_equal(
                    rayArcIndices, opticalArcCount, name="finishedArcMask"
                )
                faMask = tf.logical_and(finishedArcMask, chooseArc)
                finishedRays_Arc = tf.boolean_mask(reactedRays_Arc, faMask)
                finishedRays = tf.cond(
                    tf.reduce_any(faMask),
                    lambda: tf.concat([finishedRays, finishedRays_Arc], axis=0),
                    lambda: finishedRays,
                )

        # conjugate to finished rays
        with tf.name_scope("reactedRays") as scope:
            reactedRays = tf.zeros([0, 6], dtype=tf.float64)

            if b_usingSegments:
                chooseSegment = tf.logical_and(
                    tf.logical_not(finishedSegmentMask), chooseSegment
                )
                reactedRays_Segment = tf.boolean_mask(
                    reactedRays_Segment, chooseSegment, name="reactedRays_Segment"
                )
                boundarySegments = tf.boolean_mask(
                    boundarySegments, chooseSegment, name="boundarySegments"
                )
                reactedRays = tf.cond(
                    tf.reduce_any(chooseSegment),
                    lambda: tf.concat([reactedRays, reactedRays_Segment], axis=0),
                    lambda: reactedRays,
                )

            if b_usingArcs:
                chooseArc = tf.logical_and(tf.logical_not(finishedArcMask), chooseArc)
                reactedRays_Arc = tf.boolean_mask(
                    reactedRays_Arc, chooseArc, name="reactedRays_Arc"
                )
                arcV = tf.boolean_mask(arcV, chooseArc, name="arcV")
                boundaryArcs = tf.boolean_mask(
                    boundaryArcs, chooseArc, name="boundaryArcs"
                )
                reactedRays = tf.cond(
                    tf.reduce_any(chooseArc),
                    lambda: tf.concat([reactedRays, reactedRays_Arc], axis=0),
                    lambda: reactedRays,
                )

        # calculate the norm of the surface
        with tf.name_scope("norm") as scope:
            norm = tf.zeros([0], dtype=tf.float64)

            if b_usingSegments:
                normSegment = (
                    tf.atan2(
                        boundarySegments[:, 3] - boundarySegments[:, 1],
                        boundarySegments[:, 2] - boundarySegments[:, 0],
                        name="normSegment",
                    )
                    + PI / 2
                )
                norm = tf.cond(
                    tf.reduce_any(chooseSegment),
                    lambda: tf.concat([norm, normSegment], axis=0),
                    lambda: norm,
                )

            if b_usingArcs:
                normArc = tf.where(
                    tf.less(boundaryArcs[:, 4], 0), arcV + PI, arcV, name="normArc"
                )
                normArc = tf.floormod(normArc, 2 * PI)
                norm = tf.cond(
                    tf.reduce_any(chooseArc),
                    lambda: tf.concat([norm, normArc], axis=0),
                    lambda: norm,
                )

        with tf.name_scope("refractiveIndex") as scope:
            # calculate the refractive index for every material and ray
            wavelengths = reactedRays[:, 4]
            nstack = tf.stack(
                [each(wavelengths) for each in materials], axis=1, name="nstack"
            )
            rayRange = tf.range(
                tf.shape(reactedRays)[0], dtype=tf.int32, name="rayRange"
            )

            # select just the correct entry for n_in and n_out
            if b_usingSegments and b_usingArcs:
                n_in_indices = tf.concat(
                    [boundarySegments[:, 4], boundaryArcs[:, 5]],
                    axis=0,
                    name="n_in_indices",
                )
            else:
                if b_usingSegments:
                    n_in_indices = boundarySegments[:, 4]
                if b_usingArcs:
                    n_in_indices = boundaryArcs[:, 5]
            n_in_indices = tf.cast(n_in_indices, tf.int32)
            n_in_indices = tf.transpose(tf.stack([rayRange, n_in_indices]))
            n_in = tf.gather_nd(nstack, n_in_indices, name="n_in")

            if b_usingSegments and b_usingArcs:
                n_out_indices = tf.concat(
                    [boundarySegments[:, 5], boundaryArcs[:, 6]],
                    axis=0,
                    name="n_out_indices",
                )
            else:
                if b_usingSegments:
                    n_out_indices = boundarySegments[:, 5]
                if b_usingArcs:
                    n_out_indices = boundaryArcs[:, 6]
            n_out_indices = tf.cast(n_out_indices, tf.int32)
            n_out_indices = tf.transpose(tf.stack([rayRange, n_out_indices]))
            n_out = tf.gather_nd(nstack, n_out_indices, name="n_out")

        activeRays = react(reactedRays, norm, n_in, n_out)
        return reactedRays, activeRays, finishedRays, deadRays


# ------------------------------------------------------------------------------------------------------------------

"""
react

reactedRays: a tensor holding ray data for reacting rays that have been projected into the boundaries.
norm: the norm of the boundary that each ray reacts with
n_in, n_out: the refractive index associated with the boundary each ray is reacting with.  n_in refers to the 
    material opposite to the norm vector, and n_out to the material that the norm vector points into.

return: a tensor holding ray data for the new rays generated from reacting the reacted rays with the boundaries.
"""


def react(reactedRays, norm, n_in, n_out):
    with tf.name_scope("ray_reactions") as scope:

        with tf.name_scope("rayStart") as scope:
            xstart = reactedRays[:, 2]
            ystart = reactedRays[:, 3]

        with tf.name_scope("theta1") as scope:
            theta1 = tf.atan2(
                reactedRays[:, 1] - reactedRays[:, 3],
                reactedRays[:, 0] - reactedRays[:, 2],
                name="theta1",
            )
            theta1 = norm - theta1

            # basically just theta1 % 2PI, make sure that we measured the correct angle between the two directions
            with tf.name_scope("mod_2PI") as scope:
                theta1 = tf.where(theta1 > PI, theta1 - (2 * PI), theta1)
                theta1 = tf.where(theta1 < -PI, theta1 + (2 * PI), theta1)

        # distinguish internal/external refractions
        internalMask = tf.greater_equal(
            tf.abs(theta1), PI / 2, name="internalRefractionsMask"
        )
        n = tf.where(internalMask, n_in / n_out, n_out / n_in, name="n")
        # these wheres selects between internal and external refraction
        norm = tf.where(internalMask, norm, norm + PI, "normInternal_ExternalSelector")
        theta1 = tf.where(
            internalMask, theta1 + PI, theta1, "theta1Internal_ExternalSelector"
        )

        theta2 = tf.multiply(tf.sin(theta1), n, name="theta2")

        # the where selects between TIR and internal refraction
        theta2 = tf.where(
            tf.less_equal(tf.abs(theta2), 1.0),
            norm - tf.asin(theta2),
            norm + theta1 + PI,
            name="TIR_selector",
        )

        with tf.name_scope("rayEnd") as scope:
            xend = xstart + tf.cos(theta2)
            yend = ystart + tf.sin(theta2)

        # xend = xstart + tf.cos(norm)
        # yend = ystart + tf.sin(norm)

        return tf.stack(
            [xstart, ystart, xend, yend, reactedRays[:, 4], reactedRays[:, 5]],
            axis=1,
            name="activeRays",
        )


# ------------------------------------------------------------------------------------------------------------------

"""
rayTrace

maximumLoopDepth: The number of passes that will be performed during the ray tracing.  This number should be as 
    small as possible, since each ray tracing path adds many operations to the graph.  If you care that your rays
    reach a target, add one to the value of this parameter, since a ray reacting with the last optical surface 
    will need another pass to be projected into the target and added to the list of finished rays.
b_terminateIfFinished: If true, can terminate the loop early, as soon as all rays have either finished or died. 
    Still respects maximumLoopDepth.  If false, always runs for maximimLoopDepth cycles
    
for explanation of the other parameters, see rayTraceSinglePass.

return:
reactedRays, activeRays, finishedRays, deadRays, counter.  Counter is the number of passes actually executed.  See 
    rayTraceSinglePass for an explanation for the other return values.  The rays returned here have shape (..., 6)
    where the last element is an idenfifying label added so that rays can be tracked through the ray tracer, since
    the rays will be reordered and sorted into different tensors depending on how they trace.  The label on a ray
    in one of the outputs is simply the position of its ancestor in the input
"""


def rayTrace(
    rays,
    boundarySegments,
    boundaryArcs,
    targetSegments,
    targetArcs,
    materials,
    maximumLoopDepth,
    b_terminateIfFinished=True,
    epsilion=1e-6,
):

    # error check the shape of the inputs
    if rays.shape[1] != 5:
        raise ValueError(
            "rayTrace: rays had incorrect shape {}.  Shape must be (..., 5)".format(
                rays.shape
            )
        )
    if boundarySegments is not None and boundarySegments.shape[1] != 6:
        raise ValueError(
            "rayTrace: boundarySegments had incorrect shape {}.  Shape must be (..., 6)".format(
                boundarySegments.shape
            )
        )
    if boundaryArcs is not None and boundaryArcs.shape[1] != 7:
        raise ValueError(
            "rayTrace: boundaryArcs had incorrect shape {}.  Shape must be (..., 7)".format(
                boundarySegments.shape
            )
        )
    if targetSegments is not None and targetSegments.shape[1] != 4:
        raise ValueError(
            "rayTrace: targetSegments had incorrect shape {}.  Shape must be (..., 4)".format(
                targetSegments.shape
            )
        )
    if targetArcs is not None and targetArcs.shape[1] != 5:
        raise ValueError(
            "rayTrace: targetArcs had incorrect shape {}.  Shape must be (..., 5)".format(
                targetArcs.shape
            )
        )

    with tf.name_scope("labelRays") as scope:
        rays = tf.reshape(rays, [-1, 5])
        labels = tf.range(tf.shape(rays)[0])
        labels = tf.reshape(labels, [-1, 1])
        labels = tf.cast(labels, dtype=tf.float64)
        rays = tf.concat([rays, labels], axis=1)

    # setup the stopping condition
    def stoppingCondition(counter, reactedRays, activeRays, finishedRays, deadRays):
        with tf.name_scope("loopCondition") as scope:
            if b_terminateIfFinished:
                activeRayCount = tf.shape(activeRays)[0]
                return tf.logical_and(
                    tf.less(counter, maximumLoopDepth), tf.not_equal(activeRayCount, 0)
                )
            else:
                return tf.less(counter, maximumLoopDepth)

    # setup initial values for the loop variables
    raySize = [0, rays.shape[1]]
    activeRays = rays
    reactedRays = tf.zeros(raySize, dtype=tf.float64)
    finishedRays = tf.zeros(raySize, dtype=tf.float64)
    deadRays = tf.zeros(raySize, dtype=tf.float64)

    # setup the loop body
    def loopBody(counter, reactedRays, activeRays, finishedRays, deadRays):
        with tf.name_scope("incrementCounter") as scope:
            counter += 1
        reactedRaysOut, activeRays, finishedRaysOut, deadRaysOut = rayTraceSinglePass(
            activeRays,
            boundarySegments,
            boundaryArcs,
            targetSegments,
            targetArcs,
            materials,
            epsilion=epsilion,
        )

        reactedRays = tf.concat(
            [reactedRays, reactedRaysOut], 0, name="concatReactedRays"
        )
        finishedRays = tf.concat(
            [finishedRays, finishedRaysOut], 0, name="concatFinishedRays"
        )
        deadRays = tf.concat([deadRays, deadRaysOut], 0, name="concatDeadRays")

        return counter, reactedRays, activeRays, finishedRays, deadRays

    counterInitial = tf.constant(0)
    # raySize = raySize[1].value
    raySize = raySize[1]
    counter, reactedRays, activeRays, finishedRays, deadRays = tf.while_loop(
        stoppingCondition,
        loopBody,
        [0, reactedRays, activeRays, finishedRays, deadRays],
        shape_invariants=[
            counterInitial.get_shape(),
            tf.TensorShape([None, raySize]),
            tf.TensorShape([None, raySize]),
            tf.TensorShape([None, raySize]),
            tf.TensorShape([None, raySize]),
        ],
        name="RayTrace",
    )

    return reactedRays, activeRays, finishedRays, deadRays, counter
