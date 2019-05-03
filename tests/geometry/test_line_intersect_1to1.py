import pytest

import tensorflow as tf
import numpy as np

import tfrt.geometry as geometry

def test_intersecting_lines(session, count=100):
    # create three sets of points
    common_point = np.random.uniform(-1.0, 1.0, size=[count, 2])
    first_point = np.random.uniform(-10.0, 10.0, size=[count, 2])
    second_point = np.random.uniform(-10.0, 10.0, size=[count, 2])
    
    # take two random points along the line first-common and second-common
    first_start_p = np.random.uniform(-10.0, 10.0, size=[count,1])
    first_end_p = np.random.uniform(-10.0, 10.0, size=[count,1])
    second_start_p = np.random.uniform(-10.0, 10.0, size=[count,1])
    second_end_p = np.random.uniform(-10.0, 10.0, size=[count,1])
    
    first_start = common_point + first_start_p * (first_point - common_point)
    first_end = common_point + first_end_p * (first_point - common_point)
    second_start = common_point + second_start_p * (second_point - common_point)
    second_end = common_point + second_end_p * (second_point - common_point)
    
    # build the two sets of lines from these points
    first_line = np.concatenate((first_start, first_end), axis=1)
    second_line = np.concatenate((second_start, second_end), axis=1)
    
    # extract the intersections.  They should be equal to the common point
    (x, y), valid_intersection, u, v = geometry.line_intersect_1_to_1(
        first_line, second_line
    )
    
    # check that all pairs are marked valid.  The likelihood of two of these lines 
    # being parallel is vanishingly small, since first and second point would have had
    # to have been randomly chosen as colinear, though it is technically possible.
    all_valid = tf.reduce_all(valid_intersection)
    assert session.run(all_valid)
    
    # check that the intersection is common_point.  Actually will do this by measuring
    # the distance between the two and making sure it is very small
    common_x, common_y = tf.unstack(common_point, axis=1)
    distance = tf.sqrt((x-common_x)**2 + (y-common_y)**2)
    max_distance = tf.reduce_max(distance)
    assert session.run(max_distance) < 1e-4
