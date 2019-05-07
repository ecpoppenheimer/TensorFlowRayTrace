import pytest
import math

import tensorflow as tf
import numpy as np

import tfrt.geometry as geometry

PI = math.pi

def test_two_intersection_case(session, count=100):
    # create the circles
    xc = np.random.uniform(-10.0, 10.0, size=[count,])
    yc = np.random.uniform(-10.0, 10.0, size=[count,])
    radius = np.random.uniform(0.1, 2.0, size=[count,])
    circles = np.stack((xc, yc, radius), axis=1)
    
    # select two points on each circle
    angle1 = np.random.uniform(0, 2*PI, size=[count,])
    angle2 = np.random.uniform(0, 2*PI, size=[count,])
    
    p1x = xc + radius*np.cos(angle1)
    p1y = yc + radius*np.sin(angle1)
    p1 = np.stack((p1x, p1y), axis=1)
    p2x = xc + radius*np.cos(angle2)
    p2y = yc + radius*np.sin(angle2)
    p2 = np.stack((p2x, p2y), axis=1)
    
    # take two random ponits along the line p1 p2
    start_param = np.random.uniform(-10.0, 10.0, size=[count, 1])
    end_param = np.random.uniform(-10.0, 10.0, size=[count, 1])
    
    start = p1 + start_param * (p2 - p1)
    end = p1 + end_param * (p2 - p1)
    
    # build the lines from the above randomly selected points
    lines = np.concatenate((start, end), axis=1)
    
    # extract the intersections.
    plus, minus = geometry.line_circle_intersect_1to1(lines, circles)
    
    # check that every intersection in both sets is marked valid.
    all_valid_plus = tf.reduce_all(plus["valid"])
    assert session.run(all_valid_plus)
    all_valid_minus = tf.reduce_all(minus["valid"])
    assert session.run(all_valid_minus)
    
    # check that the intersections are not the same, by checking that the distance
    # between both solutions points is large enough
    plus_x, plus_y = plus["coords"]
    minus_x, minus_y = minus["coords"]
    distance = tf.sqrt((plus_x - minus_x)**2 + (plus_y - minus_y)**2)
    min_distance = tf.reduce_min(distance)
    assert session.run(min_distance) > 1e-6
    
    # We don't know which intersection matches to which point.  But if we check all
    # four options (p1->plus, p2->plus, p1->minus, p2->minus) exactly two distances
    # should be very small
    ones = tf.ones((count,))
    zeros = tf.zeros((count,))
    d1 = tf.sqrt((plus_x - p1x)**2 + (plus_y - p1y)**2)
    d1small = tf.where(tf.less(d1, 1e-6), ones, zeros)
    d2 = tf.sqrt((plus_x - p2x)**2 + (plus_y - p2y)**2)
    d2small = tf.where(tf.less(d2, 1e-6), ones, zeros)
    d3 = tf.sqrt((minus_x - p1x)**2 + (minus_y - p1y)**2)
    d3small = tf.where(tf.less(d3, 1e-6), ones, zeros)
    d4 = tf.sqrt((minus_x - p2x)**2 + (minus_y - p2y)**2)
    d4small = tf.where(tf.less(d4, 1e-6), ones, zeros)
    
    dsum = d1small + d2small + d3small + d4small
    dequals2 = tf.reduce_all(tf.equal(dsum, 2*ones))
    assert session.run(dequals2)

# ------------------------------------------------------------------------------------

def test_one_intersection_case(session, count=100):
    # create the circles
    xc = np.random.uniform(-10.0, 10.0, size=[count,])
    yc = np.random.uniform(-10.0, 10.0, size=[count,])
    radius = np.random.uniform(0.1, 2.0, size=[count,])
    circles = np.stack((xc, yc, radius), axis=1)
    
    # select a point on each circle
    angle = np.random.uniform(0, 2*PI, size=[count,])
    
    x = xc + radius*np.cos(angle)
    y = yc + radius*np.sin(angle)
    p = np.stack((x, y), axis=1)
    
    # generate a second point to make a line tangent to the circle
    second_angle = angle+PI/2.0
    second_x = x + np.cos(second_angle)
    second_y = y + np.sin(second_angle)
    second_p = np.stack((second_x, second_y), axis=1)
    
    # take two random points along the line p second_p
    start_param = np.random.uniform(-10.0, 10.0, size=[count, 1])
    end_param = np.random.uniform(-10.0, 10.0, size=[count, 1])
    
    start = p + start_param * (second_p - p)
    end = p + end_param * (second_p - p)
    
    # build the lines from the above randomly selected points
    lines = np.concatenate((start, end), axis=1)
    
    # extract the intersections.
    # It is a little hard to detect tangency.  And I am using float32 (I think?)
    # for my math, so the default epsilion is a little too low.
    plus, minus = geometry.line_circle_intersect_1to1(lines, circles, epsilion=1e-6)
    
    # check that every intersection in both sets is marked valid.
    all_valid_plus = tf.reduce_all(plus["valid"])
    assert session.run(all_valid_plus)
    all_valid_minus = tf.reduce_all(minus["valid"])
    assert session.run(all_valid_minus)
    
    # check that the intersections are the same, by checking that the distance
    # between both solutions points is small
    plus_x, plus_y = plus["coords"]
    minus_x, minus_y = minus["coords"]
    distance = tf.sqrt((plus_x - minus_x)**2 + (plus_y - minus_y)**2)
    max_distance = tf.reduce_max(distance)
    assert session.run(max_distance) < 1e-6
    
    # we have already checked that the solutions are the same, so now check that one
    # is correct.
    distance = tf.sqrt((plus_x - x)**2 + (plus_y - y)**2)
    max_distance = tf.reduce_max(distance)
    assert session.run(max_distance) < 1e-6
    
def test_zero_intersection_case(session, count=100):
    # create the circles
    xc = np.random.uniform(-10.0, 10.0, size=[count,])
    yc = np.random.uniform(-10.0, 10.0, size=[count,])
    radius = np.random.uniform(0.1, 2.0, size=[count,])
    circles = np.stack((xc, yc, radius), axis=1)
    
    # select a point not on each circle
    angle = np.random.uniform(0, 2*PI, size=[count,])
    
    x = xc + 2.0*radius*np.cos(angle)
    y = yc + 2.0*radius*np.sin(angle)
    p = np.stack((x, y), axis=1)
    
    # generate a second point to make a line tangent to the circle
    second_angle = angle+PI/2.0
    second_x = x + np.cos(second_angle)
    second_y = y + np.sin(second_angle)
    second_p = np.stack((second_x, second_y), axis=1)
    
    # take two random points along the line p second_p
    start_param = np.random.uniform(-10.0, 10.0, size=[count, 1])
    end_param = np.random.uniform(-10.0, 10.0, size=[count, 1])
    
    start = p + start_param * (second_p - p)
    end = p + end_param * (second_p - p)
    
    # build the lines from the above randomly selected points
    lines = np.concatenate((start, end), axis=1)
    
    # extract the intersections.
    plus, minus = geometry.line_circle_intersect_1to1(lines, circles)
    
    # check that every intersection in both sets is marked invalid.
    all_valid_plus = tf.reduce_any(plus["valid"])
    assert not session.run(all_valid_plus)
    all_valid_minus = tf.reduce_any(minus["valid"])
    assert not session.run(all_valid_minus)
    
    
    
    
    
    
