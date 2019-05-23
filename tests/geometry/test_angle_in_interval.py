import pytest
import math

import tensorflow as tf
import numpy as np

import tfrt.geometry as geometry

PI = np.array(math.pi, dtype=np.float64)

float_formatter = lambda x: "%.10f" % x
np.set_printoptions(formatter={"float_kind": float_formatter})


def generate_angles_in_interval(start, stop, count):
    stop = tf.where(tf.less(stop, start), stop + 2 * PI, stop)
    angles = tf.linspace(start, stop, count)
    angles = tf.where(tf.greater(angles, PI), angles - 2 * PI, angles)
    return angles


def generate_angles_outside_interval(start, stop, count):
    start = tf.where(tf.equal(stop, start), start + 2 * PI, start)
    return generate_angles_in_interval(stop, start, count + 2)[1:-1]


@pytest.mark.parametrize(
    "start",
    [0.0, 0.00001, 1.0 / 4.0, 0.99999, 1.0, -0.00001, -1.0 / 4.0, -0.99999, -1.0],
)
@pytest.mark.parametrize(
    "stop",
    [0.0, 0.00001, 1.0 / 4.0, 0.99999, 1.0, -0.00001, -1.0 / 4.0, -0.99999, -1.0],
)
def test_interval(session, start, stop, count=11):
    start = start * PI
    stop = stop * PI
    print(f"start: {math.degrees(start)}, stop: {math.degrees(stop)}")

    # generate the angles, paying attention to the special cases where one of the
    # sets must be empty
    included_angles = generate_angles_in_interval(start, stop, count)
    excluded_angles = generate_angles_outside_interval(start, stop, count)
    if start == -PI and stop == PI:
        excluded_angles = tf.zeros([0], dtype=tf.float64)
    elif start == PI and stop == -PI:
        included_angles = tf.zeros([0], dtype=tf.float64)

    is_included = geometry.angle_in_interval(included_angles, start, stop)
    is_excluded = geometry.angle_in_interval(excluded_angles, start, stop)

    print("===================")
    print("included angles:")
    print(np.degrees(session.run(included_angles)))
    print(session.run(is_included))

    print("===================")
    print("excluded angles:")
    print(np.degrees(session.run(excluded_angles)))
    print(session.run(is_excluded))

    all_included = tf.reduce_all(is_included)
    all_excluded = tf.logical_not(tf.reduce_any(is_excluded))

    assert session.run(all_included)
    assert session.run(all_excluded)


# with tf.Session() as session:
#    test_interval(session, .99999, -.99999, 11)
