import pytest

import tensorflow as tf
import numpy as np

import tfrt.sources as sources

PI = sources.PI


@pytest.mark.parametrize(
    "distribution_type",
    [
        sources.StaticUniformAngularDistribution,
        sources.RandomUniformAngularDistribution,
        sources.StaticLambertianAngularDistribution,
        sources.RandomLambertianAngularDistribution,
    ],
)
@pytest.mark.parametrize(
    "min_angle,max_angle",
    [
        (0, PI / 4.0),
        (-PI / 2.0, PI / 2.0),
        pytest.param(PI / 4.0, 0, marks=pytest.mark.xfail),
        pytest.param(-2 * PI, 2 * PI, marks=pytest.mark.xfail),
        pytest.param([-1], [1], marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize(
    "sample_count",
    [
        1,
        1.5,
        10,
        pytest.param(0, marks=pytest.mark.xfail),
        pytest.param(-1, marks=pytest.mark.xfail),
        pytest.param(-10, marks=pytest.mark.xfail),
        pytest.param([5, 5], marks=pytest.mark.xfail),
    ],
)
def test_with_constants(session, distribution_type, min_angle, max_angle, sample_count):
    distribution = distribution_type(min_angle, max_angle, sample_count)
    distribution.build()
    result = session.run(distribution.angles)
    assert distribution.angles.dtype == tf.float64
    if distribution.ranks is not None:
        assert distribution.ranks.dtype == tf.float64
    assert result.shape == np.reshape(np.floor(sample_count), (1,))


# =====================================================================================

feed_min_angle = tf.placeholder(tf.float64)
feed_max_angle = tf.placeholder(tf.float64)
feed_sample_count = tf.placeholder(tf.int64)


@pytest.mark.parametrize(
    "distribution_type",
    [
        sources.StaticUniformAngularDistribution,
        sources.RandomUniformAngularDistribution,
        sources.StaticLambertianAngularDistribution,
        sources.RandomLambertianAngularDistribution,
    ],
)
@pytest.mark.parametrize(
    "min_angle,max_angle",
    [
        (0, PI / 4.0),
        (-PI / 2.0, PI / 2.0),
        pytest.param(PI / 4.0, 0, marks=pytest.mark.xfail),
        pytest.param(-2 * PI, 2 * PI, marks=pytest.mark.xfail),
        pytest.param([-1], [1], marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize(
    "sample_count",
    [
        1,
        1.5,
        10,
        pytest.param(0, marks=pytest.mark.xfail),
        pytest.param(-1, marks=pytest.mark.xfail),
        pytest.param(-10, marks=pytest.mark.xfail),
        pytest.param([5, 5], marks=pytest.mark.xfail),
    ],
)
def test_with_feeds(session, distribution_type, min_angle, max_angle, sample_count):
    distribution = distribution_type(feed_min_angle, feed_max_angle, feed_sample_count)
    distribution.build()
    result = session.run(
        distribution.angles,
        feed_dict={
            feed_min_angle: min_angle,
            feed_max_angle: max_angle,
            feed_sample_count: sample_count,
        },
    )
    assert result.shape == np.reshape(np.floor(sample_count), (1,))
