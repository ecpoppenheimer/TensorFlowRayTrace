import pytest

import tensorflow as tf
import numpy as np

import tfrt.sources as sources

PI = sources.PI

# =====================================================================================
# Beam base point distribution


@pytest.mark.parametrize(
    "distribution_type", [sources.StaticUniformBeam, sources.RandomUniformBeam]
)
@pytest.mark.parametrize(
    "beam_start,beam_end",
    [
        (0.0, 1.0),
        (-1.0, 0.0),
        (1, 2),
        (0, 0),
        pytest.param(1, -1, marks=pytest.mark.xfail),
        pytest.param((-1,), (1,), marks=pytest.mark.xfail),
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
        pytest.param((5, 5), marks=pytest.mark.xfail),
    ],
)
def test_beam_with_constants(
    session, distribution_type, beam_start, beam_end, sample_count
):
    distribution = distribution_type(beam_start, beam_end, sample_count)
    distribution.build()
    result = session.run(distribution.base_points)
    assert distribution.base_points[0].dtype == tf.float64
    if distribution.ranks is not None:
        assert distribution.ranks.dtype == tf.float64
    assert len(result) == 2
    assert result[0].shape == np.reshape(np.floor(sample_count), (1,))
    assert result[1].shape == np.reshape(np.floor(sample_count), (1,))


# -------------------------------------------------------------------------------------

feed_beam_start = tf.placeholder(tf.float64)
feed_beam_end = tf.placeholder(tf.float64)
feed_sample_count = tf.placeholder(tf.int64)


@pytest.mark.parametrize(
    "distribution_type", [sources.StaticUniformBeam, sources.RandomUniformBeam]
)
@pytest.mark.parametrize(
    "beam_start,beam_end",
    [
        (0.0, 1.0),
        (-1.0, 0.0),
        (1, 2),
        (0, 0),
        pytest.param(1, -1, marks=pytest.mark.xfail),
        pytest.param((-1,), (1,), marks=pytest.mark.xfail),
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
        pytest.param((5, 5), marks=pytest.mark.xfail),
    ],
)
def test_beam_with_feeds(
    session, distribution_type, beam_start, beam_end, sample_count
):
    distribution = distribution_type(feed_beam_start, feed_beam_end, feed_sample_count)
    distribution.build()
    result = session.run(
        distribution.base_points,
        feed_dict={
            feed_beam_start: beam_start,
            feed_beam_end: beam_end,
            feed_sample_count: sample_count,
        },
    )
    assert distribution.base_points[0].dtype == tf.float64
    if distribution.ranks is not None:
        assert distribution.ranks.dtype == tf.float64
    assert len(result) == 2
    assert result[0].shape == np.reshape(np.floor(sample_count), (1,))
    assert result[1].shape == np.reshape(np.floor(sample_count), (1,))


# -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "distribution_type", [sources.StaticUniformBeam, sources.RandomUniformBeam]
)
def test_beam_setting_angle(session, distribution_type):
    distribution = distribution_type(-1, 1, 5)
    distribution.central_angle = PI / 4.0
    distribution.build()
    session.run(distribution.base_points)


@pytest.mark.parametrize(
    "distribution_type", [sources.StaticUniformBeam, sources.RandomUniformBeam]
)
def test_beam_setting_angle_many(session, distribution_type):
    distribution = distribution_type(-1, 1, 5, central_angle=-PI / 4.0)
    distribution.central_angle = PI / 4.0
    distribution.central_angle = PI / 2.0
    distribution.build()
    session.run(distribution.base_points)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "distribution_type", [sources.StaticUniformBeam, sources.RandomUniformBeam]
)
def test_beam_setting_angle_fail(session, distribution_type):
    distribution = distribution_type(-1, 1, 5)
    distribution.build()
    distribution.central_angle = PI / 4.0
    session.run(distribution.base_points)


# =====================================================================================
# Aperature base point distribution


@pytest.mark.parametrize(
    "distribution_type",
    [sources.StaticUniformAperaturePoints, sources.RandomUniformAperaturePoints],
)
@pytest.mark.parametrize(
    "start_point,end_point",
    [
        ((0.0, 1.0), (1.0, 1.0)),
        ((0.0, 0.0), (0.0, 0.0)),
        ((1, 1), (-1, -1)),
        pytest.param(1, -1, marks=pytest.mark.xfail),
        pytest.param((-1,), (1,), marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), (1, 2, 3), marks=pytest.mark.xfail),
        pytest.param([[1, 2, 3]], [[1, 2, 3]], marks=pytest.mark.xfail),
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
        pytest.param((5, 5), marks=pytest.mark.xfail),
    ],
)
def test_aperature_with_constants(
    session, distribution_type, start_point, end_point, sample_count
):
    distribution = distribution_type(start_point, end_point, sample_count)
    distribution.build()
    result = session.run(distribution.base_points)
    assert distribution.base_points[0].dtype == tf.float64
    if distribution.ranks is not None:
        assert distribution.ranks.dtype == tf.float64
    assert len(result) == 2
    assert result[0].shape == np.reshape(np.floor(sample_count), (1,))
    assert result[1].shape == np.reshape(np.floor(sample_count), (1,))


# -------------------------------------------------------------------------------------

feed_start_point = tf.placeholder(tf.float64)
feed_end_point = tf.placeholder(tf.float64)


@pytest.mark.parametrize(
    "distribution_type",
    [sources.StaticUniformAperaturePoints, sources.RandomUniformAperaturePoints],
)
@pytest.mark.parametrize(
    "start_point,end_point",
    [
        ((0.0, 1.0), (1.0, 1.0)),
        ((0.0, 0.0), (0.0, 0.0)),
        ((1, 1), (-1, -1)),
        pytest.param(1, -1, marks=pytest.mark.xfail),
        pytest.param((-1,), (1,), marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), (1, 2, 3), marks=pytest.mark.xfail),
        pytest.param([[1, 2, 3]], [[1, 2, 3]], marks=pytest.mark.xfail),
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
        pytest.param((5, 5), marks=pytest.mark.xfail),
    ],
)
def test_aperature_with_feeds(
    session, distribution_type, start_point, end_point, sample_count
):
    distribution = distribution_type(
        feed_start_point, feed_end_point, feed_sample_count
    )
    distribution.build()
    result = session.run(
        distribution.base_points,
        feed_dict={
            feed_start_point: start_point,
            feed_end_point: end_point,
            feed_sample_count: sample_count,
        },
    )
    assert distribution.base_points[0].dtype == tf.float64
    if distribution.ranks is not None:
        assert distribution.ranks.dtype == tf.float64
    assert len(result) == 2
    assert result[0].shape == np.reshape(np.floor(sample_count), (1,))
    assert result[1].shape == np.reshape(np.floor(sample_count), (1,))
