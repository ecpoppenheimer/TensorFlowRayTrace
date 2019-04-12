import pytest

import tensorflow as tf
import numpy as np

import tfrt.sources as sources

PI = sources.PI


@pytest.fixture(scope="module")
def sample_angular_distribution():
    return sources.StaticUniformAngularDistribution(-1, 1, 5)


@pytest.fixture(scope="module")
def sample_base_point_distribution():
    return sources.StaticUniformBeam(-1, 1, 5)


@pytest.fixture(scope="module")
def sample_wavelengths():
    return np.linspace(0.4, 0.5, 5)


feed_center = tf.placeholder(tf.float64)
feed_central_angle = tf.placeholder(tf.float64)

# -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "center",
    [
        (0, 0),
        (1.0, 1.0),
        pytest.param(0, marks=pytest.mark.xfail),
        pytest.param((0,), marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), marks=pytest.mark.xfail),
        pytest.param([[1, 1]], marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize(
    "central_angle",
    [
        0,
        PI / 2.0,
        pytest.param((0,), marks=pytest.mark.xfail),
        pytest.param([[1, 1]], marks=pytest.mark.xfail),
    ],
)
def test_simple_params_constant(
    session,
    center,
    central_angle,
    sample_angular_distribution,
    sample_base_point_distribution,
    sample_wavelengths,
):
    source = sources.AngularSource(
        center,
        central_angle,
        sample_angular_distribution,
        sample_base_point_distribution,
        sample_wavelengths,
        dense=True,
    )
    result = session.run(source.rays)
    assert source.rays.dtype == tf.float64
    assert result.shape == (125, 5)


# -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "center",
    [
        (0, 0),
        (1.0, 1.0),
        pytest.param(0, marks=pytest.mark.xfail),
        pytest.param((0,), marks=pytest.mark.xfail),
        pytest.param((1, 2, 3), marks=pytest.mark.xfail),
        pytest.param([[1, 1]], marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize(
    "central_angle",
    [
        0,
        PI / 2.0,
        pytest.param((0,), marks=pytest.mark.xfail),
        pytest.param([[1, 1]], marks=pytest.mark.xfail),
    ],
)
def test_simple_params_feed(
    session,
    center,
    central_angle,
    sample_angular_distribution,
    sample_base_point_distribution,
    sample_wavelengths,
):
    source = sources.AngularSource(
        feed_center,
        feed_central_angle,
        sample_angular_distribution,
        sample_base_point_distribution,
        sample_wavelengths,
        dense=True,
    )
    result = session.run(
        source.rays, feed_dict={feed_center: center, feed_central_angle: central_angle}
    )
    assert source.rays.dtype == tf.float64
    assert result.shape == (125, 5)


# -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "angular_distribution",
    [
        sources.StaticUniformAngularDistribution(-1, 1, 1),
        sources.StaticUniformAngularDistribution(-1, 1, 5),
        pytest.param(
            sources.ManualAngularDistribution([[0, 1, 2], [0, 1, 2]]),
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.parametrize(
    "base_point_distribution",
    [
        sources.StaticUniformBeam(-1, 1, 1),
        sources.StaticUniformBeam(-1, 1, 5),
        pytest.param(
            sources.ManualBasePointDistribution(
                [[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]
            ),
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            sources.ManualBasePointDistribution([[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5]]),
            marks=pytest.mark.xfail,
        ),
    ],
)
@pytest.mark.parametrize(
    "wavelengths",
    [
        np.linspace(0.4, 0.5, 1),
        np.linspace(1, 2, 5),
        pytest.param(0, marks=pytest.mark.xfail),
        pytest.param(np.array([[1, 1]]), marks=pytest.mark.xfail),
    ],
)
def test_dense_angular_source(
    session, angular_distribution, base_point_distribution, wavelengths
):
    source = sources.AngularSource(
        (0.0, 0.0),
        0.0,
        angular_distribution,
        base_point_distribution,
        wavelengths,
        dense=True,
    )
    angle_result = session.run(angular_distribution.angles)
    base_point_result = session.run(base_point_distribution.base_points)
    result = session.run(source.rays)
    assert source.rays.dtype == tf.float64
    assert result.shape[1] == 5
    assert result.shape[0] == (
        wavelengths.shape[0] * angle_result.shape[0] * base_point_result[0].shape[0]
    )


# -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wavelengths,angular_distribution,base_point_distribution",
    [
        (
            np.array([0.4]),
            sources.StaticUniformAngularDistribution(-1, 1, 1),
            sources.StaticUniformBeam(-1, 1, 1),
        ),
        (
            np.linspace(0.4, 0.5, 5),
            sources.StaticUniformAngularDistribution(-1, 1, 5),
            sources.StaticUniformBeam(-1, 1, 5),
        ),
        pytest.param(
            np.linspace(0.4, 0.5, 5),
            sources.StaticUniformAngularDistribution(-1, 1, 4),
            sources.StaticUniformBeam(-1, 1, 4),
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_undense_angular_source(
    session, angular_distribution, base_point_distribution, wavelengths
):
    source = sources.AngularSource(
        (0.0, 0.0),
        0.0,
        angular_distribution,
        base_point_distribution,
        wavelengths,
        dense=False,
    )
    result = session.run(source.rays)
    assert source.rays.dtype == tf.float64
    assert result.shape[1] == 5
    assert result.shape[0] == wavelengths.shape[0]
