import pytest

import tensorflow as tf
import numpy as np

import tfrt.sources as sources

PI = sources.PI

@pytest.mark.parametrize("start_points", [
    sources.StaticUniformAperaturePoints((0, -1), (0, 1), 1),
    sources.StaticUniformAperaturePoints((0, -1), (0, 1), 5),
    pytest.param(sources.ManualBasePointDistribution(
        [[0, 1, 2], [0, 1, 2]],
        [[0, 1, 2], [0, 1, 2]]
        ), marks=pytest.mark.xfail
    ),
    pytest.param(sources.ManualBasePointDistribution(
        [[1, 2, 3, 4, 5]],
        [[1, 2, 3, 4, 5]]
        ), marks=pytest.mark.xfail
    )
])
@pytest.mark.parametrize("end_points", [
    sources.StaticUniformAperaturePoints((1, -1), (1, 1), 1),
    sources.StaticUniformAperaturePoints((1, -1), (1, 1), 5),
    pytest.param(sources.ManualBasePointDistribution(
        [[4, 5, 6], [4, 5, 6]],
        [[4, 5, 6], [4, 5, 6]]
        ), marks=pytest.mark.xfail
    ),
    pytest.param(sources.ManualBasePointDistribution(
        [[1, 2, 3, 4, 5]],
        [[1, 2, 3, 4, 5]]
        ), marks=pytest.mark.xfail
    )
])
@pytest.mark.parametrize("wavelengths", [
    np.linspace(.4, .5, 1),
    np.linspace(1, 2, 5),
    pytest.param(0, marks=pytest.mark.xfail),
    pytest.param(np.array([[1,1]]), marks=pytest.mark.xfail)
])
def test_dense_aperature_source(session, start_points, end_points, wavelengths):
    source = sources.AperatureSource(start_points, end_points, wavelengths, dense=True)
    start_result = session.run(start_points.base_points)
    end_result = session.run(end_points.base_points)
    rays_result = session.run(source.rays)
    assert source.rays.dtype == tf.float64
    assert rays_result.shape[1] == 5
    assert rays_result.shape[0] == (wavelengths.shape[0] * start_result[0].shape[0] * 
        end_result[0].shape[0])

#--------------------------------------------------------------------------------------

@pytest.mark.parametrize("wavelengths,start_points,end_points", [
    (
        np.array([.4]),
        sources.StaticUniformAperaturePoints((0, -1), (0, 1), 1),
        sources.StaticUniformAperaturePoints((1, -1), (1, 1), 1)
    ),
    (
        np.linspace(.4, .5, 5),
        sources.StaticUniformAperaturePoints((0, -1), (0, 1), 5),
        sources.StaticUniformAperaturePoints((1, -1), (1, 1), 5)
    ),
    pytest.param(
        np.linspace(.4, .5, 5),
        sources.StaticUniformAperaturePoints((0, -1), (0, 1), 4),
        sources.StaticUniformAperaturePoints((1, -1), (1, 1), 5),
        marks=pytest.mark.xfail
    ),
    pytest.param(
        np.linspace(.4, .5, 5),
        sources.StaticUniformAperaturePoints((0, -1), (0, 1), 5),
        sources.StaticUniformAperaturePoints((1, -1), (1, 1), 4),
        marks=pytest.mark.xfail
    )
])
def test_undense_angular_source(session, start_points, end_points, wavelengths):
    source = sources.AperatureSource(start_points, end_points, wavelengths, dense=False)
    rays_result = session.run(source.rays)
    rays_result = session.run(source.rays)
    assert source.rays.dtype == tf.float64
    assert rays_result.shape[1] == 5
    assert rays_result.shape[0] == wavelengths.shape[0]

