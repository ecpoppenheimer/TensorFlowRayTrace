import pytest

import matplotlib.pyplot as plt

import tfrt


@pytest.fixture(scope="function")
def ax():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    yield ax
    plt.close(fig)


@pytest.mark.parametrize("units", ["um", "nm"])
def test_valid_units_for_ray_drawer(ax, units):
    tfrt.drawing.RayDrawer(ax, units=units)


@pytest.mark.parametrize("units", ["foo", "micron", "inch", "mile", "km", "m", "eric"])
def test_invalid_units_for_ray_drawer(ax, units):
    with pytest.raises(ValueError):
        tfrt.drawing.RayDrawer(ax, units=units)
