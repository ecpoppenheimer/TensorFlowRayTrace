import pytest

import matplotlib.pyplot as plt

import tfrt


def test_can_construct_ray_drawer_with_no_rays(ax):
    tfrt.drawing.RayDrawer(ax)


@pytest.mark.parametrize("units", ["um", "nm"])
def test_valid_units_for_ray_drawer(ax, units):
    tfrt.drawing.RayDrawer(ax, units=units)


@pytest.mark.parametrize("units", ["foo", "micron", "inch", "mile", "km", "m", "eric"])
def test_invalid_units_for_ray_drawer(ax, units):
    with pytest.raises(ValueError):
        tfrt.drawing.RayDrawer(ax, units=units)
