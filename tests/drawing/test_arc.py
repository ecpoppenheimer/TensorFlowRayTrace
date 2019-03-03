import pytest


import tfrt


def test_can_instantiate_arc_drawer_with_no_arc_data(ax):
    tfrt.drawing.ArcDrawer(ax)
