import pytest


import tfrt


def test_can_construct_arc_drawer_with_no_arcs(ax):
    tfrt.drawing.ArcDrawer(ax)
