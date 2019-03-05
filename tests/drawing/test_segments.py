import pytest


import tfrt


def test_can_construct_segment_drawer_with_no_segments(ax):
    tfrt.drawing.SegmentDrawer(ax)
