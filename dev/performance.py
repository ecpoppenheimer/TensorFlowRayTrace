from math import pi as PI
import timeit

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

import tfrt.distributions as distributions
import tfrt.boundaries as boundaries
import tfrt.drawing as drawing
import tfrt.materials as materials
import tfrt.engine as eng
import tfrt.sources as sources
import tfrt.operation as op

if __name__ == "__main__":
    # build the segment boundary
    segment_points = distributions.StaticUniformAperaturePoints(
        (.1, -2.),
        (.1, 2.),
        10
    )        
    segments = boundaries.ManualSegmentBoundary()
    def build_segs(pts=segment_points):
        return (
            pts.base_points_x[:-1],
            pts.base_points_y[:-1],
            pts.base_points_x[1:],
            pts.base_points_y[1:]
        )
    segments.update_function = build_segs
    segments.update_handles.append(segment_points.update)
    eng.annotation_helper(segments, "mat_in", 1, "x_start", dtype=tf.int64)
    eng.annotation_helper(segments, "mat_out", 0, "x_start", dtype=tf.int64)
    
    # build the target
    target = boundaries.ManualSegmentBoundary()
    target.feed_segments(np.array([[10, -5, 10, 5]], dtype=np.float64))
    target.frozen=True
    
    # build the source rays
    beam_points = distributions.StaticUniformBeam(-1.5, 1.5, 10)
    angles = distributions.StaticUniformAngularDistribution(0, 0, 1)
    source = sources.AngularSource(
        (-1.0, 0.0), 0.0, angles, beam_points, drawing.RAINBOW_6
    )
    
    # build the system
    system = eng.OpticalSystem2D()
    system.optical_segments = [segments]
    system.sources = [source]
    system.target_segments = [target]
    system.materials = [
        {"n": materials.vacuum},
        {"n": materials.acrylic}
    ]
    
    trace_engine = eng.OpticalEngine(
        2,
        [op.StandardReaction()],
        simple_ray_inheritance={"angle_ranks", "wavelength"}
    )
    trace_engine.optical_system = system
    system.update()
    trace_engine.validate_system()

    def test_trace():
        system.update()
        trace_engine.ray_trace(2)
        trace_engine.clear_ray_history()
    
    # seems like the first trace takes much longer
    test_trace()
    
    # test various ammounts of rays
    for rays, segs in ((10, 11), (10000, 11), (10, 1001), (10000, 1001)):
        beam_points.sample_count = rays
        segment_points.sample_count = segs
        system.update()
        time = timeit.timeit(test_trace, number=20)
        ray_count = system.sources["x_start"].shape[0]
        seg_count = system.optical_segments["x_start"].shape[0]
        print(f"Tracing {ray_count} rays with {seg_count} optical segments 20 times took "
            f"{time} seconds. (average of {time/20.0} seconds per trace.)"
        )
    
    
    plt.show()
    
    
    
    
