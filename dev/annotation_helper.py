import numpy as np
import tensorflow as tf

import tfrt.distributions as distributions
import tfrt.boundaries as boundaries
import tfrt.engine as eng
import tfrt.sources as sources

# build the source rays
beam_points = distributions.StaticUniformBeam(-.5, .5, 3)
angles = distributions.StaticUniformAngularDistribution(-.2, .2, 7)
source = sources.AngularSource((-1.0, 0.0), 0.0, angles, beam_points, [500])

def print_source():
    for key in source.keys():
        print(f"{key}: {source[key].shape}")

print(f"source rays printout")
print_source()
print("------------")

eng.annotation_helper(source, "foo", 1, "x_start")
source.update()

print(f"use annotate helper:")
print_source()
print("------------")

beam_points.sample_count = 5
source.update()

print(f"change beam point count:")
print_source()
print("------------")

angles.sample_count = 2
source.update()

print(f"change angle count:")
print_source()
print("------------")
