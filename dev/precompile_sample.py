import tensorflow as tf
import numpy as np

import tfrt.sources as sources
import tfrt.distributions as distributions
import tfrt.drawing as drawing

PI = sources.PI

source = sources.PrecompiledSource(2, do_downsample=False)

angles = distributions.RandomUniformAngularDistribution(-PI/4, PI/4, 5)
sampling_source = sources.PointSource(2, (0, 0), 0, angles, [drawing.YELLOW])

samples = [sampling_source.snapshot() for i in range(5)]
source.from_samples(samples)

for field, item in source.items():
    print(f"{field}: {item}")
