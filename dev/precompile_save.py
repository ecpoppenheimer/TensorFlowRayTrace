import numpy as np
import tensorflow as tf

import tfrt.sources as sources
import tfrt.distributions as distributions
from tfrt.drawing import RAINBOW_6

angles = distributions.StaticUniformAngularDistribution(-1, 1, 11)
base_points = distributions.StaticUniformBeam(-1, 1, 9)
source = sources.AngularSource(
    2,
    (0, 0),
    0,
    angles,
    base_points,
    RAINBOW_6,
    dense=True
)

print("source printout:")
for key, value in source.items():
    print(f"{key}: {value.shape}")
    
pcs = sources.PrecompiledSource(source)
print("precompiled source printout:")
for key, value in pcs.items():
    print(f"{key}: {value.shape}")
    
pcs.save("./data/precompiled_source_test.dat")
