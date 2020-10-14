import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tfrt.analyze as analyze
import tfrt.distributions as distributions

x_start = -4.0
x_end = 4.0
y_start = -4.0
y_end = 4.0
x_bins = 25
y_bins = 25

sample_domain = ((x_start, x_end, x_bins), (y_start, y_end, y_bins))
comparator_domain = np.array(((x_start, x_end), (y_start, y_end)), dtype=np.float64)
sample_count = 100000

def sample_density_function(x, y):
    #return tf.exp(-(x*x+y*y))
    return tf.ones_like(x)
    
def goal_density_function(x, y):
    #return tf.exp(-(x*x+y*y))
    return tf.ones_like(x)
    
generator = distributions.ArbitraryDistribution(sample_density_function, sample_domain)
sample_x, sample_y = generator(
    tf.random.uniform((sample_count,), x_start, x_end, dtype=tf.float64),
    tf.random.uniform((sample_count,), y_start, y_end, dtype=tf.float64)
)

comparator = analyze.DistributionDifferential(
    goal_density_function,
    comparator_domain,
    oob_penalty=lambda x: 0.005*x*x + 0.001*tf.ones_like(x),
    #oob_penalty=lambda x: tf.zeros_like(x),
    #oob_penalty=None,
    x_bins=x_bins,
    y_bins=y_bins
)
print("overlap = ", comparator(sample_x, sample_y))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(
    comparator._eval_grid_x, comparator._eval_grid_y, comparator._goal, color="red",
    rstride=1, cstride=1
)
ax.plot_surface(
    comparator._eval_grid_x, comparator._eval_grid_y, comparator.saved_histo, color="blue",
    rstride=1, cstride=1
)

plt.show()
