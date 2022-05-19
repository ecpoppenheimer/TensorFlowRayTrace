import numpy as np
import matplotlib.pyplot as plt

import tfrt.distributions as distributions

# parameters
x_range = (-3.0, 3.0)
y_range = (-2.0, 2.0)
x_res = 60
y_res = 40
sample_count = 5000

# define an evaluation grid, and define the density
x = np.linspace(x_range[0], x_range[1], x_res)[:, None]
y = np.linspace(y_range[0], y_range[1], y_res)[None, :]
#density = np.exp(-(x**2/4 + y**2))
density = np.exp(-y**2) * np.abs(np.cos(x))

# set up the plot.  Plot the density in the first panel
fig, axes = plt.subplots(1, 3)
axes[0].imshow(density.T, origin="lower")

# make the cdf
cdf = distributions.CumulativeDensityFunction((x_range, y_range), density)
#cdf = distributions.ArbitraryDistribution(density, (x_range, y_range))

# make a random sample of points, which live in the domain (0, 0) -> (1, 1)
random_sample = np.random.uniform(0.0, 1.0, (sample_count, 2))

# evaluate the cdf on the random sample, and plot into the second pane
mapped_sample = cdf(random_sample)
axes[1].scatter(mapped_sample[:, 0], mapped_sample[:, 1])
axes[1].set_xlim(x_range)
axes[1].set_ylim(y_range)
axes[1].set_aspect("auto")

# evaluate the inverse cdf on the mapped sample, and plot into the third pane
flattened_sample = cdf.icdf(mapped_sample)
axes[2].scatter(flattened_sample[:, 0], flattened_sample[:, 1])
axes[2].set_xlim((-.2, 1.2))
axes[2].set_ylim((-.2, 1.2))
axes[2].set_aspect("auto")

plt.show()
