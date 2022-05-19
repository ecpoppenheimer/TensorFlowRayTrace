import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tfrt.distributions as distributions

# parameters
x_range = (-1.5, 1.5)
y_range = (-1.25, 1.25)
x_res = 120
y_res = 100
sample_count = 400

# define the evaluation grid
x = np.linspace(x_range[0], x_range[1], x_res)[:, None]
y = np.linspace(y_range[0], y_range[1], y_res)[None, :]

# define the density for the given
given_density = np.zeros((x_res, y_res))
given_density[np.sqrt(x**2+y**2) < 1] = 1.0

# define the density for the goal
theta = np.arctan2(y, x)
r = 4 + np.cos(5 * theta) + .15 * np.cos(10 * theta)
goal_density = np.zeros((x_res, y_res))
goal_density[np.sqrt(x**2+y**2) < r / 4] = 1.0

# set up the plot.  Plot the density in the first panel
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=(1, 1, 2))

# lot the given density
ax00 = fig.add_subplot(gs[0, 0])
ax00.imshow(given_density.T, origin="lower")
ax00.set_aspect("equal")
ax00.set_title("Given Input")

# make the cdf
given_cdf = distributions.CumulativeDensityFunction((x_range, y_range), given_density)

# make a random sample of points, which live in the domain (0, 0) -> (1, 1)
random_sample = np.random.uniform(0.0, 1.0, (sample_count, 2))

# evaluate the cdf on the random sample, and plot into the second pane
mapped_sample = given_cdf(random_sample)
ax10 = fig.add_subplot(gs[1, 0])
ax10.scatter(mapped_sample[:, 0], mapped_sample[:, 1])
ax10.set_xlim(x_range)
ax10.set_ylim(y_range)
ax10.set_aspect("equal")
ax10.set_title("Sampled Input")

# plot the goal density
ax01 = fig.add_subplot(gs[0, 1])
ax01.imshow(goal_density.T, origin="lower")
ax01.set_aspect("equal")
ax01.set_title("Desired Output")

# make and sample the goal cdf
goal_cdf = distributions.CumulativeDensityFunction((x_range, y_range), goal_density)
mapped_goal = goal_cdf(random_sample)

# plot the goal sample
ax11 = fig.add_subplot(gs[1, 1])
ax11.scatter(mapped_goal[:, 0], mapped_goal[:, 1])
ax11.set_xlim(x_range)
ax11.set_ylim(y_range)
ax11.set_aspect("equal")
ax11.set_title("Sampled Goal")

# plot the transformation
ax2 = fig.add_subplot(gs[:,2])
ax2.set_title("Transformation")
ax2.scatter(mapped_sample[:, 0], mapped_sample[:, 1], color="blue")
ax2.scatter(mapped_goal[:, 0], mapped_goal[:, 1], color="green")
ax2.set_xlim(x_range)
ax2.set_ylim(y_range)
ax2.set_aspect("equal")

# draw arrows between the two sets of points
for i in range(sample_count):
    ax2.arrow(
        mapped_sample[i, 0],
        mapped_sample[i, 1],
        mapped_goal[i, 0] - mapped_sample[i, 0],
        mapped_goal[i, 1] - mapped_sample[i, 1],
        color="red",
        head_width=.03,
        head_length=.05,
        head_starts_at_zero=False
    )

plt.show()
