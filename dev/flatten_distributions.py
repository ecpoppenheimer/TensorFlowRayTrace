import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.distributions as dist

PI = math.pi

grid_limits = ((-3, 3, 50), (-3, 3, 50))
hermite_ij = (0, 2)
point_cloud_samples = 50000
arrow_plot_samples = 200

def density_function(x, y):
    return np.exp(-(x*x)) * np.abs(np.cos(PI*y/2))
    #return np.exp(-(x*x + y*y))
    #return np.abs(np.sin(y)*np.sin(x))
    #return np.maximum(1 - (x*x + y*y), 1e-6)
    #return x + np.abs(x) + 1e-10
    
point_cloud = dist.ArbitraryDistribution(density_function, grid_limits)

# set up a mpl plot
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))
ax1.set_aspect("equal")
ax2.set_aspect("equal")
ax3.set_aspect("equal")
ax4.set_aspect("equal")
ax5.set_aspect("equal")
ax6.set_aspect("equal")

# plot the density function
x_grid = np.linspace(*grid_limits[0])
y_grid = np.linspace(*grid_limits[1])
x_grid, y_grid = np.meshgrid(x_grid, y_grid)
plt.sca(ax1)
plt.contourf(x_grid, y_grid, density_function(x_grid, y_grid))
plt.title("input density function")

# plot the input as a histogram
uniform_x = np.random.uniform(*grid_limits[0][:-1], point_cloud_samples)
uniform_y = np.random.uniform(*grid_limits[1][:-1], point_cloud_samples)
sampled_x, sampled_y = point_cloud(uniform_x, uniform_y)

plt.sca(ax2)
plt.hist2d(
    sampled_x,
    sampled_y,
    bins=(grid_limits[0][2], grid_limits[1][2])
)
plt.title("input distribution histogram")

# plot the input map
uniform_x = np.random.uniform(*grid_limits[0][:-1], arrow_plot_samples)
uniform_y = np.random.uniform(*grid_limits[1][:-1], arrow_plot_samples)
test_x, test_y = point_cloud(uniform_x, uniform_y)

plt.sca(ax3)
plt.scatter(test_x, test_y, s=16.0, color="red")
plt.scatter(uniform_x, uniform_y, s=16.0, color="blue")
for i in range(arrow_plot_samples):
    plt.arrow(
        uniform_x[i], 
        uniform_y[i], 
        test_x[i] - uniform_x[i],
        test_y[i] - uniform_y[i],
        length_includes_head=True,
        color="green",
        head_width=.2
        )
plt.title("input map")

# plot a histogram of the flattened samples
plt.sca(ax4)
flat_samples_x, flat_samples_y = dist.flatten_distribution(sampled_x, sampled_y, grid_limits)
plt.hist2d(
    flat_samples_x,
    flat_samples_y,
    bins=(grid_limits[0][2], grid_limits[1][2])
)
plt.title("flattened output histogram")

# plot the inverse of input map, but re-computed using flatten_distribution
flat_test_x, flat_test_y = dist.flatten_distribution(test_x, test_y, grid_limits)
# re-map flat_test from (0,1) to the grid limits
flat_test_x -= 0.5
flat_test_x *= grid_limits[0][1] - grid_limits[0][0]
flat_test_y -= 0.5
flat_test_y *= grid_limits[1][1] - grid_limits[1][0]

plt.sca(ax6)
plt.scatter(flat_test_x, flat_test_y, s=16.0, color="red")
plt.scatter(test_x, test_y, s=16.0, color="blue")
for i in range(arrow_plot_samples):
    plt.arrow(
        test_x[i], 
        test_y[i], 
        flat_test_x[i] - test_x[i],
        flat_test_y[i] - test_y[i],
        length_includes_head=True,
        color="green",
        head_width=.15
        )
plt.title("output map")

# plot a scatter plot of a subset of flat_samples
plt.sca(ax5)
plt.scatter(flat_samples_x[:500], flat_samples_y[:500], s=16.0, color="red")
plt.title("flattened sample")

plt.show()
