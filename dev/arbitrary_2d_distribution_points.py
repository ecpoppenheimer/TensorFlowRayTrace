import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.special import hermite
import imageio

import tfrt.distributions as dist

PI = math.pi

grid_limits = np.array(((-3, 3, 50), (-3, 3, 50)))
hermite_ij = (0, 2)
point_cloud_samples = 50000

# define the density function that we will use for testing
def density_function(x, y):
    #return np.exp(-(x*x)) * np.abs(np.cos(PI*y/2))
    #return np.exp(-(x*x + y*y))
    #return np.abs(np.sin(y)*np.sin(x))
    #return np.maximum(1 - (x*x + y*y), 1e-6)
    #return (hermite(hermite_ij[0])(x) * hermite(hermite_ij[1])(y))**2 * np.exp(-(x*x + y*y)) + 1e-10
    return x + np.abs(x) + 1e-10

# define the coordinate grid that will be used for evaluation and plotting
x, y = np.meshgrid(
    np.linspace(*grid_limits[0,:2], grid_limits[0,2]),
    np.linspace(*grid_limits[1,:2], grid_limits[1,2])
)
        
point_cloud = dist.ArbitraryDistribution(density_function, grid_limits)
#point_cloud = dist.ArbitraryDistribution("./data/full_hex_blob.png", grid_limits[:,:2])
rand_x = np.random.uniform(grid_limits[0,0], grid_limits[0,1], point_cloud_samples)
rand_y = np.random.uniform(grid_limits[1,0], grid_limits[1,1], point_cloud_samples)
sampled_x, sampled_y = point_cloud(rand_x, rand_y)
    
# set up the figure
gs = mpl.gridspec.GridSpec(3, 2)

fig = plt.figure(figsize=(8, 12))
plt.tight_layout()
distribution_ax = fig.add_subplot(gs[0, 0])
distribution_ax.set_aspect("equal")
histo_ax = fig.add_subplot(gs[0, 1])
histo_ax.set_aspect("equal")
scatter_ax = fig.add_subplot(gs[1:, :])
scatter_ax.set_aspect("equal")

# plot the raw, analytic distribution
distribution_ax.contourf(x, y, density_function(x, y))

# plot the x_quantile
histo_ax.hist2d(sampled_x, sampled_y, range=grid_limits[:,:2], bins=grid_limits[:,2], vmin=0)

# scatter plot the point cloud
scatter_ax.scatter(sampled_x, sampled_y)
scatter_ax.set_xbound(grid_limits[0,:2])
scatter_ax.set_ybound(grid_limits[1,:2])

plt.show()

#with np.printoptions(precision=3, suppress=True):
