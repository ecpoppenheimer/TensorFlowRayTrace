import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.distributions as dist

gs = mpl.gridspec.GridSpec(3, 2)

fig = plt.figure(figsize=(12, 10))
plt.tight_layout()
histo_ax = fig.add_subplot(gs[0, 0])
histo_ax.set_aspect("equal")
slice_ax = fig.add_subplot(gs[0, 1])
scatter_ax = fig.add_subplot(gs[1:, :])
scatter_ax.set_aspect("equal")

#mesh = dist.RandomUniformSquare(2, 251)
#mesh = dist.StaticUniformSquare(2, 51)
#mesh = dist.StaticUniformCircle(10000)
mesh = dist.RandomUniformCircle(10000)
x_points, y_points = tf.unstack(mesh.points, axis=1)

# plot the histogram
h = histo_ax.hist2d(x_points, y_points, bins=19, vmin=0)

# plot a slice of the histogram through the center
slice_ax.plot(h[0][10])
slice_ax.set_ylim(bottom=0)

# plot the points themselves
scatter_ax.scatter(x_points, y_points)

plt.show()
