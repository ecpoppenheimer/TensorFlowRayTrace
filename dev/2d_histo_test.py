import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tfrt.analyze as analyze

x = .5*tf.random.normal((10000,))
y = tf.random.normal((10000,))
x_bins = 10
y_bins = 20
limits = np.array(((-5, 5), (-5, 5)))

h = analyze.histogram2D(x, y, limits, x_bins=x_bins, y_bins=y_bins, dtype=tf.float32)
print("histo dtype: ", h.dtype)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_aspect("equal")
x_grid, y_grid = np.meshgrid(
    np.linspace(limits[0,0], limits[0,1], x_bins),
    np.linspace(limits[1,0], limits[1,1], y_bins)
)
ax.pcolormesh(x_grid, y_grid, h)
print("h shape: ", h.shape)
plt.show()
