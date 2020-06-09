import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.distributions as distributions
import tfrt.drawing as drawing

# generate the base points
x_size = 1
y_size = 1
base_points = distributions.ImageBasePoints(
    "./data/greyblob.png", 
    x_size=x_size, 
    y_size=y_size
)

# set up the plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_aspect("equal")
ax.set_xbound(-1.2*x_size, 1.2*x_size)
ax.set_ybound(-1.2*y_size, 1.2*y_size)
drawing.disable_figure_key_commands()

plt.scatter(base_points.points[:,0], base_points.points[:,1], c="red")
plt.show()
