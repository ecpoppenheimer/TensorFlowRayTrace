import math

import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

import tfrt.drawing as drawing
import tfrt.distributions as distributions

PI = math.pi

outline = False
if outline: # generate an outlined square and circle
    # generate the two distributions
    points_per_side = 13
    point_count = 4 * (points_per_side - 1)
    circle_rad = .8
    square_rad = 1

    # make the square
    square_side = np.linspace(-square_rad, square_rad, points_per_side)
    ones_side = np.ones((points_per_side,))
    sq_top = np.stack((square_side, square_rad * ones_side), axis=1)
    sq_bottom = np.stack((square_side, -square_rad * ones_side), axis=1)
    sq_right = np.stack((square_rad * ones_side[1:-1], square_side[1:-1]), axis=1)
    sq_left = np.stack((-square_rad * ones_side[1:-1], square_side[1:-1]), axis=1)
    square = np.concatenate([sq_top, sq_bottom, sq_right, sq_left], axis=0)

    # make the circle
    angles = np.linspace(0, 2*PI, point_count, endpoint=False)
    circle = circle_rad * np.stack((np.cos(angles), np.sin(angles)), axis=1)
else: # generate a filled in square and circle
    points_per_side = 15
    point_count = points_per_side**2
    circle_rad = .8
    square_rad = 1
    
    square = distributions.StaticUniformSquare(square_rad, points_per_side).points.numpy()
    circle = distributions.StaticUniformCircle(point_count, radius=circle_rad).points.numpy()

# shuffle both distributions, to make sure that the map function is working well.
np.random.shuffle(circle)
np.random.shuffle(square)

# set up the plot
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_aspect("equal")
ax_scale = max(circle_rad, square_rad) * 1.15
ax.set_xbound(-ax_scale, ax_scale)
ax.set_ybound(-ax_scale, ax_scale)
drawing.disable_figure_key_commands()

plt.scatter(square[:,0], square[:,1], c="red")
plt.scatter(circle[:,0], circle[:,1], c="blue")

# compute the map
mapped_circle = distributions.transform_map(square, circle)

# set up key commands to swap between the two sets of arrows
arrows = []

def original_map():
    global arrows
    for arrow in arrows:
        arrow.remove()
    arrows = []

    for i in range(point_count):
        sq = square[i]
        cir = circle[i]
        arrows.append(
            plt.arrow(
                cir[0],
                cir[1],
                sq[0] - cir[0],
                sq[1] - cir[1],
                color="green",
                head_width=.05,
                head_length=.05,
                length_includes_head=True
            )
        )
    drawing.redraw_current_figure()
        
def transformed_map():
    global arrows
    for arrow in arrows:
        arrow.remove()
    arrows = []

    for i in range(point_count):
        sq = square[i]
        cir = mapped_circle[i]
        arrows.append(
            plt.arrow(
                cir[0],
                cir[1],
                sq[0] - cir[0],
                sq[1] - cir[1],
                color="green",
                head_width=.05,
                head_length=.05,
                length_includes_head=True
            )
        )
    drawing.redraw_current_figure()
transformed_map()

def on_key(event):
    if event.key == 'a':
        original_map()
    elif event.key == 'd':
        transformed_map()
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()











