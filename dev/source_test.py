import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

import tfrt.sources as sources
import tfrt.drawing as drawing

PI = sources.PI

def on_key(event, session, drawer, source_rays):
    # Message loop called whenever a key is pressed on the figure

    # Extract the center and facing
    center = session.run(source_rays.center)
    facing = session.run(source_rays.facing)
    
    if event.key == "w":
        # move the center up
        center[1] = center[1] + .1
        session.run(
            source_rays.assign_center,
            feed_dict={source_rays.center_placeholder: center}
        )
    elif event.key == "s":
        # move the center down
        center[1] = center[1] - .1
        session.run(
            source_rays.assign_center,
            feed_dict={source_rays.center_placeholder: center}
        )
    elif event.key == "a":
        # move the center left
        center[0] = center[0] - .1
        session.run(
            source_rays.assign_center,
            feed_dict={source_rays.center_placeholder: center}
        )
    elif event.key == "d":
        # move the center right
        center[0] = center[0] + .1
        session.run(
            source_rays.assign_center,
            feed_dict={source_rays.center_placeholder: center}
        )
    elif event.key == "q":
        # rotate source ccw
        facing += .1
        session.run(
            source_rays.assign_facing,
            feed_dict={source_rays.facing_placeholder: facing}
        )
    elif event.key == "e":
        # rotate source cw
        facing -= .1
        session.run(
            source_rays.assign_facing,
            feed_dict={source_rays.facing_placeholder: facing}
        )
        
        

    drawer.rays = session.run(source_rays._rays)
    drawer.draw()
    drawing.redraw_current_figure()

if __name__ == "__main__":
    drawing.disable_figure_key_commands()
    with tf.Session() as session:

        # set up the figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(9, 9))

        # configure axes
        ax.set_aspect("equal")
        ax.set_xbound(-2, 2)
        ax.set_ybound(-2, 2)
        
        # Build the variables that allow the source to be parametric
        center = tf.get_variable("source_center", initializer=(0.0, 0.0))
        facing = tf.get_variable(
            "source_facing",
            initializer=(0.0,)
        )
        
        # build the source rays
        source_rays = sources.StaticSource(
            center,
            (500,),
            7,
            angular_facing=facing,
            angular_lower_cutoff=-PI/4.0,
            angular_upper_cutoff=PI/4.0,
            angular_distribution="uniform",
            beam_lower=-1.0,
            beam_upper=1.0,
            beam_samples=5,
            beam_distribution="uniform",
            name="test_source"
        )
        
        # Store center setting machinery inside source_rays
        source_rays.center = center
        source_rays.center_placeholder = tf.placeholder(tf.float32, shape=(2,), 
            name="center_placeholder")
        source_rays.assign_center = tf.assign(
            source_rays.center,
            source_rays.center_placeholder
        )
        
        source_rays.facing = facing
        source_rays.facing_placeholder = tf.placeholder(tf.float32, shape=(1,), 
            name="facing_placeholder")
        source_rays.assign_facing = tf.assign(
            source_rays.facing,
            source_rays.facing_placeholder
        )
        
        # initialize global variables
        session.run(tf.global_variables_initializer())

        # set up drawer
        drawer = drawing.RayDrawer(
            ax, 
            rays=session.run(source_rays._rays)
        )
        drawer.draw()
        
        # Print the the public helper attributes
        print("angles")
        print(np.degrees(session.run(source_rays._angles)))
        print("angle ranks")
        print(session.run(source_rays._angle_ranks))
        print("beam ranks")
        print(session.run(source_rays._beam_ranks))
        
        # hand over to user
        fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, session, 
            drawer, source_rays
        ))
        plt.show()
