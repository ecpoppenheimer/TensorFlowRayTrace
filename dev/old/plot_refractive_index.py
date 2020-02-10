import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

import tfrt.materials as materials

if __name__ == "__main__":
    # set up the figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))

    # evaluate the refractive index data
    min_wavelength = 310
    max_wavelength = 4600
    pointCount = 200

    x = np.linspace(min_wavelength, max_wavelength, pointCount)
    wavelengths = tf.constant(x, dtype=tf.float64)
    n = materials.soda_lime(wavelengths)

    with tf.Session() as session:
        y = session.run(n)

    print(y)
    plt.plot(x, y)
    plt.show()
