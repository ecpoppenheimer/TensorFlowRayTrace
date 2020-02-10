import numpy as np
import tensorflow as tf
import math

import tfrt.geometry as geometry

PI = geometry.PI

with tf.Session() as session:
    count = 33
    angles = np.linspace(-4 * PI, 4 * PI, count)
    start = np.full_like(angles, PI / 4.0)
    end = np.full_like(angles, 3 * PI / 2.0)

    answer = geometry.angle_in_interval(angles, start, end)

    answer_result = session.run(answer)
    for a, s, e, b in zip(angles, start, end, answer_result):
        print(
            f"angle {math.degrees(a):.1f} ({math.degrees(a) % 360:.1f}) in "
            "{math.degrees(s)} -> {math.degrees(e)}: {b}"
        )

    print("============")
    angles = [7 * PI / 2.0, 3 * PI / 2.0, -1 * PI / 2.0, -5 * PI / 2.0]
    modded = tf.floormod(angles, 2 * PI)
    modded_result = session.run(modded)

    for a, m in zip(angles, modded_result):
        print(f"angle {math.degrees(a):.25f} => {math.degrees(m):.25f}")

    print("============")
    angles = np.array(
        [7 * PI / 2.0, 3 * PI / 2.0, -1 * PI / 2.0, -5 * PI / 2.0], dtype=np.float32
    )
    modded = tf.floormod(angles, 2 * PI)
    modded_result = session.run(modded)

    for a, m in zip(angles, modded_result):
        print(f"angle {math.degrees(a):.25f} => {math.degrees(m):.25f}")

    print("============")
    angles = np.array(
        [7 * PI / 2.0, 3 * PI / 2.0, -1 * PI / 2.0, -5 * PI / 2.0], dtype=np.float64
    )
    modded = tf.floormod(angles, 2 * PI)
    modded_result = session.run(modded)

    for a, m in zip(angles, modded_result):
        print(f"angle {math.degrees(a):.25f} => {math.degrees(m):.25f}")
