import tfrt.ray_engine as engine
import tensorflow as tf

a = engine.RaySet(
    {}, 
    {
        "x_start": tf.random.uniform((3,), 1, 3),
        "x_end": tf.random.uniform((3,), 10, 30)
    },
    2
)
b = engine.RaySet(
    {},
    {
        "x_start": tf.random.uniform((3,), 4, 6),
        "y_end": tf.random.uniform((3,), 400, 600)
    },
    2
)

print(type(c["x_start"]))

print("a:")
for sig in a.signature:
    print(f"{sig}: {a[sig]}")
print("b:")
for sig in b.signature:
    print(f"{sig}: {b[sig]}")
print("c:")
for sig in b.signature:
    print(f"{sig}: {c[sig]}")
print("--------------------")
print("a:")
for sig in a.signature:
    print(f"{sig}: {a[sig]}")
print("b:")
for sig in b.signature:
    print(f"{sig}: {b[sig]}")
print("c:")
for sig in b.signature:
    print(f"{sig}: {c[sig]}")
print("--------------------")    
print("a:")
for sig in a.signature:
    print(f"{sig}: {a[sig]}")
print("b:")
for sig in b.signature:
    print(f"{sig}: {b[sig]}")
print("c:")
for sig in b.signature:
    print(f"{sig}: {c[sig]}")
