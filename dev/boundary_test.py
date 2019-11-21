import tfrt.boundary as boundary

b = boundary.ManualBoundary()
b.feed_segments([(1, 10, 0, 0), (2, 20, 0, 0), (3, 30, 0, 0)])
print(b["x_start"])
print(b["y_start"])
print(b["x_end"])
print(b["y_end"])
