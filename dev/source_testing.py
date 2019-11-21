import tensorflow as tf
import tfrt.sources as sources

PI = sources.PI

ang_dist = sources.RandomUniformAngularDistribution(-PI/2.0, PI/2.0, 3)
bp_dist = sources.StaticUniformBeam(-1.0, 1.0, 4)
print(ang_dist.angles)
print(bp_dist.base_points_x)
source = sources.AngularSource((0.0, 0.0), PI/2.0, ang_dist, bp_dist, [1.0], dense=True)
print("randomizing...")
print(source["x_end"])
source.update()
print("randomizing...")
print(source["x_end"])
source.update()
print("randomizing...")
print(source["x_end"])
print("--------------------------")
print("killing recursive update")
source.recursively_update = False
source.update()
print("randomizing...")
print(source["x_end"])
source.update()
print("randomizing...")
print(source["x_end"])
source.update()
print("randomizing...")
print(source["x_end"])
