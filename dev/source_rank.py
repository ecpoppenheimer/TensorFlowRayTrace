import tfrt.distributions as dist
import tfrt.sources as sources

d = dist.StaticUniformAngularDistribution(-1, 1, 3)
s = sources.PointSource(2, (0, 0), 0, d, [.5, .6, .7], dense=True, rank_source=("angle", "ranks"))

print(f"source._ranks: {s._ranks}")
print(f"source[ranks]: {s['ranks']}")
