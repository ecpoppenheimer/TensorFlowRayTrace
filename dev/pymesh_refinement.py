import pymesh

"""
not sure what I am doing with docker, and I know I am doing the wrong thing with the folder,
but this works well enough.

docker run -it --rm -v `pwd`:/stl pymesh/pymesh bash
cd /stl
python pymesh_refinement.py
"""

mesh = pymesh.load_mesh("./stl/raw_disk.stl")


def refine_mesh(mesh, cycles, long_threshold, **kwargs):
    """
    kwargs can be: abs_threshold, rel_threshold, preserve_feature
    """
    for i in range(cycles):
        mesh, _ = pymesh.split_long_edges(mesh, long_threshold)
        mesh, _ = pymesh.collapse_short_edges(mesh, **kwargs)
        
    return mesh
    
mesh = refine_mesh(mesh, 10, .08, rel_threshold=.9, preserve_feature=True)

pymesh.save_mesh("./stl/processed_disk_large.stl", mesh)
