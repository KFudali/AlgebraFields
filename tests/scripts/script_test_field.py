import numpy as np
import pygmsh
import meshio

with pygmsh.occ.Geometry() as geom:
    cube = geom.add_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.1)
    mesh = geom.generate_mesh()

mesh.cells

print(f"Points shape: {mesh.points.shape}")
for cell_block in mesh.cells:
    print(f"{cell_block.type}: {cell_block.data.shape}")