import numpy as np
from space.field.operators import Dt, Dx
from discr import fd

import space
import tools.geometry

n = 100
grid = tools.geometry.StructuredGridND(shape=(n, n), spacing=(0.1, 0.1))
domain = fd.domain.FDDomain(grid)
discretization = fd.FDDiscretization(domain)
top = domain.left_boundary(ax = 0); bot = domain.right_boundary(ax = 0)
left = domain.left_boundary(ax = 1); right = domain.right_boundary(ax = 1)

eq_space = space.FieldSpace(discretization)

F = eq_space.scalar_field(init_value = 0.0)

lhs = Dx.laplace(F)
rhs = eq_space.scalar_field()

les = space.equation.LES(lhs.operator, rhs.value())
les.add_bcs((
    eq_space.bcs.dirichlet(top, 10.0),
    eq_space.bcs.dirichlet(bot, 0.0),
    eq_space.bcs.neumann(left, -20.0),
    eq_space.bcs.neumann(right, 20.0)
))

out = les.solve().eval()
print("finished")

import matplotlib.pyplot as plt

u = out
nx, ny = grid.shape
dx, dy = grid.spacing
x = np.linspace(0, (nx-1)*dx, nx)
y = np.linspace(0, (ny-1)*dy, ny)
X, Y = np.meshgrid(x, y)
U = u.reshape(X.shape)
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, U, cmap='viridis', edgecolor='k', linewidth=0.5)
ax1.set_title("Conjugate Gradient")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u")
fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, label='u')
plt.tight_layout()
plt.show()

# Fix hacked rhs[0] in LES