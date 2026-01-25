import numpy as np
import discr
import space
import tools.algebra
import tools.geometry

n = 50
fd_grid = tools.geometry.grid.StructuredGridND(shape=(n, n), spacing=(0.1, 0.1))
fd_domain = discr.fd.FDDomain(fd_grid)

top, bot = fd_domain.grid_boundaries(ax = 0)
left, right = fd_domain.grid_boundaries(ax = 1)

fd_disc = discr.fd.FDDiscretization(fd_domain)
eq_space = space.EquationSpace(fd_disc)


bc_bot = eq_space.bcs.dirichlet(bot, value = 0)
bc_top = eq_space.bcs.dirichlet(top, value = 10)
bc_right = eq_space.bcs.neumann(right, value = -10)
bc_left = eq_space.bcs.neumann(left, value = 10)
F = eq_space.field()
F.apply_bc(bc_left)
F.apply_bc(bc_right)

F.apply_bc(bc_bot)
F.apply_bc(bc_top)

rhs = eq_space.field()
rhs.update(
    tools.algebra.expr.CallableExpression(
        rhs.shape, lambda: 10* np.ones(shape = rhs.shape)
    )
).eval()
## lambda d^2/dxdx F = f ##
les = space.system.LESExpr(
    F.operator.laplace(), rhs.value()
)
[les.add_bc(bc) for bc in F.bcs]
les_solve = les.solve()
F.update(les_solve).eval()
arr = F.value().eval()


import matplotlib.pyplot as plt
u = F.value().eval()[0]
u[np.unravel_index(bc_top.boundary.ids, fd_grid.shape)] = 10
u[np.unravel_index(bc_bot.boundary.ids, fd_grid.shape)] = 0
nx, ny = fd_grid.shape
dx, dy = fd_grid.spacing
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

