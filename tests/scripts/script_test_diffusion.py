import discr
import numpy as np
import space
import tools.geometry
import matplotlib.pyplot as plt

n = 4
fd_grid = tools.geometry.grid.StructuredGridND(shape=(n, n), spacing=(0.1, 0.1))
fd_domain = discr.fd.FDDomain(fd_grid)

top, bot = fd_domain.grid_boundaries(ax = 0)
left, right = fd_domain.grid_boundaries(ax = 1)

fd_disc = discr.fd.FDDiscretization(fd_domain)
eq_space = space.EquationSpace(fd_disc)

bc_right = eq_space.bcs.neumann(right, value = 0)
bc_left = eq_space.bcs.neumann(left, value = 0)
bc_bot = eq_space.bcs.dirichlet(bot, value = 0)
bc_top = eq_space.bcs.dirichlet(top, value = 10)
F = eq_space.field()
F.apply_bc(bc_left)
F.apply_bc(bc_right)
F.apply_bc(bc_bot)
F.apply_bc(bc_top)

dFdt = space.time.explicit.EulerTimeDerivative(F)
# ## lambda d^2/dxdx F(t) = (F(t) - F(t-1)) / dt

lam = 1.0
op =  dFdt.op().Ax() - (lam * F.operator.laplace()) 
rhs = dFdt.op().b()
solve_les = space.system.LESExpr(op, rhs)
solve_les.add_bc(bc_left)
solve_les.add_bc(bc_right)
solve_les.add_bc(bc_top)
solve_les.add_bc(bc_bot)

# It does not work for neumann because current implementation fixes values like
#  # apply_neumann(left_ids, 1, 1, A)
# apply_neumann(right_ids, 1, -1, A)
# The time term should still be in the global matrix and we eliminate it forcefully

dt = 0.01
for t in np.arange(0.01, 1.0, dt):
    F.update(solve_les.solve()).eval()
    F.advance(dt)
    dFdt.advance(dt)

u = F.prev_value(1).eval()[0]
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