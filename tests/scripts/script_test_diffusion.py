import discr
import space
import tools.geometry

n = 10
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

lam = 0.01
solve_les = space.system.LESExpr(
    (lam * F.operator.laplace()) - dFdt.op().Ax(), -dFdt.op().b()
)

for dt in range(0, 1, 0.01):
    F.update(solve_les.solve()).eval()