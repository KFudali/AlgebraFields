from tools.geometry import grid
import discr
import space

n = 1000
fd_grid = grid.StructuredGridND(shape=(n, n), spacing=(0.1, 0.1))
fd_domain = discr.fd.FDDomain(fd_grid)

top, bot = fd_domain.grid_boundaries(ax = 0)
left, right = fd_domain.grid_boundaries(ax = 1)

fd_disc = discr.fd.FDDiscretization(fd_domain)
eq_space = space.EquationSpace(fd_disc)

bc_right = eq_space.bcs.dirichlet(right, value = 0)
bc_left = eq_space.bcs.dirichlet(left, value = 0)
bc_bot = eq_space.bcs.dirichlet(bot, value = 0)
bc_top = eq_space.bcs.dirichlet(top, value = 10)
F = eq_space.field()
F.apply_bc(bc_left)
F.apply_bc(bc_right)
F.apply_bc(bc_bot)
F.apply_bc(bc_top)

rhs = eq_space.field()
# ## lambda d^2/dxdx F = f ##

les = space.system.LESExpr(
    F.operator.laplace(), rhs.value()
)
[les.add_bc(bc) for bc in F.bcs]
les_solve = les.solve()
F.update(les_solve).eval()
arr = F.value().eval()