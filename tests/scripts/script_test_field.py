import space
import model.geometry.grid as grid
import model.discretization as discretization
import space.expr as expr

fd_grid = grid.StructuredGridND(shape=(10, 10), spacing=(0.01, 0.01))
fd_domain = discretization.fd.FDDomain(fd_grid)

top, bot = fd_domain.grid_boundaries(ax = 0)
left, right = fd_domain.grid_boundaries(ax = 1)

fd_disc = discretization.fd.FDDiscretization(fd_domain)
eq_space = space.EquationSpace(fd_disc)

bc_left = eq_space.bcs.dirichlet(left, value = 10)
bc_right = eq_space.bcs.dirichlet(right, value = 0)
bc_top = eq_space.bcs.dirichlet(top, value = 0)
bc_bot = eq_space.bcs.dirichlet(bot, value = 0)

F = eq_space.fields.scalar()
F.apply_bc(bc_left)
F.apply_bc(bc_right)
F.apply_bc(bc_top)
F.apply_bc(bc_bot)

rhs = eq_space.fields.scalar()

# ## lambda d^2/dxdx F = f ##
solve_les = expr.solve.LESSolve(
    F.operator.laplace(),
    rhs.value()
)
[solve_les.apply_bc(bc) for bc in F.bcs]

f_update = expr.field.FieldUpdate(field = F, value = solve_les.solve())
f_update.eval()