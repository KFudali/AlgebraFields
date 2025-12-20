import space
import expr
import mesh
import domain
import discretization

grid = mesh.structured.StructuredGridND(shape=(100, 100), spacing=(0.01, 0.01))
fd_domain = domain.fd.FDDomain(grid)

left, right = fd_domain.grid_boundaries(ax = 0)
top, bot = fd_domain.grid_boundaries(ax = 1)

fd_disc = discretization.fd.FDDiscretization(fd_domain)
eq_space = space.EquationSpace(fd_disc)

bc_left = eq_space.bcs.dirichlet(left, value = 10)
bc_right = eq_space.bcs.dirichlet(right, value = 10)
bc_top = eq_space.bcs.neumann(top, value = 0)
bc_bot = eq_space.bcs.neumann(bot, value = 0)

F = eq_space.fields.scalar()
F.apply_bc(bc_left)
F.apply_bc(bc_right)
F.apply_bc(bc_top)
F.apply_bc(bc_top)

rhs = eq_space.fields.scalar()

# ## lambda d^2/dxdx F = f ##
solve_les = expr.solve.LESSolve(
    F.operator.laplace(),
    rhs.value()
)
[solve_les.apply_bc(bc) for bc in F.bcs]

f_update = expr.FieldUpdate(field = F, value = solve_les.solve())
f_update.eval()
# plot(F)
