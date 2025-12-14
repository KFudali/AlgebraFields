import numpy as np
import space
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

# ## lambda d^2/dxdx F = f ##
# f = eq_space.new_scalar_field(init_value = 0.0)
# alpha = eq_space.scalar(value = 0.01)

# system = expr.linear_system(
#     A = F.operator().laplace().linop().A,
#     b = f.value()
# )
# [system.apply_bc(bc) for bc in F.bcs]

# calc_LES = expr.solve.LESSolve(system)
# f_update = expr.field.FieldUpdate(field = F, value = calc_LES)
# f_update.eval()

# plot(F)