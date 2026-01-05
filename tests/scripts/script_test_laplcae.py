import model.geometry.grid as grid
import model.discrete as discrete
import space
import algebra

fd_grid = grid.StructuredGridND(shape=(4, 4), spacing=(0.01, 0.01))
fd_domain = discrete.fd.FDDomain(fd_grid)

top, bot = fd_domain.grid_boundaries(ax = 0)
left, right = fd_domain.grid_boundaries(ax = 1)

fd_disc = discrete.fd.FDDiscretization(fd_domain)
eq_space = space.EquationSpace(fd_disc)

bc_right = eq_space.bcs.dirichlet(right, value = 0)
bc_left = eq_space.bcs.dirichlet(left, value = 10)
bc_top = eq_space.bcs.neumann(top, value = 0)
bc_bot = eq_space.bcs.neumann(bot, value = 0)

F = eq_space.field()
F.apply_bc(bc_left)
F.apply_bc(bc_right)
F.apply_bc(bc_top)
F.apply_bc(bc_bot)

rhs = eq_space.field()

laplace = eq_space.operators.laplace()
# ## lambda d^2/dxdx F = f ##
les = algebra.system.LES(laplace, rh
solve_les = space.expr.SolveLES(



solve_les = space..LES(
    eq_space.operators.laplace(), rhs.current_value()
)
[solve_les.apply_bc(bc) for bc in F.bcs]
F.update_current(solve_les.solve()).eval()
print('success!')