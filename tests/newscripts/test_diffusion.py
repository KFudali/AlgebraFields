import numpy as np
import space
import tools.geometry
import matplotlib.pyplot as plt
from discretization import fd

n = 4
grid = tools.geometry.StructuredGridND(shape=(n, n), spacing=(0.1, 0.1))
domain = fd.domain.FDDomain(grid)
disc = fd.FDDiscretization(domain)
top = domain.left_boundary(ax = 0); bot = domain.right_boundary(ax = 0)
left = domain.left_boundary(ax = 1); right = domain.right_boundary(ax = 1)

top_bc = disc.bcs.dirichlet(top, 10.0); bot_bc = disc.bcs.dirichlet(bot, 0.0)
left_bc = disc.bcs.neumann(top, 0.0); right_bc = disc.bcs.neumann(bot, 0.0)


eq_space = space.EquationSpace(fd_disc)
F = eq_space.scalar_field()
dFdt = F.dt.explicit.euler()

# ## lambda d^2/dxdx F(t) = (F(t) - F(t-1)) / dt
lam = 1.0
op =  dFdt - (lam * F.operator.laplace()) 
les = space.system.LESExpr(op, rhs=algebra.ZeroExpression(eq_space.shape))
les.apply_bcs([top_bc, bot_bc, left_bc, right_bc])

dt = 0.01
for t in np.arange(0.01, 1.0, dt):
    F.update(solve_les.solve()).eval()
    F.advance(dt)
    dFdt.advance(dt)