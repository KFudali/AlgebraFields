import numpy as np

from space.field.operators import Dt, Dx
from discr import fd

import space
import tools.geometry

n = 20
grid = tools.geometry.StructuredGridND(shape=(n, n), spacing=(0.1, 0.1))
domain = fd.domain.FDDomain(grid)
discretization = fd.FDDiscretization(domain)
top = domain.left_boundary(ax = 0); bot = domain.right_boundary(ax = 0)
left = domain.left_boundary(ax = 1); right = domain.right_boundary(ax = 1)

eq_space = space.FieldSpace(discretization)

F = eq_space.scalar_field(init_value = 0.0)

diff = 1.0
dx = Dx.laplace(F).operator * diff
dt  = Dt.euler(F).operator
lhs = dt - dx
rhs = eq_space.scalar_field(init_value = 0.0).value()

les = space.equation.LES(lhs, rhs)
les.add_bcs((
    eq_space.bcs.dirichlet(top, 10.0),
    eq_space.bcs.dirichlet(bot, 0.0),
    eq_space.bcs.neumann(left, -20.0),
    eq_space.bcs.neumann(right, 20.0)
))

F_monitor = space.monitors.FieldMonitor(F)
for time in eq_space.time.loop(0.0, 1.0, dt = 0.01):
    F.update(les.solve()).eval()

F_monitor.playback()