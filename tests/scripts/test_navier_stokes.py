import numpy as np

from space.field.operators import Dt, Dx
from discr import fd

import space
import tools.geometry

n = 20
grid = tools.geometry.StructuredGridND(shape=(n, n), spacing=(0.1, 0.1))
domain = fd.domain.FDDomain(grid)
discretization = fd.FDDiscretization(domain)
top = domain.left_boundary(ax = 0)
bot = domain.right_boundary(ax = 0)
left = domain.left_boundary(ax = 1)
right = domain.right_boundary(ax = 1)

eq_space = space.FieldSpace(discretization)

nu = 0.001
F = eq_space.vector_field(init_value = 0.0)
u = eq_space.vector_field(init_value = 0.0)
fi = eq_space.scalar_field(init_value = 0.0)
p = eq_space.scalar_field(init_value = 0.0)


#Step 1
dudt = Dt.euler(u)
u_eq = space.equation.LES(
     - (nu * Dx.laplace(u)), 
    -Dx.grad(p) + F
)
fi_eq = space.equation.LES(
    Dx.laplace(fi), 
    (dudt.alpha(0) / eq_space.time.dt()) * Dx.div(u)
)
F_monitor = space.monitors.FieldMonitor(F)
for time in eq_space.time.loop(0.0, 1.0, dt = 0.01):
    u.update(u_eq.solve()).eval()
    fi.update(fi_eq.solve()).eval()
    p.update(p.update(p.value() + fi.value() - nu * Dx.div(u))).eval()