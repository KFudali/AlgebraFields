Project started as a prototying sandbox for generic Algerba and PDE library that i intend to implement in Rust or C++ for my numerical enviroments (https://github.com/KFudali/OcafCAD and finally https://github.com/PawelekPro/MeshGeneratingTool). 

The idea is that even though numerical and PDE libraries already exists (FEniCS, OpenFOAM, DUNE) none of them really satisfied my need for very abstract interface to algebra. Here i intend to wrap all actions in generic Expr or Step classes that all do operations on Field in EquationSpace objects. In other words i am trying to express the assembly of numerical methods without fixing that to specific discretization (FEM, FD, FVM). Below is a snippet from (currently only) test scripts to demonstrate what interfaces i am going for:
```python
fd_grid = grid.StructuredGridND(shape=(10, 10), spacing=(0.01, 0.01))
fd_domain = discretization.fd.FDDomain(fd_grid)

top, bot = fd_domain.grid_boundaries(ax=0)
left, right = fd_domain.grid_boundaries(ax=1)

fd_disc = discretization.fd.FDDiscretization(fd_domain)
eq_space = space.EquationSpace(fd_disc)

bc_left  = eq_space.bcs.dirichlet(left,  value=10)
bc_right = eq_space.bcs.dirichlet(right, value=0)
bc_top   = eq_space.bcs.dirichlet(top,   value=0)
bc_bot   = eq_space.bcs.dirichlet(bot,   value=0)

F = eq_space.fields.scalar()
F.apply_bc(bc_left)
F.apply_bc(bc_right)
F.apply_bc(bc_top)
F.apply_bc(bc_bot)

rhs = eq_space.fields.scalar()

# lambda d^2/dx^2 F = f
solve_les = expr.solve.LESSolve(
    F.operator.laplace(),
    rhs.value()
)

for bc in F.bcs:
    solve_les.apply_bc(bc)

f_update = expr.field.FieldUpdate(
    field=F,
    value=solve_les.solve()
)
f_update.eval()
```
