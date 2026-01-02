import space
import model.geometry.grid as grid
import model.discretization as discretization
import space.expr as expr

fd_grid = grid.StructuredGridND(shape=(4, 4), spacing=(0.01, 0.01))
fd_domain = discretization.fd.FDDomain(fd_grid)

top, bot = fd_domain.grid_boundaries(ax = 0)
left, right = fd_domain.grid_boundaries(ax = 1)

fd_disc = discretization.fd.FDDiscretization(fd_domain)
eq_space = space.EquationSpace(fd_disc)

bc_right = eq_space.bcs.dirichlet(right, value = 0)
bc_left = eq_space.bcs.dirichlet(left, value = 10)
bc_top = eq_space.bcs.neumann(top, value = 0)
bc_bot = eq_space.bcs.neumann(bot, value = 0)

time_series = ConstStepTimeSeries(0.0, 100.0, 0.01)

F = eq_space.fields.transient.scalar(time_series)

F.apply_bc(bc_left)
F.apply_bc(bc_right)
F.apply_bc(bc_top)
F.apply_bc(bc_bot)

# ## lambda d^2/dxdx F(t+1) = (F(t+1) - F(t)) / dt
# ## F(t+1) - dt * lambda * d^2/dxdx F(t+1) = F(t)
dFdte = eq_space.fields.time.derivatives.explicit.euler(field=F, order=1, time_series = time_series)
dFdti = eq_space.fields.time.derivatives.implicit.euler(field=F, order=1, time_series = time_series)

ts = time_series.time_step(0)

F.initialize(0, ts = time_series.first)
dFdti.initialize(0, ts = time_series.first)

solve_les = expr.solve.LESSolve(
    dFdti.at(ts).linop()  * lam * F.at(ts).operator.laplace(),
    rhs.value()
)
[solve_les.apply_bc(bc) for bc in F.bcs]
f_update = expr.field.FieldUpdate(field = F, value = solve_les.solve())
progress_ts = time_series.progress_ts(ts)

les_step = Step(solve_les)
field_update_step = Step(f_update)
progress_ts_step = Step(progress_ts)
algorithm = Algorithm([les_step, field_update_step, progress_ts_step])

integrator = TimeIntegrator(time_series)
integrator.run(algorithm)

# Centralized time design
tw = eq_space.current_time_window()
# dFdte = eq_space.fields.time.derivatives.explicit.euler(field=F, order=1)
dFdti = eq_space.fields.time.derivatives.implicit.euler(field=F, order=1)
F.initialize(0, ts = time_series.first)
dFdti.initialize(0, ts = time_series.first)

solve_les = expr.solve.LESSolve(
    dFdti.at(tw).linop()  * lam * F.at(tw).operator.laplace(),
    rhs.value()
)
[solve_les.apply_bc(bc) for bc in F.bcs]
f_update = expr.field.FieldUpdate(field = F, value = solve_les.solve())

les_step = Step(solve_les)
field_update_step = Step(f_update)
algorithm = Algorithm(
    [les_step, field_update_step]
)

integrator = ConstTimeStepIntegrator(start = 0.0, end = 100.0, dt = 0.01)
integrator.run(algorithm)

