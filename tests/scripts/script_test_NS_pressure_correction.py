import spacer
import model.geometry.grid as grid
import model.discretization as discretization
import spacer.expr as expr

fd_grid = grid.StructuredGridND(shape=(4, 4), spacing=(0.01, 0.01))
fd_domain = discretization.fd.FDDomain(fd_grid)

top, bot = fd_domain.grid_boundaries(ax = 0)
left, right = fd_domain.grid_boundaries(ax = 1)

fd_disc = discretization.fd.FDDiscretization(fd_domain)
eq_space = spacer.EquationSpace(fd_disc)

time_series = ConstStepTimeSeries(0.0, 100.0, 0.01)

prev_ts = time_series.step(0)
ts = time_series.step(1)

U_hat = eq_space.fields.transient.vector(time_series)
F = eq_space.fields.transient.vector(time_series)
p = eq_space.fields.transient.scalar(time_series)
fi = eq_space.fields.scalar(time_series)

u_hat_bc_right = eq_space.bcs.dirichlet(right, value = 0)
u_hat_bc_left = eq_space.bcs.dirichlet(left, value = 0)
u_hat_bc_top = eq_space.bcs.dirichlet(top, value = 0)
u_hat_bc_bot = eq_space.bcs.dirichlet(bot, value = 0)

U_hat.apply_bc(u_hat_bc_right)
U_hat.apply_bc(u_hat_bc_left)
U_hat.apply_bc(u_hat_bc_top)
U_hat.apply_bc(u_hat_bc_bot)

fi_bc_right = eq_space.bcs.neumann(right, value = 0)
fi_bc_left = eq_space.bcs.neumann(left, value = 0)
fi_bc_top = eq_space.bcs.neumann(top, value = 0)
fi_bc_bot = eq_space.bcs.neumann(bot, value = 0)

fi.apply_bc(fi_bc_right)
fi.apply_bc(fi_bc_top)
fi.apply_bc(fi_bc_left)
fi.apply_bc(fi_bc_right)

dU_hat_dt = eq_space.time.derivative.imex.standard(order = 2)
ni = eq_space.scalar(0.01)

#------------------------------------------------#
A = dU_hat_dt.at(ts).linop() + Op.Laplace(U_hat.at(ts))
rhs = Op.Grad(p.at(prev_ts)) + f.at(ts)
system = LESSystem(A, rhs, bcs = F.bcs)
u_hat_les = expr.solve.LESSolve(system)
U_hat_update = U_hat.at(ts).update(u_hat_les.solve())
#------------------------------------------------#
A = fi.operator.laplace()
rhs = 3/ (2 * ts.size) * U_hat.at(ts).operator.div().value()
fi_les = expr.solve.LESSolve(A, rhs)
[u_hat_les.apply_bc(bc) for bc in U_hat.bcs]
fi_update = fi.update(fi_les.solve)
#------------------------------------------------#
p_sum = p.at(prev_ts).value() + fi.value() - nu * U_hat.at(ts).operator.grad().value()
p_update = p.at(ts).update(p_sum)
#------------------------------------------------#

u_hat_update_step = Step(U_hat_update)
fi_update_step = Step(fi_update)
p_update_step = Step(p_update)

pressure_correction = Algorithm(
    u_hat_update_step,
    fi_update_step,
    p_update_step
)

integrator = TimeIntegrator(time_series)
integrator.run(pressure_correction)


prev_ts = adaptive_time_series.step(0)
ts = adaptive_time_series.next_step(dt=0.01)
...algorithm definition goes here...
adapt_ts_step = Step(adaptive_time_series.set_next_step_size(some_calculation...))
pressure_correction = Algorithm(
    u_hat_update_step,
    fi_update_step,
    p_update_step,
    adapt_ts_step
)
integrator = TimeIntegrator(time_series)
integrator.run(pressure_correction)

(time integrator does something like if not time_series.finished():
    algorithm.execute(), 
    time_series.progress(), 
    some logging..
)