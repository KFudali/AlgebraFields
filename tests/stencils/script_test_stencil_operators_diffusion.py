from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt
import numpy as np
import copy

from tools.geometry import StructuredGridND
from stencils.fd_stencil_operator import FDStencilOperator
from stencils.fd_stencil import FDStencil
from discr.fd import FDDiscretization
from discr.fd.domain import FDDomain
from discr.core.domain import BoundaryId

N = 20
h = 0.1
grid = StructuredGridND((N, N), (h,h))
domain = FDDomain(grid)
discr = FDDiscretization(domain)

def apply_neumann(op: FDStencilOperator, bid: BoundaryId, g: float) -> np.ndarray:
    stencil = op._stencils[bid]
    boundary = domain.boundary(bid)
    h = boundary.grid.ax_spacing(boundary.axis)
    b = np.zeros(grid.shape)
    for offset, value in stencil.ax_contrib(boundary.axis).copy().items():
        if offset * boundary.inward_dir < 0:
            contrib = stencil._contrib[boundary.axis][offset]
            stencil._contrib[boundary.axis][-offset] += contrib
            stencil._contrib[boundary.axis].pop(offset)
            b.flat[boundary.ids] = - contrib * 2 * h * g
    return b

def apply_dirichlet(op: FDStencilOperator, bid: BoundaryId, value: float) -> np.ndarray:
    stencil = op._stencils[bid]
    boundary = domain.boundary(bid)
    u = np.zeros(grid.shape)
    u.flat[boundary.ids] = value
    rhs = np.zeros_like(u)
    for ax in range(0, grid.ndim):
        if ax != boundary.axis: 
            stencil._contrib[ax] = {0: 0}
    stencil.apply(u, rhs)
    stencil._contrib[boundary.axis] = {0: 1}
    return rhs

top, bottom = domain.grid_boundaries(0)
left, right = domain.grid_boundaries(1)
bids = [top, bottom, left, right]

lap = {-1: 1 / h**2, 0: -2 / h**2, 1: 1 / h**2}
lap_stencil = FDStencil({0: lap, 1: copy.deepcopy(lap)})
laplace = FDStencilOperator(discr)
laplace.set_interior_stencil(lap_stencil)
for bid in bids: 
    laplace.set_boundary_stencil(bid, copy.deepcopy(lap_stencil))

dt = 0.01
ones_stencil = FDStencil({0: {0: 1 / dt}, 1: {0: 0}})
ones_op = FDStencilOperator(discr)
ones_op.set_interior_stencil(ones_stencil)
for bid in bids: 
    ones_op.set_boundary_stencil(bid, copy.deepcopy(ones_stencil))
diff = 0.01
op = -diff * laplace + ones_op

rhs = np.zeros(grid.shape)
rhs -= apply_dirichlet(op, top, 10)
rhs -= apply_dirichlet(op, bottom, 0)
rhs -= apply_neumann(op, left, 20)
rhs -= apply_neumann(op, right, -20)

def matvec(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    op.apply(x.reshape(grid.shape), out.reshape(grid.shape))
    return out
linop = LinearOperator(shape=(N*N, N*N), matvec=matvec, dtype=float)

top_ids = domain.boundary(top).ids
bot_ids = domain.boundary(bottom).ids
prev_x = np.zeros_like(rhs)
res = []
for t in np.arange(0.01, 1.0, dt):
    t_rhs = prev_x.copy() / dt
    t_rhs += rhs
    prev_x, info = cg(linop, t_rhs.flat, maxiter=1000, rtol = 1e-8)
    prev_x = prev_x.reshape(t_rhs.shape)
    u = prev_x.copy()
    u.flat[top_ids] = 10.0
    u.flat[bot_ids] = 0.0
    res.append(u)

import matplotlib.animation as animation

fields = [u.reshape((N, N)) for u in res]
vmin = min(f.min() for f in fields)
vmax = max(f.max() for f in fields)

fig, ax = plt.subplots()
cont = ax.contourf(fields[0], levels=20)
cbar = fig.colorbar(cont, ax=ax)

def update(frame):
    ax.clear()
    ax.contourf(fields[frame], levels=20)
    ax.set_title(f"t = {frame * dt:.2f}")

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(fields),
    interval=100,
    blit=False
)

plt.show()
res[-1].flat[top_ids] = 10
res[-1].flat[bot_ids] = 0
U = res[-1].reshape(N, N)
x = np.linspace(0, (N-1)*h, N)
y = np.linspace(0, (N-1)*h, N)
X, Y = np.meshgrid(x, y)
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, U, cmap='viridis', edgecolor='k', linewidth=0.5)
ax1.set_title("Conjugate Gradient")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u")
fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, label='u')
plt.tight_layout()
plt.show()