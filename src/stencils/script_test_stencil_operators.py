from scipy.sparse.linalg import cg, LinearOperator
import numpy as np
import copy

from tools.geometry.grid import StructuredGridND
from stencils.fd_stencil_operator import FDStencilOperator
from stencils.fd_stencil import FDStencil
from discr.fd import FDDiscretization, FDDomain
from discr.core.domain import BoundaryId

N = 10
h = 0.1
grid = StructuredGridND((N, N), (h,h))
domain = FDDomain(grid)
discr = FDDiscretization(domain)

def apply_neumann(op: FDStencilOperator, bid: BoundaryId, value: float) -> np.ndarray:
    stencil = op._stencils[bid]
    boundary = domain.boundary(bid)
    h = boundary.grid.ax_spacing(boundary.axis)
    b = np.zeros(grid.shape)
    for offset, value in stencil.ax_contrib(boundary.axis).copy().items():
        if offset * boundary.inward_dir < 0:
            contrib = stencil._contrib[boundary.axis][offset]
            stencil._contrib[boundary.axis][-offset] += contrib
            stencil._contrib[boundary.axis].pop(offset)
            b.flat[boundary.ids] = - contrib * 2 * h * value
    return b

def apply_dirichlet(op: FDStencilOperator, bid: BoundaryId, value: float) -> np.ndarray:
    stencil = op._stencils[bid]
    boundary = domain.boundary(bid)
    u = np.zeros(grid.shape)
    u.flat[boundary.ids] = value
    rhs = np.zeros_like(u)
    stencil.apply(u, rhs)
    stencil._contrib[0] = {0: 1}
    for ax in range(1, grid.ndim):
        stencil._contrib[ax] = {0: 0}
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

rhs = -apply_dirichlet(laplace, top, 10)
rhs -= apply_dirichlet(laplace, bottom, 0)
rhs -= apply_neumann(laplace, left, 0)
rhs -= apply_neumann(laplace, right, 0)

def matvec(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    laplace.apply(x.reshape(grid.shape), out.reshape(grid.shape))
    return out

linop = LinearOperator(shape=(N*N, N*N), matvec=matvec, dtype=float)
x, info = cg(linop, rhs.flatten(), maxiter=100, rtol = 1e-8)

u = x.copy()
u[domain.boundary(top).ids] = 10
u[domain.boundary(bottom).ids] = 0
U = u.reshape(N, N)
import matplotlib.pyplot as plt
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
