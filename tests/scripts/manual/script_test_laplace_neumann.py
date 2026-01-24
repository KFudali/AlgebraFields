import numpy as np
from scipy.sparse.linalg import cg, gmres, LinearOperator
import matplotlib.pyplot as plt

N = 50
N2 = N*N
shape = (N, N)
dx = 0.1
dy = 0.1
ni, nj = shape
top_ids = np.arange(0,N)
bot_ids = np.arange(N*(N-1), N*N)
left_ids = np.arange(0, N*N, N)
right_ids = np.arange(N-1, N*N, N)
left_ids = left_ids[1:-1]
right_ids = right_ids[1:-1]
dirichlet_ids = np.unique(np.concatenate([top_ids, bot_ids,]))
neumann_ids = np.unique(np.concatenate([left_ids, right_ids]))
boundary_ids = np.unique(np.concatenate([top_ids, bot_ids, left_ids, right_ids]))
all_ids = np.arange(N2)
interior_ids = np.setdiff1d(all_ids, boundary_ids)

A = np.eye(N2, N2) * (-2 / dx**2 - 2 / dy**2)
A += (np.eye(N2, N2, -1) + np.eye(N2, N2, 1)) / dx**2
A += (np.eye(N2, N2, -(N)) + np.eye(N2, N2, (N))) / dy**2
rhs = np.zeros(shape = np.prod(shape))
rhs[top_ids] = 10
A[dirichlet_ids, :] = 0
A[dirichlet_ids, dirichlet_ids] = 1

def apply_neumann(ids, ax, inward_dir, A):
    for i in ids:
        idx = list(np.unravel_index(i, shape))
        idx_in = idx.copy()
        idx_in[ax] += inward_dir
        j_in = np.ravel_multi_index(idx_in, shape)
        A[i, j_in] = 2 / dy**2
        A[i, j_in - 2 * inward_dir] = 0 

def apply_neumann_rhs(ids, value, rhs_array):
    for i in ids:
        rhs_array[i] += 2 * value / dy

apply_neumann(left_ids, 1, 1, A)
apply_neumann(right_ids, 1, -1, A)
dirichlet_rhs_flux = A @ rhs
apply_neumann_rhs(left_ids, 10, rhs)
apply_neumann_rhs(right_ids, 10, rhs)
rhs -= dirichlet_rhs_flux

def matvec(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out = A @ x
    return out

linop = LinearOperator(shape=(A.shape), matvec=matvec, dtype=float)
x, info = gmres(linop, rhs, maxiter=100, rtol = 1e-8)
u = x.copy()
u[top_ids] = 10
u[bot_ids] = 0
U = u.reshape(N, N)

import matplotlib.pyplot as plt
x = np.linspace(0, (N-1)*dx, N)
y = np.linspace(0, (N-1)*dy, N)
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