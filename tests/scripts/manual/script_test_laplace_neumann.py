import numpy as np
from scipy.sparse.linalg import cg, gmres, LinearOperator
import matplotlib.pyplot as plt

N = 100
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
A[dirichlet_ids, :] = 0
A[dirichlet_ids, dirichlet_ids] = 1

rhs = np.zeros(shape = np.prod(shape))

def apply_neumann(ids, ax, inward_dir, value):
    for i in ids:
        A[i, :] = 0.0
        idx = list(np.unravel_index(i, shape))
        idx_in = idx.copy()
        idx_in[ax] += inward_dir
        j = np.ravel_multi_index(idx_in, shape)
        A[i, i] = -inward_dir / dy
        A[i, j] =  inward_dir / dy
        rhs[i] = value
apply_neumann(right_ids, 1, -1, 10)
apply_neumann(left_ids, 1, 1, 10)
rhs[top_ids] = 10
mod_rhs = rhs - A @ rhs
direct_x = np.linalg.solve(A, mod_rhs)
def matvec(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out = A @ x
    return out
linop = LinearOperator(shape=(A.shape), matvec=matvec, dtype=float)
x, info = gmres(linop, mod_rhs, maxiter=100, rtol = 1e-8)
u = x.copy()
dir_u = direct_x.copy()
u[top_ids] = 10
u[bot_ids] = 0
dir_u[top_ids] = 10
dir_u[bot_ids] = 0

U_direct = dir_u.reshape(N, N)
U_cg = u.reshape(N, N)

x = np.linspace(0, (N-1)*dx, N)
y = np.linspace(0, (N-1)*dy, N)
X, Y = np.meshgrid(x, y)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

c1 = axes[0].contourf(X, Y, U_direct, levels=50)
fig.colorbar(c1, ax=axes[0])
axes[0].set_title("Direct solve")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_aspect("equal")

c2 = axes[1].contourf(X, Y, U_cg, levels=50)
fig.colorbar(c2, ax=axes[1])
axes[1].set_title("Conjugate Gradient solve")
axes[1].set_xlabel("x")
axes[1].set_aspect("equal")

plt.tight_layout()
plt.show()
