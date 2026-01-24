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
boundary_ids = np.unique(np.concatenate([top_ids, bot_ids, left_ids, right_ids]))
all_ids = np.arange(N2)
interior_ids = np.setdiff1d(all_ids, boundary_ids)

A = np.eye(N2, N2) * (-2 / dx**2 - 2 / dy**2)
A += (np.eye(N2, N2, -1) + np.eye(N2, N2, 1)) / dx**2
A += (np.eye(N2, N2, -(N)) + np.eye(N2, N2, (N))) / dy**2
A[boundary_ids, :] = 0
A[boundary_ids, boundary_ids] = 1
rhs = np.zeros(shape = np.prod(shape))
rhs[top_ids] = 10
rhs -= A @ rhs
def matvec(x):
    return A @ x
linop = LinearOperator(shape=(A.shape), matvec=matvec, dtype=float)
x, info = cg(linop, rhs, maxiter=100, rtol = 1e-8)
u = x
u[top_ids] = 10
x = np.linspace(0, (N-1)*dx, N)
y = np.linspace(0, (N-1)*dy, N)
X, Y = np.meshgrid(x, y)
U = u.reshape(X.shape)
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