import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt

N = 4
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
A[boundary_ids, :] = 0= 1

A[boundary_ids, boundary_ids] 
rhs = np.zeros(shape = np.prod(shape))
rhs[top_ids] = 10

mod_rhs = rhs - A @ rhs

def matvec(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out = A @ x
    return out
linop = LinearOperator(shape=(A.shape), matvec=matvec, dtype=float)
x, info = cg(linop, mod_rhs, maxiter=100, rtol = 1e-8)
u = x
u[top_ids] = 10
x = np.linspace(0, (N-1)*dx, N)
y = np.linspace(0, (N-1)*dy, N)
X, Y = np.meshgrid(x, y)
U = u.reshape(N,N)
plt.figure(figsize=(6, 5))
cont = plt.contourf(X, Y, U, levels=50)
plt.colorbar(cont, label="u")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Laplace equation (FD, Dirichlet BC)")
plt.axis("equal")
plt.show()