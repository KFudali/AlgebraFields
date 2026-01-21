import numpy as np
from scipy.sparse.linalg import cg, gmres, LinearOperator
import matplotlib.pyplot as plt

N = 10
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

L = np.eye(N2, N2) * (-2 / dx**2 - 2 / dy**2)
L += (np.eye(N2, N2, -1) + np.eye(N2, N2, 1)) / dx**2
L += (np.eye(N2, N2, -(N)) + np.eye(N2, N2, (N))) / dy**2

rhs = np.zeros(shape = np.prod(shape))
rhs[top_ids] = 10
dt = 0.01
# ## lambda d^2/dxdx F(t) = (F(t) - F(t-1)) / dt
I = np.eye(N2, N2)
diff = 0.01
A = -diff * L - I /dt
A[boundary_ids, :] = 0
A[boundary_ids, boundary_ids] = 1
prev_x = np.zeros_like(rhs)
F = np.zeros(N2)
F[interior_ids] = 5.0

def matvec(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out = A @ x
    return out
linop = LinearOperator(shape=(A.shape), matvec=matvec, dtype=float)

prev_x = np.zeros_like(rhs)
direct_res = []
for t in np.arange(0.01, 1.01, dt):
    rhs = prev_x.copy() / dt - F
    rhs[top_ids] = 10
    prev_x = np.linalg.solve(A, rhs)
    direct_res.append(prev_x)

prev_x = np.zeros_like(rhs)
cg_res = []
for t in np.arange(0.01, 1.01, dt):
    rhs = prev_x.copy() / dt - F
    rhs[top_ids] = 10
    prev_x, info = gmres(linop, rhs, maxiter=100, rtol = 1e-8)
    cg_res.append(prev_x)

import matplotlib.animation as animation

fields = [u.reshape((N, N)) for u in cg_res]
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
