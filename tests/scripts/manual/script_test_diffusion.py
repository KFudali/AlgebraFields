import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt

N = 20
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
dirichlet_ids = np.unique(np.concatenate([top_ids, bot_ids]))
neumann_ids = np.unique(np.concatenate([left_ids, right_ids]))
boundary_ids = np.unique(np.concatenate([top_ids, bot_ids, left_ids, right_ids]))
all_ids = np.arange(N2)
interior_ids = np.setdiff1d(all_ids, boundary_ids)

L = np.eye(N2, N2) * (-2 / dx**2 - 2 / dy**2)
L += (np.eye(N2, N2, -1) + np.eye(N2, N2, 1)) / dx**2
L += (np.eye(N2, N2, -(N)) + np.eye(N2, N2, (N))) / dy**2

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

apply_neumann(left_ids, 1, 1, L)
apply_neumann(right_ids, 1, -1, L)
L[dirichlet_ids, :] = 0
L[dirichlet_ids, dirichlet_ids] = 1

def lap_rhs():
    dir_rhs = np.zeros_like(rhs)
    dir_rhs[top_ids] = 10
    lap_rhs = -L @ dir_rhs
    apply_neumann_rhs(left_ids, 10, lap_rhs)
    apply_neumann_rhs(right_ids, -10, lap_rhs)
    return lap_rhs
rhs = np.zeros(shape = np.prod(shape))
rhs[top_ids] = 10
rhs += lap_rhs()
steady_sol = np.linalg.solve(L, rhs)

dt = 0.01
# -dt * lambda * d^2/d^2 + I = u(t-1)
diff = 1
cL = -(dt * diff)
I = np.eye(N2, N2)
A =  cL * L + I
A[dirichlet_ids, dirichlet_ids] = 1

def matvec(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out = A @ x
    return out
linop = LinearOperator(shape=(A.shape), matvec=matvec, dtype=float)

prev_x = np.zeros_like(rhs)
prev_x[top_ids] = 10
res = []
for t in np.arange(0.01, 1.0, dt):
    rhs = prev_x.copy()
    rhs += cL * lap_rhs()
    rhs[top_ids] = 0
    prev_x, info = cg(linop, rhs, maxiter=100, rtol = 1e-8)
    prev_x[top_ids] = 10
    res.append(prev_x)

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
res[-1][top_ids] = 10
res[-1][bot_ids] = 0
U = res[-1].reshape(N, N)
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