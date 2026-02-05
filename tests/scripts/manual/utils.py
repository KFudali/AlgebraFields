import numpy as np

def add_ghost_row(
    field: np.ndarray, ax: int, dir: int,
):
    assert field.ndim == 2
    N0, N1 = field.shape
    if ax == 0:
        ghost = np.zeros((1, N1), dtype=field.dtype)
    else:
        ghost = np.zeros((N0, 1), dtype=field.dtype)

    if dir == +1:
        extended = np.concatenate((field, ghost), axis=ax)
    else:
        extended = np.concatenate((ghost, field), axis=ax)

    boundary_slice = [slice(None), slice(None)]
    ghost_slice = [slice(None), slice(None)]
    pre_boundary_slice = [slice(None), slice(None)]
    if dir == -1:          # top
        pre_boundary_slice[ax] = 2
        boundary_slice[ax] = 1
        ghost_slice[ax] = 0
    else:                  # bottom
        pre_boundary_slice[ax] = -3
        boundary_slice[ax] = -2
        ghost_slice[ax] = -1
    return extended, tuple(pre_boundary_slice), tuple(boundary_slice), tuple(ghost_slice)


def laplace(field: np.ndarray, h: float, cut_boundary = True) -> np.ndarray:
    center = (slice(1, -1),) * 2
    out = np.zeros_like(field)
    for axis in range(2):
        plus = list(center)
        minus = list(center)
        plus[axis] = slice(2, None)
        minus[axis] = slice(None, -2)
        out[center] += (
            field[tuple(plus)] - 2.0 * field[center] + field[tuple(minus)]
        ) / h**2
    if cut_boundary:
        return out[1:-1, 1:-1]
    else:
        return out