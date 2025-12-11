import numpy as np

class StructuredGrid2D:
    def __init__(self, nx: int, ny: int, dx: float, dy: float):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self._grid = np.zeros((nx, ny), dtype=float)
    
    @property
    def grid(self) -> np.ndarray:
        return self._grid

    def id(self, i: int, j: int) -> int:
        return i * self.ny + j

    def offset_id(
        self, idx: int, x: int = 0, y: int = 0
    ) -> int:
        i, j = self.ij(idx)
        i_new = i - x 
        j_new = j - y

        i_new = max(0, min(self.nx - 1, i_new))
        j_new = max(0, min(self.ny - 1, j_new))

        return self.id(i_new, j_new)


    def ij(self, id: int):
        return divmod(id, self.ny)

    @property
    def size(self):
        return self.nx * self.ny

    @property
    def shape(self) -> tuple[int, int]:
        return self.nx, self.ny

    @property
    def top_ids(self) -> np.ndarray:
        return np.array([self.id(i, self.ny-1) for i in range(self.nx)])

    @property
    def bottom_ids(self) -> np.ndarray:
        return np.array([self.id(i, 0) for i in range(self.nx)])

    @property
    def left_ids(self) -> np.ndarray:
        return np.array([self.id(0, j) for j in range(self.ny)])

    @property
    def right_ids(self) -> np.ndarray:
        return np.array([self.id(self.nx-1, j) for j in range(self.ny)])

    @property
    def interior_ids(self) -> np.ndarray:
        ids = []
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                ids.append(self.id(i, j))
        return np.array(ids)
