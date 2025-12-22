import numpy as np
from itertools import product

class StructuredGridND:
    def __init__(self, shape: tuple[int, ...], spacing: tuple[float, ...]):
        self._shape = tuple(shape)
        self._spacing = tuple(spacing)
        self._ndim = len(self._shape)

        assert len(self._shape) == len(self._spacing)

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def spacing(self) -> tuple[int, ...]:
        return self._spacing

    def ax_spacing(self, axis: int) -> float:
        return self._spacing[axis]

    def id(self, *idx) -> int:
        """Convert ND index → flat index"""
        return np.ravel_multi_index(idx, self.shape)

    def idx(self, flat_id: int) -> tuple[int, ...]:
        """Convert flat index → ND index"""
        return np.unravel_index(flat_id, self.shape)

    def offset_id(self, flat_id: int, offset: tuple[int, ...]) -> int:
        idx = np.array(self.idx(flat_id))
        offset = np.array(offset)

        new_idx = idx - offset
        new_idx = np.clip(new_idx, 0, np.array(self.shape) - 1)

        return self.id(*new_idx)
    
    def _boundary_ids_fixed(self, axis: int, fixed: int) -> np.ndarray:
        ranges = [
            [fixed] if i == axis else range(self.shape[i])
            for i in range(self.ndim)
        ]
        ids = [self.id(*idx) for idx in product(*ranges)]
        return np.array(ids, dtype=int)

    def left_ids(self, axis: int) -> np.ndarray:
        return self._boundary_ids_fixed(axis, 0)

    def right_ids(self, axis: int) -> np.ndarray:
        return self._boundary_ids_fixed(axis, self.shape[axis] - 1)

    @property
    def boundary(self):
        """
        Returns a dict:
        {
            (axis, "min"): ids,
            (axis, "max"): ids,
            ...
        }
        """
        bd = {}
        for axis in range(self.ndim):
            bd[(axis, "min")] = self.left_ids(axis)
            bd[(axis, "max")] = self.right_ids(axis)
        return bd

    @property
    def boundary(self):
        """
        Returns a dict:
        {
            (axis, "min"): ids,
            (axis, "max"): ids,
            ...
        }
        """
        bd = {}
        for axis in range(self.ndim):
            bd[(axis, "min")] = self.boundary_ids(axis, "min")
            bd[(axis, "max")] = self.boundary_ids(axis, "max")
        return bd

    @property
    def interior_ids(self) -> np.ndarray:
        ranges = [
            range(1, n - 1) if n > 2 else []
            for n in self.shape
        ]

        if any(len(r) == 0 for r in ranges):
            return np.array([], dtype=int)

        ids = [
            self.id(*idx)
            for idx in product(*ranges)
        ]
        return np.array(ids, dtype=int)
