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

    def flat_id(self, *idx) -> int:
        """Convert ND index → flat index"""
        return np.ravel_multi_index(idx, self.shape)

    def flat_id_arr(self, idx_arr: np.ndarray) -> np.ndarray:
        """
        Convert ND index array → flat index array
        idx_arr shape: (N, ndim)
        returns shape: (N,)
        """
        idx_arr = np.asarray(idx_arr)
        if idx_arr.ndim != 2 or idx_arr.shape[1] != len(self.shape):
            raise ValueError(
                f"Expected idx_arr of shape (N, {len(self.shape)}), "
                f"got {idx_arr.shape}"
            )
        return np.ravel_multi_index(idx_arr.T, self.shape)
    
    def idx(self, flat_id: int) -> tuple[int, ...]:
        """Convert flat index → ND index"""
        return np.unravel_index(flat_id, self.shape)

    def idx_arr(self, flat_ids: np.ndarray) -> np.ndarray: 
        """Convert flat index array → rray of dim x ids"""
        return np.stack((np.unravel_index(flat_ids, self.shape)), axis=1)

    def offset_flat_id(self, flat_id: int, offset: tuple[int, ...]) -> int:
        idx = np.array(self.idx(flat_id))
        offset = np.array(offset)

        new_idx = idx - offset
        new_idx = np.clip(new_idx, 0, np.array(self.shape) - 1)

        return self.flat_id(*new_idx)
    
    def _id_slice(self, axis: int, slice_pos: int) -> np.ndarray:
        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"axis must be in [0, {self.ndim - 1}]")

        if slice_pos < 0 or slice_pos >= self.shape[axis]:
            raise ValueError(
                f"slice_pos must be in [0, {self.shape[axis] - 1}]"
            )

        ranges = [
            range(self.shape[d]) if d != axis else [slice_pos]
            for d in range(self.ndim)
        ]

        idx_nd = np.array(list(product(*ranges)), dtype=int)
        return self.flat_id_arr(idx_nd)


    def left_ids(self, axis: int) -> np.ndarray:
        return self._id_slice(axis, 0)

    def right_ids(self, axis: int) -> np.ndarray:
        return self._id_slice(axis, self.shape[axis] - 1)

    @property
    def interior_ids(self) -> np.ndarray:
        if any(n <= 2 for n in self.shape):
            return np.array([], dtype=int)

        ranges = [range(1, n - 1) for n in self.shape]
        idx_nd = np.array(list(product(*ranges)), dtype=int)

        return self.flat_id_arr(idx_nd)

    @property
    def boundary_ids(self) -> np.ndarray:
        ids = []
        for axis in range(self.ndim):
            ids.append(self.left_ids(axis))
            ids.append(self.right_ids(axis))
        return np.unique(np.concatenate(ids))
