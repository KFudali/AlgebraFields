import numpy as np
from ..core import Boundary, BoundaryId

class FDBoundary(Boundary):
    def __init__(
        self,
        boundary_id: BoundaryId,
        ids: np.ndarray
    ):
        super().__init__(boundary_id)
        self._ids = ids

    @property
    def ids(self) -> np.ndarray: return self._ids