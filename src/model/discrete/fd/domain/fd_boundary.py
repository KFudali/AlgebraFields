import numpy as np
from model.domain import BoundaryId, Boundary
from model.geometry.grid import StructuredGridND

class FDBoundary(Boundary):
    def __init__(
        self,
        boundary_id: BoundaryId,
        ids: np.ndarray,
        ax: int,
        inward_dir: int,
        grid_spacing: float
    ):
        super().__init__(boundary_id)
        self._ids = ids
        self._ax = ax
        self._inward_dir = inward_dir
        self._spacing = grid_spacing
    
    @property
    def grid_spacing(self) -> float:
        return self._spacing

    @property
    def axis(self) -> int:
        return self._ax

    @property
    def inward_dir(self) -> float:
        return self._inward_dir

    @property
    def ids(self) -> np.ndarray: return self._ids
    
