from typing import Optional

from discr.core.domain import BoundaryId, Domain
from tools.geometry import StructuredGridND
from .fd_boundary import FDBoundary
from tools.region import boundary_region

class FDDomain(Domain):
    def __init__(self, structured_grid: StructuredGridND):
        self._grid = structured_grid
        self._boundaries = dict[BoundaryId, FDBoundary]()
        self._next_boundary_id = 1
        self._mark_boundaries()

    def _mark_boundaries(self):
        for ax in range(self._grid.ndim):
            left_id = self._alloc_boundary_id()
            right_id = self._alloc_boundary_id()
            self._add_boundary(left_id, ax, dir=-1)
            self._add_boundary(right_id, ax, dir=1)

    def _alloc_boundary_id(self) -> BoundaryId:
        bid = BoundaryId(self._next_boundary_id)
        self._next_boundary_id += 1
        return bid

    def _add_boundary(self, bid: BoundaryId, ax: int, dir: int) -> FDBoundary:
        include_corners = ax == 0
        region = boundary_region(self._grid.shape, ax, dir, include_corners)
        b = FDBoundary(bid, region, ax, -dir, self.grid)
        self._boundaries[bid] = b
        return b

    def left_boundary(self, ax: int) -> BoundaryId:
        return list(self.boundaries.keys())[2 * ax]
    
    def right_boundary(self, ax: int) -> BoundaryId:
        return list(self.boundaries.keys())[2 * ax + 1]

    @property
    def grid(self) -> StructuredGridND: return self._grid

    @property
    def boundaries(self) -> dict[BoundaryId, FDBoundary]: return self._boundaries

    def boundary(self, boundary_id: BoundaryId) -> Optional[FDBoundary]:
        return self._boundaries.get(boundary_id, None)