import numpy as np
from typing import Optional

from model.domain import SubDomain, SubDomainId, Domain
from model.geometry.grid import StructuredGridND

from .fd_boundary import FDBoundary, BoundaryId
from .fd_subdomain import FDSubDomain, SubDomainId

class FDDomain(Domain):
    def __init__(self, structured_grid: StructuredGridND):
        self._grid = structured_grid
        self._subdomains = dict[SubDomainId, FDSubDomain]()
        self._boundaries = dict[BoundaryId, FDBoundary]()
        self._grid_boundaries = dict[int, tuple[BoundaryId, BoundaryId]]()
        self._next_boundary_id = 1
        self._next_subdomain_id = 1
        self._mark_grid_boundaries()
        self._mark_grid_interior()

    @property
    def grid(self) -> StructuredGridND: return self._grid

    @property
    def subdomains(self) -> list[SubDomainId]:
        return list(self._subdomains.values())

    @property
    def boundaries(self) -> list[BoundaryId]: pass
    
    def grid_boundaries(self, ax: int) -> tuple[BoundaryId, BoundaryId]:
        return self._grid_boundaries[ax]
    
    def interior_id(self) -> SubDomainId:
        return self._interior_subdomain_id

    def _mark_grid_interior(self):
        subdomain_id = self.mark_as_subdomain(self.grid.interior_ids)
        self._interior_subdomain_id = subdomain_id

    def _mark_grid_boundaries(self):
        for ax in range(self._grid.ndim):
            left = self._grid.left_ids(axis=ax)
            right = self._grid.right_ids(axis=ax)
            left_id = self.mark_as_boundary(left)
            right_id = self.mark_as_boundary(right)
            self._grid_boundaries[ax] = (left_id, right_id)

    def mark_as_subdomain(self, ids: np.ndarray) -> FDSubDomain:
        subdomain_id = SubDomainId(self._next_subdomain_id)
        subdomain = FDSubDomain(subdomain_id, ids)
        self._subdomains[subdomain_id] = subdomain
        self._next_subdomain_id = self._next_subdomain_id + 1
        return subdomain

    def mark_as_boundary(self, ids: np.ndarray) -> FDBoundary:
        boundary_id = BoundaryId(self._next_boundary_id)
        boundary = FDBoundary(boundary_id, ids)
        self._boundaries[boundary_id] = boundary
        self._next_boundary_id = self._next_boundary_id + 1
        return boundary

    def subdomain(self, subdomain_id: SubDomainId) -> Optional[FDSubDomain]:
        return self._subdomains.get(subdomain_id, None)
    
    def boundary(self, boundary_id: BoundaryId) -> Optional[FDBoundary]:
        return self._boundaries.get(boundary_id, None)