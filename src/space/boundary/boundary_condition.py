from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..space_bound import SpaceBound, Space

import domain
class BoundaryCondition(SpaceBound):
    def __init__(
        self,
        space: Space,
        boundary: domain.Boundary
    ):
        super().__init__(space)
        self._boundary = boundary

BC = BoundaryCondition