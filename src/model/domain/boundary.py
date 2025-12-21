from .boundary_id import BoundaryId

class Boundary():
    def __init__(
        self,
        boundary_id: BoundaryId
    ):
        self._id = boundary_id

    @property
    def id(self) -> BoundaryId: return self._id
