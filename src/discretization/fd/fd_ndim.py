from ..discretization import Discretization
from mesh.structured import StructuredGridND

class FDDiscretization(Discretization):
    def __init__(
        self,
        grid: StructuredGridND
    ):
        self._grid = grid

    