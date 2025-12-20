from discretization import Discretization
class Space():
    def __init__(
        self,
        discretization: Discretization
    ):
        self._disc = discretization

    @property
    def ndim(self) -> int: pass
    
    @property
    def shape(self) -> tuple[int, ...]: pass

    @property
    def discretization(self) -> Discretization: return self._disc