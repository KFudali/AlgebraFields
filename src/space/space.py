from discretization import Discretization
class Space():
    def __init__(
        self,
        discretization: Discretization
    ):
        self._disc = discretization

    @property
    def discretization(self) -> Discretization: return self._disc