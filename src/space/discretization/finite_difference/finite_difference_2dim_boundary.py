from ..boundary_discretization import BoundaryDiscretization
import utils.math 

class FDBoundaryDisc2D(BoundaryDiscretization):
    def __init__(self):
        super().__init__()

    def div(self) -> utils.math.LinOp:
        pass

    def curl(self) -> List[utils.math.LinOp]:
        pass

    def laplace(self) -> utils.math.LinOp:
        pass