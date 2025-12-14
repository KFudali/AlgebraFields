from ..discretization import Discretization
from ....mesh.element import ElementMesh2D


class FEMDiscretization2D(Discretization):
    def __init__(
        self,
        mesh: ElementMesh2D,
        base_function_order: int = 1
    ):
        super().__init__()