from abc import ABC, abstractmethod
from typing import List
import utils.math

class BoundaryDiscretization(ABC):
    @abstractmethod
    def grad(self) -> List[utils.math.LinOp]:
        pass

    @abstractmethod
    def div(self) -> utils.math.LinOp:
        pass

    @abstractmethod
    def curl(self) -> List[utils.math.LinOp]:
        pass

    @abstractmethod
    def laplace(self) -> utils.math.LinOp:
        pass