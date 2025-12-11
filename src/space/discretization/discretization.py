from abc import ABC, abstractmethod
from typing import List, Callable
import numpy as np

from .boundary_discretization import BoundaryDiscretization

class Discretization(ABC):
    @property
    @abstractmethod
    def boundary(self) -> BoundaryDiscretization: pass

    @abstractmethod
    def n_values(self) -> int:
        pass

    @abstractmethod
    def grad(self) -> List[np.ndarray]:
        pass

    @abstractmethod
    def div(self) -> np.ndarray:
        pass

    @abstractmethod
    def curl(self) -> List[np.ndarray]:
        pass

    @abstractmethod
    def laplace(self) -> np.ndarray:
        pass

    @abstractmethod
    def mass(self) -> np.ndarray:
        pass

    @abstractmethod
    def integrate(
        self, 
        f: Callable[[np.ndarray], np.ndarray], 
        domain_id: int = None
    ) -> float:
        pass
