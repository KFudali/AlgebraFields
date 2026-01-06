from abc import ABC, abstractmethod
from tools.algebra import Operator, Expression

class EqSystem(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def solve(self) -> Expression: pass