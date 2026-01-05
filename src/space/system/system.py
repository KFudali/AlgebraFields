from abc import ABC, abstractmethod
from algebra import Operator, Expression

class EqSystem(ABC):
    def __init__(self):
        super().__init__()

    def solve(self) -> Expression: pass