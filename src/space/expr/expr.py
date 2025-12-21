from abc import abstractmethod, ABC

class Expr(ABC):
    @abstractmethod
    def eval(self): pass