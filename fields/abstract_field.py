import abc

class AbstractField(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def laplace(self) -> "AbstractField":
        pass
    
    @abc.abstractmethod
    def gradient(self) -> "AbstractField":
        pass
    
    @abc.abstractmethod
    def partial_derivative(self):
        pass