from abc import ABC, abstractmethod

class Step(ABC):
    @abstractmethod
    def execute(self): pass