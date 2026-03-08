import algebra
from typing import TypeVar

class DiscreteOperator(algebra.Operator): pass

TDiscreteOperator = TypeVar("TDiscreteOperator", bound=DiscreteOperator)