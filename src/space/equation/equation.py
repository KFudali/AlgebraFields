import algebra

class Equation():
    def __init__(self, operator: algebra.Operator):
        self._op = operator
    
    def solve(self) -> algebra.Expression: pass

# class FieldEquation():
#     def solve(self) -> FieldExpression: pass