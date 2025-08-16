import math


class Variable:
    def __init__(self, value=None, requires_grad=False, _op=None, _inputs=None):
        self._value = value
        self.requires_grad = requires_grad
        self._op = _op
        self._inputs = _inputs or []
        self._backward = None
        self.grad = 0.0 if requires_grad else None

    @property
    def value(self):
        if self._value is not None: return self._value
        if self._op is None: raise ValueError("No operation defined for computation")
        self._value = self._op(*[inp.value for inp in self._inputs])
        return self._value

    def backward(self):
        _ = self.value

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._inputs:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for v in topo:
            if v.requires_grad: v.grad = 0.0

        self.grad = 1.0

        for v in reversed(topo):
            if v._backward: v._backward()

    def __repr__(self):
        val_str = self._value if self._value is not None else "lazy"
        return f"Variable(value={val_str}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(value=other)

        def op(a, b):
            return a + b

        out = Variable(
            _op=op,
            _inputs=[self, other],
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += 1.0 * out.grad
            if other.requires_grad:
                other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * Variable(value=-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(value=other)

        def op(a, b):
            return a * b

        out = Variable(
            _op=op,
            _inputs=[self, other],
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += other.value * out.grad
            if other.requires_grad:
                other.grad += self.value * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        other = other if isinstance(other, Variable) else Variable(value=other)

        def op(a, b):
            return a**b

        out = Variable(
            _op=op,
            _inputs=[self, other],
            requires_grad=self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad and self.value != 0:
                self.grad += (
                    other.value * (self.value ** (other.value - 1))
                ) * out.grad
            if other.requires_grad and self.value > 0:
                other.grad += (out.value * math.log(self.value)) * out.grad

        out._backward = _backward
        return out

    def __rpow__(self, other):
        return Variable(value=other) ** self

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return Variable(value=other) * (self**-1)

    def exp(self):
        def op(a):
            return math.exp(a)

        out = Variable(_op=op, _inputs=[self], requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.value * out.grad

        out._backward = _backward
        return out
