class Tensor:

    def __init__(
        self,
        data,
        _children=(),
        op_="",
        label="",
        dtype=np.float32,
        keep_gradient=False,
    ):
        if not isinstance(data, np.ndarray):
            raise ValueError("Numpy Array Expected")
        self.data = data
        self.shape = data.shape
        self.size = data.size
        self.prev_ = set(_children)
        self._backward = lambda: None
        self.grad = 0
        self.keep_gradient = keep_gradient
        self.dtype = dtype
        self.op_ = op_
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data + other.data,
            _children=(self, other),
            op_="+",
        )

        def __backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = __backward
        return out

    def __matmul__(self, other):
        assert isinstance(other, Tensor)
        t = Tensor(self.data @ other.data, _children=(self, other), op_="@")

        def __backward():
            self.grad += other.data.T
            other.grad += self.data.T

        t._backward = __backward
        return t

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data * other.data,
            _children=(self, other),
            op_="*",
        )

        def __backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = __backward

        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        t = (math.e ** (2 * self.data) - 1) / (math.e ** (2 * self.data) + 1)
        out = Tensor(t, _children=(self,), op_="tanh")

        def __backward():
            self.grad = 1 - t**2

        out._backward = __backward
        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def exp(self):
        x = self.data
        out = Tensor(math.exp(x), (self,), "exp")

        def __backward():
            self.grad += out.data * out.grad

        out._backward = __backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev_:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def trace(root):
    nodes, edges = set(), set()

    def _build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev_:
                edges.add((child, v))
                _build(child)

    _build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="png", graph_attr={"rankdir": "LR"})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f}"
            % (
                n.label,
                n.data,
                n.grad,
            ),
            shape="record",
        )
        if n.op_:
            dot.node(name=uid + n.op_, label=n.op_)
            dot.edge(uid + n.op_, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op_)

    return dot
