"""Reference implementations for notebook 04 -- backpropagation.

Every backward function takes the upstream gradient `dout` first, then whatever
it cached on the forward pass, and returns the gradient with respect to each
input. That is the signature every framework uses.
"""
import numpy as np


# ---------------------------------------------------------------- the gates
def add_backward(dout, x, y):
    """z = x + y. The add gate is a gradient DISTRIBUTOR: both inputs get dout."""
    return dout * np.ones_like(x), dout * np.ones_like(y)


def mul_backward(dout, x, y):
    """z = x * y. The multiply gate is a SWAPPER: each input gets the other's value."""
    return dout * y, dout * x


def max_backward(dout, x, y):
    """z = max(x, y). The max gate is a ROUTER: all of dout to the winner, 0 to the loser.

    Ties send the gradient to x, matching numpy's `>=` convention here.
    """
    mask = (x >= y)
    return dout * mask, dout * (~mask)


def sigmoid_backward(dout, x):
    """s = 1/(1+e^-x).  ds/dx = s(1-s) -- expressed via the OUTPUT, not the input."""
    s = 1.0 / (1.0 + np.exp(-x))
    return dout * s * (1.0 - s)


def relu_backward(dout, x):
    """ReLU passes gradient where the input was positive and blocks it elsewhere."""
    return dout * (x > 0)


def matmul_backward(dout, x, w):
    """out = x.dot(w).  x (N, D), w (D, M), dout (N, M) -> (dx, dw).

    The full Jacobian would be (N, M, N, D). Never build it -- the answer is
    always two matrix multiplies, and you can recover them from shapes alone.
    """
    dx = dout.dot(w.T)      # (N, M) x (M, D) -> (N, D), matching x
    dw = x.T.dot(dout)      # (D, N) x (N, M) -> (D, M), matching w
    return dx, dw


# ------------------------------------------------- a full staged computation
def staged_forward(w0, w1, w2, x0, x1):
    """f = sigmoid(w0*x0 + w1*x1 + w2). Returns (f, cache of every intermediate)."""
    a = w0 * x0
    b = w1 * x1
    c = a + b
    d = c + w2
    f = 1.0 / (1.0 + np.exp(-d))
    return f, (w0, w1, w2, x0, x1, a, b, c, d, f)


def staged_backward(cache):
    """Hand-rolled backward pass. Returns (dw0, dw1, dw2, dx0, dx1)."""
    w0, w1, w2, x0, x1, a, b, c, d, f = cache

    dd = f * (1.0 - f)            # through the sigmoid, starting from df = 1
    dc = dd * 1.0                 # d = c + w2  -> add distributes
    dw2 = dd * 1.0
    da = dc * 1.0                 # c = a + b   -> add distributes
    db = dc * 1.0
    dw0, dx0 = da * x0, da * w0   # a = w0 * x0 -> multiply swaps
    dw1, dx1 = db * x1, db * w1
    return dw0, dw1, dw2, dx0, dx1


def branching_backward(dout, x, y, z):
    """f = (x + y) * max(y, z).  Note y is used TWICE -- gradients ADD.

    Returns (dx, dy, dz).
    """
    s = x + y
    m = np.maximum(y, z)

    ds = dout * m                 # multiply gate swaps
    dm = dout * s

    dx = ds                       # add gate distributes
    dy = ds                       # ... first contribution to y

    y_wins = (y >= z)
    dy = dy + dm * y_wins         # ... second contribution, ADDED not overwritten
    dz = dm * (~y_wins)
    return dx, dy, dz


# ---------------------------------------------------------------- test cases
def _pair(seed=0, shape=(4, 3)):
    rng = np.random.RandomState(seed)
    return (rng.randn(*shape), rng.randn(*shape), rng.randn(*shape))


def _gate_case(seed=0, shape=(4, 3)):
    dout, x, y = _pair(seed, shape)
    return (dout, x, y)


def _unary_case(seed=0, shape=(5, 4)):
    rng = np.random.RandomState(seed)
    return (rng.randn(*shape), rng.randn(*shape))


def _matmul_case(seed=0, n=6, d=4, m=3):
    rng = np.random.RandomState(seed)
    return (rng.randn(n, m), rng.randn(n, d), rng.randn(d, m))


def _staged_case(seed=0):
    rng = np.random.RandomState(seed)
    return tuple(rng.randn(5))


def _staged_back_case(seed=0):
    return (staged_forward(*_staged_case(seed))[1],)


def _branch_case(seed=0, shape=(4, 3)):
    rng = np.random.RandomState(seed)
    return (rng.randn(*shape), rng.randn(*shape), rng.randn(*shape),
            rng.randn(*shape))


CASES = {
    "add_backward":      lambda: [_gate_case(0), _gate_case(1, (2, 5))],
    "mul_backward":      lambda: [_gate_case(0), _gate_case(1, (2, 5))],
    "max_backward":      lambda: [_gate_case(0), _gate_case(1, (2, 5))],
    "sigmoid_backward":  lambda: [_unary_case(0), _unary_case(1, (3, 3))],
    "relu_backward":     lambda: [_unary_case(0), _unary_case(1, (3, 3))],
    "matmul_backward":   lambda: [_matmul_case(0), _matmul_case(1, 4, 7, 2)],
    "staged_forward":    lambda: [_staged_case(0), _staged_case(3)],
    "staged_backward":   lambda: [_staged_back_case(0), _staged_back_case(3)],
    "branching_backward": lambda: [_branch_case(0), _branch_case(2, (3, 2))],
}
