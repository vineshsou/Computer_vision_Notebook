"""Numerical gradient checking -- the single most useful debugging tool in this course.

If your analytic gradient disagrees with the numeric one, your analytic gradient
is wrong. This is how you find out before wasting an hour wondering why the loss
will not go down.
"""
import numpy as np


def rel_error(a, b, eps=1e-8):
    """Relative error between two arrays. The standard CS231n comparison metric.

    Rule of thumb for gradient checks on float64:
        > 1e-2   almost certainly a bug
        1e-2 .. 1e-4  uncomfortable; fine only if there are kinks (ReLU, max)
        1e-4 .. 1e-7  usually fine
        < 1e-7   great
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.max(np.abs(a - b) / denom))


def eval_numerical_gradient(f, x, h=1e-5, verbose=False):
    """Centred-difference gradient of a SCALAR-valued f at x. Modifies nothing.

    f: callable taking an array like x and returning a float
    x: np.ndarray -- gradient is computed elementwise, so this is O(2 * x.size)
       function evaluations. Only use it on small arrays.
    """
    x = x.astype(np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old = x[ix]
        x[ix] = old + h
        fxph = f(x)
        x[ix] = old - h
        fxmh = f(x)
        x[ix] = old
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """Numeric version of a backward pass for an ARRAY-valued f.

    Computes  dL/dx  given  df = dL/d(f(x)),  by finite-differencing f and
    contracting with df. This is what you use to check layer backward passes.
    """
    x = x.astype(np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old = x[ix]
        x[ix] = old + h
        pos = np.array(f(x), dtype=np.float64, copy=True)
        x[ix] = old - h
        neg = np.array(f(x), dtype=np.float64, copy=True)
        x[ix] = old
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5, seed=0):
    """Spot-check a few random coordinates instead of the whole array.

    Use this when x is big (e.g. a 3073 x 10 weight matrix) and a full numeric
    gradient would take minutes. Prints each comparison and returns the worst
    relative error seen.
    """
    rng = np.random.RandomState(seed)
    worst = 0.0
    for _ in range(num_checks):
        ix = tuple(rng.randint(m) for m in x.shape)
        old = x[ix]
        x[ix] = old + h
        fxph = f(x)
        x[ix] = old - h
        fxmh = f(x)
        x[ix] = old

        num = (fxph - fxmh) / (2 * h)
        ana = analytic_grad[ix]
        err = abs(num - ana) / max(abs(num) + abs(ana), 1e-12)
        worst = max(worst, err)
        print("  numeric %+.6f   analytic %+.6f   rel error %.2e" % (num, ana, err))
    return worst
