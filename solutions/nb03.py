"""Reference implementations for notebook 03 -- optimization and SGD.

train_linear takes an explicit list of batch index arrays rather than drawing
them itself. That makes the loop deterministic and gradable: what is being
tested is the update rule, not how you happened to call the RNG.
"""
import numpy as np


def make_batches(num_train, batch_size, num_iters, seed=0):
    """Deterministic list of index arrays, sampled with replacement."""
    rng = np.random.RandomState(seed)
    return [rng.choice(num_train, batch_size, replace=True)
            for _ in range(num_iters)]


def predict_linear(W, X):
    """Highest-scoring class per row. X (N, D), W (D, C) -> (N,) int labels."""
    return np.argmax(X.dot(W), axis=1)


def train_linear(loss_fn, W0, X, y, batches, learning_rate, reg):
    """Vanilla minibatch SGD. Returns (W, loss_history)."""
    W = W0.copy()
    loss_history = []
    for idx in batches:
        loss, grad = loss_fn(W, X[idx], y[idx], reg)
        W -= learning_rate * grad          # the entire algorithm, one line
        loss_history.append(loss)
    return W, loss_history


def loss_along_direction(loss_fn, W, direction, alphas, X, y, reg):
    """Slice the loss surface along W + alpha * direction."""
    out = np.zeros(len(alphas))
    for i, a in enumerate(alphas):
        out[i] = loss_fn(W + a * direction, X, y, reg)[0]
    return out


def loss_surface_2d(loss_fn, W, d1, d2, alphas, betas, X, y, reg):
    """Loss on the plane W + a*d1 + b*d2. Returns (len(betas), len(alphas))."""
    Z = np.zeros((len(betas), len(alphas)))
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            Z[i, j] = loss_fn(W + a * d1 + b * d2, X, y, reg)[0]
    return Z


# ---------------------------------------------------------------- test cases
def _softmax(W, X, y, reg):
    """Local copy so nb03's cases do not depend on the learner's nb02 work."""
    n = X.shape[0]
    s = X.dot(W)
    s -= s.max(axis=1, keepdims=True)
    e = np.exp(s)
    p = e / e.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(n), y]).sum() / n + reg * np.sum(W * W)
    ds = p.copy()
    ds[np.arange(n), y] -= 1.0
    ds /= n
    return loss, X.T.dot(ds) + 2.0 * reg * W


def _train_case(seed=0, n=40, d=6, c=3, iters=25):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    W0 = rng.randn(d, c) * 0.01
    batches = make_batches(n, 8, iters, seed=seed + 100)
    return ((_softmax, W0, X, y, batches, 0.1, 0.01), {})


def _predict_case(seed=0, n=15, d=5, c=4):
    rng = np.random.RandomState(seed)
    return (rng.randn(d, c), rng.randn(n, d))


def _dir_case(seed=0, n=30, d=5, c=3):
    rng = np.random.RandomState(seed)
    W = rng.randn(d, c) * 0.1
    X, y = rng.randn(n, d), rng.randint(0, c, n)
    direction = rng.randn(d, c)
    alphas = np.linspace(-1, 1, 11)
    return ((_softmax, W, direction, alphas, X, y, 0.0), {})


CASES = {
    "predict_linear": lambda: [_predict_case(0), _predict_case(1, 8, 3, 6)],
    "train_linear": lambda: [_train_case(0), _train_case(2, 30, 4, 5, 15)],
    "loss_along_direction": lambda: [_dir_case(0), _dir_case(1)],
}
