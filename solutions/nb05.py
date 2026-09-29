"""Reference implementations for notebook 05 -- modular layers and a 2-layer net.

The layer API used from here to the end of the series:

    out, cache = layer_forward(x, *params)
    dx, *dparams = layer_backward(dout, cache)

Forward returns whatever backward will need. Nothing else is shared between
them. That contract is the whole reason deep networks are composable.
"""
import numpy as np


# ------------------------------------------------------------------ layers
def affine_forward(x, w, b):
    """x may be (N, d_1, ..., d_k); it is flattened to (N, D) with D = prod(d_i)."""
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    out = x_flat.dot(w) + b
    return out, (x, w, b)


def affine_backward(dout, cache):
    x, w, b = cache
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    dx = dout.dot(w.T).reshape(x.shape)   # restore the ORIGINAL shape
    dw = x_flat.T.dot(dout)
    db = np.sum(dout, axis=0)             # bias is broadcast over N -> sum over N
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    return out, x


def relu_backward(dout, cache):
    x = cache
    return dout * (x > 0)


def affine_relu_forward(x, w, b):
    """The 'sandwich' layer -- what every framework calls a Linear+ReLU block."""
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    return out, (fc_cache, relu_cache)


def affine_relu_backward(dout, cache):
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    return affine_backward(da, fc_cache)


def softmax_loss(x, y):
    """x (N, C) raw scores, y (N,) labels. Returns (loss, dx) averaged over N."""
    N = x.shape[0]
    shifted = x - np.max(x, axis=1, keepdims=True)
    log_sum = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - log_sum
    loss = -np.sum(log_probs[np.arange(N), y]) / N

    dx = np.exp(log_probs)
    dx[np.arange(N), y] -= 1.0
    dx /= N
    return loss, dx


def svm_loss(x, y):
    """Hinge loss as a layer, for comparison. Returns (loss, dx)."""
    N = x.shape[0]
    correct = x[np.arange(N), y][:, np.newaxis]
    margins = np.maximum(0.0, x - correct + 1.0)
    margins[np.arange(N), y] = 0.0
    loss = np.sum(margins) / N

    dx = (margins > 0).astype(x.dtype)
    dx[np.arange(N), y] = -np.sum(dx, axis=1)
    dx /= N
    return loss, dx


# ------------------------------------------------------------ the whole net
class TwoLayerNet(object):
    """affine - relu - affine - softmax."""

    def __init__(self, input_dim=3072, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0, seed=None):
        rng = np.random.RandomState(seed)
        self.reg = reg
        self.params = {
            'W1': weight_scale * rng.randn(input_dim, hidden_dim),
            'b1': np.zeros(hidden_dim),
            'W2': weight_scale * rng.randn(hidden_dim, num_classes),
            'b2': np.zeros(num_classes),
        }

    def loss(self, X, y=None):
        """y is None -> return scores. Otherwise return (loss, grads)."""
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        h, cache1 = affine_relu_forward(X, W1, b1)
        scores, cache2 = affine_forward(h, W2, b2)
        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        grads = {}
        dh, grads['W2'], grads['b2'] = affine_backward(dscores, cache2)
        _, grads['W1'], grads['b1'] = affine_relu_backward(dh, cache1)
        grads['W1'] += self.reg * W1     # d/dW of 0.5*reg*W^2 is reg*W
        grads['W2'] += self.reg * W2
        return loss, grads


def sgd_momentum_step(w, dw, config):
    """One SGD+momentum update. config holds learning_rate, momentum, velocity."""
    v = config.get('velocity', np.zeros_like(w))
    v = config['momentum'] * v - config['learning_rate'] * dw
    config['velocity'] = v
    return w + v, config


def train_net(model, X, y, X_val, y_val, learning_rate=1e-3, momentum=0.9,
              num_epochs=10, batch_size=200, lr_decay=0.95, seed=0, verbose=True):
    """Minimal solver. Returns a history dict."""
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    iters_per_epoch = max(N // batch_size, 1)
    configs = {p: {'learning_rate': learning_rate, 'momentum': momentum}
               for p in model.params}
    hist = {'loss': [], 'train_acc': [], 'val_acc': []}

    for ep in range(num_epochs):
        perm = rng.permutation(N)
        for it in range(iters_per_epoch):
            idx = perm[it * batch_size:(it + 1) * batch_size]
            loss, grads = model.loss(X[idx], y[idx])
            hist['loss'].append(loss)
            for p in model.params:
                model.params[p], configs[p] = sgd_momentum_step(
                    model.params[p], grads[p], configs[p])
        for p in configs:
            configs[p]['learning_rate'] *= lr_decay

        tr = float(np.mean(np.argmax(model.loss(X[:1000]), axis=1) == y[:1000]))
        va = float(np.mean(np.argmax(model.loss(X_val), axis=1) == y_val))
        hist['train_acc'].append(tr)
        hist['val_acc'].append(va)
        if verbose:
            print("  epoch %2d/%d   loss %.4f   train %.3f   val %.3f"
                  % (ep + 1, num_epochs, loss, tr, va))
    return hist


# ---------------------------------------------------------------- test cases
def _affine_case(seed=0, shape=(4, 2, 3), m=5):
    rng = np.random.RandomState(seed)
    x = rng.randn(*shape)
    D = int(np.prod(shape[1:]))
    return (x, rng.randn(D, m), rng.randn(m))


def _affine_back_case(seed=0, shape=(4, 2, 3), m=5):
    x, w, b = _affine_case(seed, shape, m)
    rng = np.random.RandomState(seed + 50)
    out, cache = affine_forward(x, w, b)
    return (rng.randn(*out.shape), cache)


def _relu_case(seed=0, shape=(5, 6)):
    return (np.random.RandomState(seed).randn(*shape),)


def _relu_back_case(seed=0, shape=(5, 6)):
    rng = np.random.RandomState(seed)
    x = rng.randn(*shape)
    _, cache = relu_forward(x)
    return (rng.randn(*shape), cache)


def _ar_case(seed=0):
    return _affine_case(seed, (6, 4), 3)


def _ar_back_case(seed=0):
    x, w, b = _affine_case(seed, (6, 4), 3)
    rng = np.random.RandomState(seed + 7)
    out, cache = affine_relu_forward(x, w, b)
    return (rng.randn(*out.shape), cache)


def _loss_case(seed=0, n=10, c=5):
    rng = np.random.RandomState(seed)
    return (rng.randn(n, c) * 2.0, rng.randint(0, c, n))


CASES = {
    "affine_forward":       lambda: [_affine_case(0), _affine_case(1, (3, 7), 2)],
    "affine_backward":      lambda: [_affine_back_case(0), _affine_back_case(1, (3, 7), 2)],
    "relu_forward":         lambda: [_relu_case(0), _relu_case(1, (2, 8))],
    "relu_backward":        lambda: [_relu_back_case(0), _relu_back_case(1, (2, 8))],
    "affine_relu_forward":  lambda: [_ar_case(0), _ar_case(2)],
    "affine_relu_backward": lambda: [_ar_back_case(0), _ar_back_case(2)],
    "softmax_loss":         lambda: [_loss_case(0), _loss_case(1, 7, 3)],
    "svm_loss":             lambda: [_loss_case(0), _loss_case(1, 7, 3)],
}
