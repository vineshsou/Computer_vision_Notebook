"""Reference implementations for notebook 07 -- training neural networks, part I.

Activation functions, weight initialisation, and batch normalisation.
"""
import numpy as np


# ------------------------------------------------------- activation functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh_forward(x):
    out = np.tanh(x)
    return out, out


def tanh_backward(dout, cache):
    out = cache
    return dout * (1.0 - out ** 2)


def leaky_relu_forward(x, alpha=0.01):
    out = np.where(x > 0, x, alpha * x)
    return out, (x, alpha)


def leaky_relu_backward(dout, cache):
    x, alpha = cache
    return dout * np.where(x > 0, 1.0, alpha)


# ---------------------------------------------------------- initialisation
def init_weights(fan_in, fan_out, mode, seed=0):
    """Return a (fan_in, fan_out) weight matrix under the named scheme.

    'zeros'  -- broken by symmetry, included so you can watch it fail
    'small'  -- 0.01 * randn, vanishes with depth
    'large'  -- 1.0  * randn, saturates/explodes with depth
    'xavier' -- randn / sqrt(fan_in),      for tanh/sigmoid
    'he'     -- randn * sqrt(2 / fan_in),  for ReLU
    """
    rng = np.random.RandomState(seed)
    if mode == 'zeros':
        return np.zeros((fan_in, fan_out))
    if mode == 'small':
        return 0.01 * rng.randn(fan_in, fan_out)
    if mode == 'large':
        return 1.0 * rng.randn(fan_in, fan_out)
    if mode == 'xavier':
        return rng.randn(fan_in, fan_out) / np.sqrt(fan_in)
    if mode == 'he':
        return rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
    raise ValueError("unknown mode %r" % mode)


def forward_stats(x, weights, activation):
    """Push x through a stack of weight matrices, recording activations.

    weights    : list of (D, D) arrays
    activation : callable applied after each matmul
    Returns a list of the activation array produced by each layer.
    """
    acts = []
    h = x
    for w in weights:
        h = activation(h.dot(w))
        acts.append(h)
    return acts


# ------------------------------------------------------- batch normalisation
def batchnorm_forward(x, gamma, beta, bn_param):
    """x (N, D). bn_param has 'mode', and optionally 'eps', 'momentum',
    'running_mean', 'running_var'. Returns (out, cache).
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    N, D = x.shape

    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    if mode == 'train':
        mu = x.mean(axis=0)
        var = x.var(axis=0)
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = (x - mu) * inv_std
        out = gamma * x_hat + beta

        # exponential moving average, used at test time
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        cache = (x, x_hat, gamma, mu, var, inv_std, eps)

    elif mode == 'test':
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        cache = None
    else:
        raise ValueError('bad batchnorm mode %r' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    """-> (dx, dgamma, dbeta).

    The simplified form. Derived by differentiating through mu and var rather
    than backpropping the graph node by node -- far shorter and much faster.
    """
    x, x_hat, gamma, mu, var, inv_std, eps = cache
    N = x.shape[0]

    dbeta = dout.sum(axis=0)
    dgamma = (dout * x_hat).sum(axis=0)

    dx_hat = dout * gamma
    dx = inv_std / N * (N * dx_hat
                        - dx_hat.sum(axis=0)
                        - x_hat * (dx_hat * x_hat).sum(axis=0))
    return dx, dgamma, dbeta


def batchnorm_forward_naive_graph(x, gamma, beta, eps=1e-5):
    """The same forward, staged node by node -- used in the notebook to show
    that the simplified backward agrees with a literal graph backward."""
    N, D = x.shape
    mu = x.sum(axis=0) / N
    xc = x - mu
    sq = xc ** 2
    var = sq.sum(axis=0) / N
    sqrtvar = np.sqrt(var + eps)
    inv_std = 1.0 / sqrtvar
    x_hat = xc * inv_std
    out = gamma * x_hat + beta
    return out, (x, mu, xc, sq, var, sqrtvar, inv_std, x_hat, gamma, eps)


# ---------------------------------------------------------------- test cases
def _act_case(seed=0, shape=(5, 4)):
    return (np.random.RandomState(seed).randn(*shape),)


def _act_back_case(seed=0, shape=(5, 4), which='tanh'):
    rng = np.random.RandomState(seed)
    x = rng.randn(*shape)
    if which == 'tanh':
        _, cache = tanh_forward(x)
    else:
        _, cache = leaky_relu_forward(x)
    return (rng.randn(*shape), cache)


def _init_case(seed=0):
    return (20, 30, 'he', 1)


def _stats_case(seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randn(40, 12)
    ws = [init_weights(12, 12, 'xavier', seed=i) for i in range(4)]
    return ((x, ws, np.tanh), {})


def _bn_case(seed=0, n=8, d=5, mode='train'):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d) * 3.0 + 7.0
    return (x, rng.randn(d) + 1.0, rng.randn(d),
            {'mode': mode, 'eps': 1e-5, 'momentum': 0.9,
             'running_mean': np.zeros(d), 'running_var': np.ones(d)})


def _bn_back_case(seed=0, n=8, d=5):
    x, gamma, beta, bn_param = _bn_case(seed, n, d)
    _, cache = batchnorm_forward(x, gamma, beta, dict(bn_param))
    return (np.random.RandomState(seed + 21).randn(n, d), cache)


CASES = {
    "tanh_forward":        lambda: [_act_case(0), _act_case(1, (3, 6))],
    "tanh_backward":       lambda: [_act_back_case(0), _act_back_case(1, (3, 6))],
    "leaky_relu_forward":  lambda: [_act_case(0), _act_case(1, (3, 6))],
    "leaky_relu_backward": lambda: [_act_back_case(0, which='lrelu'),
                                    _act_back_case(1, (3, 6), 'lrelu')],
    "init_weights": lambda: [(20, 30, 'he', 1), (16, 8, 'xavier', 2),
                             (5, 5, 'small', 3), (5, 5, 'zeros', 0)],
    "forward_stats": lambda: [_stats_case(0), _stats_case(4)],
    "batchnorm_forward": lambda: [_bn_case(0), _bn_case(1, 12, 3),
                                  _bn_case(2, 6, 4, 'test')],
    "batchnorm_backward": lambda: [_bn_back_case(0), _bn_back_case(1, 12, 3)],
}
