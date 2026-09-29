"""Reference implementations for notebook 08 -- optimizers, schedules, dropout.

Every update rule has the same signature, the one every framework uses:

    next_w, config = rule(w, dw, config)

`config` carries the hyperparameters AND the optimizer's own state (velocity,
running averages, step count), so the rule itself is stateless.
"""
import numpy as np


# ---------------------------------------------------------------- optimizers
def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    return w - config['learning_rate'] * dw, config


def sgd_momentum(w, dw, config=None):
    """v = mu*v - lr*dw ;  w += v

    The velocity accumulates a running direction, so consistent gradient
    components build up while oscillating ones cancel.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v

    config['velocity'] = v
    return next_w, config


def nesterov_momentum(w, dw, config=None):
    """Look-ahead momentum, in the standard reformulated form.

    Plain momentum steps with the gradient at the CURRENT point. Nesterov
    evaluates it where the momentum is about to take you, which lets it start
    correcting a moment earlier.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))
    mu, lr = config['momentum'], config['learning_rate']

    v_prev = v
    v = mu * v - lr * dw
    next_w = w - mu * v_prev + (1 + mu) * v

    config['velocity'] = v
    return next_w, config


def adagrad(w, dw, config=None):
    """Per-parameter learning rate scaled by 1/sqrt(sum of squared gradients).

    The denominator only ever grows, so the effective step size decays
    monotonically to zero -- which is why AdaGrad stalls on deep nets.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('epsilon', 1e-8)
    cache = config.get('cache', np.zeros_like(w))

    cache = cache + dw ** 2
    next_w = w - config['learning_rate'] * dw / (np.sqrt(cache) + config['epsilon'])

    config['cache'] = cache
    return next_w, config


def rmsprop(w, dw, config=None):
    """AdaGrad with a LEAKY accumulator, so the step size can recover."""
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    cache = config.get('cache', np.zeros_like(w))

    cache = config['decay_rate'] * cache + (1 - config['decay_rate']) * dw ** 2
    next_w = w - config['learning_rate'] * dw / (np.sqrt(cache) + config['epsilon'])

    config['cache'] = cache
    return next_w, config


def adam(w, dw, config=None):
    """RMSProp + momentum + bias correction.

    m is a running mean of the gradient (the momentum part).
    v is a running mean of its square (the RMSProp part).
    Both start at zero, so early estimates are biased toward zero; the
    /(1 - beta**t) terms undo exactly that bias. Skipping them makes the first
    few hundred steps far too small.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    m = config.get('m', np.zeros_like(w))
    v = config.get('v', np.zeros_like(w))
    t = config.get('t', 0)

    t += 1
    beta1, beta2 = config['beta1'], config['beta2']
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    next_w = w - config['learning_rate'] * m_hat / (np.sqrt(v_hat) + config['epsilon'])

    config['m'], config['v'], config['t'] = m, v, t
    return next_w, config


# ------------------------------------------------------- learning-rate schedules
def lr_step_decay(lr0, epoch, drop=0.5, every=5):
    return lr0 * (drop ** (epoch // every))


def lr_exponential_decay(lr0, epoch, k=0.1):
    return lr0 * np.exp(-k * epoch)


def lr_inverse_decay(lr0, epoch, k=0.1):
    return lr0 / (1.0 + k * epoch)


def lr_cosine(lr0, epoch, total_epochs):
    return 0.5 * lr0 * (1.0 + np.cos(np.pi * epoch / total_epochs))


def lr_warmup_cosine(lr0, epoch, total_epochs, warmup=5):
    if epoch < warmup:
        return lr0 * (epoch + 1) / float(warmup)
    prog = (epoch - warmup) / float(max(total_epochs - warmup, 1))
    return 0.5 * lr0 * (1.0 + np.cos(np.pi * prog))


# ---------------------------------------------------------------- dropout
def dropout_forward(x, dropout_param):
    """Inverted dropout.

    dropout_param: 'p' = probability of KEEPING a unit, 'mode', optional 'seed'.

    Scaling by 1/p at TRAIN time (rather than by p at test time) is what
    "inverted" means, and it is the reason test-time code is completely
    unchanged -- which is what you want, because test-time cost matters.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if dropout_param.get('seed') is not None:
        np.random.seed(dropout_param['seed'])

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        cache = (dropout_param, mask)
    elif mode == 'test':
        out = x
        cache = (dropout_param, None)
    else:
        raise ValueError('bad dropout mode %r' % mode)
    return out.astype(x.dtype, copy=False), cache


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    if dropout_param['mode'] == 'train':
        return dout * mask
    return dout


# ---------------------------------------------------------------- test cases
def _opt_case(seed=0, shape=(4, 5), extra=None):
    rng = np.random.RandomState(seed)
    cfg = {'learning_rate': 1e-2}
    if extra:
        cfg.update(extra)
    return (rng.randn(*shape), rng.randn(*shape), cfg)


def _drop_case(seed=0, mode='train', p=0.6):
    rng = np.random.RandomState(seed)
    return (rng.randn(10, 8),
            {'p': p, 'mode': mode, 'seed': 123})


def _drop_back_case(seed=0, mode='train'):
    x, dp = _drop_case(seed, mode)
    _, cache = dropout_forward(x, dict(dp))
    return (np.random.RandomState(seed + 3).randn(*x.shape), cache)


CASES = {
    "sgd":               lambda: [_opt_case(0), _opt_case(1, (3, 3))],
    "sgd_momentum":      lambda: [_opt_case(0, extra={'momentum': 0.9}),
                                  _opt_case(1, (3, 3), {'momentum': 0.5})],
    "nesterov_momentum": lambda: [_opt_case(0, extra={'momentum': 0.9}),
                                  _opt_case(1, (3, 3), {'momentum': 0.5})],
    "adagrad":           lambda: [_opt_case(0), _opt_case(1, (3, 3))],
    "rmsprop":           lambda: [_opt_case(0, extra={'decay_rate': 0.99}),
                                  _opt_case(1, (3, 3), {'decay_rate': 0.9})],
    "adam":              lambda: [_opt_case(0, extra={'learning_rate': 1e-3}),
                                  _opt_case(1, (3, 3), {'learning_rate': 1e-3, 't': 5})],
    "lr_step_decay":        lambda: [(1.0, 0), (1.0, 7), (1.0, 12)],
    "lr_exponential_decay": lambda: [(1.0, 0), (1.0, 7)],
    "lr_inverse_decay":     lambda: [(1.0, 0), (1.0, 7)],
    "lr_cosine":            lambda: [(1.0, 0, 20), (1.0, 10, 20), (1.0, 20, 20)],
    "dropout_forward":  lambda: [_drop_case(0), _drop_case(1, 'test'),
                                 _drop_case(2, 'train', 0.25)],
    "dropout_backward": lambda: [_drop_back_case(0), _drop_back_case(1, 'test')],
}


# ------------------------------------------------- a solver taking any rule
from solutions.nb05 import (affine_forward, affine_backward, relu_forward,
                            relu_backward, softmax_loss)


class DropoutNet(object):
    """affine - relu - [dropout] - affine - softmax.

    dropout_keep=1.0 disables dropout entirely.
    """

    def __init__(self, input_dim=3072, hidden_dim=200, num_classes=10,
                 reg=0.0, dropout_keep=1.0, seed=0):
        rng = np.random.RandomState(seed)
        self.reg = reg
        self.dropout_keep = dropout_keep
        self.params = {
            'W1': rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
            'b1': np.zeros(hidden_dim),
            'W2': rng.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim),
            'b2': np.zeros(num_classes),
        }

    def loss(self, X, y=None, mode='test'):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        a, cache_a = affine_forward(X, W1, b1)
        h, cache_h = relu_forward(a)

        use_drop = self.dropout_keep < 1.0
        if use_drop:
            dp = {'p': self.dropout_keep, 'mode': mode}
            h, cache_d = dropout_forward(h, dp)

        scores, cache_s = affine_forward(h, W2, b2)
        if y is None:
            return scores

        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        grads = {}
        dh, grads['W2'], grads['b2'] = affine_backward(dscores, cache_s)
        if use_drop:
            dh = dropout_backward(dh, cache_d)
        da = relu_backward(dh, cache_h)
        _, grads['W1'], grads['b1'] = affine_backward(da, cache_a)
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        return loss, grads


def train_with_rule(model, X, y, X_val, y_val, rule, config,
                    num_epochs=10, batch_size=200, seed=0,
                    lr_schedule=None, verbose=False):
    """Train `model` using any update rule with the (w, dw, config) signature.

    lr_schedule(epoch) -> learning rate, applied at the start of each epoch.
    """
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    iters = max(N // batch_size, 1)
    configs = {p: dict(config) for p in model.params}
    hist = {'loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    for ep in range(num_epochs):
        if lr_schedule is not None:
            for p in configs:
                configs[p]['learning_rate'] = lr_schedule(ep)
        hist['lr'].append(configs[list(configs)[0]]['learning_rate'])

        perm = rng.permutation(N)
        for it in range(iters):
            idx = perm[it * batch_size:(it + 1) * batch_size]
            loss, grads = model.loss(X[idx], y[idx], mode='train')
            hist['loss'].append(loss)
            for p in model.params:
                model.params[p], configs[p] = rule(model.params[p], grads[p],
                                                   configs[p])

        tr = float(np.mean(np.argmax(model.loss(X[:1000]), axis=1) == y[:1000]))
        va = float(np.mean(np.argmax(model.loss(X_val), axis=1) == y_val))
        hist['train_acc'].append(tr)
        hist['val_acc'].append(va)
        if verbose:
            print("  epoch %2d  loss %.4f  train %.3f  val %.3f"
                  % (ep + 1, loss, tr, va))
    return hist
