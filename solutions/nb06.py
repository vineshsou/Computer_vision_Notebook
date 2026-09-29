"""Reference implementations for notebook 06 -- convolution and pooling from scratch.

Layout convention: NCHW, the same one PyTorch uses.
    x : (N, C, H, W)
    w : (F, C, HH, WW)   F filters, each spanning ALL C input channels
    b : (F,)
    out : (N, F, H_out, W_out)
"""
import numpy as np


def conv_output_size(H, W, HH, WW, stride, pad):
    """The formula. Returns (H_out, W_out) as floats so you can spot non-integers."""
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride
    return H_out, W_out


def conv_forward_naive(x, w, b, conv_param):
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # pad only the two spatial axes
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    hs, ws = i * stride, j * stride
                    window = xp[n, :, hs:hs + HH, ws:ws + WW]
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]

    return out, (x, w, b, conv_param)


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape

    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dxp = np.zeros_like(xp)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            db[f] += np.sum(dout[n, f])
            for i in range(H_out):
                for j in range(W_out):
                    hs, ws = i * stride, j * stride
                    d = dout[n, f, i, j]
                    # the multiply gate: each input gets the other's value
                    dw[f] += xp[n, :, hs:hs + HH, ws:ws + WW] * d
                    # overlapping windows reuse the same pixels -> ACCUMULATE
                    dxp[n, :, hs:hs + HH, ws:ws + WW] += w[f] * d

    dx = dxp[:, :, pad:pad + H, pad:pad + W] if pad > 0 else dxp
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    ph, pw = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_out = 1 + (H - ph) // stride
    W_out = 1 + (W - pw) // stride

    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    hs, ws = i * stride, j * stride
                    out[n, c, i, j] = np.max(x[n, c, hs:hs + ph, ws:ws + pw])
    return out, (x, pool_param)


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    ph, pw = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape

    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    hs, ws = i * stride, j * stride
                    window = x[n, c, hs:hs + ph, ws:ws + pw]
                    # the max gate routes all gradient to the winner
                    mask = (window == np.max(window))
                    dx[n, c, hs:hs + ph, ws:ws + pw] += mask * dout[n, c, i, j]
    return dx


# ------------------------------------------------------------------ im2col
def im2col_indices(x_shape, HH, WW, stride, pad):
    """Precompute the (k, i, j) fancy-index arrays that gather every patch."""
    N, C, H, W = x_shape
    H_out = (H + 2 * pad - HH) // stride + 1
    W_out = (W + 2 * pad - WW) // stride + 1

    i0 = np.repeat(np.arange(HH), WW)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(H_out), W_out)
    j0 = np.tile(np.arange(WW), HH * C)
    j1 = stride * np.tile(np.arange(W_out), H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), HH * WW).reshape(-1, 1)
    return k, i, j


def im2col(x, HH, WW, stride, pad):
    """Flatten every receptive field into a column. -> (C*HH*WW, N*H_out*W_out)."""
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    k, i, j = im2col_indices(x.shape, HH, WW, stride, pad)
    cols = xp[:, k, i, j]                      # (N, C*HH*WW, H_out*W_out)
    C = x.shape[1]
    return cols.transpose(1, 2, 0).reshape(C * HH * WW, -1)


def col2im(cols, x_shape, HH, WW, stride, pad):
    """Scatter columns back, ADDING where receptive fields overlapped."""
    N, C, H, W = x_shape
    Hp, Wp = H + 2 * pad, W + 2 * pad
    xp = np.zeros((N, C, Hp, Wp), dtype=cols.dtype)
    k, i, j = im2col_indices(x_shape, HH, WW, stride, pad)
    cols_r = cols.reshape(C * HH * WW, -1, N).transpose(2, 0, 1)
    np.add.at(xp, (slice(None), k, i, j), cols_r)
    return xp if pad == 0 else xp[:, :, pad:pad + H, pad:pad + W]


def conv_forward_im2col(x, w, b, conv_param):
    """Same result as conv_forward_naive, as one matrix multiply."""
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    cols = im2col(x, HH, WW, stride, pad)          # (C*HH*WW, N*H_out*W_out)
    res = w.reshape(F, -1).dot(cols) + b.reshape(-1, 1)
    out = res.reshape(F, H_out, W_out, N).transpose(3, 0, 1, 2)
    return np.ascontiguousarray(out), (x, w, b, conv_param, cols)


# ---------------------------------------------------------------- test cases
def _conv_case(seed=0, N=2, C=3, H=6, W=6, F=4, HH=3, WW=3, stride=1, pad=1):
    rng = np.random.RandomState(seed)
    return (rng.randn(N, C, H, W), rng.randn(F, C, HH, WW), rng.randn(F),
            {'stride': stride, 'pad': pad})


def _conv_back_case(seed=0, **kw):
    x, w, b, cp = _conv_case(seed, **kw)
    out, cache = conv_forward_naive(x, w, b, cp)
    return (np.random.RandomState(seed + 11).randn(*out.shape), cache)


def _pool_case(seed=0, N=2, C=3, H=6, W=6):
    rng = np.random.RandomState(seed)
    return (rng.randn(N, C, H, W),
            {'pool_height': 2, 'pool_width': 2, 'stride': 2})


def _pool_back_case(seed=0, **kw):
    x, pp = _pool_case(seed, **kw)
    out, cache = max_pool_forward_naive(x, pp)
    return (np.random.RandomState(seed + 13).randn(*out.shape), cache)


def _size_case(seed=0):
    return (32, 32, 5, 5, 1, 2)


CASES = {
    "conv_output_size": lambda: [(32, 32, 5, 5, 1, 2), (7, 9, 3, 3, 2, 0)],
    "conv_forward_naive": lambda: [
        _conv_case(0),
        _conv_case(1, N=1, C=2, H=7, W=5, F=3, HH=3, WW=3, stride=2, pad=1)],
    "conv_backward_naive": lambda: [
        _conv_back_case(0),
        _conv_back_case(1, N=1, C=2, H=7, W=5, F=3, HH=3, WW=3, stride=2, pad=1)],
    "max_pool_forward_naive":  lambda: [_pool_case(0), _pool_case(1, 1, 2, 8, 8)],
    "max_pool_backward_naive": lambda: [_pool_back_case(0),
                                        _pool_back_case(1, N=1, C=2, H=8, W=8)],
    "im2col": lambda: [
        (np.random.RandomState(0).randn(2, 3, 5, 5), 3, 3, 1, 1),
        (np.random.RandomState(1).randn(1, 2, 6, 4), 2, 2, 2, 0)],
    "conv_forward_im2col": lambda: [
        _conv_case(0),
        _conv_case(1, N=1, C=2, H=7, W=5, F=3, HH=3, WW=3, stride=2, pad=1)],
}
