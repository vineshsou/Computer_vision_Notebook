"""Reference implementations for notebook 02 -- SVM and Softmax linear classifiers.

Shape convention (CS231n assignment 1):
    X : (N, D)   rows are examples, last column is the bias-trick 1
    W : (D, C)   columns are class weight vectors
    scores = X.dot(W)  ->  (N, C)
"""
import numpy as np


def svm_loss_naive(W, X, y, reg):
    """Multiclass SVM loss with delta = 1, explicit loops."""
    dW = np.zeros_like(W)
    num_train, num_classes = X.shape[0], W.shape[1]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1.0
            if margin > 0:
                loss += margin
                # this wrong class was over the margin: push it down,
                # and push the correct class up by the same amount
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2.0 * reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    num_train = X.shape[0]
    scores = X.dot(W)                                        # (N, C)
    correct = scores[np.arange(num_train), y][:, np.newaxis]  # (N, 1)
    margins = np.maximum(0.0, scores - correct + 1.0)
    margins[np.arange(num_train), y] = 0.0                    # no loss for correct class
    loss = np.sum(margins) / num_train + reg * np.sum(W * W)

    # binary mask of which margins were violated
    mask = (margins > 0).astype(W.dtype)
    # the correct class absorbs -1 for every violated margin in its row
    mask[np.arange(num_train), y] = -np.sum(mask, axis=1)
    dW = X.T.dot(mask) / num_train + 2.0 * reg * W
    return loss, dW


def softmax_loss_naive(W, X, y, reg):
    dW = np.zeros_like(W)
    num_train, num_classes = X.shape[0], W.shape[1]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)              # numeric stability shift
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)
        loss += -np.log(probs[y[i]])
        for j in range(num_classes):
            dW[:, j] += (probs[j] - (j == y[i])) * X[i]

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2.0 * reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    loss = np.sum(-np.log(probs[np.arange(num_train), y])) / num_train
    loss += reg * np.sum(W * W)

    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1.0
    dscores /= num_train
    dW = X.T.dot(dscores) + 2.0 * reg * W
    return loss, dW


def train_linear(loss_fn, X, y, num_classes, learning_rate=1e-7, reg=2.5e4,
                 num_iters=500, batch_size=200, seed=0, verbose=False):
    """Plain minibatch SGD. Provided here; you write this properly in notebook 03."""
    rng = np.random.RandomState(seed)
    num_train, dim = X.shape
    W = 0.001 * rng.randn(dim, num_classes)
    history = []
    for it in range(num_iters):
        idx = rng.choice(num_train, batch_size, replace=True)
        loss, grad = loss_fn(W, X[idx], y[idx], reg)
        W -= learning_rate * grad
        history.append(loss)
        if verbose and it % 100 == 0:
            print("  iter %4d / %d   loss %.4f" % (it, num_iters, loss))
    return W, history


# ---------------------------------------------------------------- test cases
def _wxy(seed=0, n=12, d=7, c=4, reg=0.05):
    rng = np.random.RandomState(seed)
    W = rng.randn(d, c) * 0.1
    X = rng.randn(n, d)
    y = rng.randint(0, c, n)
    return (W, X, y, reg)


CASES = {
    "svm_loss_naive":          lambda: [_wxy(0), _wxy(1, 20, 5, 3, 0.0)],
    "svm_loss_vectorized":     lambda: [_wxy(0), _wxy(1, 20, 5, 3, 0.0)],
    "softmax_loss_naive":      lambda: [_wxy(0), _wxy(1, 20, 5, 3, 0.0)],
    "softmax_loss_vectorized": lambda: [_wxy(0), _wxy(1, 20, 5, 3, 0.0)],
}
