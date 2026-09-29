"""Reference implementations for notebook 01 -- k-Nearest Neighbour."""
import numpy as np


def compute_distances_two_loops(X, X_train):
    num_test, num_train = X.shape[0], X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            dists[i, j] = np.sqrt(np.sum((X[i] - X_train[j]) ** 2))
    return dists


def compute_distances_one_loop(X, X_train):
    num_test = X.shape[0]
    dists = np.zeros((num_test, X_train.shape[0]))
    for i in range(num_test):
        # X[i] is (D,), X_train is (N, D) -> broadcasting gives (N, D)
        dists[i, :] = np.sqrt(np.sum((X_train - X[i]) ** 2, axis=1))
    return dists


def compute_distances_no_loops(X, X_train):
    # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2
    test_sq = np.sum(X ** 2, axis=1)[:, np.newaxis]        # (num_test, 1)
    train_sq = np.sum(X_train ** 2, axis=1)[np.newaxis, :]  # (1, num_train)
    cross = X.dot(X_train.T)                                # (num_test, num_train)
    sq = test_sq + train_sq - 2 * cross
    # floating point can push exact zeros a hair negative
    return np.sqrt(np.maximum(sq, 0))


def compute_distances_l1(X, X_train):
    # No algebraic shortcut exists for L1 -- |a-b| does not expand. The best you
    # can do without a loop is a (num_test, num_train, D) intermediate, which is
    # usually too big, so one loop is the practical answer.
    num_test = X.shape[0]
    dists = np.zeros((num_test, X_train.shape[0]))
    for i in range(num_test):
        dists[i, :] = np.sum(np.abs(X_train - X[i]), axis=1)
    return dists


def predict_labels(dists, y_train, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test, dtype=np.int64)
    for i in range(num_test):
        closest_y = y_train[np.argsort(dists[i])[:k]]
        # bincount + argmax breaks ties toward the smaller label, matching CS231n
        y_pred[i] = np.argmax(np.bincount(closest_y))
    return y_pred


def cross_validate_knn(X_train, y_train, k_choices, num_folds=5):
    """Returns {k: [acc_fold_0, ...]} using the same folds for every k."""
    X_folds = np.array_split(X_train, num_folds)
    y_folds = np.array_split(y_train, num_folds)
    accuracies = {k: [] for k in k_choices}
    for f in range(num_folds):
        X_val_f, y_val_f = X_folds[f], y_folds[f]
        X_tr_f = np.concatenate([X_folds[i] for i in range(num_folds) if i != f])
        y_tr_f = np.concatenate([y_folds[i] for i in range(num_folds) if i != f])
        dists = compute_distances_no_loops(X_val_f, X_tr_f)
        for k in k_choices:
            pred = predict_labels(dists, y_tr_f, k=k)
            accuracies[k].append(float(np.mean(pred == y_val_f)))
    return accuracies


# ---------------------------------------------------------------- test cases
def _xy(seed=0, nt=7, ntr=11, d=5):
    rng = np.random.RandomState(seed)
    return rng.randn(nt, d), rng.randn(ntr, d)


CASES = {
    "compute_distances_two_loops": lambda: [_xy(0), _xy(1, 3, 4, 9)],
    "compute_distances_one_loop":  lambda: [_xy(0), _xy(1, 3, 4, 9)],
    "compute_distances_no_loops":  lambda: [_xy(0), _xy(1, 3, 4, 9)],
    "compute_distances_l1":        lambda: [_xy(0), _xy(1, 3, 4, 9)],
    "predict_labels": lambda: [
        ((np.random.RandomState(2).rand(6, 12),
          np.random.RandomState(3).randint(0, 4, 12)), {"k": 1}),
        ((np.random.RandomState(2).rand(6, 12),
          np.random.RandomState(3).randint(0, 4, 12)), {"k": 5}),
    ],
    "cross_validate_knn": lambda: [
        ((np.random.RandomState(4).randn(40, 6),
          np.random.RandomState(5).randint(0, 3, 40), [1, 3, 5]), {"num_folds": 4}),
    ],
}
