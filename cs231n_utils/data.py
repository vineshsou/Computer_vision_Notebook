"""CIFAR-10 loading, sized for a CPU-only machine.

Downloads the canonical python pickle version once into cs231n/data/ and caches
the decoded arrays as a single .npz so subsequent loads are instant.
"""
import os
import pickle
import warnings
import tarfile
import urllib.request

import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
TGZ = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
RAW_DIR = os.path.join(DATA_DIR, "cifar-10-batches-py")
CACHE = os.path.join(DATA_DIR, "cifar10.npz")

CLASSES = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def _download(verbose=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(TGZ):
        if verbose:
            print("Downloading CIFAR-10 (~170 MB, one time)...")

        def hook(blocks, bs, total):
            if verbose and total > 0 and blocks % 200 == 0:
                pct = 100.0 * blocks * bs / total
                print("\r  %5.1f%%" % min(pct, 100.0), end="")

        urllib.request.urlretrieve(URL, TGZ, reporthook=hook)
        if verbose:
            print("\r  100.0%  done")
    if not os.path.exists(RAW_DIR):
        if verbose:
            print("Extracting...")
        with tarfile.open(TGZ) as t:
            t.extractall(DATA_DIR)


def _decode():
    """Read the 6 pickle batches into arrays. Returns uint8 NHWC + int64 labels."""
    xs, ys = [], []
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*align.*")
    for i in range(1, 6):
        with open(os.path.join(RAW_DIR, "data_batch_%d" % i), "rb") as f:
            d = pickle.load(f, encoding="latin1")
        xs.append(d["data"])
        ys.append(d["labels"])
    Xtr = np.concatenate(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    ytr = np.concatenate(ys).astype(np.int64)

    with open(os.path.join(RAW_DIR, "test_batch"), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    Xte = np.array(d["data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    yte = np.array(d["labels"], dtype=np.int64)
    return Xtr.astype(np.uint8), ytr, Xte.astype(np.uint8), yte


def load_cifar10_raw(verbose=True):
    """Full CIFAR-10 as uint8 NHWC images in [0, 255]. Cached after first call."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"]
    _download(verbose)
    Xtr, ytr, Xte, yte = _decode()
    np.savez_compressed(CACHE, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    if verbose:
        print("Cached decoded arrays to", os.path.basename(CACHE))
    return Xtr, ytr, Xte, yte


def get_cifar10(num_train=5000, num_val=500, num_test=500, flatten=True,
                subtract_mean=True, seed=0, verbose=True):
    """A CPU-sized CIFAR-10 split, preprocessed the way CS231n assignment 1 does.

    The training set is subsampled because a 50k x 3072 float64 distance matrix
    is not something you want on a laptop. Defaults train in seconds, not hours.

    Returns dict with X_train/y_train/X_val/y_val/X_test/y_test, and 'mean_image'
    when subtract_mean is on.

    flatten=True  -> (N, 3072) float64, for kNN / linear / fully-connected nets
    flatten=False -> (N, 3, 32, 32) float32 NCHW, for convnets
    """
    Xtr, ytr, Xte, yte = load_cifar10_raw(verbose=verbose)
    rng = np.random.RandomState(seed)

    idx = rng.permutation(len(Xtr))
    tr_idx = idx[:num_train]
    val_idx = idx[num_train:num_train + num_val]
    te_idx = rng.permutation(len(Xte))[:num_test]

    X_train, y_train = Xtr[tr_idx], ytr[tr_idx]
    X_val, y_val = Xtr[val_idx], ytr[val_idx]
    X_test, y_test = Xte[te_idx], yte[te_idx]

    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)
    X_test = X_test.astype(np.float64)

    out = {}
    if subtract_mean:
        mean_image = X_train.mean(axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
        out["mean_image"] = mean_image

    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_val = X_val.reshape(len(X_val), -1)
        X_test = X_test.reshape(len(X_test), -1)
    else:
        X_train = X_train.transpose(0, 3, 1, 2).astype(np.float32)
        X_val = X_val.transpose(0, 3, 1, 2).astype(np.float32)
        X_test = X_test.transpose(0, 3, 1, 2).astype(np.float32)

    out.update(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
               X_test=X_test, y_test=y_test)
    if verbose:
        print("train %s  val %s  test %s" %
              (X_train.shape, X_val.shape, X_test.shape))
    return out
