"""Plot helpers shared by the series.

Colors are the documented validated categorical palette, assigned in fixed slot
order and never cycled. Two or more series always get a legend; up to four also
get a direct label at the end of the line, so identity is never colour-alone.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Categorical slots, fixed order. Light-surface steps.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8887"
GRID = "#e4e3df"


def setup_plots():
    """Call once at the top of a notebook."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
    })


def _direct_label(ax, x, y, text, color):
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                va="center", ha="left", fontsize=8, color=color, weight="semibold",
                clip_on=False)


def plot_series(ax, xs, series, title, xlabel, ylabel, direct_labels=True):
    """series: list of (name, values). Legend for >=2; direct labels for <=4."""
    n = len(series)
    for i, (name, ys) in enumerate(series):
        c = SERIES[i % len(SERIES)]
        ax.plot(xs, ys, color=c, label=name, solid_capstyle="round")
        if direct_labels and n <= 4 and len(ys):
            _direct_label(ax, xs[-1], ys[-1], name, c)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    if n >= 2:
        ax.legend(loc="best")
    if direct_labels and n <= 4:
        ax.margins(x=0.12)
    return ax


def plot_loss_curves(loss_history, train_acc=None, val_acc=None, smooth=0):
    """Standard two-panel training diagnostic: loss over iterations, accuracy over epochs."""
    has_acc = train_acc is not None
    fig, axes = plt.subplots(1, 2 if has_acc else 1,
                             figsize=(10 if has_acc else 5.2, 3.4))
    axes = np.atleast_1d(axes)

    y = np.asarray(loss_history, dtype=float)
    if smooth and len(y) > smooth:
        k = np.ones(smooth) / smooth
        y = np.convolve(y, k, mode="valid")
    # one series -> no legend box, the title names it
    axes[0].plot(np.arange(len(y)), y, color=SERIES[0], solid_capstyle="round")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].set_axisbelow(True)

    if has_acc:
        ep = np.arange(1, len(train_acc) + 1)
        s = [("train", train_acc)]
        if val_acc is not None:
            s.append(("val", val_acc))
        plot_series(axes[1], ep, s, "Accuracy", "epoch", "accuracy")
    fig.tight_layout()
    return fig


def show_images(images, labels=None, class_names=None, ncols=10, title=None,
                scale=1.0):
    """Grid of images. Accepts uint8 or float, NHWC or NCHW, with or without mean removed."""
    imgs = np.asarray(images)
    if imgs.ndim == 4 and imgs.shape[1] in (1, 3) and imgs.shape[-1] not in (1, 3):
        imgs = imgs.transpose(0, 2, 3, 1)
    n = len(imgs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 0.85 * scale, nrows * 0.95 * scale))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        im = imgs[i].astype(np.float64)
        lo, hi = im.min(), im.max()
        if lo < 0 or hi > 255 or (hi <= 1.0 and lo >= 0.0 and hi > 0):
            im = (im - lo) / max(hi - lo, 1e-12)   # rescale for display only
        else:
            im = im / 255.0
        ax.imshow(np.clip(im.squeeze(), 0, 1),
                  cmap="gray" if im.ndim == 2 or im.shape[-1] == 1 else None)
        if labels is not None:
            name = (class_names[labels[i]] if class_names is not None
                    else str(labels[i]))
            ax.set_title(name, fontsize=7, color=INK_2, pad=2)
    if title:
        fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_weight_templates(W, class_names, title="Learned class templates"):
    """Visualise a (3073, 10) or (3072, 10) linear classifier weight matrix as images."""
    Wv = W[:-1, :] if W.shape[0] % 1024 == 1 else W
    Wv = Wv.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    return show_images(Wv, labels=np.arange(Wv.shape[0]),
                       class_names=class_names, ncols=10, title=title, scale=1.3)


def plot_norm_balls(point=(1.6, 1.0)):
    """L1 diamond vs L2 circle unit balls in R^2, plus the two paths from the
    origin to `point` -- straight line (L2) vs axis-aligned steps (L1)."""
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    diamond = np.array([[1, 0, -1, 0, 1], [0, 1, 0, -1, 0]], dtype=float)

    ax.plot(circle[0], circle[1], color=SERIES[0], label=r"L2 ball: $x^2+y^2=1$")
    ax.plot(diamond[0], diamond[1], color=SERIES[1], label=r"L1 ball: $|x|+|y|=1$")
    ax.fill(circle[0], circle[1], color=SERIES[0], alpha=0.08)
    ax.fill(diamond[0], diamond[1], color=SERIES[1], alpha=0.08)

    px, py = point
    ax.plot([0, px], [0, py], color=SERIES[0], linewidth=2.2, solid_capstyle="round")
    ax.plot([0, px, px], [0, 0, py], color=SERIES[1], linewidth=2.2,
            linestyle="--", solid_capstyle="round")
    ax.scatter([0, px], [0, py], color=INK, zorder=5, s=25)
    ax.annotate("point", xy=(px, py), xytext=(6, 4), textcoords="offset points",
                fontsize=8, color=INK_2)

    l2, l1 = float(np.hypot(px, py)), float(abs(px) + abs(py))
    ax.annotate("L2 = %.2f" % l2, xy=(px * 0.5, py * 0.5 + 0.18),
                color=SERIES[0], fontsize=8, weight="semibold")
    ax.annotate("L1 = %.2f" % l1, xy=(px * 0.45, -0.3),
                color=SERIES[1], fontsize=8, weight="semibold")

    lim = max(abs(px), abs(py)) + 0.8
    ax.set_xlim(-lim, lim + 0.6); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=GRID, linewidth=0.8); ax.axvline(0, color=GRID, linewidth=0.8)
    ax.set_title("Unit balls: what 'distance 1' means, per metric")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_distance_concentration(dims=(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 3072),
                                n_points=500, seed=0):
    """(max dist - min dist) / min dist from a random query to n_points random
    points, as dimension grows. The classic concentration-of-distance picture:
    in high dimensions the nearest and farthest neighbour become equidistant."""
    rng = np.random.RandomState(seed)
    ratios = []
    for d in dims:
        pts = rng.rand(n_points, d)
        q = rng.rand(1, d)
        dists = np.sqrt(np.sum((pts - q) ** 2, axis=1))
        ratios.append((dists.max() - dists.min()) / dists.min())

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(dims, ratios, color=SERIES[0], marker="o", markersize=4,
            solid_capstyle="round")
    ax.set_xscale("log", base=2)
    ax.set_title("Nearest and farthest neighbour become equidistant")
    ax.set_xlabel("dimension"); ax.set_ylabel("(max dist − min dist) / min dist")
    ax.set_axisbelow(True)
    if 3072 in dims:
        i = dims.index(3072)
        ax.annotate("CIFAR pixel space,\nd = 3072", xy=(3072, ratios[i]),
                    xytext=(-95, 16), textcoords="offset points", fontsize=7.5,
                    color=SERIES[1])
    fig.tight_layout()
    return fig


def plot_knn_regions(X_train, y_train, dist_fn, predict_fn, k_list,
                     class_names=None, grid_res=160, pad=1.2):
    """Colour the 2D plane by kNN vote for each k in k_list.

    dist_fn(X, X_train) -> (num_X, num_train) distances.
    predict_fn(dists, y_train, k=k) -> (num_X,) predicted labels.
    Both are meant to be the notebook's own vectorised implementations
    (wrapped with `use(...)` so this runs even before the TODOs are filled in).
    """
    x_min, x_max = X_train[:, 0].min() - pad, X_train[:, 0].max() + pad
    y_min, y_max = X_train[:, 1].min() - pad, X_train[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                         np.linspace(y_min, y_max, grid_res))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    classes = np.unique(y_train)
    cmap = ListedColormap([SERIES[i % len(SERIES)] for i in range(len(classes))])

    n = len(k_list)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.3))
    axes = np.atleast_1d(axes)

    d = dist_fn(grid, X_train)
    for ax, k in zip(axes, k_list):
        pred = predict_fn(d, y_train, k=k).reshape(xx.shape)
        ax.contourf(xx, yy, pred, levels=np.arange(len(classes) + 1) - 0.5,
                   cmap=cmap, alpha=0.35)
        ax.contour(xx, yy, pred, levels=np.arange(len(classes) + 1) - 0.5,
                  colors=INK, linewidths=0.6, alpha=0.5)
        for ci, c in enumerate(classes):
            m = y_train == c
            ax.scatter(X_train[m, 0], X_train[m, 1], color=SERIES[ci % len(SERIES)],
                      s=16, edgecolor=INK, linewidth=0.3,
                      label=(class_names[c] if class_names else str(c)))
        ax.set_title("k = %d" % k)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    axes[0].legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    fig.suptitle("Nearest-neighbour decision regions in R² "
                "(the same voting rule partitions R³⁰⁷²)",
                fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_score_geometry(w, b, X=None, y=None, class_names=None, ax=None):
    """2D linear score f(x) = w.x + b as a signed-distance picture: the
    decision line, its normal vector w, and the two shaded half-planes."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.6, 4.6))

    w = np.asarray(w, dtype=float)
    lim = float(np.abs(X).max() * 1.3) if X is not None else 4.0

    xs = np.linspace(-lim, lim, 220)
    XX, YY = np.meshgrid(xs, xs)
    scores = w[0] * XX + w[1] * YY + b
    ax.contourf(XX, YY, scores, levels=[scores.min() - 1, 0, scores.max() + 1],
               colors=[SERIES[1], SERIES[0]], alpha=0.12)
    ax.contour(XX, YY, scores, levels=[0], colors=[INK], linewidths=1.6)

    w_norm2 = float(w @ w)
    foot = -b * w / w_norm2                     # point on the line nearest the origin
    arrow_len = lim * 0.3
    tip = foot + w / np.sqrt(w_norm2) * arrow_len
    ax.annotate("", xy=tuple(tip), xytext=tuple(foot),
               arrowprops=dict(arrowstyle="-|>", color=SERIES[2], lw=2.2), zorder=6)
    ax.annotate(r"$w$  ($\perp$ to the boundary)", xy=tuple(tip), xytext=(0, 11),
               textcoords="offset points", ha="center", color=SERIES[2], fontsize=8,
               weight="semibold", zorder=6,
               bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.82))

    if X is not None and y is not None:
        classes = np.unique(y)
        for ci, c in enumerate(classes):
            m = y == c
            ax.scatter(X[m, 0], X[m, 1], color=SERIES[ci % len(SERIES)], s=22,
                      edgecolor=INK, linewidth=0.3,
                      label=(class_names[c] if class_names else str(c)))
        ax.legend(loc="upper left", fontsize=7)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(r"Score $= w\cdot x + b$: a hyperplane with normal $w$")
    ax.set_axisbelow(True)
    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def plot_hyperplane_regions(W, b, X=None, y=None, class_names=None, lim=None, ax=None):
    """Toy 2D multiclass linear classifier: colour R^2 by argmax(W.x + b).

    W: (2, C), b: (C,). Each column of W is one class's score direction; the
    arrows show direction only (magnitude is normalised for visibility, since
    only the *differences* w_i - w_j set the boundaries)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.2, 5.2))

    if lim is None:
        lim = float(np.abs(X).max() * 1.4) if X is not None else 4.0

    xs = np.linspace(-lim, lim, 320)
    XX, YY = np.meshgrid(xs, xs)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    scores = grid @ W + b
    winner = np.argmax(scores, axis=1).reshape(XX.shape)

    C = W.shape[1]
    cmap = ListedColormap([SERIES[i % len(SERIES)] for i in range(C)])
    ax.contourf(XX, YY, winner, levels=np.arange(C + 1) - 0.5, cmap=cmap, alpha=0.25)
    ax.contour(XX, YY, winner, levels=np.arange(C + 1) - 0.5, colors=INK, linewidths=0.9)

    origin = np.zeros(2)
    arrow_len = lim * 0.32
    for c in range(C):
        wc = W[:, c]
        direction = wc / (np.linalg.norm(wc) + 1e-12) * arrow_len
        ax.annotate("", xy=direction, xytext=origin,
                   arrowprops=dict(arrowstyle="-|>", color=SERIES[c % len(SERIES)], lw=2))
        name = class_names[c] if class_names else ("w_%d" % c)
        ax.annotate(name, xy=direction, xytext=(4, 4), textcoords="offset points",
                   fontsize=8, color=SERIES[c % len(SERIES)], weight="semibold")

    if X is not None and y is not None:
        for c in range(C):
            m = y == c
            ax.scatter(X[m, 0], X[m, 1], color=SERIES[c % len(SERIES)], s=20,
                      edgecolor=INK, linewidth=0.3)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(r"argmax$_j\,(w_j \cdot x + b_j)$: each class owns a convex region")
    ax.set_axisbelow(True)
    if own_fig:
        fig.tight_layout()
        return fig
    return ax
