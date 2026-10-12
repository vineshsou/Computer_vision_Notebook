# CS231n — Deep Learning for Computer Vision, from scratch

Nine notebooks tracking Stanford CS231n lectures 1–9. Everything is built in
**numpy first**, gradient-checked, and only then handed to PyTorch — so by the
time you call `nn.Conv2d` you have written its forward and backward yourself.

This sits alongside the classical-CV notebooks in the parent directory
(`01_pixels_and_arrays` … `19_opencv_api_reference`). Those cover what images
*are*; this covers how a model learns from them.

## How the notebooks work

Cells marked `TODO` are yours to fill in. Under each one:

```python
grade('nb02', 'svm_loss_naive', svm_loss_naive)
```

compares your function against a reference on fixed inputs and prints a
relative error. Two things make this usable:

- **Nothing breaks while a TODO is empty.** Downstream cells call
  `use('nb02', 'svm_loss_naive', svm_loss_naive)`, which returns your version
  **if it is correct** and the reference otherwise. So you can run any notebook
  top to bottom on day one and read the outputs, then go back and fill things
  in. It falls back on *incorrectness*, not just on `None` — a stub that
  returns a well-shaped array of zeros is still caught.
- **You can look, but not by accident.** `show_solution('nb02',
  'svm_loss_naive')` prints the reference. Solutions live in `solutions/` and
  are never imported into your namespace by the grader.

Beyond `grade`, most sections also check your work the way you would in real
work — **numeric gradient checks**, known image kernels, sanity values like
`-log(1/10)`. Those are the habits worth keeping; `grade` is the training
wheels.

## The notebooks

| # | Notebook | CS231n | Runtime |
|---|---|---|---|
| 01 | `01_image_classification_knn.ipynb` | Lec 2 | ~14 s |
| 02 | `02_linear_classifiers_svm_softmax.ipynb` | Lec 3 | ~8 s |
| 03 | `03_optimization_sgd.ipynb` | Lec 3 | ~32 s |
| 04 | `04_backpropagation.ipynb` | Lec 4 | ~2 s |
| 05 | `05_neural_networks.ipynb` | Lec 4–5 | ~9 s |
| 06 | `06_convnets_from_scratch.ipynb` | Lec 5 | ~3 s |
| 07 | `07_training_neural_networks_1.ipynb` | Lec 6 | ~2 s |
| 08 | `08_optimizers_and_regularization.ipynb` | Lec 7 | ~4 min |
| 09 | `09_pytorch_and_architectures.ipynb` | Lec 8–9 | ~3 min |

Runtimes are for this machine (CPU-only, 24 torch threads) with all TODOs
unfilled. 08 and 09 are slow because they actually train things.

### What you implement

| Notebook | Functions |
|---|---|
| 01 | L2 distances (2-loop → 1-loop → 0-loop), L1, k-NN voting, cross-validation |
| 02 | SVM and softmax loss + gradient, naive and vectorized |
| 03 | minibatch SGD loop, prediction, loss-surface slicing |
| 04 | add/mul/max gates, sigmoid, ReLU, matmul backward, staged + branching backward |
| 05 | affine, ReLU, sandwich layers, softmax/SVM loss layers, `TwoLayerNet` |
| 06 | conv forward/backward, max-pool forward/backward, `im2col` |
| 07 | tanh, leaky ReLU, 5 init schemes, activation statistics, batchnorm fwd/bwd |
| 08 | momentum, Nesterov, AdaGrad, RMSProp, Adam, 4 LR schedules, dropout |
| 09 | param/FLOP counts, receptive field, `ResidualBlock.forward` |

## Results you should reproduce

Same data, same loss, same optimizer family — only the architecture changes:

| Notebook | Model | CIFAR-10 accuracy |
|---|---|---|
| 01 | k-NN on raw pixels | ~0.29 |
| 02–03 | linear (softmax) classifier | ~0.39 |
| 05 | 2-layer fully-connected net | ~0.46 |
| 09 | small ResNet + augmentation, 10 epochs | **~0.74** |
| 09 | the same, 25 epochs | **~0.82** |

(01–05 use a 5,000-image training subset; 09 uses 20,000.)

## Setup

Uses the existing `cv-env` in the parent directory. No new dependencies —
numpy, matplotlib and torch are already installed.

```bash
cd cs231n
../cv-env/Scripts/python.exe -m jupyter notebook     # Windows
```

**Run Jupyter from inside `cs231n/`.** The notebooks do
`sys.path.insert(0, os.path.abspath('.'))` to find `cs231n_utils`, so the
working directory has to be this one.

CIFAR-10 (~170 MB) downloads automatically on first use into `data/` and is
then cached as a single `.npz`, so it only happens once.

### Running on another machine (e.g. a MacBook)

The whole project is in the repo; `data/` is gitignored and re-downloads itself.

```bash
git clone https://github.com/vineshsou/Computer_vision_Notebook.git
cd Computer_vision_Notebook
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib torch jupyterlab
jupyter lab
```

As above, start Jupyter from the repo root. On Apple Silicon, PyTorch can use
the `mps` device; CPU also works.

**Keeping machines in sync.** Notebooks are JSON, so editing the same one on
two machines without syncing causes painful merge conflicts.

- Before you stop: `git add -A && git commit -m "..." && git push`
- Before you start: `git pull`
- Optional: `pip install nbstripout && nbstripout --install` strips cell
  outputs on commit, which keeps diffs readable.

## Layout

```
cs231n/
├── 01..09_*.ipynb          the notebooks
├── cs231n_utils/
│   ├── data.py             CIFAR-10 download, cache, CPU-sized splits
│   ├── gradcheck.py        rel_error, numeric gradients, sparse spot-checks
│   ├── checks.py           grade() / use() / show_solution()
│   └── viz.py              shared plot style and helpers
├── solutions/
│   └── nb01.py .. nb09.py  reference implementations + their test cases
└── data/                   CIFAR-10 (downloaded, not in version control)
```

## A note on the numbers

Every result in these notebooks was produced by running the notebook, not
asserted from memory. Where a result contradicts the usual story it is
reported as it came out, with an explanation:

- In notebook 01 the **one-loop** distance version is *slower* than the
  two-loop one on this machine — vectorizing is not automatically faster, and
  the reason (a 49 MB temporary per iteration, versus BLAS doing one big
  matmul) is the actual lesson.
- In notebook 08 **dropout does not help** on the setup tested, and three
  seeds show the differences are inside the noise band. The methodology point
  — check the seed spread before believing a regularization result — is worth
  more than a tidier number would have been.
- In notebook 09 **augmentation buys nothing at 10 epochs** (val 0.74 either
  way) and nearly 8 points at 25 (0.746 → 0.824). What it changes at the short
  budget is the train/val gap. Judge a regularizer at the budget you intend to
  train at.
- The weight-template and first-layer-weight pictures in notebooks 02 and 05
  are described as they actually render at these training budgets, which is
  blurrier than the canonical CS231n figures. Where the textbook picture needs
  the full 50,000 images to appear, that is said rather than implied.
