"""The self-check harness.

Every TODO in these notebooks has a reference implementation living in
cs231n/solutions/nbNN.py, together with the test inputs it should be checked on.

    grade('nb02', 'svm_loss_naive', svm_loss_naive)   # compare yours to reference
    show_solution('nb02', 'svm_loss_naive')           # reveal the reference

Nothing here ever imports the reference into your namespace, so you cannot
accidentally 'pass' a check with the answer.
"""
import importlib
import inspect
import sys
import os
import traceback

import numpy as np

from .gradcheck import rel_error

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

PASS_TOL = 1e-6

_GREEN = "PASS"
_RED = "FAIL"


def _load(module):
    mod = importlib.import_module("solutions." + module)
    importlib.reload(mod)
    return mod


def _is_stub(result):
    """A not-yet-written TODO returns None (a bare `pass`), or all-None tuple."""
    if result is None:
        return True
    if isinstance(result, tuple) and len(result) and all(r is None for r in result):
        return True
    return False


def _compare(got, want, path="out"):
    """Recursively diff two return values. Yields (path, rel_error_or_None, ok)."""
    if want is None:
        # e.g. a layer that legitimately caches nothing in test mode
        yield (path, None, got is None,
               "" if got is None else "expected None, got %s" % type(got).__name__)
        return
    if got is None:
        yield (path, None, False, "returned None where a value was expected")
        return
    if isinstance(want, tuple):
        if not isinstance(got, tuple):
            yield (path, None, False, "expected a tuple of %d, got %s"
                   % (len(want), type(got).__name__))
            return
        if len(got) != len(want):
            yield (path, None, False, "expected %d return values, got %d"
                   % (len(want), len(got)))
            return
        for i, (g, w) in enumerate(zip(got, want)):
            for item in _compare(g, w, "%s[%d]" % (path, i)):
                yield item
        return

    if isinstance(want, dict):
        for k in sorted(want):
            if not isinstance(got, dict) or k not in got:
                yield (path + "[%r]" % k, None, False, "missing key")
                continue
            for item in _compare(got[k], want[k], path + "[%r]" % k):
                yield item
        return

    if isinstance(want, str) or isinstance(want, bool):
        yield (path, None, got == want,
               "" if got == want else "got %r, expected %r" % (got, want))
        return

    try:
        g = np.asarray(got, dtype=np.float64)
        w = np.asarray(want, dtype=np.float64)
    except (TypeError, ValueError):
        # non-numeric payload (strings, callables, objects) -- compare directly
        same = bool(got == want)
        yield (path, None, same, "" if same else "got %r, expected %r" % (got, want))
        return

    if g.shape != w.shape:
        yield (path, None, False, "shape %s, expected %s" % (g.shape, w.shape))
        return
    err = rel_error(g, w)
    yield (path, err, err <= PASS_TOL, "")


def grade(module, name, user_fn, tol=None, verbose=True):
    """Run user_fn against the reference on the module's stored test cases.

    Returns True if every case matched within tolerance.
    """
    tol = PASS_TOL if tol is None else tol
    mod = _load(module)
    if not hasattr(mod, name):
        print("[?] no reference named %r in solutions/%s.py" % (name, module))
        return False
    ref = getattr(mod, name)
    cases = mod.CASES.get(name)
    if cases is None:
        print("[?] no test cases registered for %r" % name)
        return False

    arglists = cases()
    all_ok = True
    for ci, args in enumerate(arglists):
        label = "%s  case %d/%d" % (name, ci + 1, len(arglists))
        pos, kw = _split_args(args)
        try:
            got = user_fn(*_copy(pos), **_copy(kw))
        except NotImplementedError:
            print("[ ] %s -- not implemented yet" % label)
            return False
        except Exception:
            print("[%s] %s -- raised:" % (_RED, label))
            traceback.print_exc(limit=2)
            return False

        if _is_stub(got):
            print("[ ] %s -- returned None (still a stub)" % label)
            return False

        want = ref(*_copy(pos), **_copy(kw))
        for path, err, ok, msg in _compare(got, want):
            ok = ok if err is None else (err <= tol)
            all_ok &= ok
            if not ok:
                print("[%s] %s  %s  %s" % (_RED, label, path,
                                           msg or "rel error %.3e" % err))
            elif verbose:
                print("[%s] %s  %s  rel error %.3e"
                      % (_GREEN, label, path, err if err is not None else 0.0))
    if all_ok:
        print("--> %s looks correct." % name)
    else:
        print("--> %s does not match the reference. "
              "Run show_solution('%s', '%s') when you want to compare."
              % (name, module, name))
    return all_ok


def _copy(x):
    if isinstance(x, dict):
        return {k: _copy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_copy(v) for v in x]
        return type(x)(t) if not isinstance(x, tuple) else tuple(t)
    if isinstance(x, np.ndarray):
        return x.copy()
    return x


def show_solution(module, name):
    """Print the reference implementation. Use it to compare, not to copy."""
    mod = _load(module)
    obj = getattr(mod, name, None)
    if obj is None:
        print("no reference named %r in solutions/%s.py" % (name, module))
        return
    print(inspect.getsource(obj))


def list_solutions(module):
    mod = _load(module)
    for k in sorted(mod.CASES):
        print(" ", k)


def _split_args(args):
    """Interpret one registered case as (positional, keyword).

    A case is (pos_tuple, kwargs_dict) ONLY when it is a 2-tuple whose first
    element is itself a tuple and whose second is a dict. Without the
    first-is-a-tuple condition, a perfectly ordinary 2-argument call like
    max_pool_forward_naive(x, pool_param) gets mistaken for a kwargs pair.
    """
    if (isinstance(args, tuple) and len(args) == 2
            and isinstance(args[0], tuple) and isinstance(args[1], dict)):
        return args[0], args[1]
    return args, {}


def _run_cases(module, name, user_fn):
    """Run user_fn on every registered case. Returns (status, list of (got, want)).

    status is 'stub' (nothing written yet), 'error', or 'ran'.
    """
    mod = _load(module)
    ref = getattr(mod, name, None)
    cases = mod.CASES.get(name) if hasattr(mod, "CASES") else None
    if ref is None or cases is None:
        return "missing", []

    pairs = []
    for args in cases():
        pos, kw = _split_args(args)
        try:
            got = user_fn(*_copy(pos), **_copy(kw))
        except NotImplementedError:
            return "stub", []
        except Exception:
            return "error", []
        want = ref(*_copy(pos), **_copy(kw))
        if _is_stub(got) or _looks_untouched(got, want):
            return "stub", []
        pairs.append((got, want))
    return "ran", pairs


def _looks_untouched(got, want):
    """A stub that pre-allocates `np.zeros(...)` and returns it unmodified.

    Without this, `use()` would happily accept an all-zero distance matrix and
    every downstream cell would run on garbage instead of falling back.
    """
    def all_zero(v):
        if isinstance(v, tuple):
            return all(all_zero(x) for x in v)
        if isinstance(v, dict):
            return len(v) == 0 or all(all_zero(x) for x in v.values())
        if isinstance(v, (list,)):
            return len(v) == 0
        try:
            a = np.asarray(v, dtype=np.float64)
        except Exception:
            return False
        return a.size == 0 or not np.any(a)

    return all_zero(got) and not all_zero(want)


def grade(module, name, user_fn, tol=None, verbose=True):
    """Run user_fn against the reference on the module's stored test cases.

    Returns True if every case matched within tolerance.
    """
    tol = PASS_TOL if tol is None else tol
    status, pairs = _run_cases(module, name, user_fn)

    if status == "missing":
        print("[?] no reference or test cases registered for %r" % name)
        return False
    if status == "stub":
        print("[ ] %s -- not filled in yet." % name)
        return False
    if status == "error":
        print("[%s] %s raised:" % (_RED, name))
        try:
            mod = _load(module)
            pos, kw = _split_args(mod.CASES[name]()[0])
            user_fn(*_copy(pos), **_copy(kw))
        except Exception:
            traceback.print_exc(limit=3)
        return False

    all_ok = True
    for ci, (got, want) in enumerate(pairs):
        label = "%s  case %d/%d" % (name, ci + 1, len(pairs))
        for path, err, ok, msg in _compare(got, want):
            ok = ok if err is None else (err <= tol)
            all_ok &= ok
            if not ok:
                print("[%s] %s  %s  %s" % (_RED, label, path,
                                           msg or "rel error %.3e" % err))
            elif verbose:
                print("[%s] %s  %s  rel error %.3e"
                      % (_GREEN, label, path, err if err is not None else 0.0))
    if all_ok:
        print("--> %s looks correct." % name)
    else:
        print("--> %s does not match the reference. "
              "Run show_solution('%s', '%s') when you want to compare."
              % (name, module, name))
    return all_ok


def use(module, name, user_fn, quiet=False):
    """Return user_fn if it is CORRECT, otherwise the reference implementation.

    This is what keeps a notebook runnable while TODOs are unfilled or wrong.
    Falling back on incorrectness (not merely on `None`) matters: a stub that
    pre-allocates `np.zeros(...)` returns a perfectly well-shaped array of
    garbage, and every downstream cell would quietly consume it.
    """
    mod = _load(module)
    ref = getattr(mod, name, None)
    if ref is None:
        return user_fn

    status, pairs = _run_cases(module, name, user_fn)
    if status == "missing":
        return user_fn

    ok = status == "ran"
    if ok:
        for got, want in pairs:
            for _, err, good, _msg in _compare(got, want):
                good = good if err is None else (err <= PASS_TOL)
                ok &= good

    if not ok:
        if not quiet:
            why = {"stub": "not filled in", "error": "raises",
                   "ran": "does not match"}.get(status, status)
            print("(falling back to the reference %s -- yours %s)" % (name, why))
        return ref
    return user_fn
