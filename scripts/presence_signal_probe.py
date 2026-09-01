#!/usr/bin/env python3
"""Model-free go/no-go probe for PRESENCE / IN-FRAME / RANGE from CSI.

Purpose
-------
Before spending epochs on a neural net, ask the cheap question: is there ANY
learnable relationship between the CSI window and the coarse labels we now care
about? A k-NN classifier answers this with no training and no hyperparameters.

The three labels are derived from the paired ground truth:
  IN_FRAME  - the visible skeleton does not touch a frame border
  RANGE     - bbox diagonal above/below the median (a proxy for close/far)
  PRESENCE  - n_persons > 0   (usually degenerate on existing data: see audit)

Methodology guards
------------------
* TEMPORAL SPLIT, not random. Adjacent CSI windows are ~identical, so a random
  split leaks the test set into training and produces meaningless high scores.
  We train on the first 70% of each capture and test on the last 30%.
* MAJORITY BASELINE. Reported alongside every score. A 70/30 label split means
  a constant predictor already scores 70%, so raw accuracy is not evidence.
* SHUFFLE CONTROL. The same probe on shuffled labels. If the real score is not
  clearly above the shuffled score, the apparent signal is an artifact.
* BALANCED ACCURACY is the headline number - it is insensitive to imbalance.

Interpretation
--------------
    real >> shuffle AND real >> majority   -> real signal, proceed to training
    real ~= shuffle                        -> no signal in these features
    real ~= majority                       -> classifier only learned the prior

Usage
-----
    python scripts/presence_signal_probe.py
    python scripts/presence_signal_probe.py --k 7 --paired data/paired
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

VIS_T = 0.5
EDGE = 0.02


def visible_pts(kp, conf):
    return [(kp[j][0], kp[j][1]) for j in range(17)
            if (kp[j][2] if len(kp[j]) > 2 else conf[j]) >= VIS_T]


def bbox_diag(pts):
    if not pts:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def is_clipped(pts):
    if not pts:
        return True
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs) <= EDGE or max(xs) >= 1 - EDGE
            or min(ys) <= EDGE or max(ys) >= 1 - EDGE)


def load_paired(path: Path):
    """Return (X, meta) for one capture, preserving temporal order."""
    X, meta = [], []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            csi = r.get("csi")
            kp = r.get("kp")
            if not csi or not kp:
                continue
            conf = r.get("conf") or [1.0] * 17
            pts = visible_pts(kp, conf)
            X.append(np.asarray(csi, dtype=np.float32))
            meta.append({
                "in_frame": 0 if is_clipped(pts) else 1,
                "diag": bbox_diag(pts),
                "present": 1 if r.get("n_persons_mode", 0) > 0 else 0,
            })
    if not X:
        return None, None
    n = min(x.shape[0] for x in X)
    return np.stack([x[:n] for x in X]), meta


def knn_predict(Xtr, ytr, Xte, k):
    """Cosine-distance k-NN majority vote."""
    a = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-9)
    b = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
    sim = b @ a.T
    idx = np.argsort(-sim, axis=1)[:, :k]
    votes = ytr[idx]
    return (votes.mean(axis=1) >= 0.5).astype(int)


def balanced_accuracy(y, p):
    accs = []
    for c in (0, 1):
        m = y == c
        if m.sum() > 0:
            accs.append((p[m] == c).mean())
    return float(np.mean(accs)) if accs else 0.0


def evaluate(X, y, k, rng):
    """Temporal 70/30 split; returns (real_bal, shuf_bal, major_bal, n_test)."""
    n = len(y)
    cut = int(0.7 * n)
    if cut < k + 1 or n - cut < 5:
        return None
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = y[:cut], y[cut:]
    if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
        return None

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    real = balanced_accuracy(yte, knn_predict(Xtr, ytr, Xte, k))
    ysh = ytr.copy()
    rng.shuffle(ysh)
    shuf = balanced_accuracy(yte, knn_predict(Xtr, ysh, Xte, k))
    maj = balanced_accuracy(yte, np.full_like(yte, int(round(ytr.mean()))))
    return real, shuf, maj, len(yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", default="data/paired")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    files = sorted(Path(args.paired).glob("*.paired.jsonl"))
    if not files:
        print("no paired files under", args.paired)
        return 1

    print("=" * 90)
    print("PRESENCE / IN-FRAME / RANGE  -  MODEL-FREE SIGNAL PROBE".center(90))
    print("=" * 90)
    print(f"\nk={args.k}   temporal 70/30 split per capture   metric = balanced accuracy\n")

    tasks = ("in_frame", "range", "present")
    agg = {t: [] for t in tasks}

    for f in files:
        X, meta = load_paired(f)
        if X is None:
            continue
        print(f"  {f.name[:44]:<46} n={len(meta):<5} dim={X.shape[1]}")
        med = float(np.median([m["diag"] for m in meta]))
        for t in tasks:
            if t == "in_frame":
                y = np.array([m["in_frame"] for m in meta])
            elif t == "range":
                y = np.array([1 if m["diag"] > med else 0 for m in meta])
            else:
                y = np.array([m["present"] for m in meta])

            bal = y.mean()
            res = evaluate(X, y, args.k, rng)
            if res is None:
                print(f"       {t:<10} skipped (single class or too few samples;"
                      f" positive rate {bal:.2f})")
                continue
            real, shuf, maj, nte = res
            delta = real - max(shuf, maj)
            flag = "SIGNAL" if delta > 0.08 else "weak" if delta > 0.03 else "none"
            agg[t].append((real, shuf, maj))
            print(f"       {t:<10} real={real:.3f}  shuffle={shuf:.3f}  majority={maj:.3f}"
                  f"  delta={delta:+.3f}  [{flag}]")
        print()

    print("-" * 90)
    print("AGGREGATE (mean across captures)\n")
    print(f"  {'task':<12}{'real':>8}{'shuffle':>10}{'majority':>10}{'delta':>9}   verdict")
    print("  " + "-" * 72)
    for t in tasks:
        if not agg[t]:
            print(f"  {t:<12}{'--':>8}{'--':>10}{'--':>10}{'--':>9}   no evaluable captures")
            continue
        r = float(np.mean([a[0] for a in agg[t]]))
        s = float(np.mean([a[1] for a in agg[t]]))
        m = float(np.mean([a[2] for a in agg[t]]))
        d = r - max(s, m)
        v = ("REAL SIGNAL - worth training" if d > 0.08 else
             "weak - needs better features/data" if d > 0.03 else
             "NO SIGNAL in these features")
        print(f"  {t:<12}{r:8.3f}{s:10.3f}{m:10.3f}{d:+9.3f}   {v}")
    print("\n  NOTE: this data is the pre-fix 56-subcarrier amplitude-only slice")
    print("        (~26.5% of available signal, no phase). A positive result here")
    print("        is strong evidence; a negative result is INCONCLUSIVE.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
