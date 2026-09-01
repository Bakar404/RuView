#!/usr/bin/env python3
"""Model-free test for ANY (incl. non-linear) CSI->pose relationship.

Correlation only detects linear structure, and a neural net failing to learn
could always be blamed on architecture or learning rate. This probe removes
both objections:

  k-NN probe   For each val sample, find its k nearest neighbours in CSI space
               (among train samples) and predict the average of their poses.
               If CSI carries pose information in ANY smooth form, neighbours
               in CSI space have similar poses and this beats the mean pose.
               It has no weights, no optimiser and nothing to tune.

  shuffle ctrl The same probe with labels randomly permuted. This is the
               null distribution. If the real score sits inside the shuffled
               spread, the "signal" is an artefact of the metric.

  temporal     Autocorrelation of CSI across the time axis. Real physical
               channel state is smooth over ~100 ms; white noise is not.
               If adjacent windows are as different as distant ones, the
               capture is noise-dominated and no model can succeed.

Scored with the canonical PCK from metrics_core.rs so numbers are directly
comparable to the trainer's val_pck and to baseline_pck.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

VISIBILITY_THRESHOLD = 0.5
L_HIP, R_HIP = 11, 12


def torso_scale(kp: np.ndarray) -> float:
    """Hip-to-hip distance, bbox-diagonal fallback (metrics_core.rs:144-165)."""
    if kp[L_HIP, 2] >= VISIBILITY_THRESHOLD and kp[R_HIP, 2] >= VISIBILITY_THRESHOLD:
        d = float(np.linalg.norm(kp[L_HIP, :2] - kp[R_HIP, :2]))
        if d > 1e-6:
            return d
    vis = kp[kp[:, 2] >= VISIBILITY_THRESHOLD]
    if vis.shape[0] < 2:
        return 0.0
    span = vis[:, :2].max(axis=0) - vis[:, :2].min(axis=0)
    return float(np.hypot(*span))


def pck(pred: np.ndarray, gt: np.ndarray, thr: float = 0.2) -> float:
    """Canonical PCK; unscoreable samples contribute 0, never 1."""
    correct = total = 0
    for p, g in zip(pred, gt):
        s = torso_scale(g)
        if s <= 1e-6:
            continue
        for j in range(g.shape[0]):
            if g[j, 2] < VISIBILITY_THRESHOLD:
                continue
            total += 1
            if float(np.linalg.norm(p[j, :2] - g[j, :2])) <= thr * s:
                correct += 1
    return correct / total if total else 0.0


def load(root: Path):
    X, K, order = [], [], []
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for adir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            c, k = adir / "wifi_csi.npy", adir / "gt_keypoints.npy"
            if c.exists() and k.exists():
                X.append(np.load(c).reshape(-1))
                K.append(np.load(k)[0])
                order.append(f"{sdir.name}/{adir.name}")
    return np.array(X, dtype=np.float64), np.array(K, dtype=np.float64), order


def knn_predict(Xtr, Ktr, Xva, k):
    # Standardise so no single loud subcarrier dominates the distance.
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xva - mu) / sd
    # ||a-b||^2 = |a|^2 + |b|^2 - 2ab
    d = (B * B).sum(1)[:, None] + (A * A).sum(1)[None, :] - 2.0 * B @ A.T
    idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    return Ktr[idx].mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/mmfi-single")
    ap.add_argument("--val-subject", default="S02")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--shuffles", type=int, default=25)
    args = ap.parse_args()

    X, K, order = load(Path(args.root))
    is_val = np.array([o.startswith(args.val_subject + "/") for o in order])
    Xtr, Ktr = X[~is_val], K[~is_val]
    Xva, Kva = X[is_val], K[is_val]
    print(f"  train {Xtr.shape[0]}   val {Xva.shape[0]} (held out {args.val_subject})\n")

    mean_pose = Ktr.mean(axis=0)
    mean_pose[:, 2] = 1.0
    base = pck(np.repeat(mean_pose[None], Xva.shape[0], axis=0), Kva)
    print(f"  mean-pose baseline : {base*100:6.2f}%")

    real = pck(knn_predict(Xtr, Ktr, Xva, args.k), Kva)
    print(f"  k-NN on real CSI   : {real*100:6.2f}%   (k={args.k})")

    rng = np.random.default_rng(0)
    scores = []
    for _ in range(args.shuffles):
        perm = rng.permutation(Ktr.shape[0])
        scores.append(pck(knn_predict(Xtr, Ktr[perm], Xva, args.k), Kva))
    scores = np.array(scores)
    print(f"  k-NN, labels shuffled: {scores.mean()*100:6.2f}% "
          f"+/- {scores.std()*100:.2f}%  (n={args.shuffles})")

    hi = scores.mean() + 2 * scores.std()
    print(f"\n  null 95% upper bound : {hi*100:6.2f}%")
    if real > hi and real > base:
        print("  => CSI carries pose information above chance.")
    else:
        print("  => NO signal: real CSI is indistinguishable from shuffled labels.")

    print("\n  --- temporal autocorrelation of CSI ---")
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    for lag in (1, 2, 5, 10, 25, 50):
        if lag >= Z.shape[0]:
            break
        a, b = Z[:-lag], Z[lag:]
        num = (a * b).sum(1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
        print(f"    lag {lag:3d}: cosine similarity {float(np.mean(num/den)):+.4f}")
    print("    (physical channel state should decay smoothly from ~1.0;")
    print("     values near 0 at lag 1 mean the capture is noise-dominated)")


if __name__ == "__main__":
    main()
