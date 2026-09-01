#!/usr/bin/env python3
"""Diagnose whether the CSI tensors carry any between-sample signal.

If a pose model collapses to a constant output, there are only two families of
cause: the input does not vary in a way the model can use, or the training
objective is degenerate. This checks the first.

Reports:
  * global amplitude scale (raw magnitudes matter: an un-normalised input with
    huge mean and tiny relative variance is nearly constant to a linear layer)
  * between-sample variance vs within-sample variance -- if between-sample
    variance is negligible, every window looks identical to the network
  * correlation between CSI features and keypoint targets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_all(root: Path, limit: int | None = None):
    csi, kp = [], []
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for adir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            c = adir / "wifi_csi.npy"
            k = adir / "gt_keypoints.npy"
            if c.exists() and k.exists():
                csi.append(np.load(c).reshape(-1))
                kp.append(np.load(k)[0])
                if limit and len(csi) >= limit:
                    return np.array(csi), np.array(kp)
    return np.array(csi), np.array(kp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/mmfi-single")
    ap.add_argument("--limit", type=int, default=800)
    args = ap.parse_args()

    X, K = load_all(Path(args.root), args.limit)
    if X.size == 0:
        raise SystemExit(f"No data under {args.root}")

    print(f"  samples: {X.shape[0]}   features/sample: {X.shape[1]}\n")

    print("  --- amplitude scale ---")
    print(f"    min={X.min():.4f}  max={X.max():.4f}  mean={X.mean():.4f}  std={X.std():.4f}")

    # Between- vs within-sample variance.
    per_sample_mean = X.mean(axis=1)
    between = float(np.var(per_sample_mean))
    within = float(np.mean(np.var(X, axis=1)))
    print("\n  --- variance decomposition ---")
    print(f"    between-sample var (of per-sample means): {between:.6e}")
    print(f"    within-sample  var (mean over samples)  : {within:.6e}")
    ratio = between / within if within > 0 else 0.0
    print(f"    ratio between/within                    : {ratio:.6f}")

    # Per-feature: how much does each feature vary ACROSS samples?
    feat_std = X.std(axis=0)
    print("\n  --- per-feature across-sample std ---")
    print(f"    mean={feat_std.mean():.6e}  max={feat_std.max():.6e}  "
          f"min={feat_std.min():.6e}")
    rel = feat_std.mean() / (abs(X.mean()) + 1e-12)
    print(f"    mean std / mean amplitude (relative variation): {rel:.6e}")

    # Are any two samples actually different?
    d01 = float(np.abs(X[0] - X[1]).mean()) if X.shape[0] > 1 else 0.0
    dfar = float(np.abs(X[0] - X[-1]).mean())
    print(f"\n    mean|x0-x1|   = {d01:.6e}")
    print(f"    mean|x0-xN|   = {dfar:.6e}")

    # Correlation of top-varying CSI features with keypoint coords.
    print("\n  --- CSI vs keypoint correlation ---")
    y = K[:, :, :2].reshape(K.shape[0], -1)
    idx = np.argsort(feat_std)[-64:]
    best = 0.0
    for j in range(y.shape[1]):
        yy = y[:, j]
        if np.std(yy) < 1e-9:
            continue
        for i in idx:
            xx = X[:, i]
            if np.std(xx) < 1e-12:
                continue
            c = abs(float(np.corrcoef(xx, yy)[0, 1]))
            if c > best:
                best = c
    print(f"    max |corr| over 64 most-variable features x 34 targets: {best:.4f}")


if __name__ == "__main__":
    main()
