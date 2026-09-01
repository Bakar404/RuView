#!/usr/bin/env python3
"""Mean-pose baseline for PCK@0.2 — the ADR-291 paired baseline.

model_gates.rs:546 fails any reported metric that is surfaced without a paired
mean-pose/majority baseline. A pose model that collapses to predicting a single
constant pose can score surprisingly well on PCK, so a headline number is
meaningless until compared against exactly that trivial predictor.

This replicates metrics_core.rs::pck_canonical bit-for-bit:
  * visible iff visibility >= 0.5                       (:85 VISIBILITY_THRESHOLD)
  * torso = || kp[11] - kp[12] ||                       (:144 canonical_torso_size)
  * fallback = bbox diagonal of visible keypoints       (:158)
  * correct iff ||pred - gt|| <= threshold * torso      (:199-212)
  * unscoreable sample => 0.0, never 1.0                (:196)

Split mirrors the trainer's subject-disjoint split: train on all subjects except
--val-subject, evaluate on that one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

VISIBILITY_THRESHOLD = 0.5
MIN_REFERENCE_EXTENT = 1e-6
CANON_LEFT_HIP = 11
CANON_RIGHT_HIP = 12


def bbox_diagonal(kp: np.ndarray, vis: np.ndarray) -> float:
    m = vis >= VISIBILITY_THRESHOLD
    if not m.any():
        return 0.0
    xs, ys = kp[m, 0], kp[m, 1]
    w = max(float(xs.max() - xs.min()), 0.0)
    h = max(float(ys.max() - ys.min()), 0.0)
    return float(np.hypot(w, h))


def torso_size(kp: np.ndarray, vis: np.ndarray) -> float | None:
    if (vis[CANON_LEFT_HIP] >= VISIBILITY_THRESHOLD
            and vis[CANON_RIGHT_HIP] >= VISIBILITY_THRESHOLD):
        d = float(np.hypot(kp[CANON_LEFT_HIP, 0] - kp[CANON_RIGHT_HIP, 0],
                           kp[CANON_LEFT_HIP, 1] - kp[CANON_RIGHT_HIP, 1]))
        if d > MIN_REFERENCE_EXTENT:
            return d
    d = bbox_diagonal(kp, vis)
    return d if d > MIN_REFERENCE_EXTENT else None


def pck(pred: np.ndarray, gt: np.ndarray, vis: np.ndarray, thr: float) -> tuple[int, int]:
    t = torso_size(gt, vis)
    if t is None:
        return 0, 0
    lim = thr * t
    m = vis >= VISIBILITY_THRESHOLD
    if not m.any():
        return 0, 0
    d = np.hypot(pred[m, 0] - gt[m, 0], pred[m, 1] - gt[m, 1])
    return int((d <= lim).sum()), int(m.sum())


def load_subject(root: Path, subject: str) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    sdir = root / subject
    if not sdir.is_dir():
        return out
    for adir in sorted(sdir.iterdir()):
        f = adir / "gt_keypoints.npy"
        if f.exists():
            arr = np.load(f)          # [T, 17, 3]
            frame = arr[0]            # loader reads t_start only
            out.append((frame[:, :2].astype(np.float64), frame[:, 2].astype(np.float64)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/mmfi")
    ap.add_argument("--val-subject", default="S04")
    ap.add_argument("--threshold", type=float, default=0.2)
    args = ap.parse_args()

    root = Path(args.root)
    subjects = sorted(p.name for p in root.iterdir() if p.is_dir())
    train_subj = [s for s in subjects if s != args.val_subject]

    train = [s for sub in train_subj for s in load_subject(root, sub)]
    val = load_subject(root, args.val_subject)
    if not train or not val:
        print("Missing train or val data")
        return

    # Visibility-weighted mean pose over the training subjects.
    num = np.zeros((17, 2))
    den = np.zeros((17, 1))
    for kp, vis in train:
        m = (vis >= VISIBILITY_THRESHOLD).astype(np.float64)[:, None]
        num += kp * m
        den += m
    mean_pose = np.divide(num, np.maximum(den, 1e-9))

    strategies = {
        "mean-pose (train)": mean_pose,
        "image-centre (0.5,0.5)": np.full((17, 2), 0.5),
    }

    print(f"  train: {len(train)} samples from {', '.join(train_subj)}")
    print(f"  val  : {len(val)} samples from {args.val_subject}")
    print(f"  PCK@{args.threshold} (canonical, torso-normalised)\n")

    for name, pred in strategies.items():
        c = t = 0
        per_sample = []
        for kp, vis in val:
            ci, ti = pck(pred, kp, vis, args.threshold)
            c += ci
            t += ti
            per_sample.append(ci / ti if ti else 0.0)
        micro = c / t if t else 0.0
        macro = float(np.mean(per_sample)) if per_sample else 0.0
        print(f"  {name:<26} micro={micro * 100:5.2f}%   macro={macro * 100:5.2f}%   "
              f"({c}/{t} joints)")


if __name__ == "__main__":
    main()
