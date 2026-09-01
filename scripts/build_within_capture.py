#!/usr/bin/env python3
"""Build a within-capture MM-Fi tree by chunking ONE capture into pseudo-subjects.

Why: MmFiDataset::subject_disjoint_split assigns whole subjects to train/test
(dataset.rs:596-605), so a single capture would be unsplittable. Chunking one
capture into N contiguous temporal blocks lets the trainer hold one block out
while every sample keeps the SAME camera geometry.

That isolates the question the cross-capture run could not answer: does the CSI
carry pose information when the label coordinate frame is held constant?

Blocks are contiguous in time (never interleaved), so held-out windows are not
temporal neighbours of training windows.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/mmfi/S03", help="single-capture subject dir")
    ap.add_argument("--out", default="data/mmfi-single")
    ap.add_argument("--chunks", type=int, default=5)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"Not a directory: {src}")

    actions = sorted(p for p in src.iterdir() if p.is_dir())
    if not actions:
        raise SystemExit(f"No action dirs under {src}")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    n = len(actions)
    per = (n + args.chunks - 1) // args.chunks

    for c in range(args.chunks):
        block = actions[c * per:(c + 1) * per]
        if not block:
            continue
        sdir = out / f"S{c + 1:02d}"
        for i, a in enumerate(block, start=1):
            dst = sdir / f"A{i:04d}"
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("wifi_csi.npy", "gt_keypoints.npy"):
                if (a / f).exists():
                    shutil.copy2(a / f, dst / f)
        print(f"  S{c + 1:02d}: {len(block):4d} clips  "
              f"(source {block[0].name}..{block[-1].name})")

    print(f"\n  {n} clips -> {args.chunks} contiguous blocks under {out.resolve()}")


if __name__ == "__main__":
    main()
