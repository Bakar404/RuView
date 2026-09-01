#!/usr/bin/env python3
"""Expand the scalar `conf` in *.paired.jsonl into a per-keypoint array.

Why: scripts/train-wiflow-supervised.js weights its supervised loss per joint
(`loss += conf[k] * (lx + ly)`, :1098) and only accepts a per-keypoint array
when `conf.length >= 17` (:578). It reads keypoints as [x, y] only (:567-571),
so the visibility channel written by align-ground-truth.js is otherwise
discarded and masked joints would train as if they were real labels.

Folding visibility into conf gives the JS trainer the same masking semantics
the Rust trainer gets from keypoint column 2 (dataset.rs:506-508).

conf[k] = visibility[k] * window_confidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NUM_KEYPOINTS = 17


def convert(path: Path, dry_run: bool = False) -> tuple[int, int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    converted = skipped = masked_joints = 0

    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        kp = rec.get("kp") or rec.get("keypoints")
        if not kp or len(kp) < NUM_KEYPOINTS:
            skipped += 1
            out.append(line)
            continue

        base = rec.get("conf", 1.0)
        if isinstance(base, list):
            out.append(line)  # already per-keypoint
            skipped += 1
            continue

        conf = []
        for k in range(NUM_KEYPOINTS):
            vis = kp[k][2] if len(kp[k]) > 2 else 1.0
            conf.append(round(float(vis) * float(base), 6))
            if vis < 0.5:
                masked_joints += 1

        rec["conf_window"] = base
        rec["conf"] = conf
        out.append(json.dumps(rec))
        converted += 1

    if not dry_run and converted:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")

    return converted, skipped, masked_joints


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/paired")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*.paired.jsonl"))
    if not files:
        print(f"No *.paired.jsonl under {args.dir}")
        return

    t_conv = t_mask = 0
    for f in files:
        conv, skip, masked = convert(f, args.dry_run)
        t_conv += conv
        t_mask += masked
        pct = (100.0 * masked / (conv * NUM_KEYPOINTS)) if conv else 0.0
        print(f"  {f.name:<48} {conv:5d} converted  {skip:4d} skipped  "
              f"{masked:6d} masked joints ({pct:.1f}%)")

    print(f"\n  TOTAL: {t_conv} samples, {t_mask} masked joints "
          f"({100.0 * t_mask / max(1, t_conv * NUM_KEYPOINTS):.1f}% of all joints)")
    if args.dry_run:
        print("  (dry run - nothing written)")


if __name__ == "__main__":
    main()
