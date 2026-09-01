#!/usr/bin/env python3
"""Check whether the 3-node round-robin phase is stable across windows.

probe_frame_interleave.py showed frame t and frame t+3 are ~0.996 similar
inside a window: three sources are interleaved on the frame axis. That alone
is survivable IF the phase is constant, because then feature index f always
means node (f mod 3) and a model can learn per-node weights.

The aligner slices windows in raw arrival order
(align-ground-truth.js:303-307). Any dropped or reordered packet rotates the
phase for every window after it. Feature index f then refers to a DIFFERENT
node from one window to the next, and the input space is scrambled: no model
and no nearest-neighbour method can recover pose from it.

Method: take frame 0 of a reference window and find which frame offset of
every other window it best matches. A stable phase puts the best match at
offset 0 (mod 3) essentially always. A drifting phase spreads the best match
uniformly over {0,1,2}.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load(src: Path, limit: int):
    out = []
    for f in sorted(src.glob("*.paired.jsonl")):
        rows = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            a = np.asarray(r["csi"], dtype=np.float64)
            sh = r.get("csi_shape")
            if sh and len(sh) == 2:
                a = a.reshape(sh)
            rows.append(a)
            if len(rows) >= limit:
                break
        if rows:
            out.append((f.name, rows))
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/paired")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--period", type=int, default=3)
    args = ap.parse_args()

    files = load(Path(args.src), args.limit)
    P = args.period

    grand = Counter()
    for name, rows in files:
        if len(rows) < 5:
            continue
        ref = rows[0][0]  # frame 0 of the first window in this capture
        best = Counter()
        for w in rows[1:]:
            scores = [ncc(ref, w[o]) for o in range(min(P, w.shape[0]))]
            best[int(np.argmax(scores))] += 1
        total = sum(best.values())
        dist = "  ".join(f"off{o}={100*best[o]/total:5.1f}%" for o in range(P))
        print(f"  {name[:44]:44s} n={total:4d}  {dist}")
        grand.update(best)

    tot = sum(grand.values())
    if not tot:
        raise SystemExit("no windows compared")
    print("\n  --- pooled over all captures ---")
    for o in range(P):
        print(f"    best match at offset {o}: {100*grand[o]/tot:5.1f}%")
    top = 100 * max(grand.values()) / tot
    print(f"\n    dominant offset share: {top:.1f}%  (uniform = {100/P:.1f}%)")
    if top > 85:
        print("    => phase is STABLE: feature index reliably identifies a node.")
    elif top < 55:
        print("    => phase DRIFTS: feature index does NOT identify a node.")
        print("       The input space is scrambled across windows; this alone")
        print("       is sufficient to destroy all learnable CSI->pose structure.")
    else:
        print("    => phase partially stable; some windows are rotated.")


if __name__ == "__main__":
    main()
