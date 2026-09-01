#!/usr/bin/env python3
"""Test whether the CSI frame axis interleaves multiple sources.

Each paired record holds csi as [window_frames, subcarriers] in time_major
order. If those frames are a genuine time series from ONE link, frame t is
most similar to frame t+1 and similarity decays smoothly with lag.

If instead the recorder round-robins packets from N transmitters (3 ESP32
nodes here) into a single stream, and the aligner stacks them in arrival
order, then frame t belongs to node (t mod N). Similarity then PEAKS at
lag N, 2N, 3N... and is LOW at lags that cross node boundaries. That
sawtooth is unmistakable and cannot be produced by physical channel decay.

Also clusters frames with k-means to see if they fall into N tight groups,
and reports how group membership maps onto frame index mod N.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_windows(src: Path, limit: int):
    out = []
    for f in sorted(src.glob("*.paired.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            shape = r.get("csi_shape")
            arr = np.asarray(r["csi"], dtype=np.float64)
            if shape and len(shape) == 2:
                arr = arr.reshape(shape)
            out.append(arr)
            if len(out) >= limit:
                return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/paired")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    W = load_windows(Path(args.src), args.limit)
    if not W:
        raise SystemExit("no windows loaded")
    F, S = W[0].shape
    print(f"  windows: {len(W)}   frames/window: {F}   subcarriers: {S}\n")

    print("  --- within-window frame-to-frame similarity by lag ---")
    print("    (one link -> smooth decay from ~1.0;")
    print("     N interleaved nodes -> peaks at multiples of N)\n")
    sims = {}
    for lag in range(1, min(F, 13)):
        vals = []
        for w in W:
            a, b = w[:-lag], w[lag:]
            an = a - a.mean(axis=1, keepdims=True)
            bn = b - b.mean(axis=1, keepdims=True)
            num = (an * bn).sum(axis=1)
            den = np.linalg.norm(an, axis=1) * np.linalg.norm(bn, axis=1) + 1e-12
            vals.append(np.mean(num / den))
        sims[lag] = float(np.mean(vals))
        bar = "#" * max(0, int((sims[lag] + 1) * 30))
        print(f"    lag {lag:2d}: {sims[lag]:+.4f}  {bar}")

    peak = max(sims, key=lambda k: sims[k])
    print(f"\n    highest similarity at lag {peak}")
    if peak > 1:
        print(f"    => consistent with {peak} interleaved sources on the frame axis")
    else:
        print("    => lag 1 dominant: frame axis behaves like a single time series")

    # Do frames split into tight groups aligned to index mod N?
    print("\n  --- frame clustering vs (frame index mod N) ---")
    X = np.concatenate([w for w in W[:200]], axis=0)
    idx = np.concatenate([np.arange(w.shape[0]) for w in W[:200]])
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-8)
    for N in (2, 3, 4):
        means = np.stack([Xn[idx % N == r].mean(axis=0) for r in range(N)])
        # Between-group separation relative to within-group spread.
        between = float(np.mean([np.linalg.norm(means[i] - means[j])
                                 for i in range(N) for j in range(i + 1, N)]))
        within = float(np.mean([np.linalg.norm(Xn[idx % N == r] - means[r], axis=1).mean()
                                for r in range(N)]))
        print(f"    N={N}: between-group dist {between:8.3f} | "
              f"within-group {within:8.3f} | ratio {between/(within+1e-12):.4f}")
    print("    (ratio near 0 = index mod N explains nothing;")
    print("     ratio >> 0 = frames genuinely grouped by position mod N)")


if __name__ == "__main__":
    main()
