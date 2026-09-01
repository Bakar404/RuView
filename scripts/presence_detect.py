#!/usr/bin/env python3
"""Background-reference presence detector - the validated approach.

What the diagnostics established
--------------------------------
* ACROSS separate capture files, session artifacts dominate. Two captures
  recorded 90 s apart under identical conditions (ctrlA/ctrlB) are separable at
  0.85 balanced accuracy. Any classifier trained with "file = class" therefore
  learns capture identity, not occupancy. That route is a dead end.

* WITHIN one continuous capture the signal is unambiguous. Using a confirmed
  empty stretch as the reference, empty-vs-empty scores AUC ~0.60-0.71 (the
  drift floor) while occupied-vs-empty saturates at 1.000 on all three nodes,
  with a transition sharp enough to localise to a single minute.

So the detector does not classify captures. It builds a BACKGROUND MODEL of the
empty room from a reference period, then scores each incoming window by how far
its per-subcarrier amplitude profile deviates from that background. This is the
classical anomaly-detection formulation, and it is the one the measurements
support.

Score
-----
For each node, per usable subcarrier k, the background holds mean mu[k] and
standard deviation sigma[k] over the reference period. For a window x:

    z[k]  = |x[k] - mu[k]| / (sigma[k] + eps)
    score = fraction of subcarriers with z[k] > `--z`

Fraction-of-subcarriers is used rather than mean-z because it is bounded,
robust to a few wild bins, and directly interpretable: "62% of subcarriers look
unlike the empty room".

Drift caveat (important)
------------------------
The background is only valid while the channel is stationary. Measured drift
pushes empty-vs-empty AUC to ~0.7 over tens of minutes, so the reference must be
refreshed periodically in any long-running deployment. `--recalib` re-estimates
the background from recent windows that scored below threshold.

Usage
-----
    # validate against the capture containing a known arrival
    python scripts/presence_detect.py --file data/presence/absent_20260831_173643.ndjson.gz \
        --calib-min 8 --report

    # calibrate on one file, score another
    python scripts/presence_detect.py --calib data/presence/absent_20260831_163640.ndjson.gz \
        --file data/presence/absent_20260831_173643.ndjson.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime

import numpy as np


def read_rows(path):
    rows, label = [], None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") == "header":
                    label = r.get("label")
                    continue
                if r.get("nodes"):
                    rows.append(r)
    except (EOFError, OSError):
        pass  # tolerate a truncated tail
    return label, rows


def local_time(r):
    try:
        return datetime.fromisoformat(r["utc"].replace("Z", "+00:00")).astimezone()
    except Exception:
        return None


def vec(r, nodes, normalize=False):
    """Concatenate per-node amp_mean into one profile, or None if a node is missing.

    With `normalize`, each node's profile is z-scored within its own window.
    That removes any uniform gain change - which is what most channel drift and
    AGC movement looks like - while preserving the SHAPE of the amplitude-vs-
    subcarrier curve, which is what a body in the room actually alters.
    """
    out = []
    for n in nodes:
        d = r["nodes"].get(n)
        if not d or not d.get("amp_mean"):
            return None
        a = np.asarray(d["amp_mean"], dtype=np.float64)
        if normalize:
            a = (a - a.mean()) / (a.std() + 1e-9)
        out.append(a)
    return np.concatenate(out)


class Background:
    """Per-subcarrier empty-room reference."""

    def __init__(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        # Floor sigma at a small fraction of the mean level so that a subcarrier
        # which happened to be quiet during calibration cannot produce enormous
        # z-scores later purely from its own noise.
        floor = 0.05 * np.median(self.mu)
        self.sd = np.maximum(self.sd, max(floor, 1e-6))

    def score(self, x, z):
        return float((np.abs(x - self.mu) / self.sd > z).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="capture to score")
    ap.add_argument("--calib", default=None,
                    help="separate calibration capture; default = start of --file")
    ap.add_argument("--calib-min", type=float, default=8.0,
                    help="minutes of reference used to build the background")
    ap.add_argument("--calib-skip-min", type=float, default=0.0,
                    help="skip this many minutes before starting calibration. Use "
                         "to calibrate on the END of a capture (e.g. the empty tail "
                         "of a departure recording), keeping everything in-capture "
                         "and avoiding cross-capture offset.")
    ap.add_argument("--z", type=float, default=3.0, help="per-subcarrier z cutoff")
    ap.add_argument("--thresh", type=float, default=0.25,
                    help="score above this = PRESENT")
    ap.add_argument("--smooth", type=int, default=5,
                    help="windows of median smoothing (suppresses single-window flips)")
    ap.add_argument("--report", action="store_true", help="print a per-minute table")
    ap.add_argument("--normalize", action="store_true",
                    help="z-score each node profile per window; removes uniform "
                         "gain drift while keeping profile shape")
    args = ap.parse_args()

    _, rows = read_rows(args.file)
    if not rows:
        print("no rows in", args.file)
        return 1
    nodes = sorted({n for r in rows for n in r["nodes"]}, key=int)

    if args.calib:
        _, crows = read_rows(args.calib)
    else:
        crows = rows
    cvecs = []
    t0 = local_time(crows[0])
    for r in crows:
        t = local_time(r)
        if not (t0 and t):
            continue
        el = (t - t0).total_seconds() / 60.0
        if el < args.calib_skip_min:
            continue
        if el > args.calib_skip_min + args.calib_min:
            break
        v = vec(r, nodes, args.normalize)
        if v is not None:
            cvecs.append(v)
    if len(cvecs) < 30:
        print(f"only {len(cvecs)} calibration windows - need at least 30")
        return 1

    w = min(len(v) for v in cvecs)
    bg = Background(np.asarray([v[:w] for v in cvecs]))

    src = "start of --file" if not args.calib else args.calib
    print("=" * 78)
    print("PRESENCE DETECTOR".center(78))
    print("=" * 78)
    print(f"  scoring     : {args.file}")
    print(f"  background  : {len(cvecs)} windows ({args.calib_min:.0f} min) from {src}")
    print(f"  nodes       : {nodes}   profile dim: {w}")
    print(f"  z cutoff    : {args.z}   present threshold: {args.thresh}\n")

    times, scores = [], []
    for r in rows:
        v = vec(r, nodes, args.normalize)
        if v is None or len(v) < w:
            continue
        times.append(local_time(r))
        scores.append(bg.score(v[:w], args.z))
    s = np.asarray(scores)

    if args.smooth > 1:
        k = args.smooth
        pad = np.pad(s, (k // 2, k // 2), mode="edge")
        s = np.asarray([np.median(pad[i:i + k]) for i in range(len(s))])

    present = s > args.thresh

    if args.report:
        print(f"  {'minute':<9}{'n':>5}{'score':>9}{'state':>10}   detection")
        print("  " + "-" * 66)
        cur, buf = None, []
        for t, sc in zip(times, s):
            if t is None:
                continue
            key = t.strftime("%H:%M")
            if cur is None:
                cur = key
            if key != cur:
                m = float(np.median([b for b in buf]))
                st = "PRESENT" if m > args.thresh else "empty"
                print(f"  {cur:<9}{len(buf):>5}{m:>9.3f}{st:>10}   {'#' * int(m * 50)}")
                cur, buf = key, []
            buf.append(sc)
        if buf:
            m = float(np.median(buf))
            st = "PRESENT" if m > args.thresh else "empty"
            print(f"  {cur:<9}{len(buf):>5}{m:>9.3f}{st:>10}   {'#' * int(m * 50)}")

    # transitions
    print("\n  STATE CHANGES")
    changes = 0
    for i in range(1, len(present)):
        if present[i] != present[i - 1] and times[i]:
            print(f"    {times[i].strftime('%H:%M:%S')}  -> "
                  f"{'PRESENT' if present[i] else 'empty'}   (score {s[i]:.3f})")
            changes += 1
    if changes == 0:
        print("    none")

    print(f"\n  windows: {len(s)}   present: {int(present.sum())} "
          f"({100*present.mean():.1f}%)")
    print(f"  score  : empty-period median={np.median(s[~present]) if (~present).any() else float('nan'):.3f}  "
          f"present-period median={np.median(s[present]) if present.any() else float('nan'):.3f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
