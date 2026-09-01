#!/usr/bin/env python3
"""Session timeline over presence captures - sanity check and return-time finder.

Two jobs:

1. VALIDATION. Print a chronological, per-minute summary of the whole session
   (occupied -> transition -> absent -> return). If the empty-apartment period
   is visibly separable from the occupied period in these aggregate statistics,
   presence detection is real and a classifier is worth training. If the phases
   look identical here, no model will save it.

2. TRIMMING. The final `absent` file kept recording after the occupant came
   home, so its tail is mislabelled. Rather than guessing a cut time from the
   wall clock, locate the return from the signal itself: walking into the
   apartment produces a large, abrupt rise in cross-frame amplitude variability
   on every node at once. `--trim` rewrites the file keeping only rows before
   that point (minus a safety margin).

Metrics per window (aggregated over nodes)
------------------------------------------
  std_med  median over subcarriers of amp_std  - the MOTION proxy. High while
           someone moves, near the noise floor in an empty room.
  amp_med  median over subcarriers of amp_mean - the STATIC level. Shifts when
           a body persistently reshapes the multipath profile even at rest.
  rssi     mean RSSI across nodes.

Usage
-----
    python scripts/presence_timeline.py
    python scripts/presence_timeline.py --trim --margin-min 5
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import shutil
import statistics as st
from datetime import datetime


def read_rows(path):
    """Read a capture, tolerating a truncated gzip tail.

    A logger killed mid-write leaves the stream without its end-of-stream
    marker. Everything before the cut is still valid, so decode what we can and
    stop at the first error rather than discarding the whole capture.
    """
    label = None
    rows = []
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
        pass  # truncated tail: keep the rows already recovered
    return label, rows


def row_stats(r):
    """Aggregate one window across nodes -> (std_med, amp_med, rssi)."""
    stds, amps, rssis = [], [], []
    for _, d in sorted(r["nodes"].items()):
        sd = d.get("amp_std") or []
        am = d.get("amp_mean") or []
        if sd:
            stds.append(st.median(sd))
        if am:
            amps.append(st.median(am))
        if d.get("rssi") is not None:
            rssis.append(d["rssi"])
    if not stds or not amps:
        return None
    return (st.mean(stds), st.mean(amps),
            st.mean(rssis) if rssis else float("nan"))


def parse_utc(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/presence")
    ap.add_argument("--bucket-min", type=float, default=2.0)
    ap.add_argument("--trim", action="store_true",
                    help="rewrite the last absent file, cutting at the detected return")
    ap.add_argument("--margin-min", type=float, default=5.0,
                    help="extra minutes to discard before the detected return")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.ndjson.gz")),
                   key=lambda p: os.path.getctime(p))
    if not files:
        print("no captures found")
        return 1

    print("=" * 92)
    print("SESSION TIMELINE".center(92))
    print("=" * 92)
    print(f"\n  {'time':<10}{'label':<12}{'win':>6}{'std_med':>10}{'amp_med':>10}"
          f"{'rssi':>8}   motion")
    print("  " + "-" * 86)

    per_label = {}
    last_absent = None

    for path in files:
        label, rows = read_rows(path)
        if not rows:
            print(f"  {os.path.basename(path)[:40]:<40} EMPTY - skipped")
            continue
        if label == "absent":
            last_absent = (path, rows)

        bucket = []
        bstart = None
        for r in rows:
            s = row_stats(r)
            if s is None:
                continue
            t = parse_utc(r.get("utc", ""))
            if bstart is None:
                bstart = t
            bucket.append((t, s))
            per_label.setdefault(label, []).append(s)
            if t and bstart and (t - bstart).total_seconds() >= args.bucket_min * 60:
                emit(bucket, label)
                bucket, bstart = [], None
        if bucket:
            emit(bucket, label)

    print("\n" + "=" * 92)
    print("PER-LABEL SUMMARY".center(92))
    print("=" * 92)
    print(f"\n  {'label':<14}{'windows':>9}{'std_med':>12}{'amp_med':>12}{'rssi':>9}")
    print("  " + "-" * 58)
    for lab in sorted(per_label):
        v = per_label[lab]
        print(f"  {lab:<14}{len(v):>9}{st.median([x[0] for x in v]):>12.3f}"
              f"{st.median([x[1] for x in v]):>12.2f}"
              f"{st.mean([x[2] for x in v]):>9.1f}")

    if "absent" in per_label and "occupied" in per_label:
        a = st.median([x[0] for x in per_label["absent"]])
        o = st.median([x[0] for x in per_label["occupied"]])
        print(f"\n  MOTION SEPARATION  occupied/absent = {o/max(a,1e-9):.2f}x")
        aa = st.median([x[1] for x in per_label["absent"]])
        oa = st.median([x[1] for x in per_label["occupied"]])
        print(f"  LEVEL  SEPARATION  occupied/absent = {oa/max(aa,1e-9):.3f}x")
        if o / max(a, 1e-9) > 1.5:
            print("\n  -> Motion clearly separates the classes. Presence detection is viable.")
        else:
            print("\n  -> Motion alone does NOT separate them; the classifier will have to")
            print("     rely on the per-subcarrier profile shape, not aggregate variance.")

    # ---- return detection / trimming -------------------------------------
    if last_absent:
        path, rows = last_absent
        stats = [(parse_utc(r.get("utc", "")), row_stats(r)) for r in rows]
        stats = [(t, s) for t, s in stats if s]
        if len(stats) > 30:
            base = st.median([s[0] for _, s in stats[:len(stats) // 2]])
            thresh = max(base * 2.5, base + 0.5)
            hit = None
            run = 0
            for t, s in stats:
                if s[0] > thresh:
                    run += 1
                    if run >= 5:
                        hit = t
                        break
                else:
                    run = 0
            print("\n" + "=" * 92)
            print("RETURN DETECTION".center(92))
            print("=" * 92)
            print(f"\n  file      : {os.path.basename(path)}")
            print(f"  baseline  : std_med = {base:.3f} (empty-room floor)")
            print(f"  threshold : {thresh:.3f}")
            if hit:
                print(f"  DETECTED  : sustained motion from {hit.astimezone().strftime('%H:%M:%S')} local")
                print("  -> the signal itself shows the moment someone walked in.")
                if args.trim:
                    cut = hit.timestamp() - args.margin_min * 60
                    tmp = path + ".tmp"
                    kept = 0
                    with gzip.open(path, "rt", encoding="utf-8") as src, \
                         gzip.open(tmp, "wt", encoding="utf-8") as dst:
                        for line in src:
                            try:
                                r = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if r.get("type") == "header":
                                dst.write(line)
                                continue
                            t = parse_utc(r.get("utc", ""))
                            if t and t.timestamp() < cut:
                                dst.write(line)
                                kept += 1
                    shutil.move(tmp, path)
                    print(f"  TRIMMED   : kept {kept} of {len(rows)} windows "
                          f"(cut {args.margin_min:.0f} min before return)")
            else:
                print("  no sustained motion spike found above threshold")
    print()
    return 0


def emit(bucket, label):
    if not bucket:
        return
    t0 = bucket[0][0]
    sd = st.median([b[1][0] for b in bucket])
    am = st.median([b[1][1] for b in bucket])
    rs = st.mean([b[1][2] for b in bucket])
    bar = "#" * min(40, int(sd * 8))
    ts = t0.astimezone().strftime("%H:%M:%S") if t0 else "--"
    print(f"  {ts:<10}{label:<12}{len(bucket):>6}{sd:>10.3f}{am:>10.2f}{rs:>8.1f}   {bar}")


if __name__ == "__main__":
    raise SystemExit(main())
