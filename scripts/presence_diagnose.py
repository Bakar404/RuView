#!/usr/bin/env python3
"""Diagnose WHERE the present-vs-absent signal lives (if anywhere).

The session timeline showed that the median-over-subcarriers of `amp_std` is
almost identical for `occupied` and `absent` (~2.7 vs ~2.65). That is a
median over 166 subcarriers x 3 nodes, so it would hide a signal that is
concentrated in a few subcarriers or on a single node. Before concluding there
is no signal, look in the places the median cannot see.

Checks
------
1. PER NODE. Node 1 sits by the desk/router, node 3 across the room. A body at
   the desk should perturb them unequally.
2. DISTRIBUTION TAILS. p90/p99 of amp_std rather than the median: motion may
   move only the most sensitive subcarriers.
3. PER SUBCARRIER discriminative power. For each (node, subcarrier), how well
   does amp_mean alone separate occupied from absent? Reported as AUC, which is
   threshold-free and insensitive to class imbalance. AUC 0.5 = no information,
   1.0 = perfect. This finds a signal concentrated in a handful of bins.
4. DRIFT CONTROL. The same AUC computed between the FIRST and SECOND HALF of
   the absent capture - where the true answer is "no difference, both empty".
   Any AUC materially above 0.5 there is pure temporal drift, and sets the bar
   that a real presence signal must clear. This is the single most important
   number in the script: a presence AUC of 0.8 means nothing if the drift
   control also reaches 0.8.

Usage
-----
    python scripts/presence_diagnose.py
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np


def read_rows(path):
    label, rows = None, []
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
        pass
    return label, rows


def auc(pos, neg):
    """Threshold-free separability via the Mann-Whitney U statistic."""
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    a = np.concatenate([pos, neg])
    order = a.argsort()
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    n1 = len(pos)
    r1 = ranks[:n1].sum()
    u = r1 - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * len(neg)))


def collect(data_dir):
    by_label = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(data_dir, "*.ndjson.gz"))):
        lab, rows = read_rows(p)
        if rows:
            by_label[lab].append((os.path.basename(p), rows))
    return by_label


def node_matrix(rows, node, field):
    out = []
    for r in rows:
        d = r["nodes"].get(node)
        if d and d.get(field):
            out.append(d[field])
    if not out:
        return np.zeros((0, 0))
    w = min(len(v) for v in out)
    return np.asarray([v[:w] for v in out], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/presence")
    args = ap.parse_args()

    by_label = collect(args.data)
    if not by_label:
        print("no data")
        return 1

    print("=" * 88)
    print("PRESENCE SIGNAL DIAGNOSIS".center(88))
    print("=" * 88)
    for lab in sorted(by_label):
        n = sum(len(r) for _, r in by_label[lab])
        print(f"  {lab:<14}{len(by_label[lab])} file(s), {n} windows")

    occ = [r for _, rows in by_label.get("occupied", []) for r in rows]
    abs_rows = [r for _, rows in by_label.get("absent", []) for r in rows]
    if not occ or not abs_rows:
        print("\nneed both 'occupied' and 'absent' to compare")
        return 1

    nodes = sorted({n for r in occ + abs_rows for n in r["nodes"]}, key=int)

    # ---- 1 & 2: per-node motion statistics --------------------------------
    print("\n[1] amp_std percentiles per node (motion energy)\n")
    print(f"    {'node':<6}{'label':<12}{'p50':>9}{'p90':>9}{'p99':>9}{'max':>9}")
    print("    " + "-" * 54)
    for nd in nodes:
        for lab, rows in (("occupied", occ), ("absent", abs_rows)):
            M = node_matrix(rows, nd, "amp_std")
            if M.size == 0:
                continue
            print(f"    {nd:<6}{lab:<12}{np.percentile(M,50):>9.3f}"
                  f"{np.percentile(M,90):>9.3f}{np.percentile(M,99):>9.3f}{M.max():>9.3f}")
        print()

    # ---- 3: per-subcarrier AUC, occupied vs absent ------------------------
    print("[2] Per-subcarrier separability (amp_mean), occupied vs absent\n")
    print(f"    {'node':<6}{'bins':>6}{'AUC max':>10}{'AUC p95':>10}{'AUC med':>10}"
          f"{'|AUC-.5|>0.2':>14}")
    print("    " + "-" * 58)
    real_auc = {}
    for nd in nodes:
        A = node_matrix(occ, nd, "amp_mean")
        B = node_matrix(abs_rows, nd, "amp_mean")
        if A.size == 0 or B.size == 0:
            continue
        w = min(A.shape[1], B.shape[1])
        au = np.array([auc(A[:, k], B[:, k]) for k in range(w)])
        real_auc[nd] = au
        strong = int((np.abs(au - 0.5) > 0.2).sum())
        print(f"    {nd:<6}{w:>6}{np.abs(au-0.5).max()+0.5:>10.3f}"
              f"{np.percentile(np.abs(au-0.5),95)+0.5:>10.3f}"
              f"{np.percentile(np.abs(au-0.5),50)+0.5:>10.3f}{strong:>14}")

    # ---- 4: drift control --------------------------------------------------
    print("\n[3] DRIFT CONTROL - first half vs second half of 'absent'")
    print("    (both halves are an empty room: any separability here is drift)\n")
    half = len(abs_rows) // 2
    A1, A2 = abs_rows[:half], abs_rows[half:]
    print(f"    {'node':<6}{'bins':>6}{'AUC max':>10}{'AUC p95':>10}{'AUC med':>10}"
          f"{'|AUC-.5|>0.2':>14}")
    print("    " + "-" * 58)
    drift_auc = {}
    for nd in nodes:
        A = node_matrix(A1, nd, "amp_mean")
        B = node_matrix(A2, nd, "amp_mean")
        if A.size == 0 or B.size == 0:
            continue
        w = min(A.shape[1], B.shape[1])
        au = np.array([auc(A[:, k], B[:, k]) for k in range(w)])
        drift_auc[nd] = au
        strong = int((np.abs(au - 0.5) > 0.2).sum())
        print(f"    {nd:<6}{w:>6}{np.abs(au-0.5).max()+0.5:>10.3f}"
              f"{np.percentile(np.abs(au-0.5),95)+0.5:>10.3f}"
              f"{np.percentile(np.abs(au-0.5),50)+0.5:>10.3f}{strong:>14}")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 88)
    print("VERDICT".center(88))
    print("=" * 88)
    for nd in nodes:
        if nd not in real_auc or nd not in drift_auc:
            continue
        r = float(np.percentile(np.abs(real_auc[nd] - 0.5), 95))
        d = float(np.percentile(np.abs(drift_auc[nd] - 0.5), 95))
        margin = r - d
        tag = ("REAL - presence exceeds drift" if margin > 0.10 else
               "MARGINAL" if margin > 0.04 else
               "INDISTINGUISHABLE FROM DRIFT")
        print(f"  node {nd}: presence={r+0.5:.3f}  drift={d+0.5:.3f}  "
              f"margin={margin:+.3f}   {tag}")
    print("\n  A presence AUC only matters to the extent it EXCEEDS the drift AUC.")
    print("  If the margin is near zero, a classifier would be learning what time")
    print("  it is, not whether anyone is home.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
