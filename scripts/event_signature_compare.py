#!/usr/bin/env python3
"""Compare the signature of an unexplained event against a KNOWN human signature.

Context
-------
Three events were observed in the existing captures:

  A. KNOWN HUMAN - arrival at 17:49:55 in absent_20260831_173643. The occupant
     confirmed coming home, so this is ground truth for "person in the room".
  B. KNOWN HUMAN - departure around 16:24-16:29 in transition_20260831_161637.
  C. UNEXPLAINED - a 20-minute excursion from 16:49 to 17:09 inside
     absent_20260831_163640, during a period the apartment was confirmed empty.

Question: is C the same kind of physical event as A?

Method
------
For each event, compute a standardised deviation profile relative to a nearby
empty baseline drawn from the SAME capture (so no cross-capture offset):

    delta[k] = (mean_event[k] - mean_base[k]) / sd_base[k]

then compare profiles with Pearson correlation. A person at a given location
perturbs a specific, repeatable set of subcarriers on each node. If C shares A's
spatial/spectral pattern, it is most likely also a body (a neighbour through the
wall, or someone in the hallway - this repo does through-wall sensing). If the
correlation is near zero, C is a different phenomenon (interference, an
appliance, router rate adaptation) and should be treated as a false positive.

Also reports per-node deviation magnitude, since a person outside the apartment
should load the nodes nearest that wall rather than all three equally.

Usage
-----
    python scripts/event_signature_compare.py
"""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime

import numpy as np

DATA = "data/presence"


def read_rows(path):
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
                if r.get("nodes"):
                    rows.append(r)
    except (EOFError, OSError):
        pass
    return rows


def minute_of(r):
    d = datetime.fromisoformat(r["utc"].replace("Z", "+00:00")).astimezone()
    return d.hour * 60 + d.minute + d.second / 60.0


def seg(rows, a, b):
    return [r for r in rows if a <= minute_of(r) < b]


def mat(rows, node, field="amp_mean"):
    v = [r["nodes"][node][field] for r in rows
         if node in r["nodes"] and r["nodes"][node].get(field)]
    if not v:
        return np.zeros((0, 0))
    w = min(len(x) for x in v)
    return np.asarray([x[:w] for x in v], dtype=np.float64)


def delta_profile(event_rows, base_rows, nodes):
    """Standardised per-subcarrier deviation, concatenated across nodes."""
    out, per_node = [], {}
    for nd in nodes:
        E = mat(event_rows, nd)
        B = mat(base_rows, nd)
        if E.size == 0 or B.size == 0:
            return None, None
        w = min(E.shape[1], B.shape[1])
        mu = B[:, :w].mean(axis=0)
        sd = B[:, :w].std(axis=0)
        sd = np.maximum(sd, 0.05 * np.median(mu))
        d = (E[:, :w].mean(axis=0) - mu) / sd
        per_node[nd] = float(np.abs(d).mean())
        out.append(d)
    return np.concatenate(out), per_node


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()

    hour = read_rows(f"{args.data}/absent_20260831_163640.ndjson.gz")
    arrive = read_rows(f"{args.data}/absent_20260831_173643.ndjson.gz")
    trans = read_rows(f"{args.data}/transition_20260831_161637.ndjson.gz")
    if not (hour and arrive and trans):
        print("missing captures")
        return 1

    nodes = sorted({n for r in arrive for n in r["nodes"]}, key=int)
    H = 17 * 60
    G = 16 * 60

    events = {}

    # A: known human, arrival (baseline = same capture, before arrival)
    events["A_human_arrival"] = (
        seg(arrive, H + 50, H + 57), seg(arrive, H + 38, H + 48))

    # B: known human, still present early in the transition capture
    #    (baseline = the confirmed-empty tail of that same capture)
    events["B_human_departure"] = (
        seg(trans, G + 16, G + 26), seg(trans, G + 31, G + 36))

    # C: the unexplained excursion (baseline = same capture, after it ends)
    events["C_unexplained_1649"] = (
        seg(hour, G + 50, G + 68), seg(hour, H + 12, H + 32))

    # D: negative control - two genuinely empty stretches of the same capture
    events["D_empty_control"] = (
        seg(hour, H + 14, H + 22), seg(hour, H + 24, H + 32))

    print("=" * 80)
    print("EVENT SIGNATURE COMPARISON".center(80))
    print("=" * 80)

    profiles = {}
    print(f"\n  {'event':<24}{'n_evt':>7}{'n_base':>8}   per-node |deviation|")
    print("  " + "-" * 68)
    for name, (ev, base) in events.items():
        if len(ev) < 20 or len(base) < 20:
            print(f"  {name:<24}{len(ev):>7}{len(base):>8}   (insufficient)")
            continue
        d, per = delta_profile(ev, base, nodes)
        if d is None:
            continue
        profiles[name] = d
        s = "  ".join(f"n{k}={v:5.2f}" for k, v in per.items())
        print(f"  {name:<24}{len(ev):>7}{len(base):>8}   {s}")

    print("\n" + "=" * 80)
    print("PROFILE CORRELATION".center(80))
    print("=" * 80)
    keys = list(profiles)
    print(f"\n  {'':<24}" + "".join(f"{k[:14]:>16}" for k in keys))
    print("  " + "-" * (24 + 16 * len(keys)))
    for a in keys:
        row = f"  {a:<24}"
        for b in keys:
            row += f"{corr(profiles[a], profiles[b]):>16.3f}"
        print(row)

    print("\n" + "=" * 80)
    print("VERDICT".center(80))
    print("=" * 80)
    if "C_unexplained_1649" in profiles and "A_human_arrival" in profiles:
        c = corr(profiles["C_unexplained_1649"], profiles["A_human_arrival"])
        ctrl = (corr(profiles["D_empty_control"], profiles["A_human_arrival"])
                if "D_empty_control" in profiles else float("nan"))
        print(f"\n  unexplained-vs-human correlation : {c:+.3f}")
        print(f"  empty-control-vs-human           : {ctrl:+.3f}")
        if abs(c) > 0.5 and abs(c) > abs(ctrl) * 2:
            print("\n  -> The 16:49 event carries a HUMAN-LIKE signature. Most likely a")
            print("     real body outside the apartment (hallway / neighbour through")
            print("     the wall), not an instrument artifact.")
        elif abs(c) < 0.25:
            print("\n  -> The 16:49 event does NOT resemble a human. Treat it as an")
            print("     environmental false positive; the detector needs a drift-")
            print("     robust feature or periodic recalibration.")
        else:
            print("\n  -> Ambiguous. Correlation is neither clearly human nor clearly")
            print("     unrelated; more labelled events would be needed to settle it.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
