"""Match ground-truth captures to CSI recordings by timestamp overlap."""
import json
from pathlib import Path

GT = Path("data/ground-truth")
RD = Path("data/recordings")


def gt_span(p):
    lo = hi = None
    n = 0
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get("ts_ns")
            except Exception:
                continue
            if t is None:
                continue
            t /= 1e9
            n += 1
            lo = t if lo is None else min(lo, t)
            hi = t if hi is None else max(hi, t)
    return lo, hi, n


def csi_span(p):
    """Read first and last parseable timestamp without loading the file."""
    lo = hi = None
    n = 0
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get("timestamp")
            except Exception:
                continue
            if t is None:
                continue
            n += 1
            lo = t if lo is None else min(lo, t)
            hi = t if hi is None else max(hi, t)
    return lo, hi, n


csis = []
for p in sorted(RD.glob("rec_*.jsonl")):
    lo, hi, n = csi_span(p)
    if lo:
        csis.append((p, lo, hi, n))
        print(f"CSI  {p.name:<26} {lo:.1f} .. {hi:.1f}  ({hi-lo:6.1f}s, {n} frames)")

print()
pairs = []
for p in sorted(GT.glob("*.vis.jsonl")):
    lo, hi, n = gt_span(p)
    if lo is None:
        continue
    best = None
    for cp, clo, chi, cn in csis:
        ov = min(hi, chi) - max(lo, clo)
        if best is None or ov > best[1]:
            best = (cp, ov)
    frac = 100.0 * best[1] / (hi - lo) if hi > lo else 0
    status = "OK" if frac > 80 else ("PARTIAL" if frac > 30 else "NO MATCH")
    print(f"GT   {p.name:<40} {lo:.1f}..{hi:.1f} ({n:5} rec)")
    print(f"     -> {best[0].name:<26} overlap {best[1]:7.1f}s "
          f"({frac:5.1f}% of GT)  {status}")
    if frac > 30:
        pairs.append((p, best[0]))

print("\n--- alignable pairs ---")
for g, c in pairs:
    print(f"  {g.name}  <->  {c.name}")
Path("data/paired").mkdir(parents=True, exist_ok=True)
with open("data/paired/_pairs.txt", "w", encoding="utf-8") as fh:
    for g, c in pairs:
        fh.write(f"{g}\t{c}\n")
print(f"\nwrote data/paired/_pairs.txt ({len(pairs)} pairs)")
