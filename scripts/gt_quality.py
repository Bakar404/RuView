"""Quality-check a ground-truth JSONL and its temporal alignment with CSI."""
import json
import sys
import statistics as st
from collections import Counter

COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
        "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle"]

path = sys.argv[1]
recs = []
with open(path, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            recs.append(json.loads(line))

print(f"records: {len(recs)}")
ts = [r["ts_ns"] for r in recs if "ts_ns" in r]
if len(ts) > 2:
    span = (max(ts) - min(ts)) / 1e9
    d = sorted((ts[i + 1] - ts[i]) / 1e6 for i in range(len(ts) - 1))
    print(f"span   : {span:.1f} s   rate {len(ts)/span:.1f} Hz")
    print(f"gap ms : median {d[len(d)//2]:.1f}  p95 {d[int(len(d)*.95)]:.1f}  "
          f"max {max(d):.1f}")
    print(f"epoch  : {min(ts)/1e9:.3f} .. {max(ts)/1e9:.3f}")

conf = [r.get("confidence", 0) for r in recs]
vis = [r.get("n_visible", 0) for r in recs]
print(f"\nconfidence : mean {sum(conf)/len(conf):.3f}  "
      f"min {min(conf):.3f}  p10 {sorted(conf)[len(conf)//10]:.3f}")
print(f"n_visible  : mean {sum(vis)/len(vis):.2f}/17   "
      f"dist {dict(sorted(Counter(vis).items()))}")

print("\n--- per-keypoint: fraction OUTSIDE the frame (extrapolated) ---")
print(f"  {'joint':<12}{'out%':>8}{'y>1 (below)':>13}{'x out':>9}"
      f"{'mean y':>9}")
bad = 0
for i, name in enumerate(COCO):
    xs, ys = [], []
    for r in recs:
        kps = r.get("keypoints") or []
        if len(kps) <= i:
            continue
        kp = kps[i]
        xs.append(kp[0])
        ys.append(kp[1])
    if not xs:
        print(f"  {name:<12}   (no data)")
        continue
    out = sum(1 for x, y in zip(xs, ys) if not (0 <= x <= 1 and 0 <= y <= 1))
    below = sum(1 for y in ys if y > 1)
    xout = sum(1 for x in xs if not 0 <= x <= 1)
    pct = 100.0 * out / len(xs)
    if pct > 20:
        bad += 1
    flag = "  <-- mostly extrapolated" if pct > 50 else ""
    print(f"  {name:<12}{pct:7.1f}%{100.0*below/len(ys):12.1f}%"
          f"{100.0*xout/len(xs):8.1f}%{sum(ys)/len(ys):9.3f}{flag}")

print(f"\n  joints >20% out of frame: {bad}/17")
if bad == 0:
    print("  >> FRAMING GOOD - whole body captured")
elif bad <= 3:
    print("  >> FRAMING OK - minor clipping, usable")
else:
    print("  >> FRAMING POOR - raise/move camera back, too much extrapolation")
