"""Upgrade legacy [x, y] ground-truth JSONL to [x, y, visibility].

Older captures (before visibility was emitted) stored only coordinates.
MediaPipe extrapolates landmarks outside the image, so any point falling
outside the normalized [0, 1] box is a guess, not an observation. Those are
marked invisible so the trainer excludes them from the loss
(losses.rs:94-104) and from PCK (metrics_core.rs:85,114).

Usage:  python scripts/upgrade_gt_visibility.py <in.jsonl> [out.jsonl]
"""
import json
import sys
from pathlib import Path

COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
        "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle"]

src = Path(sys.argv[1])
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".vis.jsonl")

n_rec = 0
n_up = 0
masked = [0] * 17
present = [0] * 17

with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
    for line in fi:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        n_rec += 1
        kps = r.get("keypoints") or []
        new = []
        for j, kp in enumerate(kps):
            if len(kp) >= 3:
                new.append(kp)
                if kp[2] < 0.5:
                    masked[j] += 1
                present[j] += 1
                continue
            x, y = kp[0], kp[1]
            vis = 1.0 if (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) else 0.0
            if vis == 0.0:
                masked[j] += 1
            present[j] += 1
            new.append([x, y, vis])
        if new != kps:
            n_up += 1
        r["keypoints"] = new
        r["n_visible"] = sum(1 for kp in new if len(kp) > 2 and kp[2] >= 0.5)
        fo.write(json.dumps(r) + "\n")

print(f"read    : {src}")
print(f"wrote   : {dst}")
print(f"records : {n_rec}   upgraded: {n_up}")
print("\nper-joint masking:")
for j, name in enumerate(COCO):
    if not present[j]:
        continue
    pct = 100.0 * masked[j] / present[j]
    flag = "  <-- masked out" if pct > 50 else ""
    print(f"  {name:<12} {masked[j]:6}/{present[j]:<6} {pct:6.1f}% invisible{flag}")
