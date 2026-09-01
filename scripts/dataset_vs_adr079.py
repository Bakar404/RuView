#!/usr/bin/env python3
"""
Compare this installation's captured dataset against what ADR-079 specifies.

ADR-079 (docs/adr/ADR-079-camera-ground-truth-training.md:389-402) recommends
6 sessions x 5 min = 30 min -> ~180,000 CSI frames -> ~9,000 paired samples.
It also gates training on confidence > 0.5 (:211) and requires >= 3 camera
frames per CSI window (:169).

This script measures what we actually have, and - more importantly - how much
the *poses* vary. A dataset where every frame is the same pose cannot be
learned from: the mean-pose baseline already scores near-perfect, so there is
no headroom for a model to demonstrate skill. That is the trap ADR-079's own
proxy baseline fell into (PCK@20 35.3%, i.e. equal to the ADR's 35% target).

Usage:
    python scripts/dataset_vs_adr079.py
"""

import json
import math
from pathlib import Path

GT_DIR = Path("data/ground-truth")
PAIRED_DIR = Path("data/paired-v2")

ADR_TARGET_SAMPLES = 9000
ADR_TARGET_MINUTES = 30
CONF_GATE = 0.5

COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho",
        "l_elb", "r_elb", "l_wri", "r_wri", "l_hip", "r_hip",
        "l_kne", "r_kne", "l_ank", "r_ank"]


def load_gt(path):
    frames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            frames.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return frames


def torso_of(kp):
    """Hip width, matching metrics_core.rs:185-220 canonical PCK."""
    lh, rh = kp[11], kp[12]
    if lh[2] >= 0.5 and rh[2] >= 0.5:
        d = math.hypot(lh[0] - rh[0], lh[1] - rh[1])
        if d > 1e-6:
            return d
    xs = [p[0] for p in kp if p[2] >= 0.5]
    ys = [p[1] for p in kp if p[2] >= 0.5]
    if len(xs) < 2:
        return None
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def main():
    gt_files = sorted(GT_DIR.glob("*.vis.jsonl"))
    if not gt_files:
        print(f"no ground-truth captures under {GT_DIR}")
        return

    print("=" * 78)
    print("CAPTURED DATA vs ADR-079 SPECIFICATION")
    print("=" * 78)

    tot_frames = 0
    tot_seconds = 0.0
    all_conf = []
    all_kp = []
    per_capture = []

    for f in gt_files:
        frames = load_gt(f)
        if not frames:
            continue
        ts = [fr.get("ts_ns") or 0 for fr in frames]
        ts = [t / 1e9 for t in ts if t]
        dur = (max(ts) - min(ts)) if len(ts) > 1 else 0.0

        confs = []
        kps = []
        for fr in frames:
            kp = fr.get("keypoints") or fr.get("kp")
            if not kp or len(kp) != 17:
                continue
            c = fr.get("confidence")
            if c is None:
                vis = [p[2] for p in kp if len(p) > 2]
                c = sum(vis) / len(vis) if vis else 0.0
            confs.append(c)
            kps.append(kp)

        if not confs:
            continue
        tot_frames += len(frames)
        tot_seconds += dur
        all_conf.extend(confs)
        all_kp.extend(kps)
        per_capture.append((f.name, len(frames), dur, confs, kps))

    # ---- volume ----
    paired = 0
    for p in sorted(PAIRED_DIR.glob("*.paired.jsonl")):
        paired += sum(1 for ln in p.read_text(encoding="utf-8").splitlines()
                      if ln.strip())

    mins = tot_seconds / 60.0
    print(f"\n-- VOLUME --")
    print(f"  camera frames captured : {tot_frames:,}")
    print(f"  capture wall time      : {mins:.1f} min")
    print(f"  paired samples         : {paired:,}")
    print(f"  pairing yield          : {100.0*paired/max(tot_frames,1):.1f}% "
          f"of camera frames")
    print()
    print(f"  ADR-079 recommends     : {ADR_TARGET_MINUTES} min -> "
          f"{ADR_TARGET_SAMPLES:,} paired samples")
    print(f"  you have               : {mins/ADR_TARGET_MINUTES*100:.0f}% of the "
          f"time, {100.0*paired/ADR_TARGET_SAMPLES:.0f}% of the samples")

    # ---- confidence ----
    all_conf.sort()
    n = len(all_conf)

    def pct(q):
        return all_conf[min(n - 1, int(q * n))]

    passing = sum(1 for c in all_conf if c > CONF_GATE)
    print(f"\n-- CONFIDENCE (ADR-079:211 trains only on conf > {CONF_GATE}) --")
    print(f"  mean {sum(all_conf)/n:.3f}   p10 {pct(.10):.3f}   "
          f"p50 {pct(.50):.3f}   p90 {pct(.90):.3f}")
    print(f"  frames above gate      : {passing:,}/{n:,} "
          f"({100.0*passing/n:.1f}%)")

    # ---- per-keypoint visibility ----
    print(f"\n-- PER-KEYPOINT VISIBILITY (visible = v >= 0.5) --")
    for i, name in enumerate(COCO):
        vis = sum(1 for kp in all_kp if kp[i][2] >= 0.5)
        rate = 100.0 * vis / len(all_kp)
        bar = "#" * int(rate / 4)
        flag = "  <-- rarely visible" if rate < 50 else ""
        print(f"  {name:<6} {rate:5.1f}%  {bar}{flag}")

    # ---- mean-pose baseline: the number a model must beat ----
    def mean_pose_pck(kps, thresh=0.20):
        """PCK@20 of a constant predictor that always emits the mean pose.

        This is the ADR-291 / baseline_pck.py discipline: quote model PCK only
        as a delta over this. Computed per-capture, because each capture has
        its own camera framing - a global mean pose mixes coordinate frames
        (ADR-152) and is not a meaningful predictor.
        """
        if not kps:
            return None
        mk = []
        for i in range(17):
            pts = [(k[i][0], k[i][1]) for k in kps if k[i][2] >= 0.5]
            mk.append((sum(p[0] for p in pts) / len(pts),
                       sum(p[1] for p in pts) / len(pts)) if pts else None)
        hit = tot = 0
        for k in kps:
            t = torso_of(k)
            if not t:
                tot += 17          # unscoreable -> 0, never 1.0
                continue
            for i in range(17):
                if k[i][2] < 0.5 or mk[i] is None:
                    continue
                tot += 1
                if math.hypot(k[i][0] - mk[i][0], k[i][1] - mk[i][1]) < thresh * t:
                    hit += 1
        return 100.0 * hit / tot if tot else None

    print(f"\n-- MEAN-POSE PCK@20 BASELINE (what a constant predictor scores) --")
    glob_b = mean_pose_pck(all_kp)
    print(f"  pooled across all captures : {glob_b:5.2f}%   "
          f"<- mixes camera frames (ADR-152), not meaningful")
    print(f"  per capture:")
    for name, nf, dur, confs, kps in per_capture:
        b = mean_pose_pck(kps)
        short = name.replace("keypoints_", "").replace(".vis.jsonl", "")
        print(f"    {short:<26} {b:5.2f}%   ({len(kps):,} frames)")
    print(f"\n  ADR-079 sets its success target at PCK@20 > 35% (:245), but the")
    print(f"  ADR's own proxy baseline already scores 35.3% (:497). A model at")
    print(f"  35% has therefore demonstrated NO skill. Only the delta counts.")

    # ---- per capture ----
    print(f"\n-- POSE DIVERSITY (torso-normalised, mean-pose relative) --")
    mean_kp = []
    for i in range(17):
        pts = [(kp[i][0], kp[i][1]) for kp in all_kp if kp[i][2] >= 0.5]
        mean_kp.append((sum(p[0] for p in pts) / len(pts),
                        sum(p[1] for p in pts) / len(pts)) if pts else (0.0, 0.0))

    devs = []
    for kp in all_kp:
        t = torso_of(kp)
        if not t:
            continue
        for i in range(17):
            if kp[i][2] >= 0.5:
                d = math.hypot(kp[i][0] - mean_kp[i][0], kp[i][1] - mean_kp[i][1])
                devs.append(d / t)
    devs.sort()
    m = len(devs)
    within20 = sum(1 for d in devs if d < 0.20)
    print(f"  deviation from the MEAN POSE, in torso units:")
    print(f"    p50 {devs[m//2]:.3f}   p90 {devs[int(.9*m)]:.3f}   "
          f"max {devs[-1]:.3f}")
    print(f"  keypoints within 0.20 torso of the mean pose: "
          f"{100.0*within20/m:.1f}%")
    print(f"  ^ this IS the mean-pose PCK@20 baseline a model must beat")

    # ---- per capture ----
    print(f"\n-- PER CAPTURE --")
    print(f"  {'capture':<38} {'frames':>7} {'sec':>7} {'conf':>6} {'div':>7}")
    for name, nf, dur, confs, kps in per_capture:
        mk = []
        for i in range(17):
            pts = [(k[i][0], k[i][1]) for k in kps if k[i][2] >= 0.5]
            mk.append((sum(p[0] for p in pts)/len(pts),
                       sum(p[1] for p in pts)/len(pts)) if pts else (0., 0.))
        dd = []
        for k in kps:
            t = torso_of(k)
            if not t:
                continue
            for i in range(17):
                if k[i][2] >= 0.5:
                    dd.append(math.hypot(k[i][0]-mk[i][0], k[i][1]-mk[i][1]) / t)
        div = sum(dd)/len(dd) if dd else 0.0
        short = name.replace("keypoints_", "").replace(".vis.jsonl", "")
        print(f"  {short:<38} {nf:>7,} {dur:>7.1f} "
              f"{sum(confs)/len(confs):>6.3f} {div:>7.3f}")


if __name__ == "__main__":
    main()
