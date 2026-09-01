#!/usr/bin/env python3
"""End-to-end test of the phase pipeline: recording -> aligner -> MM-Fi.

Builds a synthetic `sensing_update` recording that mirrors the real server
output (3 nodes per tick, each with amplitude AND phase) plus a matching
ground-truth keypoint file, then drives the real align-ground-truth.js and
paired_to_mmfi.py over it.

Verifies three things that no unit test covers together:
  1. the aligner carries `phase` from the recording into `csi_phase`
  2. the node axis is ordered by node_id, so a given index is a fixed node
  3. the converter writes wifi_csi_phase.npy shaped like wifi_csi.npy

Node n is given the constant phase marker n/10 so misordering is detectable:
if the axis were rotated, the recovered marker order would not be 1,2,3.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "data" / "_phasetest"
N_SC = 56
NODES = [1, 2, 3]
TICK_MS = 20
N_TICKS = 400
WINDOW = 7


def build_recording(path: Path) -> None:
    t0 = 1788150000.0
    with path.open("w", encoding="utf-8") as f:
        for k in range(N_TICKS):
            ts = t0 + k * TICK_MS / 1000.0
            nodes = []
            # Emit nodes in DESCENDING id, like the real server ([3,2,1]),
            # so the test fails if the aligner trusts arrival order.
            for n in sorted(NODES, reverse=True):
                amp = [10.0 + n + math.sin(k * 0.1 + s * 0.05) for s in range(N_SC)]
                phase = [n / 10.0] * N_SC
                nodes.append({
                    "node_id": n,
                    "rssi_dbm": -42.0,
                    "position": [2.0, 0.0, 1.5],
                    "amplitude": amp,
                    "phase": phase,
                    "subcarrier_count": N_SC,
                })
            f.write(json.dumps({
                "type": "sensing_update",
                "timestamp": ts,
                "source": "esp32",
                "tick": k,
                "nodes": nodes,
                "features": {},
            }) + "\n")


def build_ground_truth(path: Path) -> None:
    t0_ns = int(1788150000.0 * 1e9)
    step_ns = int(33e6)  # ~30 fps
    n = int(N_TICKS * TICK_MS / 33)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            kp = [[0.4 + 0.1 * math.sin(i * 0.05 + j), 0.5 + 0.1 * math.cos(i * 0.05 + j), 1.0]
                  for j in range(17)]
            f.write(json.dumps({
                "ts_ns": t0_ns + i * step_ns,
                "keypoints": kp,
                "confidence": 0.9,
                "n_visible": 17,
                "n_persons": 1,
            }) + "\n")


def main() -> int:
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    rec, gt = TMP / "rec_test.jsonl", TMP / "gt_test.jsonl"
    paired, mmfi = TMP / "paired", TMP / "mmfi"
    paired.mkdir()

    build_recording(rec)
    build_ground_truth(gt)
    print(f"  synthetic recording: {N_TICKS} ticks x {len(NODES)} nodes")

    r = subprocess.run(
        ["node", str(ROOT / "scripts" / "align-ground-truth.js"),
         "--gt", str(gt), "--csi", str(rec),
         "--window-frames", str(WINDOW),
         "--output", str(paired / "gt_test.paired.jsonl")],
        capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        return 1
    for line in r.stdout.splitlines():
        if "Detected" in line or "Paired samples" in line:
            print("  " + line.strip())

    rows = [json.loads(l) for l in
            (paired / "gt_test.paired.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        print("  FAIL: aligner produced no paired samples")
        return 1

    rec0 = rows[0]
    ok = True

    if "csi_phase" not in rec0:
        print("  FAIL: paired record has no csi_phase"); ok = False
    else:
        print(f"  csi_phase present, len={len(rec0['csi_phase'])}")

    if rec0.get("csi_layout") != "time_node_major":
        print(f"  FAIL: layout {rec0.get('csi_layout')}"); ok = False
    if rec0.get("node_ids") != NODES:
        print(f"  FAIL: node_ids {rec0.get('node_ids')} != {NODES}"); ok = False
    else:
        print(f"  node_ids = {rec0['node_ids']} (ascending, as required)")

    if ok:
        ph = np.asarray(rec0["csi_phase"]).reshape(rec0["csi_shape"])
        markers = [round(float(ph[0, i, 0]) * 10) for i in range(len(NODES))]
        print(f"  recovered node markers along axis 1: {markers}")
        if markers != NODES:
            print("  FAIL: node axis is misordered (input was emitted 3,2,1)")
            ok = False
        else:
            print("  node axis correctly sorted despite reversed input order")

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "paired_to_mmfi.py"),
         "--src", str(paired), "--out", str(mmfi),
         "--window", str(WINDOW), "--subcarriers", str(N_SC), "--clean"],
        capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:]); return 1

    clip = next(mmfi.glob("S*/A*"), None)
    if clip is None:
        print("  FAIL: converter wrote no clips"); return 1
    amp_f, ph_f = clip / "wifi_csi.npy", clip / "wifi_csi_phase.npy"
    if not ph_f.exists():
        print("  FAIL: wifi_csi_phase.npy not written"); ok = False
    else:
        a, p = np.load(amp_f), np.load(ph_f)
        print(f"  wifi_csi.npy       {a.shape}")
        print(f"  wifi_csi_phase.npy {p.shape}")
        if a.shape != p.shape:
            print("  FAIL: shape mismatch"); ok = False
        if float(np.abs(p).max()) == 0.0:
            print("  FAIL: phase tensor is all zeros"); ok = False
        else:
            print(f"  phase non-zero (max |phase| = {float(np.abs(p).max()):.3f})")

    print()
    print("  RESULT: PASS - phase flows recording -> paired -> MM-Fi" if ok
          else "  RESULT: FAIL")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
