#!/usr/bin/env python3
"""Convert data/paired/*.paired.jsonl into the MM-Fi .npy tree the Rust trainer reads.

Why this exists
---------------
Two trainers coexist in this repo and they do NOT read the same thing:

  * scripts/train-wiflow-supervised.js  reads .paired.jsonl, optimises with SPSA
    (a derivative-free method, :1180) -- one scalar of gradient signal per step.
  * v2 crates/wifi-densepose-train `train`  optimises with AdamW + .backward()
    (trainer.rs:134,240) but discovers MM-Fi .npy clips (dataset.rs:352-407).

Only the Rust path does real backpropagation, so paired data has to be
materialised in the layout MmFiDataset::discover expects:

    <root>/S<nn>/A<nnnn>/wifi_csi.npy      [T, n_tx, n_rx, n_sc]  f32
                         gt_keypoints.npy  [T, 17, 3]             f32

Layout choices
--------------
Each paired sample is already a self-contained 20-frame window. MM-Fi clips are
windowed with stride 1 (dataset.rs:530-531), so packing many samples into one
clip would splice unrelated windows together. Instead every sample becomes its
own action dir with T == window_frames, yielding exactly one window per clip
(num_windows = T - window_frames + 1 = 1).

Subject id = source capture. dataset.rs splits train/test at the subject level
(:535-537), so this makes held-out captures the evaluation set -- an honest
cross-camera-geometry test rather than a leaky within-capture split.

Keypoint visibility is preserved in column 2, which is exactly what the loader
reads (dataset.rs:506-508) and what losses.rs:94-104 masks on.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

NUM_KEYPOINTS = 17
N_TX = 1
N_RX = 1


def detect_subcarriers(files: list[Path], window: int) -> int | None:
    """Infer the subcarrier width from the first well-formed paired record.

    The emitted width is a property of the capture (ESP32 HT40 now yields 166
    usable bins after null masking; legacy captures carry 56). Hard-coding it
    meant a width change silently skipped every record and produced an empty
    dataset, so it is derived from the data instead.
    """
    for src in files:
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            shape = rec.get("csi_shape")
            if not shape or len(shape) not in (2, 3):
                continue
            if len(shape) == 3:
                rows, _n_nodes, cols = shape
            else:
                rows, cols = shape
                if rec.get("csi_layout", "time_major") != "time_major":
                    rows, cols = cols, rows
            if rows == window:
                return int(cols)
    return None


def convert_file(src: Path, subject_dir: Path, window: int, n_sub: int) -> tuple[int, int]:
    written = skipped = 0

    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        csi = rec.get("csi")
        kp = rec.get("kp") or rec.get("keypoints")
        shape = rec.get("csi_shape")
        if not csi or not kp or not shape or len(shape) not in (2, 3):
            skipped += 1
            continue

        layout = rec.get("csi_layout", "time_major")
        arr = np.asarray(csi, dtype=np.float32)
        if arr.size != int(np.prod(shape)):
            skipped += 1
            continue

        if layout == "time_node_major":
            # [T, n_nodes, n_sc]: each node is a distinct physical link, so it
            # maps onto the receive-antenna axis the loader expects. Node order
            # is ascending node_id, fixed by the aligner, so a given index means
            # the same node in every sample.
            rows, n_nodes, cols = shape
            if cols != n_sub or rows != window:
                skipped += 1
                continue
            amp = arr.reshape(rows, 1, n_nodes, cols).astype(np.float32)
        else:
            rows, cols = shape
            # Normalise to [T, n_sub]
            mat = arr.reshape(rows, cols) if layout == "time_major" else arr.reshape(rows, cols).T
            if mat.shape != (window, n_sub):
                skipped += 1
                continue
            # [T, n_tx, n_rx, n_sc]
            amp = mat.reshape(window, N_TX, N_RX, n_sub).astype(np.float32)

        # Phase, when the capture carries it, must match the amplitude tensor
        # shape exactly: the loader pairs them element-wise.
        phase_arr = None
        raw_phase = rec.get("csi_phase")
        if raw_phase is not None:
            p = np.asarray(raw_phase, dtype=np.float32)
            if p.size == amp.size:
                phase_arr = p.reshape(amp.shape)

        # [17, 3] -> replicated to [T, 17, 3]; loader reads frame t_start only.
        kp_arr = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        for j in range(min(NUM_KEYPOINTS, len(kp))):
            p = kp[j]
            kp_arr[j, 0] = float(p[0])
            kp_arr[j, 1] = float(p[1])
            kp_arr[j, 2] = float(p[2]) if len(p) > 2 else 1.0
        kp_full = np.repeat(kp_arr[None, :, :], window, axis=0)

        action_dir = subject_dir / f"A{written + 1:04d}"
        action_dir.mkdir(parents=True, exist_ok=True)
        np.save(action_dir / "wifi_csi.npy", amp)
        if phase_arr is not None:
            np.save(action_dir / "wifi_csi_phase.npy", phase_arr)
        np.save(action_dir / "gt_keypoints.npy", kp_full)
        written += 1

    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/paired")
    ap.add_argument("--out", default="data/mmfi")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--subcarriers", type=int, default=None,
                    help="subcarrier width; auto-detected from the data if omitted")
    ap.add_argument("--clean", action="store_true", help="remove --out first")
    args = ap.parse_args()

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.src).glob("*.paired.jsonl"))
    if not files:
        print(f"No *.paired.jsonl under {args.src}")
        return

    total_w = total_s = 0
    n_sub = args.subcarriers
    if n_sub is None:
        n_sub = detect_subcarriers(files, args.window)
        if n_sub is None:
            print(f"Could not infer subcarrier width from {args.src} "
                  f"(no record with {args.window} frames). "
                  f"Pass --subcarriers explicitly.")
            return
        print(f"  detected subcarrier width: {n_sub}")

    for i, f in enumerate(files, start=1):
        subject = out / f"S{i:02d}"
        w, s = convert_file(f, subject, args.window, n_sub)
        total_w += w
        total_s += s
        print(f"  S{i:02d}  {f.name:<46} {w:5d} clips  {s:4d} skipped")

    if total_w == 0:
        print(f"\n  WARNING: 0 clips written — every record mismatched "
              f"window={args.window} or subcarriers={n_sub}.")

    print(f"\n  TOTAL: {total_w} clips across {len(files)} subjects ({total_s} skipped)")
    print(f"  root : {out.resolve()}")
    print(f"  shape: wifi_csi.npy [{args.window}, {N_TX}, {N_RX}, {n_sub}]  "
          f"gt_keypoints.npy [{args.window}, {NUM_KEYPOINTS}, 3]")


if __name__ == "__main__":
    main()
