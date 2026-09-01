#!/usr/bin/env python3
"""Long-duration CSI feature logger for PRESENCE / LOCATION classification.

Why this exists
---------------
Raw recording via POST /api/v1/recording/start writes ~20.5 KB per frame at
~127 frames/s = 2.6 MB/s = 9.4 GB/hour (measured). An overnight capture would
be ~75 GB, and that is the OLD 56-wide format; at 166 amplitudes + 166 phases
it is roughly 3x worse. Raw recording cannot be used for multi-hour captures.

Presence and coarse location change on a timescale of seconds, not milliseconds.
So instead of storing every frame we aggregate into fixed windows and store a
compact per-window feature vector. At the default 1 s window this is ~1000x
smaller and an 8-hour capture lands in the tens of MB.

It taps the UDP relay rather than the sensing server's WebSocket, which means:
  * no dependency on the server being up (it OOM-restarted once already);
  * raw full-width CSI, with null subcarriers masked here using the measured
    ESP32-S3 HT40 layout rather than trusting an upstream truncation;
  * zero disturbance to the live pipeline (the relay tap is a duplicate feed).

Features per window, per node
-----------------------------
  amp_mean[k]  mean amplitude of usable subcarrier k over the window
               -> the STATIC signature. A stationary body persistently reshapes
                  the amplitude-vs-subcarrier profile even with zero motion,
                  which is exactly what the server's motion-based classifier
                  cannot see (main.rs:2681 `presence: motion_score > 0.04`).
  amp_std[k]   temporal std of subcarrier k over the window
               -> the MOTION signature.
  rssi_mean, n_frames, gap flag.

Prerequisite
------------
The relay must be running with a tap:

    python scripts/udp-relay.py --listen-port 5005 --forward-port 5006 \
        --tap-host 127.0.0.1 --tap-port 5007

Usage
-----
    # empty apartment
    python scripts/presence_logger.py --label absent --duration 600

    # overnight, asleep in bed (Ctrl-C to stop, or set --duration)
    python scripts/presence_logger.py --label bed

    # working at the desk
    python scripts/presence_logger.py --label desk --duration 1800
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import signal
import socket
import struct
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

MAGIC = 0xC5110001
IQ_START = 20

_stop = False


def _on_sigint(signum, frame):
    global _stop
    _stop = True
    print("\n[stopping - flushing buffers]", flush=True)


def usable_subcarrier_indices(width: int) -> list[int]:
    """Mirror of `usable_subcarrier_indices` in main.rs.

    ESP32-S3 sends a sequence of 64-bin blocks: LLTF first, then one HT-LTF
    block per additional 64. Measured null (always exactly zero) bins, verified
    identical across all 3 nodes over 300 frames:
        LLTF block  : bin 0 (DC) and bins 27..=37      -> 12 nulls
        HT-LTF block: relative bins 29..=35            ->  7 nulls each
    For width 192 this yields nulls {0, 27-37, 93-99, 157-163} and 166 usable.
    """
    if width % 64 != 0:
        return list(range(width))
    out = []
    for block in range(width // 64):
        base = block * 64
        for rel in range(64):
            if block == 0:
                if rel == 0 or 27 <= rel <= 37:
                    continue
            else:
                if 29 <= rel <= 35:
                    continue
            out.append(base + rel)
    return out


def parse_frame(buf: bytes):
    if len(buf) < IQ_START:
        return None
    (magic,) = struct.unpack_from("<I", buf, 0)
    if magic != MAGIC:
        return None
    node_id = buf[4]
    n_ant = buf[5]
    (n_sub,) = struct.unpack_from("<H", buf, 6)
    rssi = struct.unpack_from("<b", buf, 16)[0]

    n_pairs = n_ant * n_sub
    if len(buf) < IQ_START + n_pairs * 2:
        return None

    amps = []
    for k in range(n_pairs):
        i_val = struct.unpack_from("<b", buf, IQ_START + k * 2)[0]
        q_val = struct.unpack_from("<b", buf, IQ_START + k * 2 + 1)[0]
        amps.append(math.sqrt(i_val * i_val + q_val * q_val))
    return node_id, n_sub, rssi, amps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="ground-truth class, e.g. absent | desk | bed")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5007, help="relay tap port")
    ap.add_argument("--window", type=float, default=1.0, help="seconds per feature row")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to capture; 0 = until Ctrl-C")
    ap.add_argument("--out", default="data/presence")
    ap.add_argument("--note", default="", help="free-text note stored in the header")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = outdir / f"{args.label}_{stamp}.ndjson.gz"

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as e:
        print(f"ERROR: cannot bind {args.host}:{args.port} - {e}")
        print("Is another logger already running on this tap port?")
        return 1
    sock.settimeout(2.0)

    print("=" * 74)
    print(f"PRESENCE LOGGER   label={args.label!r}")
    print("=" * 74)
    print(f"  tap        : udp://{args.host}:{args.port}")
    print(f"  window     : {args.window}s")
    print(f"  duration   : {'until Ctrl-C' if args.duration <= 0 else str(args.duration) + 's'}")
    print(f"  output     : {path}")
    print("\nWaiting for frames... (if nothing arrives, the relay is not tapping)\n")

    fh = gzip.open(path, "wt", encoding="utf-8", compresslevel=6)
    header = {
        "type": "header",
        "label": args.label,
        "note": args.note,
        "window_s": args.window,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "amp_mean[] and amp_std[] over usable subcarriers, per node",
        "source": "udp relay tap (raw, pre-truncation)",
    }
    fh.write(json.dumps(header) + "\n")

    buckets: dict[int, list[list[float]]] = defaultdict(list)
    rssis: dict[int, list[int]] = defaultdict(list)
    width_seen: dict[int, int] = {}
    usable_cache: dict[int, list[int]] = {}

    t0 = time.time()
    win_start = t0
    rows = 0
    frames = 0
    frames_win = 0
    last_report = t0
    empty_windows = 0

    try:
        while not _stop:
            if args.duration > 0 and (time.time() - t0) >= args.duration:
                break
            try:
                buf, _ = sock.recvfrom(65535)
                p = parse_frame(buf)
                if p:
                    node_id, n_sub, rssi, amps = p
                    if n_sub not in usable_cache:
                        usable_cache[n_sub] = usable_subcarrier_indices(n_sub)
                    idx = usable_cache[n_sub]
                    if len(amps) >= n_sub:
                        buckets[node_id].append([amps[i] for i in idx])
                        rssis[node_id].append(rssi)
                        width_seen[node_id] = n_sub
                        frames += 1
                        frames_win += 1
            except socket.timeout:
                pass

            now = time.time()
            if now - win_start >= args.window:
                row = {
                    "t": round(win_start - t0, 3),
                    "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "label": args.label,
                    "nodes": {},
                }
                for node_id, mat in sorted(buckets.items()):
                    if not mat:
                        continue
                    n = len(mat)
                    k = len(mat[0])
                    mean = [0.0] * k
                    for v in mat:
                        for j in range(k):
                            mean[j] += v[j]
                    mean = [m / n for m in mean]
                    if n > 1:
                        std = [0.0] * k
                        for v in mat:
                            for j in range(k):
                                d = v[j] - mean[j]
                                std[j] += d * d
                        std = [math.sqrt(s / n) for s in std]
                    else:
                        std = [0.0] * k
                    row["nodes"][str(node_id)] = {
                        "n": n,
                        "rssi": round(sum(rssis[node_id]) / len(rssis[node_id]), 1),
                        "w": width_seen[node_id],
                        "amp_mean": [round(x, 1) for x in mean],
                        "amp_std": [round(x, 2) for x in std],
                    }
                if row["nodes"]:
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                    rows += 1
                    empty_windows = 0
                else:
                    empty_windows += 1
                    if empty_windows in (5, 30, 120):
                        print(f"  !! {empty_windows} consecutive empty windows - "
                              f"check the relay tap is running", flush=True)

                buckets.clear()
                rssis.clear()
                frames_win = 0
                win_start = now
                if rows % 20 == 0:
                    fh.flush()

            if now - last_report >= 30.0:
                el = now - t0
                mb = path.stat().st_size / 1e6 if path.exists() else 0.0
                rate = mb / (el / 3600.0) if el > 0 else 0.0
                nodes = ",".join(str(n) for n in sorted(width_seen))
                print(f"  [{el/60:6.1f} min] rows={rows:6d} frames={frames:8d} "
                      f"nodes=[{nodes}] size={mb:.1f} MB ({rate:.1f} MB/h)", flush=True)
                last_report = now
    finally:
        fh.flush()
        fh.close()
        sock.close()

    el = time.time() - t0
    mb = path.stat().st_size / 1e6
    print("\n" + "-" * 74)
    print(f"  captured   : {el/60:.1f} min   rows={rows}   frames={frames}")
    print(f"  file       : {path}  ({mb:.1f} MB)")
    if rows == 0:
        print("\n  NO DATA CAPTURED. Start the relay with a tap:")
        print("    python scripts/udp-relay.py --listen-port 5005 --forward-port 5006 \\")
        print("        --tap-host 127.0.0.1 --tap-port 5007")
        return 1
    print(f"  projected  : {mb/(el/3600.0):.1f} MB/hour at this window size")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
