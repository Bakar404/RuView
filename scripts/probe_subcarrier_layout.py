#!/usr/bin/env python3
"""
Probe the true subcarrier layout of live ESP32 CSI frames.

The sensing server truncates every node's amplitude/phase vector to the first
56 bins before emitting it (main.rs .take(56)). Live frames actually report
n_subcarriers = 192 (ESP32 HT40: LLTF 64 + HT-LTF 128), so that truncation
keeps all of the DC/guard nulls while discarding real subcarriers.

This script taps the raw UDP stream *before* truncation and reports, per node:
  - the true bin count
  - which bins are always zero (null / guard)
  - which bins carry usable energy
  - how much variance the null bins inject

Run the relay with a tap first:
    python scripts/udp-relay.py --listen-port 5005 --forward-port 5006 \
        --tap-host 127.0.0.1 --tap-port 5007

Then:
    python scripts/probe_subcarrier_layout.py --port 5007 --frames 300
"""

import argparse
import socket
import struct
import sys
from collections import defaultdict

MAGIC = 0xC5110001
IQ_START = 20


def parse_frame(buf):
    if len(buf) < IQ_START:
        return None
    magic, = struct.unpack_from("<I", buf, 0)
    if magic != MAGIC:
        return None
    node_id = buf[4]
    n_ant = buf[5]
    n_sub, = struct.unpack_from("<H", buf, 6)
    seq, = struct.unpack_from("<I", buf, 12)
    rssi = struct.unpack_from("<b", buf, 16)[0]

    n_pairs = n_ant * n_sub
    if len(buf) < IQ_START + n_pairs * 2:
        return None

    amps = []
    for k in range(n_pairs):
        i_val = struct.unpack_from("<b", buf, IQ_START + k * 2)[0]
        q_val = struct.unpack_from("<b", buf, IQ_START + k * 2 + 1)[0]
        amps.append((i_val * i_val + q_val * q_val) ** 0.5)
    return node_id, n_ant, n_sub, seq, rssi, amps


def summarize_runs(indices):
    """Collapse a sorted index list into human-readable ranges."""
    if not indices:
        return "(none)"
    runs = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def pvar(xs):
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5007)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--zero-eps", type=float, default=1e-9,
                    help="amplitude at or below this counts as null")
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args.host, args.port))
    s.settimeout(args.timeout)

    per_node = defaultdict(list)
    meta = {}
    got = 0
    print(f"listening on {args.host}:{args.port} for {args.frames} frames...")
    try:
        while got < args.frames:
            data, _ = s.recvfrom(65535)
            f = parse_frame(data)
            if not f:
                continue
            node_id, n_ant, n_sub, seq, rssi, amps = f
            per_node[node_id].append(amps)
            meta[node_id] = (n_ant, n_sub, rssi, len(data))
            got += 1
    except socket.timeout:
        print(f"timed out after {got} frames", file=sys.stderr)
    finally:
        s.close()

    if not per_node:
        print("no ESP32 frames captured -- is the relay tap running?",
              file=sys.stderr)
        return 1

    union_null = None
    for node_id in sorted(per_node):
        frames = per_node[node_id]
        n_ant, n_sub, rssi, pktlen = meta[node_id]
        width = len(frames[0])
        print()
        print("=" * 68)
        print(f"NODE {node_id}   frames={len(frames)}  packet={pktlen}B  "
              f"n_antennas={n_ant}  n_subcarriers={n_sub}  width={width}  "
              f"rssi={rssi}")
        print("=" * 68)

        always_zero = []
        ever_zero = []
        for i in range(width):
            col = [fr[i] for fr in frames if i < len(fr)]
            z = sum(1 for v in col if v <= args.zero_eps)
            if z == len(col):
                always_zero.append(i)
            elif z > 0:
                ever_zero.append(i)

        usable = [i for i in range(width) if i not in set(always_zero)]
        print(f"  always-zero bins ({len(always_zero)}): "
              f"{summarize_runs(always_zero)}")
        print(f"  usable bins      ({len(usable)}): {summarize_runs(usable)}")
        if ever_zero:
            print(f"  intermittently zero ({len(ever_zero)}): "
                  f"{summarize_runs(ever_zero)}")

        v_all = sum(pvar(fr) for fr in frames) / len(frames)
        v_use = sum(pvar([fr[i] for i in usable]) for fr in frames) / len(frames)
        print(f"  mean per-frame variance  all bins : {v_all:8.2f}")
        print(f"  mean per-frame variance  usable   : {v_use:8.2f}")
        if v_use > 0:
            print(f"  null-bin inflation factor         : {v_all / v_use:8.2f}x")

        kept = [i for i in usable if i < 56]
        lost = [i for i in usable if i >= 56]
        print(f"  --- effect of the current .take(56) ---")
        print(f"  usable bins KEPT     : {len(kept)}")
        print(f"  usable bins DISCARDED: {len(lost)}  ({summarize_runs(lost)})")
        print(f"  dead bins wasted     : "
              f"{len([i for i in always_zero if i < 56])} of 56 slots")
        if usable:
            print(f"  signal retained      : "
                  f"{100.0 * len(kept) / len(usable):.1f}%")

        s_null = set(always_zero)
        union_null = s_null if union_null is None else (union_null & s_null)

    if len(per_node) > 1 and union_null is not None:
        print()
        print(f"bins null on EVERY node ({len(union_null)}): "
              f"{summarize_runs(sorted(union_null))}")
        print("^ this is the set safe to drop globally")
    return 0


if __name__ == "__main__":
    sys.exit(main())
