#!/usr/bin/env python3
"""Live background-reference presence service for the RuView observatory.

Why this exists
---------------
The shipped sensing server derives presence from motion:

    v2/crates/wifi-densepose-sensing-server/src/main.rs:2681
        presence: motion_score > 0.04

Every input to `motion_score` (main.rs:2647-2651) is a *change* feature, and the
baseline EMA (`BASELINE_EMA_ALPHA = 0.003`, main.rs:2712) absorbs a static body
within ~30 s. A person who sits still is therefore reported ABSENT. That is
exactly the failure mode that matters for a desk beside the router.

This service replaces that estimate with the background-reference detector that
was validated offline against two ground-truthed transitions in
`data/presence/` (a confirmed arrival at 17:49:55 and a confirmed departure).
It compares the *absolute* per-subcarrier amplitude profile against a calibrated
empty-room reference, so a motionless occupant stays visible indefinitely.

    score = fraction of usable subcarriers whose |x - mu| / sigma exceeds z

The fraction form is deliberate: it is bounded in [0,1], robust to a handful of
wild bins, and directly interpretable as "how much of the spectrum is disturbed".

Scoring math is kept identical to scripts/presence_detect.py so that the
offline validation carries over unchanged.

Calibration
-----------
Cross-capture calibration is NOT safe on this rig. Two 90-second captures taken
90 s apart under identical conditions were separable at 0.85 balanced accuracy,
so a reference built in one session does not transfer cleanly to another. The
background is therefore calibrated live, from this session, with the room empty,
and is persisted only so a restart does not force a re-walk-out. A persisted
reference older than --stale-hours is reported as STALE in the API.

Run
---
    python scripts/presence_service.py                # uses saved background
    python scripts/presence_service.py --calibrate 120   # leave, 120 s reference

    POST http://127.0.0.1:5008/calibrate?seconds=120  # recalibrate at runtime
    GET  http://127.0.0.1:5008/presence               # current state

Requires the UDP relay to be running with a tap:
    python scripts/udp-relay.py --listen-port 5005 --forward-port 5006 \
        --tap-host 127.0.0.1 --tap-port 5007
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import struct
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

MAGIC = 0xC5110001
IQ_START = 20
DEFAULT_BG = "data/presence/background.json"


def usable_subcarrier_indices(width: int) -> list[int]:
    """Mirror of `usable_subcarrier_indices` in main.rs.

    ESP32-S3 sends 64-bin blocks: LLTF first, then one HT-LTF block per extra 64.
    LLTF nulls bin 0 (DC) and 27..=37; each HT-LTF block nulls relative 29..=35.
    Width 192 -> 166 usable bins.
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
            elif 29 <= rel <= 35:
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


def mean_std(mat: list[list[float]]):
    n = len(mat)
    k = len(mat[0])
    mu = [0.0] * k
    for v in mat:
        for j in range(k):
            mu[j] += v[j]
    mu = [m / n for m in mu]
    if n < 2:
        return mu, [0.0] * k
    sd = [0.0] * k
    for v in mat:
        for j in range(k):
            d = v[j] - mu[j]
            sd[j] += d * d
    return mu, [math.sqrt(s / n) for s in sd]


class NodeBackground:
    """Per-subcarrier empty-room reference for one node."""

    def __init__(self, mu, sd):
        self.mu = mu
        med = sorted(mu)[len(mu) // 2] if mu else 1.0
        floor = max(0.05 * med, 1e-6)
        self.sd = [max(s, floor) for s in sd]

    def score(self, x, z):
        n = min(len(x), len(self.mu))
        if n == 0:
            return 0.0
        hits = sum(1 for j in range(n)
                   if abs(x[j] - self.mu[j]) / self.sd[j] > z)
        return hits / n

    def to_json(self):
        return {"mu": [round(v, 2) for v in self.mu],
                "sd": [round(v, 3) for v in self.sd]}

    @staticmethod
    def from_json(d):
        b = NodeBackground.__new__(NodeBackground)
        b.mu = d["mu"]
        b.sd = d["sd"]
        return b


class Detector:
    def __init__(self, args):
        self.a = args
        self.lock = threading.Lock()
        self.bg: dict[str, NodeBackground] = {}
        self.bg_time = None
        self.bg_seconds = 0.0
        self.window: dict[str, dict] = {}
        self.history = deque(maxlen=args.history)
        self.state = False
        self.raw_hist = deque(maxlen=args.smooth)
        self.calibrating = False
        self.calib_until = 0.0
        self.calib_buf = defaultdict(list)
        self.frames = 0
        self.last_frame = 0.0
        self.since = time.time()
        self.load()

    # ---------- persistence ----------
    def load(self):
        p = self.a.background
        if not os.path.exists(p):
            return
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            self.bg = {k: NodeBackground.from_json(v)
                       for k, v in d.get("nodes", {}).items()}
            self.bg_time = d.get("utc")
            self.bg_seconds = d.get("seconds", 0.0)
        except (OSError, ValueError, KeyError):
            self.bg = {}

    def save(self):
        os.makedirs(os.path.dirname(self.a.background) or ".", exist_ok=True)
        with open(self.a.background, "w", encoding="utf-8") as fh:
            json.dump({
                "utc": self.bg_time,
                "seconds": self.bg_seconds,
                "z": self.a.z,
                "nodes": {k: v.to_json() for k, v in self.bg.items()},
            }, fh)

    def stale_hours(self):
        if not self.bg_time:
            return None
        try:
            t = datetime.fromisoformat(self.bg_time.replace("Z", "+00:00"))
        except ValueError:
            return None
        return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0

    # ---------- calibration ----------
    def start_calibration(self, seconds):
        with self.lock:
            self.calibrating = True
            self.calib_until = time.time() + seconds
            self.calib_buf = defaultdict(list)
            self.bg_seconds = seconds

    def finish_calibration(self):
        bg = {}
        for node, rows in self.calib_buf.items():
            if len(rows) < 5:
                continue
            mu, sd = mean_std(rows)
            bg[node] = NodeBackground(mu, sd)
        self.calibrating = False
        if not bg:
            return False
        self.bg = bg
        self.bg_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.save()
        return True

    # ---------- scoring ----------
    def on_window(self, per_node):
        """per_node: {node: {'amp_mean': [...], 'rssi': r, 'n': k}}"""
        with self.lock:
            if self.calibrating:
                for node, d in per_node.items():
                    self.calib_buf[node].append(d["amp_mean"])
                if time.time() >= self.calib_until:
                    self.finish_calibration()
                self.window = per_node
                return

            scores = {}
            for node, d in per_node.items():
                if node in self.bg:
                    scores[node] = self.bg[node].score(d["amp_mean"], self.a.z)
            self.window = per_node

            if not scores:
                return
            # Median across nodes: one drifting node cannot force a detection.
            vals = sorted(scores.values())
            overall = vals[len(vals) // 2] if len(vals) % 2 else \
                (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
            self.raw_hist.append(overall)
            sm = sorted(self.raw_hist)
            smooth = sm[len(sm) // 2]

            new_state = smooth > self.a.thresh
            if new_state != self.state:
                self.state = new_state
                self.since = time.time()

            self.history.append({
                "t": round(time.time(), 1),
                "score": round(smooth, 4),
                "present": self.state,
            })
            self.last_scores = scores
            self.last_overall = smooth

    def snapshot(self):
        with self.lock:
            stale = self.stale_hours()
            if self.calibrating:
                status = "calibrating"
            elif not self.bg:
                status = "uncalibrated"
            elif stale is not None and stale > self.a.stale_hours:
                status = "stale"
            else:
                status = "ok"
            live = (time.time() - self.last_frame) < 5.0 if self.last_frame else False
            return {
                "status": status,
                "live": live,
                "present": bool(self.state and status in ("ok", "stale")),
                "score": round(getattr(self, "last_overall", 0.0), 4),
                "threshold": self.a.thresh,
                "per_node": {k: round(v, 4)
                             for k, v in getattr(self, "last_scores", {}).items()},
                "rssi": {k: v.get("rssi") for k, v in self.window.items()},
                "nodes": sorted(self.window.keys()),
                "held_for": round(time.time() - self.since, 1),
                "calibrated_utc": self.bg_time,
                "calibration_age_hours": round(stale, 2) if stale is not None else None,
                "calibration_seconds": self.bg_seconds,
                "calibrating_remaining": max(0.0, round(self.calib_until - time.time(), 1))
                                          if self.calibrating else 0.0,
                "frames": self.frames,
                "history": list(self.history),
                "method": "background-reference (absolute profile deviation)",
                "note": "Detects a motionless occupant; the server's motion_score "
                        "presence flag does not.",
            }


def receiver(det: Detector, args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as e:
        print(f"ERROR: cannot bind {args.host}:{args.port} - {e}")
        print("Is presence_logger.py already holding the tap?")
        os._exit(1)
    sock.settimeout(0.25)
    print(f"[tap] listening on {args.host}:{args.port}")

    buckets = defaultdict(list)
    rssis = defaultdict(list)
    usable_cache: dict[int, list[int]] = {}
    win_start = time.time()

    while True:
        try:
            buf, _ = sock.recvfrom(65535)
            p = parse_frame(buf)
            if p:
                node_id, n_sub, rssi, amps = p
                if n_sub not in usable_cache:
                    usable_cache[n_sub] = usable_subcarrier_indices(n_sub)
                idx = usable_cache[n_sub]
                if len(amps) >= n_sub:
                    buckets[str(node_id)].append([amps[i] for i in idx])
                    rssis[str(node_id)].append(rssi)
                    det.frames += 1
                    det.last_frame = time.time()
        except socket.timeout:
            pass

        now = time.time()
        if now - win_start >= args.window:
            per_node = {}
            for node, mat in buckets.items():
                if not mat:
                    continue
                mu, _ = mean_std(mat)
                per_node[node] = {
                    "amp_mean": mu,
                    "rssi": round(sum(rssis[node]) / len(rssis[node]), 1),
                    "n": len(mat),
                }
            if per_node:
                det.on_window(per_node)
            buckets.clear()
            rssis.clear()
            win_start = now


class Handler(BaseHTTPRequestHandler):
    det: Detector = None

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send({})

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/presence", "/"):
            self._send(self.det.snapshot())
        elif path == "/health":
            self._send({"ok": True})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/calibrate":
            q = parse_qs(u.query)
            secs = float(q.get("seconds", ["120"])[0])
            secs = max(10.0, min(secs, 1800.0))
            self.det.start_calibration(secs)
            self._send({"calibrating": True, "seconds": secs})
        else:
            self._send({"error": "not found"}, 404)

    def log_message(self, *a):
        pass


def bootstrap(det: "Detector", args) -> int:
    """Build the background from a recorded capture rather than from live CSI.

    Reads the same ndjson.gz that presence_logger.py writes. Tolerates a
    truncated gzip tail, which happens whenever a logger was killed mid-write.
    """
    import gzip

    rows = []
    try:
        with gzip.open(args.bootstrap_from, "rt", encoding="utf-8") as fh:
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
    if not rows:
        return 0

    def tmin(r):
        d = datetime.fromisoformat(r["utc"].replace("Z", "+00:00"))
        return d.timestamp() / 60.0

    t0 = tmin(rows[0])
    sel = [r for r in rows
           if args.bootstrap_skip_min <= (tmin(r) - t0)
           < args.bootstrap_skip_min + args.bootstrap_min]
    if not sel:
        return 0

    per_node = defaultdict(list)
    for r in sel:
        for node, d in r["nodes"].items():
            if d.get("amp_mean"):
                per_node[node].append(d["amp_mean"])

    bg = {}
    for node, mats in per_node.items():
        if len(mats) < 5:
            continue
        w = min(len(m) for m in mats)
        mu, sd = mean_std([m[:w] for m in mats])
        bg[node] = NodeBackground(mu, sd)
    if not bg:
        return 0
    det.bg = bg
    det.bg_time = sel[0]["utc"]
    det.bg_seconds = args.bootstrap_min * 60
    det.save()
    return len(sel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5007, help="relay tap port")
    ap.add_argument("--api-host", default="127.0.0.1")
    ap.add_argument("--api-port", type=int, default=5008)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--thresh", type=float, default=0.25)
    ap.add_argument("--smooth", type=int, default=5,
                    help="median filter length over 1 s windows")
    ap.add_argument("--history", type=int, default=180)
    ap.add_argument("--background", default=DEFAULT_BG)
    ap.add_argument("--stale-hours", type=float, default=12.0)
    ap.add_argument("--calibrate", type=float, default=0.0,
                    help="calibrate for N seconds on startup (room must be empty)")
    ap.add_argument("--bootstrap-from", default=None,
                    help="build the reference from a recorded presence capture "
                         "instead of live. Convenient, but a reference from an "
                         "earlier session does not transfer cleanly on this rig "
                         "(two identical captures 90s apart separated at 0.85), "
                         "so treat the result as provisional and recalibrate live.")
    ap.add_argument("--bootstrap-skip-min", type=float, default=0.0)
    ap.add_argument("--bootstrap-min", type=float, default=8.0)
    args = ap.parse_args()

    det = Detector(args)

    if args.bootstrap_from:
        n = bootstrap(det, args)
        if n:
            print(f"[calib] bootstrapped from {args.bootstrap_from} "
                  f"({n} windows, {len(det.bg)} nodes) - PROVISIONAL, recalibrate live")
        else:
            print(f"[calib] bootstrap FAILED from {args.bootstrap_from}")

    if args.calibrate > 0:
        print(f"[calib] building empty-room reference for {args.calibrate:.0f}s "
              f"- LEAVE THE ROOM NOW")
        det.start_calibration(args.calibrate)
    elif det.bg:
        age = det.stale_hours()
        print(f"[calib] loaded reference from {det.bg_time} "
              f"({age:.1f}h old, {len(det.bg)} nodes)")
    else:
        print("[calib] NO reference. POST /calibrate?seconds=120 with room empty.")

    threading.Thread(target=receiver, args=(det, args), daemon=True).start()

    Handler.det = det
    srv = ThreadingHTTPServer((args.api_host, args.api_port), Handler)
    print(f"[api ] http://{args.api_host}:{args.api_port}/presence")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
