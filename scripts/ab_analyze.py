import json, sys, math, statistics as st
from collections import defaultdict

def load(path, label):
    per = defaultdict(lambda: {"rssi": [], "mbp": [], "var": [], "ts": [], "off": [],
                               "amp_len": [], "sc": [], "seen": [], "stale": 0})
    top = {"mbp": [], "var": [], "persons": [], "conf": [], "ts": []}
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            n += 1
            f = o.get("features") or {}
            top["mbp"].append(f.get("motion_band_power", 0.0) or 0.0)
            top["var"].append(f.get("variance", 0.0) or 0.0)
            top["persons"].append(o.get("estimated_persons", 0) or 0)
            top["ts"].append(o.get("timestamp", 0.0) or 0.0)
            c = o.get("classification") or {}
            top["conf"].append(c.get("confidence", 0.0) or 0.0)

            for nd in o.get("nodes") or []:
                d = per[nd.get("node_id")]
                d["amp_len"].append(len(nd.get("amplitude") or []))
                d["sc"].append(nd.get("subcarrier_count", 0) or 0)
                sy = nd.get("sync") or {}
                if sy.get("offset_us") is not None:
                    d["off"].append(sy["offset_us"])
            for nf in o.get("node_features") or []:
                d = per[nf.get("node_id")]
                d["rssi"].append(nf.get("rssi_dbm", 0) or 0)
                ff = nf.get("features") or {}
                d["mbp"].append(ff.get("motion_band_power", 0.0) or 0.0)
                d["var"].append(ff.get("variance", 0.0) or 0.0)
                d["seen"].append(nf.get("last_seen_ms", 0) or 0)
                if nf.get("stale"):
                    d["stale"] += 1
    return {"label": label, "n": n, "top": top, "per": per}


def ms(v):
    if not v:
        return (0, 0, 0, 0)
    return (min(v), max(v), sum(v) / len(v),
            st.pstdev(v) if len(v) > 1 else 0.0)


def report(a):
    print("=" * 74)
    print(f"  {a['label']}   records={a['n']}")
    print("=" * 74)
    ts = sorted(t for t in a["top"]["ts"] if t)
    if len(ts) > 2:
        dur = ts[-1] - ts[0]
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        gaps_ms = [g * 1000 for g in gaps]
        print(f"  duration        : {dur:.1f} s")
        print(f"  update rate     : {len(ts)/dur:.1f} Hz")
        print(f"  inter-frame ms  : mean {sum(gaps_ms)/len(gaps_ms):.1f}  "
              f"med {st.median(gaps_ms):.1f}  max {max(gaps_ms):.1f}  "
              f"jitter(sd) {st.pstdev(gaps_ms):.1f}")
    for k, unit in (("mbp", "motion_band_power"), ("var", "variance"),
                    ("conf", "confidence")):
        lo, hi, mean, sd = ms(a["top"][k])
        print(f"  {unit:<17}: mean {mean:10.3f}  sd {sd:9.3f}  "
              f"min {lo:9.3f}  max {hi:10.3f}")
    p = a["top"]["persons"]
    if p:
        print(f"  estimated_persons: mean {sum(p)/len(p):.2f}  max {max(p)}  "
              f"distribution {dict(sorted({x: p.count(x) for x in set(p)}.items()))}")
    print("  --- per node ---")
    hdr = ("   id  rssi(mean/span)   motion_band(mean/sd)   amp_len  subc  "
           "sync_off_us(mean/sd)  seen_ms(max)  stale")
    print(hdr)
    for nid in sorted(a["per"]):
        d = a["per"][nid]
        rl, rh, rm, _ = ms(d["rssi"])
        _, _, mm, msd = ms(d["mbp"])
        ol, oh, om, osd = ms(d["off"])
        al = set(d["amp_len"])
        sc = set(d["sc"])
        print(f"   {nid:>2}  {rm:7.1f}/{rh-rl:<6.1f}  {mm:12.3f}/{msd:<8.3f} "
              f"{str(sorted(al)):>8} {str(sorted(sc)):>6}  "
              f"{om:9.1f}/{osd:<8.1f}  {max(d['seen']) if d['seen'] else 0:>8}  "
              f"{d['stale']}/{len(d['rssi'])}")
    return a


def compare(A, B):
    print()
    print("#" * 74)
    print("  MOTION SEPARATION VERDICT   (walking vs still)")
    print("#" * 74)
    for key, name in (("mbp", "motion_band_power"), ("var", "variance"),
                      ("conf", "confidence")):
        av = A["top"][key]
        bv = B["top"][key]
        am = sum(av) / len(av) if av else 0
        bm = sum(bv) / len(bv) if bv else 0
        asd = st.pdev if False else (st.pstdev(av) if len(av) > 1 else 0)
        ratio = (bm / am) if am else float("inf")
        # Cohen's d - effect size, the honest measure of separability
        bsd = st.pstdev(bv) if len(bv) > 1 else 0
        pool = math.sqrt((asd ** 2 + bsd ** 2) / 2) or 1e-12
        d = (bm - am) / pool
        print(f"  {name:<18} still={am:10.3f}  walk={bm:10.3f}  "
              f"ratio={ratio:6.2f}x   Cohen_d={d:7.2f}")
    print()
    for nid in sorted(set(A["per"]) | set(B["per"])):
        av = A["per"][nid]["mbp"]
        bv = B["per"][nid]["mbp"]
        am = sum(av) / len(av) if av else 0
        bm = sum(bv) / len(bv) if bv else 0
        asd = st.pstdev(av) if len(av) > 1 else 0
        bsd = st.pstdev(bv) if len(bv) > 1 else 0
        pool = math.sqrt((asd ** 2 + bsd ** 2) / 2) or 1e-12
        print(f"  node {nid}: motion_band still={am:9.3f} walk={bm:9.3f}  "
              f"ratio={(bm/am if am else 0):6.2f}x  Cohen_d={(bm-am)/pool:6.2f}")
    pa = A["top"]["persons"]
    pb = B["top"]["persons"]
    print()
    print(f"  persons  still: mean {sum(pa)/len(pa):.2f}   "
          f"walk: mean {sum(pb)/len(pb):.2f}")


if __name__ == "__main__":
    A = report(load(sys.argv[1], "CAPTURE A  (still / empty)"))
    print()
    B = report(load(sys.argv[2], "CAPTURE B  (walking)"))
    compare(A, B)
