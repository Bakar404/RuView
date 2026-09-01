import json, sys, math, statistics as st
from collections import defaultdict


def corr(a, b):
    n = len(a)
    if n < 2 or len(b) != n:
        return None
    ma = sum(a) / n
    mb = sum(b) / n
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    na = math.sqrt(sum(x * x for x in da))
    nb = math.sqrt(sum(x * x for x in db))
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(da, db)) / (na * nb)


def analyze(path, label):
    prev = {}
    stats = defaultdict(lambda: {"corr": [], "with_amp": 0, "no_amp": 0,
                                 "lens": defaultdict(int), "sc": defaultdict(int),
                                 "amp_mean": [], "amp_sd": [], "zeros": 0,
                                 "ts": []})
    total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            total += 1
            ts = o.get("timestamp", 0.0)
            for nd in o.get("nodes") or []:
                nid = nd.get("node_id")
                s = stats[nid]
                amp = nd.get("amplitude") or []
                s["lens"][len(amp)] += 1
                s["sc"][nd.get("subcarrier_count", 0)] += 1
                if not amp:
                    s["no_amp"] += 1
                    continue
                s["with_amp"] += 1
                s["ts"].append(ts)
                m = sum(amp) / len(amp)
                s["amp_mean"].append(m)
                s["amp_sd"].append(st.pstdev(amp) if len(amp) > 1 else 0.0)
                if all(v == 0 for v in amp):
                    s["zeros"] += 1
                p = prev.get(nid)
                if p is not None and len(p) == len(amp):
                    c = corr(p, amp)
                    if c is not None:
                        s["corr"].append(c)
                prev[nid] = amp

    print("=" * 78)
    print(f"  {label}    total records = {total}")
    print("=" * 78)
    for nid in sorted(stats):
        s = stats[nid]
        tot = s["with_amp"] + s["no_amp"]
        pct = 100.0 * s["with_amp"] / tot if tot else 0
        print(f"\n  --- NODE {nid} ---")
        print(f"    frames with CSI amplitude : {s['with_amp']}/{tot}  ({pct:.1f}%)")
        print(f"    amplitude lengths seen    : "
              f"{dict(sorted(s['lens'].items()))}")
        print(f"    subcarrier_count values   : "
              f"{dict(sorted(s['sc'].items()))}")
        print(f"    all-zero amplitude frames : {s['zeros']}")
        if s["amp_mean"]:
            am = s["amp_mean"]
            asd = s["amp_sd"]
            print(f"    amplitude mean            : {sum(am)/len(am):.3f}  "
                  f"(min {min(am):.3f} max {max(am):.3f})")
            print(f"    within-frame sd           : {sum(asd)/len(asd):.3f}")
        if s["ts"] and len(s["ts"]) > 2:
            t = sorted(s["ts"])
            dur = t[-1] - t[0]
            if dur > 0:
                print(f"    CSI frame rate            : "
                      f"{len(t)/dur:.2f} Hz over {dur:.1f}s")
        c = s["corr"]
        if c:
            c_sorted = sorted(c)
            print(f"    >> CONSECUTIVE-FRAME CORRELATION  n={len(c)}")
            print(f"       mean   {sum(c)/len(c):+.4f}")
            print(f"       median {c_sorted[len(c)//2]:+.4f}")
            print(f"       p10    {c_sorted[int(len(c)*0.10)]:+.4f}   "
                  f"p90 {c_sorted[int(len(c)*0.90)]:+.4f}")
            print(f"       min    {min(c):+.4f}   max {max(c):+.4f}")
            good = sum(1 for x in c if x > 0.9)
            print(f"       frames > 0.90 : {good}/{len(c)}  "
                  f"({100.0*good/len(c):.1f}%)")
        else:
            print("    >> NO CORRELATION COMPUTABLE (no consecutive CSI pairs)")
    return stats


if __name__ == "__main__":
    for p, lab in ((sys.argv[1], "CAPTURE A (still)"),
                   (sys.argv[2], "CAPTURE B (walking)")):
        analyze(p, lab)
        print()
