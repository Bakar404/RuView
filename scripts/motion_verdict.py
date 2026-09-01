import json, sys, math, statistics as st
from collections import defaultdict


def corr(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    na = math.sqrt(sum(x * x for x in da))
    nb = math.sqrt(sum(x * x for x in db))
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(da, db)) / (na * nb)


def decor(path):
    prev = {}
    out = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            for nd in o.get("nodes") or []:
                nid = nd.get("node_id")
                amp = nd.get("amplitude") or []
                if not amp:
                    continue
                p = prev.get(nid)
                if p is not None and len(p) == len(amp):
                    c = corr(p, amp)
                    if c is not None:
                        # decorrelation = motion energy in the raw CSI
                        out[nid].append(1.0 - c)
                prev[nid] = amp
    return out


def d_eff(a, b):
    sa = st.pstdev(a) if len(a) > 1 else 0
    sb = st.pstdev(b) if len(b) > 1 else 0
    pool = math.sqrt((sa * sa + sb * sb) / 2) or 1e-12
    return (sum(b) / len(b) - sum(a) / len(a)) / pool


def auc(a, b):
    """P(random walk sample > random still sample). 0.5 = no signal."""
    sa = sorted(a)
    wins = ties = 0
    import bisect
    for v in b:
        lo = bisect.bisect_left(sa, v)
        hi = bisect.bisect_right(sa, v)
        wins += lo
        ties += hi - lo
    return (wins + 0.5 * ties) / (len(a) * len(b))


A = decor(sys.argv[1])
B = decor(sys.argv[2])

print("=" * 76)
print("  MOTION DETECTION FROM RAW CSI DECORRELATION  (1 - consecutive corr)")
print("  still vs walking -- this is the physics, not the server's feature")
print("=" * 76)
print(f"\n  {'node':<6}{'still mean':>12}{'walk mean':>12}{'ratio':>9}"
      f"{'Cohen d':>10}{'AUC':>8}   verdict")
allA, allB = [], []
for nid in sorted(set(A) | set(B)):
    a, b = A[nid], B[nid]
    allA += a
    allB += b
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    ar = auc(a, b)
    v = ("STRONG" if ar > 0.75 else "usable" if ar > 0.65
         else "weak" if ar > 0.58 else "NONE")
    print(f"  {nid:<6}{ma:12.5f}{mb:12.5f}{mb/ma:9.2f}x"
          f"{d_eff(a, b):10.2f}{ar:8.3f}   {v}")

ma = sum(allA) / len(allA)
mb = sum(allB) / len(allB)
ar = auc(allA, allB)
print(f"\n  {'ALL':<6}{ma:12.5f}{mb:12.5f}{mb/ma:9.2f}x"
      f"{d_eff(allA, allB):10.2f}{ar:8.3f}")

print("\n  --- tail behaviour (large decorrelation = a body moved) ---")
for th in (0.01, 0.02, 0.05, 0.10):
    pa = 100.0 * sum(1 for x in allA if x > th) / len(allA)
    pb = 100.0 * sum(1 for x in allB if x > th) / len(allB)
    lift = (pb / pa) if pa else float("inf")
    print(f"    frames with decorrelation > {th:<5}:  "
          f"still {pa:6.2f}%   walk {pb:6.2f}%   lift {lift:6.2f}x")

print("\n  --- percentiles ---")
sa = sorted(allA)
sb = sorted(allB)
for q in (50, 75, 90, 95, 99):
    print(f"    p{q:<3}  still {sa[int(len(sa)*q/100)-1]:.5f}    "
          f"walk {sb[int(len(sb)*q/100)-1]:.5f}")
