#!/usr/bin/env python3
"""Train and honestly evaluate a PRESENCE / LOCATION classifier from CSI features.

Input
-----
The gzipped NDJSON feature files written by `scripts/presence_logger.py`,
one per capture, named `<label>_<timestamp>.ndjson.gz` under `data/presence/`.

Evaluation protocol (this is the point of the script)
-----------------------------------------------------
Naive accuracy on this kind of data is almost always a lie. Three guards:

1. LEAVE-ONE-CAPTURE-OUT (LOCO) is the headline protocol when there are >=2
   captures per class. Consecutive windows inside one capture are nearly
   identical, so a random - or even temporal - split inside a single capture
   lets the model memorise that session's channel state. LOCO forces it to
   generalise to a session it has never seen, which is the only result that
   predicts real deployment behaviour.

2. MAJORITY BASELINE and SHUFFLE CONTROL are printed next to every score.
   A 3-class problem with unbalanced captures can look impressive at 70%
   while having learned nothing but the prior.

3. BALANCED ACCURACY is the headline metric, not raw accuracy.

Feature variants
----------------
Both are evaluated, because the difference between them is diagnostic:

  raw     concatenated per-node amp_mean (+ amp_std). Carries the absolute
          amplitude profile - the static-presence signature - but is exposed
          to AGC/temperature drift between sessions.
  norm    each node's amplitude profile z-scored within its own window. This
          removes absolute level (and therefore most drift) and keeps only the
          SHAPE of the profile across subcarriers.

If `raw` scores well within a session but collapses under LOCO while `norm`
holds up, the system is drift-limited and needs periodic recalibration - a
result worth knowing before deploying anything.

Usage
-----
    python scripts/presence_train.py
    python scripts/presence_train.py --features mean+std --k 7
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import Counter, defaultdict

import numpy as np


def load_capture(path):
    """Return (X, label, node_ids) for one capture file.

    Tolerates a truncated gzip tail: a logger killed mid-write leaves no
    end-of-stream marker, but every row before the cut is still valid.
    """
    rows = []
    label = None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") == "header":
                    label = r.get("label")
                    continue
                if r.get("nodes"):
                    rows.append(r)
    except (EOFError, OSError):
        pass
    if not rows:
        return None, None, None
    if label is None:
        label = rows[0].get("label")
    nodes = sorted({n for r in rows for n in r["nodes"]}, key=int)
    return rows, label, nodes


def build_matrix(rows, nodes, mode):
    """Vectorise rows -> (n_windows, dim). Rows missing a node are dropped."""
    feats = []
    for r in rows:
        if any(n not in r["nodes"] for n in nodes):
            continue
        v = []
        for n in nodes:
            d = r["nodes"][n]
            v.extend(d["amp_mean"])
            if mode == "mean+std":
                v.extend(d["amp_std"])
        feats.append(v)
    if not feats:
        return np.zeros((0, 0), dtype=np.float32)
    width = min(len(f) for f in feats)
    return np.asarray([f[:width] for f in feats], dtype=np.float32)


def normalize_per_window(X, n_nodes, mode):
    """Z-score each node's profile inside its own window (removes level/drift)."""
    Xn = X.copy()
    blocks = n_nodes * (2 if mode == "mean+std" else 1)
    if blocks == 0 or X.shape[1] % blocks != 0:
        mu = Xn.mean(axis=1, keepdims=True)
        sd = Xn.std(axis=1, keepdims=True) + 1e-9
        return (Xn - mu) / sd
    seg = X.shape[1] // blocks
    for b in range(blocks):
        s = slice(b * seg, (b + 1) * seg)
        mu = Xn[:, s].mean(axis=1, keepdims=True)
        sd = Xn[:, s].std(axis=1, keepdims=True) + 1e-9
        Xn[:, s] = (Xn[:, s] - mu) / sd
    return Xn


def knn(Xtr, ytr, Xte, k):
    a = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-9)
    b = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
    idx = np.argsort(-(b @ a.T), axis=1)[:, :k]
    out = np.empty(len(Xte), dtype=ytr.dtype)
    for i, row in enumerate(ytr[idx]):
        out[i] = Counter(row.tolist()).most_common(1)[0][0]
    return out


def ridge_classifier(Xtr, ytr, Xte, classes, lam=1.0):
    """Closed-form one-vs-rest regularised least squares."""
    Xa = np.hstack([Xtr, np.ones((len(Xtr), 1), dtype=Xtr.dtype)])
    Xb = np.hstack([Xte, np.ones((len(Xte), 1), dtype=Xte.dtype)])
    T = np.full((len(ytr), len(classes)), -1.0, dtype=np.float64)
    for i, c in enumerate(classes):
        T[ytr == c, i] = 1.0
    d = Xa.shape[1]
    A = Xa.T @ Xa + lam * np.eye(d)
    W = np.linalg.solve(A, Xa.T @ T)
    return np.asarray(classes)[np.argmax(Xb @ W, axis=1)]


def balanced_accuracy(y, p, classes):
    rec = []
    for c in classes:
        m = y == c
        if m.sum():
            rec.append(float((p[m] == c).mean()))
    return float(np.mean(rec)) if rec else 0.0


def confusion(y, p, classes):
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for a, b in zip(y, p):
        M[idx[a], idx[b]] += 1
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/presence")
    ap.add_argument("--features", choices=["mean", "mean+std"], default="mean+std")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default="occupied,absent",
                    help="comma-separated labels to include; others are ignored. "
                         "'transition' is excluded by default because the room "
                         "state during departure is ambiguous.")
    args = ap.parse_args()
    keep = {s.strip() for s in args.labels.split(",") if s.strip()}

    files = sorted(glob.glob(os.path.join(args.data, "*.ndjson.gz")))
    if not files:
        print(f"No capture files in {args.data}/.\n")
        print("Record some first (no camera or performance needed):")
        print("  python scripts/presence_logger.py --label absent --duration 600")
        print("  python scripts/presence_logger.py --label desk   --duration 1800")
        print("  python scripts/presence_logger.py --label bed")
        return 1

    print("=" * 84)
    print("PRESENCE / LOCATION CLASSIFIER".center(84))
    print("=" * 84)
    print(f"\nfeatures={args.features}   k={args.k}\n")

    caps = []
    all_nodes = None
    for f in files:
        rows, label, nodes = load_capture(f)
        if not rows:
            print(f"  skip {os.path.basename(f)} (empty)")
            continue
        if keep and label not in keep:
            print(f"  skip {os.path.basename(f)} (label '{label}' not selected)")
            continue
        all_nodes = nodes if all_nodes is None else [n for n in all_nodes if n in nodes]
        caps.append((os.path.basename(f), label, rows))

    if not caps or not all_nodes:
        print("No usable captures.")
        return 1

    print(f"  common nodes: {all_nodes}\n")
    print(f"  {'capture':<40}{'label':<12}{'windows':>9}")
    print("  " + "-" * 62)
    data = []
    for name, label, rows in caps:
        X = build_matrix(rows, all_nodes, args.features)
        if X.shape[0] == 0:
            print(f"  {name[:38]:<40}{label:<12}{'0 (skipped)':>9}")
            continue
        print(f"  {name[:38]:<40}{label:<12}{X.shape[0]:>9}")
        data.append((name, label, X))

    if not data:
        return 1
    dim = min(X.shape[1] for _, _, X in data)
    data = [(n, l, X[:, :dim]) for n, l, X in data]
    labels = sorted({l for _, l, _ in data})
    print(f"\n  feature dim: {dim}   classes: {labels}")

    per_label = Counter(l for _, l, _ in data)
    if len(labels) < 2:
        print(f"\n  ONLY ONE CLASS ({labels[0]}). A classifier cannot be trained or")
        print("  evaluated. Record at least one capture of a different class.")
        return 1

    loco_ok = all(per_label[l] >= 2 for l in labels)
    print(f"  captures per class: {dict(per_label)}")
    if not loco_ok:
        print("\n  !! Fewer than 2 captures for some class - leave-one-capture-out is")
        print("     not possible. Falling back to a temporal 70/30 split WITHIN each")
        print("     capture, which is OPTIMISTIC: it cannot detect session overfitting.")
        print("     Record a second session per class for a trustworthy number.")

    rng = np.random.default_rng(args.seed)

    for variant in ("raw", "norm"):
        print("\n" + "=" * 84)
        print(f"  VARIANT: {variant}")
        print("=" * 84)

        prepped = []
        for n, l, X in data:
            Xv = normalize_per_window(X, len(all_nodes), args.features) if variant == "norm" else X
            prepped.append((n, l, Xv))

        results = defaultdict(list)
        folds = []

        if loco_ok:
            # Hold out one capture from EACH class simultaneously, so every test
            # set contains both classes. Holding out a single capture would make
            # the test set single-class, where balanced accuracy degenerates to
            # that class's recall and a majority predictor scores a perfect 1.0 -
            # an artifact, not a result.
            by_class = defaultdict(list)
            for i, (n, l, X) in enumerate(prepped):
                by_class[l].append(i)
            import itertools
            for combo in itertools.product(*[by_class[l] for l in labels]):
                held = set(combo)
                tr = [(l, X) for j, (n, l, X) in enumerate(prepped) if j not in held]
                if len({l for l, _ in tr}) < 2:
                    continue
                # Balance the training set: with ~8x more 'absent' than
                # 'occupied' windows, an unbalanced fit just learns the prior.
                per = defaultdict(list)
                for l, X in tr:
                    per[l].append(X)
                stacked = {l: np.vstack(v) for l, v in per.items()}
                m = min(len(v) for v in stacked.values())
                Xtr = np.vstack([v[np.linspace(0, len(v) - 1, m).astype(int)]
                                 for v in stacked.values()])
                ytr = np.concatenate([np.full(m, l) for l in stacked])
                Xte = np.vstack([prepped[j][2] for j in held])
                yte = np.concatenate([np.full(len(prepped[j][2]), prepped[j][1])
                                      for j in held])
                nm = " + ".join(sorted(prepped[j][0][:22] for j in held))
                folds.append((nm, Xtr, ytr, Xte, yte))
        else:
            for name, lab, X in prepped:
                cut = int(0.7 * len(X))
                if cut < args.k + 1 or len(X) - cut < 5:
                    continue
                folds.append((name, X[:cut], np.full(cut, lab),
                              X[cut:], np.full(len(X) - cut, lab)))
            if folds:
                Xtr = np.vstack([f[1] for f in folds])
                ytr = np.concatenate([f[2] for f in folds])
                folds = [("pooled temporal", Xtr, ytr,
                          np.vstack([f[3] for f in folds]),
                          np.concatenate([f[4] for f in folds]))]

        if not folds:
            print("  not enough data for any fold")
            continue

        for name, Xtr, ytr, Xte, yte in folds:
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            A, B = (Xtr - mu) / sd, (Xte - mu) / sd

            pk = knn(A, ytr, B, args.k)
            pr = ridge_classifier(A, ytr, B, labels)
            ysh = ytr.copy()
            rng.shuffle(ysh)
            ps = knn(A, ysh, B, args.k)
            maj = Counter(ytr.tolist()).most_common(1)[0][0]
            pm = np.full(len(yte), maj)

            bk = balanced_accuracy(yte, pk, labels)
            br = balanced_accuracy(yte, pr, labels)
            bs = balanced_accuracy(yte, ps, labels)
            bm = balanced_accuracy(yte, pm, labels)
            results["knn"].append(bk)
            results["ridge"].append(br)
            results["shuffle"].append(bs)
            results["major"].append(bm)
            best_p = pr if br >= bk else pk
            rec = {c: (float((best_p[yte == c] == c).mean()) if (yte == c).any() else float("nan"))
                   for c in labels}
            counts = {c: int((yte == c).sum()) for c in labels}
            print(f"\n  held out: {name[:60]}")
            print(f"     test n={len(yte)}  {counts}")
            print(f"     kNN={bk:.3f}  ridge={br:.3f}  shuffle={bs:.3f}  majority={bm:.3f}")
            print("     recall  " + "  ".join(f"{c}={rec[c]:.3f}" for c in labels))

        print("\n  " + "-" * 78)
        mk = float(np.mean(results["knn"]))
        mr = float(np.mean(results["ridge"]))
        ms = float(np.mean(results["shuffle"]))
        mm = float(np.mean(results["major"]))
        best = max(mk, mr)
        ctrl = max(ms, mm)
        delta = best - ctrl
        print(f"  MEAN  kNN={mk:.3f}  ridge={mr:.3f}  |  shuffle={ms:.3f}  majority={mm:.3f}")
        print(f"  DELTA over best control: {delta:+.3f}")
        if delta > 0.15:
            print("  VERDICT: STRONG SIGNAL - this is a real, usable classifier.")
        elif delta > 0.06:
            print("  VERDICT: real but modest signal - more data or better features needed.")
        else:
            print("  VERDICT: NO USABLE SIGNAL beyond the prior.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
