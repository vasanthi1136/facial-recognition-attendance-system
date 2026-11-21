#!/usr/bin/env python3
"""
Safeguard study: how do the distance threshold (T) and the consecutive-frame rule (N)
trade off correct marks against false marks?

Method (simulation on a face dataset):
  * ~20% of people are held out as "strangers" (never enrolled); the rest are enrolled.
  * LBPH is trained on the enrolled people's training images.
  * A "pass" = a person standing in front of the camera for F frames. Frames are drawn
    (with replacement) from that person's images; each frame is lost with probability
    p_miss to mimic Haar detection drop-outs.
  * The exact streak rule of main.py is applied for every (T, N) on the SAME passes.
  * Outcomes per pass: genuine -> correct mark / wrong-ID mark / missed;
                       stranger -> false mark / correctly ignored.
  * Repeated with different stranger partitions; mean +- std reported.

Usage:
  python research.py --data-dir C:\\data\\kaggle_faces [--detect]
  python research.py                      # uses your own dataset/ (needs >= 3 people)
"""
import argparse
import os

import numpy as np
import pandas as pd

import evaluate as E
import main as M

THRESHOLDS = list(range(30, 105, 5))
N_VALUES = [1, 2, 3, 5, 8]


def run_streak(labels, dists, present, T, N):
    """Vectorised version of the streak rule in AttendanceEngine.process()."""
    P, F = labels.shape
    cur = np.full(P, -1)
    streak = np.zeros(P, int)
    mark = np.full(P, -1)
    for f in range(F):
        acc = present[:, f] & (dists[:, f] <= T)
        lab = labels[:, f]
        streak = np.where(acc & (lab == cur), streak + 1, np.where(acc, 1, 0))
        cur = np.where(acc, lab, -1)
        mark = np.where((streak >= N) & (mark < 0), cur, mark)   # first mark only
    return mark


def make_passes(pool, n_pass, frames, p_miss, rng):
    """pool: {person: (labels, dists)} -> arrays [n_pass, frames]."""
    people = list(pool)
    who = rng.choice(people, size=n_pass)
    L = np.empty((n_pass, frames), int)
    D = np.empty((n_pass, frames))
    for i, k in enumerate(who):
        lab, dist = pool[k]
        idx = rng.integers(0, len(lab), size=frames)
        L[i], D[i] = lab[idx], dist[idx]
    present = rng.random((n_pass, frames)) >= p_miss
    return who, L, D, present


def one_repeat(data, args, rng):
    keys = list(data)
    n_out = max(1, int(round(len(keys) * 0.2)))
    strangers = set(rng.choice(keys, size=n_out, replace=False).tolist())
    enrolled = [k for k in keys if k not in strangers]
    splits = {k: E.split(data[k][1], args.test_ratio, args.split, rng) for k in enrolled}
    rec = E.train_lbph([im for k in enrolled for im in splits[k][0]],
                       [k for k in enrolled for _ in splits[k][0]])
    pool_g, pool_s = {}, {}
    for k in enrolled:
        p = [rec.predict(im) for im in splits[k][1]]
        pool_g[k] = (np.array([a for a, _ in p]), np.array([b for _, b in p]))
    for k in strangers:
        p = [rec.predict(im) for im in data[k][1]]
        pool_s[k] = (np.array([a for a, _ in p]), np.array([b for _, b in p]))
    who_g, Lg, Dg, Pg = make_passes(pool_g, args.passes, args.frames, args.p_miss, rng)
    _, Ls, Ds, Ps = make_passes(pool_s, args.passes, args.frames, args.p_miss, rng)
    rows = []
    for T in THRESHOLDS:
        for N in N_VALUES:
            mg = run_streak(Lg, Dg, Pg, T, N)
            ms = run_streak(Ls, Ds, Ps, T, N)
            rows.append({"T": T, "N": N,
                         "Correct mark %": (mg == who_g).mean() * 100,
                         "Wrong-ID mark %": ((mg >= 0) & (mg != who_g)).mean() * 100,
                         "Missed %": (mg < 0).mean() * 100,
                         "Stranger false-mark %": (ms >= 0).mean() * 100})
    return pd.DataFrame(rows), len(enrolled), len(strangers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--min-images", type=int, default=5)
    ap.add_argument("--split", choices=["block", "random"], default="random")
    ap.add_argument("--test-ratio", type=float, default=0.3)
    ap.add_argument("--frames", type=int, default=30, help="frames per pass")
    ap.add_argument("--passes", type=int, default=1000, help="passes per group per repeat")
    ap.add_argument("--p-miss", type=float, default=0.1, help="per-frame detection drop-out")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tag = "_" + os.path.basename(os.path.normpath(args.data_dir)) if args.data_dir else ""

    data, skipped = E.load_dataset(args.data_dir, args.detect,
                                   args.min_images if args.data_dir else 1)
    if len(data) < 3:
        raise SystemExit("Need at least 3 people (enrolled + stranger). Found %d." % len(data))
    print("=" * 74)
    print("SAFEGUARD STUDY | people: %d | images: %d | frames/pass: %d | passes/group: %d | "
          "p_miss: %.2f | repeats: %d" % (len(data), sum(len(v[1]) for v in data.values()),
                                          args.frames, args.passes, args.p_miss, args.repeats))
    reps = []
    for r in range(args.repeats):
        df, ne, ns = one_repeat(data, args, rng)
        reps.append(df)
        print("  repeat %d/%d done (enrolled=%d, strangers=%d)" % (r + 1, args.repeats, ne, ns))
    allr = pd.concat(reps)
    mean = allr.groupby(["T", "N"]).mean().round(2)
    std = allr.groupby(["T", "N"]).std().round(2).add_suffix(" (std)")
    grid = mean.join(std).reset_index()

    cols = ["Correct mark %", "Wrong-ID mark %", "Missed %", "Stranger false-mark %"]
    print("\nEFFECT OF N (consecutive frames) at the app's default threshold T=%d"
          % int(M.UNKNOWN_THRESHOLD))
    t0 = int(M.UNKNOWN_THRESHOLD)
    print(grid[grid["T"] == t0][["N"] + cols].to_string(index=False))
    print("\nEFFECT OF T (distance threshold) at N=%d" % M.CONSECUTIVE_FRAMES)
    print(grid[grid["N"] == M.CONSECUTIVE_FRAMES][["T"] + cols].to_string(index=False))

    print("\nBEST OPERATING POINTS (max correct marks s.t. stranger false-mark limit)")
    best = []
    for lim in (5.0, 1.0, 0.1):
        ok = grid[grid["Stranger false-mark %"] <= lim]
        if len(ok):
            b = ok.sort_values(["Correct mark %", "Wrong-ID mark %"], ascending=[False, True]).iloc[0]
            best.append({"Limit %": lim, "T": int(b["T"]), "N": int(b["N"]),
                         **{c: b[c] for c in cols}})
        else:
            best.append({"Limit %": lim, "T": None, "N": None})
    best = pd.DataFrame(best)
    print(best.to_string(index=False))

    # duplicate-write study (analytic): one student stays 60 s in front of the camera
    dwell = 60
    dup = pd.DataFrame([{"Cooldown (s)": c,
                         "DB write attempts / 60 s": int(dwell // max(c, M.CONSECUTIVE_FRAMES / args.fps)),
                         "Records stored": 1} for c in (0, 2, 5, 10)])
    print("\nDUPLICATE-WRITE ATTEMPTS (student present 60 s, %.0f fps, N=%d; UNIQUE(student,date) "
          "keeps 1 record)" % (args.fps, M.CONSECUTIVE_FRAMES))
    print(dup.to_string(index=False))

    out = os.path.join(M.ATTENDANCE_DIR, "safeguard_study%s.xlsx" % tag)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        grid.to_excel(w, sheet_name="Grid", index=False)
        best.to_excel(w, sheet_name="Best points", index=False)
        dup.to_excel(w, sheet_name="Cooldown", index=False)
        for c in cols:
            mean.reset_index().pivot(index="T", columns="N", values=c).to_excel(w, sheet_name=c[:28].replace("%", "pct"))
    print("\nSaved:", out)

    try:   # optional figures for the paper
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for col, fname in (("Stranger false-mark %", "fig_false_mark_vs_T%s.png" % tag),
                           ("Correct mark %", "fig_correct_mark_vs_T%s.png" % tag)):
            plt.figure(figsize=(3.5, 2.6))
            for N in N_VALUES:
                s = grid[grid.N == N]
                plt.plot(s["T"], s[col], marker="o", ms=2.5, label="N=%d" % N)
            plt.xlabel("LBPH distance threshold T")
            plt.ylabel(col)
            plt.grid(alpha=.3)
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig(os.path.join(M.ATTENDANCE_DIR, fname), dpi=300)
            plt.close()
        print("Saved figures (300 dpi) in", M.ATTENDANCE_DIR)
    except ImportError:
        print("Tip: pip install matplotlib   (to also get paper-ready figures)")
    print("=" * 74)


if __name__ == "__main__":
    M.ensure_dirs()
    M.ensure_cascade()
    main()