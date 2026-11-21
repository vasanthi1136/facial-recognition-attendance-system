#!/usr/bin/env python3
"""
Baseline comparison with repeated random splits (mean +- std).

Compares LBPH (the app's method) with Eigenfaces and Fisherfaces from OpenCV.
Each repeat: ~20% of people are held out as strangers; models are trained on the
remaining ("enrolled") people's training images and evaluated on
  * closed-set accuracy + macro-F1 (enrolled people's test images),
  * open-set EER and FRR@FAR<=1% (enrolled test images vs stranger images).
EER is threshold-free, so models with different distance scales are comparable.

Usage:
  python baselines.py --data-dir C:\\data\\orl
  python baselines.py --data-dir C:\\data\\lfw --max-images 50    # cap per person (reduces imbalance)
  python baselines.py --repeats 10 --seed 1
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd

import evaluate as E
import main as M

SMALL = (64, 64)   # Eigen/Fisherfaces work on small images (PCA/LDA memory)

MODELS = {
    "LBPH (200x200)": (lambda: cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8), None),
    "Eigenfaces (64x64)": (cv2.face.EigenFaceRecognizer_create, SMALL),
    "Fisherfaces (64x64)": (cv2.face.FisherFaceRecognizer_create, SMALL),
}


def prep(imgs, size):
    return imgs if size is None else [cv2.resize(i, size) for i in imgs]


def macro_f1(true, pred, labels):
    scores = []
    for k in labels:
        tp = ((true == k) & (pred == k)).sum()
        fp = ((true != k) & (pred == k)).sum()
        fn = ((true == k) & (pred != k)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(scores))


def eer_stats(gen_dist, gen_ok, imp_dist):
    """EER and FRR at FAR<=1%. Genuine is 'accepted' only if dist<=t AND label correct."""
    cand = np.unique(np.concatenate([gen_dist, imp_dist]))
    far = (imp_dist[:, None] <= cand[None, :]).mean(0)
    frr = 1 - ((gen_dist[:, None] <= cand[None, :]) & gen_ok[:, None]).mean(0)
    i = int(np.argmin(np.abs(far - frr)))
    ok = far <= 0.01
    return (far[i] + frr[i]) / 2 * 100, (frr[ok].min() * 100 if ok.any() else 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--min-images", type=int, default=5)
    ap.add_argument("--max-images", type=int, default=0, help="cap images per person (0 = no cap)")
    ap.add_argument("--test-ratio", type=float, default=0.3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tag = "_" + os.path.basename(os.path.normpath(args.data_dir)) if args.data_dir else ""

    data, skipped = E.load_dataset(args.data_dir, args.detect,
                                   args.min_images if args.data_dir else 1)
    if args.max_images:
        data = {k: (n, im[:args.max_images]) for k, (n, im) in data.items()}
    if len(data) < 3:
        raise SystemExit("Need at least 3 people. Found %d." % len(data))
    print("=" * 78)
    print("BASELINES | people: %d | images: %d | repeats: %d | max-images/person: %s | skipped: %d"
          % (len(data), sum(len(v[1]) for v in data.values()), args.repeats,
             args.max_images or "none", skipped))

    rows = []
    keys = list(data)
    for r in range(args.repeats):
        n_out = max(1, int(round(len(keys) * 0.2)))
        strangers = set(rng.choice(keys, size=n_out, replace=False).tolist())
        enrolled = [k for k in keys if k not in strangers]
        splits = {k: E.split(data[k][1], args.test_ratio, "random", rng) for k in enrolled}
        tr = [im for k in enrolled for im in splits[k][0]]
        trl = np.array([k for k in enrolled for _ in splits[k][0]], dtype=np.int32)
        te = [im for k in enrolled for im in splits[k][1]]
        tel = np.array([k for k in enrolled for _ in splits[k][1]])
        imp = [im for k in strangers for im in data[k][1]]
        for name, (make, size) in MODELS.items():
            try:
                rec = make()
                t0 = time.perf_counter()
                rec.train(prep(tr, size), trl)
                t_train = time.perf_counter() - t0
                t0 = time.perf_counter()
                g = [rec.predict(i) for i in prep(te, size)]
                t_pred = (time.perf_counter() - t0) / len(te) * 1000
                im_d = np.array([rec.predict(i)[1] for i in prep(imp, size)])
            except cv2.error as exc:
                print("  %s failed: %s" % (name, str(exc).splitlines()[0]))
                continue
            pred = np.array([a for a, _ in g])
            gd = np.array([b for _, b in g])
            eer, frr1 = eer_stats(gd, pred == tel, im_d)
            rows.append({"Model": name, "Repeat": r + 1,
                         "Accuracy %": (pred == tel).mean() * 100,
                         "Macro-F1": macro_f1(tel, pred, enrolled),
                         "EER %": eer, "FRR@FAR<=1% %": frr1,
                         "Train s": t_train, "Predict ms/face": t_pred})
        print("  repeat %d/%d done (enrolled=%d, strangers=%d, test imgs=%d, stranger imgs=%d)"
              % (r + 1, args.repeats, len(enrolled), len(strangers), len(te), len(imp)))

    raw = pd.DataFrame(rows)
    metrics = ["Accuracy %", "Macro-F1", "EER %", "FRR@FAR<=1% %", "Train s", "Predict ms/face"]
    g = raw.groupby("Model", sort=False)[metrics]
    mean, std = g.mean(), g.std()
    summary = pd.DataFrame({m: [("%.2f +- %.2f" if m != "Macro-F1" else "%.3f +- %.3f") % (mean.loc[k, m], std.loc[k, m])
                                for k in mean.index] for m in metrics}, index=mean.index)
    print("\nRESULTS (mean +- std over %d repeats)" % args.repeats)
    print(summary.to_string())
    print("\nLower is better for EER, FRR@FAR<=1%, Train s, Predict ms/face.")

    out = os.path.join(M.ATTENDANCE_DIR, "baseline_comparison%s.xlsx" % tag)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary")
        raw.to_excel(w, sheet_name="Per-repeat", index=False)
    print("Saved:", out)
    print("=" * 78)


if __name__ == "__main__":
    M.ensure_dirs()
    M.ensure_cascade()
    main()