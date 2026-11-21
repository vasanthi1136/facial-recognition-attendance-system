#!/usr/bin/env python3
"""
Offline evaluation of the LBPH model on the captured dataset (for the paper).

Usage:  python evaluate.py                # your own dataset/, block split, 30% test
        python evaluate.py --data-dir C:\\data\\kaggle_faces --split random   # benchmark
        (add --detect if the images are full photos rather than cropped faces)
        python evaluate.py --split random --test-ratio 0.3 --seed 1

Reports: closed-set accuracy, per-student precision/recall/F1, confusion matrix,
threshold sweep (correct accept / false reject / wrong-ID / impostor FAR),
and recognition speed. Results are also saved to attendance/evaluation_report.xlsx
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import main as M   # reuse the exact same preprocessing / paths as the app

THRESHOLDS = [40, 50, 60, 70, 80, 90, 100]


_det = None


def crop_face(gray):
    """Largest Haar-detected face (lenient minSize for external datasets)."""
    global _det
    if _det is None:
        _det = cv2.CascadeClassifier(M.CASCADE_PATH)
    faces = _det.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray[y:y + h, x:x + w]


def load_dataset(root=None, detect=False, min_images=1):
    """Return ({label: (name, [faces])}, n_skipped). Layout: root/<person>/<images>."""
    root = root or M.DATASET_DIR
    external = os.path.abspath(root) != os.path.abspath(M.DATASET_DIR)
    data, skipped = {}, 0
    for idx, folder in enumerate(sorted(os.listdir(root))):
        path = os.path.join(root, folder)
        if not os.path.isdir(path):
            continue
        if external:
            label, name = idx, folder
        else:
            try:
                label = int(folder.split("_")[0])
            except ValueError:
                continue
            name = folder.split("_", 1)[1] if "_" in folder else folder
        files = sorted(f for f in os.listdir(path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".pgm")))
        imgs = []
        for f in files:
            gray = np.array(Image.open(os.path.join(path, f)).convert("L"), dtype="uint8")
            if detect:
                gray = crop_face(gray)
                if gray is None:
                    skipped += 1
                    continue
            imgs.append(M.preprocess(gray))
        if len(imgs) >= min_images:
            data[label] = (name, imgs)
    return data, skipped


def split(imgs, ratio, mode, rng):
    n_test = max(1, int(round(len(imgs) * ratio)))
    if mode == "random":
        idx = rng.permutation(len(imgs))
        test_idx = set(idx[:n_test])
    else:   # block: last frames are test -> avoids near-duplicate neighbours leaking
        test_idx = set(range(len(imgs) - n_test, len(imgs)))
    train = [im for i, im in enumerate(imgs) if i not in test_idx]
    test = [im for i, im in enumerate(imgs) if i in test_idx]
    return train, test


def train_lbph(faces, labels):
    rec = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    rec.train(faces, np.array(labels, dtype=np.int32))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-ratio", type=float, default=0.3)
    ap.add_argument("--split", choices=["block", "random"], default="block")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", help="external dataset: root/<person>/<images> (benchmark mode)")
    ap.add_argument("--detect", action="store_true", help="Haar-crop faces from full photos")
    ap.add_argument("--min-images", type=int, default=5, help="skip people with fewer images")
    ap.add_argument("--impostor-folds", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data, skipped = load_dataset(args.data_dir, args.detect, args.min_images if args.data_dir else 1)
    if skipped:
        print("Images skipped (no face detected): %d" % skipped)
    if len(data) < 2:
        raise SystemExit("Need at least 2 students in dataset/ (ideally 5+) to evaluate.")
    names = {k: v[0] for k, v in data.items()}
    splits = {k: split(v[1], args.test_ratio, args.split, rng) for k, v in data.items()}

    tr_faces = [im for k in splits for im in splits[k][0]]
    tr_labels = [k for k in splits for _ in splits[k][0]]
    te = [(k, im) for k in splits for im in splits[k][1]]
    print("=" * 70)
    print("DATASET: %d students | train images: %d | test images: %d | split: %s"
          % (len(data), len(tr_faces), len(te), args.split))
    counts = [len(v[1]) for v in data.values()]
    if len(data) <= 12:
        print("Images per student:", {names[k]: len(v[1]) for k, v in data.items()})
    else:
        print("Images per person: min %d, mean %.1f, max %d" % (min(counts), np.mean(counts), max(counts)))

    # ---------- closed-set evaluation ----------
    rec = train_lbph(tr_faces, tr_labels)
    rows, t0 = [], time.perf_counter()
    for true, im in te:
        pred, dist = rec.predict(im)
        rows.append((true, pred, dist))
    ms_pred = (time.perf_counter() - t0) / len(te) * 1000
    res = pd.DataFrame(rows, columns=["true", "pred", "dist"])
    acc = (res.true == res.pred).mean() * 100
    print("\nCLOSED-SET ACCURACY (no threshold): %.2f%%" % acc)
    print("Mean distance - correct: %.1f | wrong: %s" % (
        res[res.true == res.pred].dist.mean(),
        "%.1f" % res[res.true != res.pred].dist.mean() if (res.true != res.pred).any() else "n/a"))

    # per-student precision / recall / F1
    per = []
    for k in data:
        tp = ((res.true == k) & (res.pred == k)).sum()
        fp = ((res.true != k) & (res.pred == k)).sum()
        fn = ((res.true == k) & (res.pred != k)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per.append({"Student": names[k], "Test imgs": int((res.true == k).sum()),
                    "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3)})
    per = pd.DataFrame(per)
    print("\nPER-STUDENT METRICS\n" + per.to_string(index=False))
    print("Macro avg  P=%.3f  R=%.3f  F1=%.3f" % (per.Precision.mean(), per.Recall.mean(), per.F1.mean()))

    labels_sorted = sorted(data)
    cm = pd.crosstab(res.true.map(names), res.pred.map(names)).reindex(
        index=[names[k] for k in labels_sorted], columns=[names[k] for k in labels_sorted],
        fill_value=0)
    print("\nCONFUSION MATRIX (rows=true, cols=predicted)\n" + cm.to_string())

    # ---------- impostor test (hold out ~20% of people as strangers, repeated) ----------
    imp = []
    keys = list(data)
    n_out = max(1, int(round(len(keys) * 0.2)))
    for _ in range(args.impostor_folds):
        out_set = set(rng.choice(keys, size=min(n_out, len(keys) - 1), replace=False).tolist())
        f = [im for kk in splits if kk not in out_set for im in splits[kk][0]]
        l = [kk for kk in splits if kk not in out_set for _ in splits[kk][0]]
        r = train_lbph(f, l)
        for k in out_set:
            imp.extend(r.predict(im)[1] for im in data[k][1])   # never seen in training
    imp = np.array(imp)

    # ---------- threshold sweep ----------
    sweep = []
    for t in THRESHOLDS:
        acc_mask = res.dist <= t
        correct = (acc_mask & (res.true == res.pred)).mean() * 100
        reject = (~acc_mask).mean() * 100
        wrong = (acc_mask & (res.true != res.pred)).mean() * 100
        far = (imp <= t).mean() * 100
        sweep.append({"Threshold": t, "Correct accept %": round(correct, 2),
                      "False reject (FRR) %": round(reject, 2),
                      "Wrong-ID %": round(wrong, 2),
                      "Impostor accept (FAR) %": round(far, 2)})
    sweep = pd.DataFrame(sweep)
    print("\nTHRESHOLD SWEEP (registered students vs impostors, LBPH distance <= T is accepted)")
    print(sweep.to_string(index=False))
    print("Impostor distances: mean %.1f, min %.1f | Genuine distances: mean %.1f"
          % (imp.mean(), imp.min(), res.dist.mean()))

    # ---------- speed ----------
    sample = te[0][1]
    t0 = time.perf_counter()
    for _ in range(200):
        rec.predict(sample)
    ms_one = (time.perf_counter() - t0) / 200 * 1000
    t0 = time.perf_counter()
    for _ in range(50):
        M.detect_faces(sample)
    ms_det = (time.perf_counter() - t0) / 50 * 1000
    speed = pd.DataFrame([{"LBPH predict (ms/face)": round(ms_one, 3),
                           "Haar detect on 200x200 (ms)": round(ms_det, 3),
                           "Approx max FPS (predict only)": round(1000 / ms_one, 1)}])
    print("\nSPEED\n" + speed.to_string(index=False))
    print("=" * 70)

    # ---------- save ----------
    tag = "_" + os.path.basename(os.path.normpath(args.data_dir)) if args.data_dir else ""
    out = os.path.join(M.ATTENDANCE_DIR, "evaluation_report%s.xlsx" % tag)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        per.to_excel(w, sheet_name="Per-student", index=False)
        cm.to_excel(w, sheet_name="Confusion")
        sweep.to_excel(w, sheet_name="Threshold sweep", index=False)
        speed.to_excel(w, sheet_name="Speed", index=False)
    print("Saved:", out)


if __name__ == "__main__":
    M.ensure_dirs()
    M.ensure_cascade()
    main()