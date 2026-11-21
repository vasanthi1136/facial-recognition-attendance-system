#!/usr/bin/env python3
"""
Download a public face benchmark and export it as  <out>/<person>/<image>.png
so it can be used with evaluate.py / research.py.

  pip install scikit-learn
  python prepare_benchmark.py --name olivetti --out C:\\data\\orl
  python prepare_benchmark.py --name lfw --out C:\\data\\lfw --min-faces 20

olivetti = the classic ORL/AT&T set (40 people x 10 images, 64x64, already cropped)
lfw      = Labeled Faces in the Wild, people with >= --min-faces images
Cite the original dataset papers and check their terms of use before publishing.
"""
import argparse
import os

import numpy as np
from PIL import Image


def export(images, targets, names, out):
    images = np.asarray(images, dtype=np.float32)
    if images.max() <= 1.0:               # sklearn returns floats in [0, 1]
        images = images * 255.0
    counts = {}
    for img, t in zip(images, targets):
        folder = os.path.join(out, names[t])
        os.makedirs(folder, exist_ok=True)
        counts[t] = counts.get(t, 0) + 1
        Image.fromarray(img.astype("uint8")).save(os.path.join(folder, "%03d.png" % counts[t]))
    print("Saved %d images of %d people to %s" % (len(images), len(counts), out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=["olivetti", "lfw"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-faces", type=int, default=20)
    a = ap.parse_args()
    if a.name == "olivetti":
        from sklearn.datasets import fetch_olivetti_faces
        d = fetch_olivetti_faces()
        export(d.images, d.target, ["person_%02d" % i for i in range(40)], a.out)
    else:
        from sklearn.datasets import fetch_lfw_people
        d = fetch_lfw_people(min_faces_per_person=a.min_faces, resize=1.0)
        export(d.images, d.target, [n.replace(" ", "_") for n in d.target_names], a.out)


if __name__ == "__main__":
    main()