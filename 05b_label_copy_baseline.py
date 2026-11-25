"""Step 5b - Training-free label-copy baseline on the official SolarFCD split (protocol P1).

Re-implements, as a stand-alone script, the lookup rule of the independent audit recorded in NOTLAR.md, section 7
(audit file critic/k04): every official Test file receives the label of its nearest official Train file(s), where
"nearest" is measured with byte identity (MD5) and the dihedral perceptual hash, without any learning.

Definitions (identical to the audit):
- Perceptual hash: imagehash.phash of the greyscale image (JPEG/MPO files decoded with PIL draft mode at 512 px) under
  all eight dihedral transformations (four rotations, with and without mirroring).
- Distance between two files: the minimum Hamming distance between the identity hash of one file and the eight hashes
  of the other, symmetrised over both directions; byte-identical files (same MD5) get distance -1.
- A Test file "has a copy in Train" if its distance to the nearest Train file is at most 4.
- Prediction: the most frequent label among the nearest Train files (all files at the minimum distance); ties are
  broken by the Train frequency of the label and then alphabetically.

Reported quantities:
- label-copy baseline: Test files that have a copy in Train AND receive the correct label from it, as a share of all
  375 Test files (article: 168/375 = 44.8%; 185 Test files have a copy, of which 90.8% are labelled correctly);
- nearest-neighbour lookup for all Test files (66.4%) and the majority-class baseline (Surface_Obstruction, 46.4%).

Usage:  python 05b_label_copy_baseline.py --solarfcd <SOLARFCD_DIR> [--out label_copy_baseline.csv] [--workers 4]
<SOLARFCD_DIR> is the root of the SolarFCD-Final archive with the folders Train, Valid and Test. Images are only read.
"""
import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

SPLITS = {"Train": "train", "Valid": "val", "Test": "test"}
T = Image.Transpose
DIHEDRAL = [[], [T.ROTATE_90], [T.ROTATE_180], [T.ROTATE_270], [T.FLIP_LEFT_RIGHT], [T.FLIP_LEFT_RIGHT, T.ROTATE_90],
            [T.FLIP_LEFT_RIGHT, T.ROTATE_180], [T.FLIP_LEFT_RIGHT, T.ROTATE_270]]


def hashes(path):
    """MD5 of the file bytes and the eight dihedral pHashes (as 64-bit integers)."""
    with open(path, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()
    im = Image.open(path)
    if im.format in ("JPEG", "MPO"):
        im.draft("L", (512, 512))
    try:
        im.seek(0)
    except Exception:
        pass
    g = im.convert("L")
    ph = []
    for ops in DIHEDRAL:
        t = g
        for op in ops:
            t = t.transpose(op)
        ph.append(int(str(imagehash.phash(t)), 16))
    return md5, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solarfcd", required=True, help="root of the SolarFCD-Final archive (Train/Valid/Test)")
    ap.add_argument("--out", default="", help="optional CSV with the per-Test-file lookup")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    rows = []
    for sdir, split in SPLITS.items():
        for cls in sorted(os.listdir(os.path.join(a.solarfcd, sdir))):
            cdir = os.path.join(a.solarfcd, sdir, cls)
            for name in sorted(os.listdir(cdir)):
                p = os.path.join(cdir, name)
                if os.path.isfile(p):
                    rows.append(dict(path=p, rel_path=f"{sdir}/{cls}/{name}", split=split, label=cls))
    H = pd.DataFrame(rows)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(hashes, H.path, chunksize=16))
    H["md5"] = [r[0] for r in res]
    ph = np.array([r[1] for r in res], dtype=np.uint64)  # files x 8

    te = np.flatnonzero(H.split == "test")
    tr = np.flatnonzero(H.split == "train")
    # symmetric dihedral distance Test x Train: min over k of d(ph0_test, phk_train) and d(phk_test, ph0_train)
    D = np.full((len(te), len(tr)), 64, dtype=np.int16)
    for k in range(8):
        np.minimum(D, np.bitwise_count(ph[te, 0][:, None] ^ ph[tr, k][None, :]).astype(np.int16), out=D)
        np.minimum(D, np.bitwise_count(ph[te, k][:, None] ^ ph[tr, 0][None, :]).astype(np.int16), out=D)
    md5 = H.md5.to_numpy()
    D[md5[te][:, None] == md5[tr][None, :]] = -1  # byte-identical copies are strictly closest

    label = H.label.to_numpy()
    train_freq = pd.Series(label[tr]).value_counts()
    out = []
    for r, i in enumerate(te):
        dmin = int(D[r].min())
        nn = tr[D[r] == dmin]
        votes = pd.Series(label[nn]).value_counts()
        top = votes[votes == votes.max()].index.tolist()
        pred = sorted(top, key=lambda c: (-train_freq[c], c))[0]
        out.append(dict(rel_path=H.rel_path[i], label=label[i], dmin_train=dmin, copy_in_train=dmin <= 4,
                        exact_copy_in_train=dmin == -1, nearest_train_labels=";".join(f"{k}:{v}" for k, v in votes.items()),
                        prediction=pred, correct=pred == label[i]))
    R = pd.DataFrame(out)
    if a.out:
        R.to_csv(a.out, index=False)

    n = len(R)
    cov = R[R.copy_in_train]
    k = int(cov.correct.sum())
    maj = R.label.value_counts()
    print(f"Official split: {len(tr)} Train, {int((H.split == 'val').sum())} Valid, {n} Test files")
    print(f"Test files with a copy in Train (MD5 or dihedral pHash distance <= 4): {len(cov)} "
          f"(byte-identical: {int(R.exact_copy_in_train.sum())})")
    print(f"Label-copy baseline: {k}/{n} = {k / n:.1%} of all Test files labelled correctly by copying the label of "
          f"their Train copy; accuracy on the {len(cov)} files with a copy {cov.correct.mean():.1%}")
    print(f"Nearest-Train-file lookup for all Test files: {int(R.correct.sum())}/{n} = {R.correct.mean():.1%}")
    print(f"Majority-class baseline ({maj.index[0]}): {int(maj.iloc[0])}/{n} = {maj.iloc[0] / n:.1%}")
    print("\nBy class (Test files, with a copy in Train, correct via the copy):")
    for c, g in R.groupby("label"):
        print(f"  {c:20s} {len(g):4d} {int(g.copy_in_train.sum()):4d} {int((g.copy_in_train & g.correct).sum()):4d}")


if __name__ == "__main__":
    main()
