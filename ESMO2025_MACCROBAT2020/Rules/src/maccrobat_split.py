"""
maccrobat_split.py
------------------
Deterministic 75/25 train/test split of MACCROBAT2020 documents.

The split is generated ONCE with a fixed seed and frozen to disk
(split_train.txt / split_test.txt). Regex are tuned on TRAIN only; TEST is
held out and must not be inspected.

Run once:  python maccrobat_split.py
"""

import os
import random
import shutil

_HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_DIR = os.path.join(_HERE, "..", "..", "MACCROBAT2020")
TRAIN_FILE = os.path.join(_HERE, "split_train.txt")
TEST_FILE = os.path.join(_HERE, "split_test.txt")
# Dossiers physiques crees a cote du dataset gold
TRAIN_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "MACCROBAT2020_train"))
TEST_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "MACCROBAT2020_test"))
SEED = 42
TEST_FRACTION = 0.25


def _doc_ids(gold_dir):
    ids = []
    for f in os.listdir(gold_dir):
        if f.endswith(".txt"):
            ids.append(os.path.splitext(f)[0])
    return sorted(ids)


def build_split(*, force=False):
    """Create the frozen split files if absent (or force=True)."""
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE) and not force:
        return load_split()
    ids = _doc_ids(GOLD_DIR)
    rng = random.Random(SEED)
    rng.shuffle(ids)
    n_test = round(len(ids) * TEST_FRACTION)
    test = sorted(ids[:n_test])
    train = sorted(ids[n_test:])
    with open(TRAIN_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(train) + "\n")
    with open(TEST_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(test) + "\n")

    # Repartition physique des paires .txt + .ann dans deux dossiers
    _materialize(train, TRAIN_DIR)
    _materialize(test, TEST_DIR)

    print(f"Split written: {len(train)} train, {len(test)} test")
    print(f"  -> {TRAIN_DIR}")
    print(f"  -> {TEST_DIR}")
    return train, test


def _materialize(ids, dest_dir):
    """Copie les .txt et .ann des documents `ids` depuis GOLD_DIR vers dest_dir
    (dossier recree a chaque fois pour eviter les residus)."""
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    for doc in ids:
        for ext in (".txt", ".ann"):
            src = os.path.join(GOLD_DIR, doc + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest_dir, doc + ext))


def load_split():
    """Return (train_ids, test_ids) as lists of doc stems."""
    with open(TRAIN_FILE, encoding="utf-8") as fh:
        train = [l.strip() for l in fh if l.strip()]
    with open(TEST_FILE, encoding="utf-8") as fh:
        test = [l.strip() for l in fh if l.strip()]
    return train, test


if __name__ == "__main__":
    build_split(force=True)
    tr, te = load_split()
    print(f"train={len(tr)}  test={len(te)}")
