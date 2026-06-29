"""
maccrobat_split.py
------------------
Deterministic 100/50 train/test split of MACCROBAT2020 documents
with iterative multi-label stratification (Sechidis et al. 2011).

Two-phase stratified split:
  Phase A: Select 150 docs from 200 via stratified sampling (seed=42)
  Phase B: Split 150 into 100 train / 50 test via stratified split (seed=42)

Each document may contain multiple entity types; the stratification
ensures that the label distribution is as balanced as possible across
train and test sets.

The split is generated ONCE with a fixed seed and frozen to disk
(split_train.txt / split_test.txt). Regex are tuned on TRAIN only; TEST is
held out and must not be inspected.

Run once:  python maccrobat_split.py
"""

import os
import re
import random
import shutil
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_DIR = os.path.join(_HERE, "..", "..", "MACCROBAT2020")
TRAIN_FILE = os.path.join(_HERE, "split_train.txt")
TEST_FILE = os.path.join(_HERE, "split_test.txt")
# Dossiers physiques crees a cote du dataset gold
TRAIN_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "MACCROBAT2020_train"))
TEST_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "MACCROBAT2020_test"))
SEED = 42
N_SELECTED = 150   # Phase A: select this many from 200
N_TRAIN = 100      # Phase B: train size
N_TEST = 50        # Phase B: test size


def _doc_ids(gold_dir):
    ids = []
    for f in os.listdir(gold_dir):
        if f.endswith(".txt"):
            ids.append(os.path.splitext(f)[0])
    return sorted(ids)


def _doc_label_sets(gold_dir, doc_ids):
    """Return {doc_id: set_of_labels} from gold .ann files."""
    doc_labels = {}
    for doc_id in doc_ids:
        ann_path = os.path.join(gold_dir, doc_id + ".ann")
        labels = set()
        if os.path.exists(ann_path):
            with open(ann_path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if not line.startswith("T"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    m = re.match(r"^(\S+)\s+", parts[1])
                    if m:
                        labels.add(m.group(1))
        doc_labels[doc_id] = labels
    return doc_labels


def _iterative_stratification(doc_ids, doc_labels, test_fraction, rng):
    """Iterative stratification for multi-label data (Sechidis et al. 2011).

    For each label, the desired number of docs per fold is computed ONCE from
    the global counts.  Labels are processed rarest-first so that rare labels
    get priority in balancing.
    """
    all_labels = sorted({l for ls in doc_labels.values() for l in ls})
    proportions = [1.0 - test_fraction, test_fraction]  # [train, test]

    # Desired count per label per fold — computed once from the full dataset
    global_label_count = Counter()
    for doc_id in doc_ids:
        for l in doc_labels[doc_id]:
            global_label_count[l] += 1
    target = {
        fold: {l: proportions[fold] * global_label_count[l] for l in all_labels}
        for fold in (0, 1)
    }

    remaining = set(doc_ids)
    folds = {0: [], 1: []}
    fold_label_counts = {0: Counter(), 1: Counter()}

    ids_shuffled = list(doc_ids)
    rng.shuffle(ids_shuffled)

    while remaining:
        # Count labels among remaining docs only (to find the rarest)
        remaining_label_counts = Counter()
        for doc_id in remaining:
            for l in doc_labels[doc_id]:
                remaining_label_counts[l] += 1

        rarest_label = None
        rarest_count = float("inf")
        for l in all_labels:
            c = remaining_label_counts.get(l, 0)
            if 0 < c < rarest_count:
                rarest_count = c
                rarest_label = l

        if rarest_label is None:
            # No labelled docs left — distribute by overall fold size
            for doc_id in ids_shuffled:
                if doc_id in remaining:
                    ratio = len(folds[1]) / max(1, len(folds[0]) + len(folds[1]))
                    chosen = 0 if ratio >= test_fraction else 1
                    folds[chosen].append(doc_id)
                    remaining.discard(doc_id)
            break

        docs_with_label = [
            d for d in ids_shuffled
            if d in remaining and rarest_label in doc_labels[d]
        ]

        for doc_id in docs_with_label:
            # For each fold, sum (target - current) across doc's labels
            need = [0.0, 0.0]
            for fold in (0, 1):
                for l in doc_labels[doc_id]:
                    need[fold] += target[fold][l] - fold_label_counts[fold].get(l, 0)

            # Assign to fold with greatest remaining need; break ties by
            # under-represented fold size
            if need[0] > need[1]:
                chosen = 0
            elif need[1] > need[0]:
                chosen = 1
            else:
                chosen = 0 if len(folds[0]) <= len(folds[1]) else 1

            folds[chosen].append(doc_id)
            for l in doc_labels[doc_id]:
                fold_label_counts[chosen][l] += 1
            remaining.discard(doc_id)

    # --- Post-hoc repair: guarantee ≥1 doc in each fold for every label ---
    for l in all_labels:
        if global_label_count[l] < 2:
            continue
        for src, dst in ((0, 1), (1, 0)):
            if fold_label_counts[dst].get(l, 0) > 0:
                continue
            candidates = [d for d in folds[src] if l in doc_labels[d]]
            if not candidates:
                continue

            def _cost(d, _src=src):
                return sum(
                    1 for lb in doc_labels[d]
                    if fold_label_counts[_src].get(lb, 0) <= 1
                )
            candidates.sort(key=_cost)
            chosen = candidates[0]
            folds[src].remove(chosen)
            folds[dst].append(chosen)
            for lb in doc_labels[chosen]:
                fold_label_counts[src][lb] -= 1
                fold_label_counts[dst][lb] += 1

    # --- Rebalance: swap docs between folds to approach 75/25 ---
    n_target_test = round(len(doc_ids) * test_fraction)
    max_swaps = 50
    for _ in range(max_swaps):
        if len(folds[1]) <= n_target_test:
            break
        # move a test doc back to train — pick the one that improves overall
        # label balance the most (i.e. whose labels are over-represented in test)
        best_doc = None
        best_score = -float("inf")
        for d in folds[1]:
            # would any label drop to 0 in test? skip
            if any(fold_label_counts[1].get(lb, 0) <= 1 for lb in doc_labels[d]):
                continue
            score = sum(
                (fold_label_counts[1].get(lb, 0) / max(1, global_label_count[lb]))
                - test_fraction
                for lb in doc_labels[d]
            )
            if score > best_score:
                best_score = score
                best_doc = d
        if best_doc is None:
            break
        folds[1].remove(best_doc)
        folds[0].append(best_doc)
        for lb in doc_labels[best_doc]:
            fold_label_counts[1][lb] -= 1
            fold_label_counts[0][lb] += 1

    return sorted(folds[0]), sorted(folds[1])


def build_split(*, force=False):
    """Create the frozen split files if absent (or force=True).

    Two-phase stratified split:
      Phase A: Select 150 docs from 200 via stratified sampling (seed=42)
               Uses iterative_stratification with fraction = 50/200 to get
               the 50 docs to IGNORE; the remaining 150 are selected.
      Phase B: Split the 150 selected into 100 train / 50 test (seed=42)
               Uses iterative_stratification with fraction = 50/150.
    """
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE) and not force:
        return load_split()
    ids = _doc_ids(GOLD_DIR)
    doc_labels = _doc_label_sets(GOLD_DIR, ids)

    # Phase A: select 150 from 200 by stratified split (ignore 50)
    rng_a = random.Random(SEED)
    ignore_fraction = (len(ids) - N_SELECTED) / len(ids)  # 50/200 = 0.25
    selected, ignored = _iterative_stratification(ids, doc_labels, ignore_fraction, rng_a)
    print(f"Phase A: {len(selected)} selected, {len(ignored)} ignored (from {len(ids)} total)")

    # Phase B: split 150 selected into 100 train / 50 test
    rng_b = random.Random(SEED)
    test_fraction_b = N_TEST / N_SELECTED  # 50/150 = 1/3
    train, test = _iterative_stratification(selected, doc_labels, test_fraction_b, rng_b)
    print(f"Phase B: {len(train)} train, {len(test)} test")

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

    _print_stratification_report(doc_labels, train, test)

    return train, test


def _print_stratification_report(doc_labels, train, test):
    """Print label distribution across train/test to verify stratification."""
    all_labels = sorted({l for ls in doc_labels.values() for l in ls})
    print(f"\n{'label':<30}{'train%':>8}{'test%':>8}{'train_n':>9}{'test_n':>8}")
    for l in all_labels:
        n_train = sum(1 for d in train if l in doc_labels[d])
        n_test = sum(1 for d in test if l in doc_labels[d])
        total = n_train + n_test
        if total == 0:
            continue
        print(f"{l:<30}{n_train/total*100:>7.1f}%{n_test/total*100:>7.1f}%"
              f"{n_train:>9}{n_test:>8}")


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
