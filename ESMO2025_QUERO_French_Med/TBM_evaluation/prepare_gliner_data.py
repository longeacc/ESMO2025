"""
Convert QUAERO FrenchMed (Brat format) -> GLiNER training format.

Multi-label stratified 75/25 split: preserves the proportion of EVERY label
between train and test (not just the dominant one).
All documents from train/MEDLINE + test/MEDLINE are pooled, then re-split.

Output files: train_gliner.json, test_gliner.json

Each document:
{
    "tokenized_text": ["Traitements", "de", ...],
    "ner": [[token_start, token_end, "LABEL"], ...],
    "relations": []   # empty — QUAERO has no relation annotations
}

Usage:
    pip install spacy numpy
    python -m spacy download fr_core_news_sm
    python prepare_gliner_data.py
"""

import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QUAERO_ROOT = Path(__file__).parent.parent / "QUAERO_FrenchMed" / "corpus"

CORPUS_DIRS = [
    QUAERO_ROOT / "train" / "MEDLINE",
    QUAERO_ROOT / "test"  / "MEDLINE",
]

VALID_LABELS = {"DISO", "PROC", "ANAT", "CHEM", "DEVI", "LIVB", "PHYS", "PHEN", "GEOG", "OBJC"}

RANDOM_SEED = 42
N_SELECTED = 150   # Phase A: select this many from pool
N_TRAIN = 100      # Phase B: train size
N_TEST = 50        # Phase B: test size

# External split file output directory (shared with Rules pipeline)
SPLIT_DIR = Path(__file__).parent.parent / "Rules" / "src"

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Brat parser
# ---------------------------------------------------------------------------

def parse_ann(ann_path: Path):
    entities = {}
    attributes = {}
    relations = []

    for line in ann_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("T"):
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            tid = parts[0]
            meta = parts[1].split()
            label = meta[0]
            if label not in VALID_LABELS:
                continue
            nums = [int(x) for token in meta[1:] for x in token.split(";")]
            entities[tid] = {"start": nums[0], "end": nums[-1], "label": label, "negated": False}

        elif line.startswith("R"):
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            meta = parts[1].split()
            rel_type = meta[0]
            args = {}
            for arg in meta[1:]:
                role, tid = arg.split(":")
                args[role] = tid
            if "Arg1" in args and "Arg2" in args:
                relations.append({"type": rel_type, "head_tid": args["Arg1"], "tail_tid": args["Arg2"]})

        elif line.startswith("A") or line.startswith("M"):
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            meta = parts[1].split()
            if len(meta) >= 2:
                attr_name = meta[0].lower()
                tid = meta[1]
                value = meta[2].lower() if len(meta) > 2 else "true"
                attributes.setdefault(tid, {})[attr_name] = value

    for tid, attrs in attributes.items():
        if tid in entities and "negated" in attrs:
            entities[tid]["negated"] = attrs["negated"] != "false"

    return entities, relations


# ---------------------------------------------------------------------------
# Character offset -> token index alignment
# ---------------------------------------------------------------------------

def char_span_to_token_span(doc, char_start: int, char_end: int):
    for mode in ("strict", "expand", "contract"):
        try:
            span = doc.char_span(char_start, char_end, alignment_mode=mode)
        except Exception:
            continue
        if span is not None and len(span) > 0 and span.start <= span.end - 1:
            return span.start, span.end - 1

    overlapping = [
        i for i, tok in enumerate(doc)
        if tok.idx < char_end and tok.idx + len(tok.text) > char_start
    ]
    if overlapping:
        return overlapping[0], overlapping[-1]

    best = min(
        range(len(doc)),
        key=lambda i: min(
            abs(doc[i].idx - char_start),
            abs(doc[i].idx + len(doc[i].text) - char_end),
        ),
    )
    return best, best


# ---------------------------------------------------------------------------
# Remove sub-spans: keep only the largest span when one is contained in another
# ---------------------------------------------------------------------------

def remove_sub_spans(ner: list[list]) -> list[list]:
    """
    If span A [ts_a, te_a] is fully contained within span B [ts_b, te_b]
    (i.e. ts_b <= ts_a and te_a <= te_b and A != B), drop A.
    """
    filtered = []
    for i, a in enumerate(ner):
        ts_a, te_a = a[0], a[1]
        is_sub = False
        for j, b in enumerate(ner):
            if i == j:
                continue
            ts_b, te_b = b[0], b[1]
            if ts_b <= ts_a and te_a <= te_b and (ts_b, te_b) != (ts_a, te_a):
                is_sub = True
                break
        if not is_sub:
            filtered.append(a)
    return filtered


# ---------------------------------------------------------------------------
# Single document conversion
# ---------------------------------------------------------------------------

def convert_doc(txt_path: Path, nlp) -> dict | None:
    text = txt_path.read_text(encoding="utf-8").strip()
    ann_path = txt_path.with_suffix(".ann")
    if not ann_path.exists():
        return None

    doc = nlp(text)
    tokens = [tok.text for tok in doc]

    entities, brat_relations = parse_ann(ann_path)

    ner = []
    tid_to_token_span = {}

    for tid, ent in entities.items():
        ts, te = char_span_to_token_span(doc, ent["start"], ent["end"])
        tid_to_token_span[tid] = (ts, te, ent["label"])
        entry = [ts, te, ent["label"]]
        if ent["negated"]:
            entry.append({"negated": True})
        ner.append(entry)

    ner = remove_sub_spans(ner)

    relations = []
    for rel in brat_relations:
        head = tid_to_token_span.get(rel["head_tid"])
        tail = tid_to_token_span.get(rel["tail_tid"])
        if head and tail:
            relations.append({"type": rel["type"], "head": list(head), "tail": list(tail)})

    return {
        "tokenized_text": tokens,
        "ner": ner,
        "relations": relations,
    }


# ---------------------------------------------------------------------------
# Multi-label iterative stratified split (Sechidis et al. 2011)
# ---------------------------------------------------------------------------

def doc_label_vector(doc: dict, all_labels: list[str]) -> np.ndarray:
    present = set(e[2] for e in doc["ner"])
    return np.array([1 if l in present else 0 for l in all_labels])


def iterative_stratified_split(docs: list[dict], all_labels: list[str],
                               train_ratio: float, seed: int,
                               n_train_target: int | None = None):
    """
    Iterative stratified split for multi-label data.
    Processes labels from rarest to most frequent.
    For each label, assigns documents to whichever fold (train/test)
    needs more of that label, preserving proportions across ALL labels.

    If n_train_target is given, post-hoc rebalancing moves docs between
    folds to reach the exact count.
    """
    rng = np.random.RandomState(seed)
    n = len(docs)

    label_matrix = np.array([doc_label_vector(d, all_labels) for d in docs])

    desired = np.zeros((2, len(all_labels)), dtype=float)
    totals = label_matrix.sum(axis=0).astype(float)
    desired[0] = totals * train_ratio
    desired[1] = totals * (1 - train_ratio)

    assignment = np.full(n, -1, dtype=int)
    label_order = np.argsort(totals)

    for l_idx in label_order:
        candidates = np.where((label_matrix[:, l_idx] == 1) & (assignment == -1))[0]
        rng.shuffle(candidates)

        for doc_idx in candidates:
            if desired[0, l_idx] >= desired[1, l_idx]:
                fold = 0  # train
            else:
                fold = 1  # test
            assignment[doc_idx] = fold
            desired[fold] -= label_matrix[doc_idx]

    unassigned = np.where(assignment == -1)[0]
    for doc_idx in unassigned:
        fold = 0 if rng.random() < train_ratio else 1
        assignment[doc_idx] = fold

    # Post-hoc rebalancing to reach exact n_train_target
    if n_train_target is not None:
        n_train_cur = int((assignment == 0).sum())
        while n_train_cur > n_train_target:
            # Move a train doc to test: pick the one whose labels are
            # most over-represented in train
            train_indices = np.where(assignment == 0)[0]
            best_idx, best_score = None, -np.inf
            for idx in train_indices:
                score = float(label_matrix[idx].sum())  # simple heuristic
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None:
                assignment[best_idx] = 1
            n_train_cur -= 1
        while n_train_cur < n_train_target:
            # Move a test doc to train
            test_indices = np.where(assignment == 1)[0]
            best_idx, best_score = None, -np.inf
            for idx in test_indices:
                score = float(label_matrix[idx].sum())
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None:
                assignment[best_idx] = 0
            n_train_cur += 1

    train_docs = [docs[i] for i in range(n) if assignment[i] == 0]
    test_docs = [docs[i] for i in range(n) if assignment[i] == 1]
    return train_docs, test_docs


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(split_name: str, docs: list[dict]):
    label_counts = defaultdict(int)
    rel_counts = defaultdict(int)
    negated = 0

    for doc in docs:
        for ent in doc["ner"]:
            label_counts[ent[2]] += 1
            if len(ent) == 4:
                negated += 1
        for rel in doc["relations"]:
            rel_counts[rel["type"]] += 1

    print(f"\n  Entities per label [{split_name}]:")
    for label in sorted(label_counts):
        print(f"    {label:<6} {label_counts[label]}")
    print(f"    Negated: {negated}")

    if rel_counts:
        print(f"  Relations per type [{split_name}]:")
        for rtype in sorted(rel_counts):
            print(f"    {rtype:<20} {rel_counts[rtype]}")
    else:
        print(f"  Relations: none found in corpus")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading spaCy fr_core_news_sm ...")
    nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner", "lemmatizer"])

    # Pool all documents from both original splits
    all_docs = []       # list of dicts (one per doc, with _source key)
    doc_sources = []    # parallel list of txt_path for doc_id extraction
    for corpus_dir in CORPUS_DIRS:
        if not corpus_dir.exists():
            print(f"SKIP: {corpus_dir} not found")
            continue
        txt_files = sorted(corpus_dir.glob("*.txt"))
        print(f"  {corpus_dir.name}: {len(txt_files)} files")
        for txt_path in txt_files:
            doc = convert_doc(txt_path, nlp)
            if doc is not None:
                # Add _source to track doc identity
                doc["_source"] = txt_path.stem
                doc["_txt_path"] = str(txt_path)
                all_docs.append(doc)
                doc_sources.append(txt_path)

    print(f"\nTotal: {len(all_docs)} documents pooled from all sources")

    # Two-phase stratified split (Sechidis 2011)
    all_labels = sorted(VALID_LABELS)

    # Phase A: select 150 from ~1666 by stratified split
    # Use iterative_stratified_split with train_ratio = N_SELECTED/total
    select_ratio = N_SELECTED / len(all_docs)
    selected_docs, ignored_docs = iterative_stratified_split(
        all_docs, all_labels, select_ratio, RANDOM_SEED,
        n_train_target=N_SELECTED,
    )
    print(f"\nPhase A: {len(selected_docs)} selected, {len(ignored_docs)} ignored")

    # Phase B: split 150 selected into 100 train / 50 test
    train_ratio_b = N_TRAIN / N_SELECTED  # 100/150 = 2/3
    train_docs, test_docs = iterative_stratified_split(
        selected_docs, all_labels, train_ratio_b, RANDOM_SEED,
        n_train_target=N_TRAIN,
    )
    print(f"Phase B: {len(train_docs)} train, {len(test_docs)} test")

    print(f"\n[train] {len(train_docs)} documents")
    print_stats("train", train_docs)

    print(f"\n[test] {len(test_docs)} documents")
    print_stats("test", test_docs)

    # Verify stratification quality
    print(f"\n  Stratification check (% of entities per label):")
    print(f"    {'Label':<8} {'Train%':>7} {'Test%':>7} {'Diff':>7}")
    print(f"    {'-'*35}")
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    for d in train_docs:
        for e in d["ner"]:
            train_counts[e[2]] += 1
    for d in test_docs:
        for e in d["ner"]:
            test_counts[e[2]] += 1
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    for label in sorted(all_labels):
        tr_pct = 100 * train_counts[label] / total_train if total_train else 0
        te_pct = 100 * test_counts[label] / total_test if total_test else 0
        diff = abs(tr_pct - te_pct)
        marker = "OK" if diff < 2 else "WARN" if diff < 5 else "BAD"
        print(f"    {label:<8} {tr_pct:>6.1f}% {te_pct:>6.1f}% {diff:>5.1f}%  {marker}")

    # Save split_train.txt and split_test.txt into Rules/src/
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_ids = [d["_source"] for d in train_docs]
    test_ids = [d["_source"] for d in test_docs]
    (SPLIT_DIR / "split_train.txt").write_text(
        "\n".join(train_ids) + "\n", encoding="utf-8"
    )
    (SPLIT_DIR / "split_test.txt").write_text(
        "\n".join(test_ids) + "\n", encoding="utf-8"
    )
    print(f"  Split IDs saved -> {SPLIT_DIR / 'split_train.txt'}")
    print(f"  Split IDs saved -> {SPLIT_DIR / 'split_test.txt'}")

    # Save also a mapping from doc_id -> txt_path for downstream scripts
    split_paths = {}
    for d in train_docs + test_docs:
        split_paths[d["_source"]] = d["_txt_path"]
    (SPLIT_DIR / "split_doc_paths.json").write_text(
        json.dumps(split_paths, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Doc paths saved -> {SPLIT_DIR / 'split_doc_paths.json'}")

    # Remove internal fields before saving GLiNER JSON
    for d in train_docs + test_docs:
        d.pop("_txt_path", None)
    # Keep _source in JSON for downstream tracking

    # Save
    for split_name, docs in [("train", train_docs), ("test", test_docs)]:
        out_path = OUT_DIR / f"{split_name}_gliner.json"
        out_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
