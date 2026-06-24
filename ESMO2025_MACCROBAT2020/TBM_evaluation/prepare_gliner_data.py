"""
Convert MACCROBAT2020 (Brat format) → GLiNER training format.

Multi-label stratified 75/25 split: preserves the proportion of EVERY label
between train and test (not just the dominant one).
Output files: train_gliner.json, test_gliner.json

Each document:
{
    "tokenized_text": ["A", "42-year-old", ...],
    "ner": [[token_start, token_end, "LABEL"], ...],
    "relations": [{"type": "MODIFY", "head": [...], "tail": [...]}, ...]
}

Usage:
    pip install spacy scikit-learn
    python -m spacy download en_core_web_sm
    python prepare_gliner_data.py
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).parent.parent / "MACCROBAT2020"

VALID_LABELS = {
    "Disease_disorder",
    "Sign_symptom",
    "Diagnostic_procedure",
    "Therapeutic_procedure",
    "Medication",
    "Biological_structure",
    "Lab_value",
    "Detailed_description",
    "Clinical_event",
    "Severity",
    "Date",
    "Duration",
    "Dosage",
    "Administration",
    "History",
    "Nonbiological_location",
    "Activity",
    "Age",
    "Sex",
    "Family_history",
    "Frequency",
    "Shape",
    "Personal_background",
    "Distance",
    "Time",
    "Subject",
    "Color",
    "Quantitative_concept",
    "Texture",
    "Qualitative_concept",
    "Area",
    "Outcome",
    "Volume",
    "Other_event",
    "Other_entity",
    "Occupation",
    "Biological_attribute",
    "Weight",
    "Height",
    "Mass",
    "Coreference",
}

TRAIN_RATIO = 0.75
RANDOM_SEED = 42

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
                if ":" in arg:
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
# Character offset → token index alignment
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
# Single document conversion — split into sentences to stay under 384 tokens
# ---------------------------------------------------------------------------

def convert_doc(txt_path: Path, nlp) -> list[dict]:
    """Return a list of sentence-level samples (one per sentence)."""
    text = txt_path.read_text(encoding="utf-8").strip()
    ann_path = txt_path.with_suffix(".ann")
    if not ann_path.exists():
        return []

    doc = nlp(text)
    entities, _relations = parse_ann(ann_path)

    # Resolve all entity spans at document level
    tid_to_doc_span = {}
    for tid, ent in entities.items():
        ts, te = char_span_to_token_span(doc, ent["start"], ent["end"])
        tid_to_doc_span[tid] = (ts, te, ent["label"], ent["negated"])

    # Split into sentences, remap entity token indices per sentence
    results = []
    for sent in doc.sents:
        sent_tokens = [tok.text for tok in sent]

        ner = []
        for tid, (ts, te, label, negated) in tid_to_doc_span.items():
            # Entity fully contained in this sentence
            if ts >= sent.start and te < sent.end:
                local_ts = ts - sent.start
                local_te = te - sent.start
                entry = [local_ts, local_te, label]
                if negated:
                    entry.append({"negated": True})
                ner.append(entry)
            # Entity spans across sentence boundary — assign to sentence
            # that contains the majority of its tokens
            elif ts < sent.end and te >= sent.start:
                overlap_start = max(ts, sent.start)
                overlap_end = min(te, sent.end - 1)
                total_len = te - ts + 1
                overlap_len = overlap_end - overlap_start + 1
                if overlap_len > total_len / 2:
                    local_ts = overlap_start - sent.start
                    local_te = overlap_end - sent.start
                    entry = [local_ts, local_te, label]
                    if negated:
                        entry.append({"negated": True})
                    ner.append(entry)

        results.append({
            "tokenized_text": sent_tokens,
            "ner": ner,
            "relations": [],
            "_source": txt_path.stem,
        })

    return results



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
    for label in sorted(label_counts, key=lambda l: -label_counts[l]):
        print(f"    {label:<30} {label_counts[label]}")
    print(f"    Negated: {negated}")

    if rel_counts:
        print(f"  Relations per type [{split_name}]:")
        for rtype in sorted(rel_counts):
            print(f"    {rtype:<20} {rel_counts[rtype]}")
    else:
        print(f"  Relations: none found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def doc_label_vector(sentences: list[dict], all_labels: list[str]) -> np.ndarray:
    """Binary vector: 1 if the document contains at least one entity of that label."""
    present = set(e[2] for s in sentences for e in s["ner"])
    return np.array([1 if l in present else 0 for l in all_labels])


def iterative_stratified_split(docs_by_file: list, all_labels: list[str],
                               train_ratio: float, seed: int):
    """
    Iterative stratified split for multi-label data.
    Algorithm (Sechidis et al. 2011 / Szymanski & Kajdanowicz 2017):
      1. For each label, compute desired count in train and test.
      2. Process labels from rarest to most frequent.
      3. For each label, distribute documents that have that label
         to whichever fold (train/test) needs more of that label.
    This ensures every label's proportion is preserved across both splits.
    """
    rng = np.random.RandomState(seed)
    n = len(docs_by_file)

    # Build label matrix (n_docs x n_labels)
    label_matrix = np.array([doc_label_vector(d, all_labels) for d in docs_by_file])

    # Desired counts per fold: [train, test]
    desired = np.zeros((2, len(all_labels)), dtype=float)
    totals = label_matrix.sum(axis=0).astype(float)
    desired[0] = totals * train_ratio
    desired[1] = totals * (1 - train_ratio)

    # Track assignment: -1 = unassigned, 0 = train, 1 = test
    assignment = np.full(n, -1, dtype=int)

    # Process labels from rarest to most frequent
    label_order = np.argsort(totals)

    for l_idx in label_order:
        # Indices of unassigned docs that have this label
        candidates = np.where((label_matrix[:, l_idx] == 1) & (assignment == -1))[0]
        rng.shuffle(candidates)

        for doc_idx in candidates:
            # Assign to the fold that needs more of this label
            if desired[0, l_idx] >= desired[1, l_idx]:
                fold = 0  # train
            else:
                fold = 1  # test
            assignment[doc_idx] = fold
            # Decrease desired counts for ALL labels of this document
            desired[fold] -= label_matrix[doc_idx]

    # Assign remaining docs (those with no valid labels) randomly
    unassigned = np.where(assignment == -1)[0]
    for doc_idx in unassigned:
        fold = 0 if rng.random() < train_ratio else 1
        assignment[doc_idx] = fold

    train_docs = [docs_by_file[i] for i in range(n) if assignment[i] == 0]
    test_docs = [docs_by_file[i] for i in range(n) if assignment[i] == 1]
    return train_docs, test_docs


def main():
    print("Loading spaCy en_core_web_sm ...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    txt_files = sorted(CORPUS_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files in {CORPUS_DIR}")
    print(f"Found {len(txt_files)} documents in {CORPUS_DIR}")

    # Each element = list of sentence-dicts for one document
    docs_by_file = []
    for txt_path in txt_files:
        sentences = convert_doc(txt_path, nlp)
        if sentences:
            docs_by_file.append(sentences)

    total_sents = sum(len(s) for s in docs_by_file)
    print(f"Converted {len(docs_by_file)} documents → {total_sents} sentences")

    # Multi-label stratified split at DOCUMENT level
    all_labels = sorted(VALID_LABELS)
    train_doc_groups, test_doc_groups = iterative_stratified_split(
        docs_by_file, all_labels, TRAIN_RATIO, RANDOM_SEED,
    )

    # Flatten sentence lists
    train_sents = [s for doc in train_doc_groups for s in doc]
    test_sents = [s for doc in test_doc_groups for s in doc]

    print(f"\n[train] {len(train_doc_groups)} documents → {len(train_sents)} sentences")
    print_stats("train", train_sents)

    print(f"\n[test] {len(test_doc_groups)} documents → {len(test_sents)} sentences")
    print_stats("test", test_sents)

    # Verify stratification quality
    print(f"\n  Stratification check (% of entities per label):")
    print(f"    {'Label':<30} {'Train%':>7} {'Test%':>7} {'Diff':>7}")
    print(f"    {'-'*51}")
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)
    for s in train_sents:
        for e in s["ner"]:
            train_counts[e[2]] += 1
    for s in test_sents:
        for e in s["ner"]:
            test_counts[e[2]] += 1
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    for label in sorted(all_labels):
        tr_pct = 100 * train_counts[label] / total_train if total_train else 0
        te_pct = 100 * test_counts[label] / total_test if total_test else 0
        diff = abs(tr_pct - te_pct)
        marker = "OK" if diff < 2 else "WARN" if diff < 5 else "BAD"
        print(f"    {label:<30} {tr_pct:>6.1f}% {te_pct:>6.1f}% {diff:>5.1f}%  {marker}")

    # Remove internal field before saving
    for s in train_sents + test_sents:
        s.pop("_source", None)

    for split_name, sents in [("train", train_sents), ("test", test_sents)]:
        out_path = OUT_DIR / f"{split_name}_gliner.json"
        out_path.write_text(json.dumps(sents, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
