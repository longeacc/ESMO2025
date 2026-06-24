"""
maccrobat_run_eval.py
---------------------
1. Annotate every MACCROBAT2020 .txt with the rule-based regex annotator,
   writing predicted .ann into a separate folder.
2. Evaluate predictions vs gold .ann by label + character-offset overlap,
   restricted to the regex-covered labels (so open-vocab entities the rules
   never target don't tank the score).

Run:
    python maccrobat_run_eval.py
"""

import os
import re
from collections import Counter, defaultdict

from maccrobat_brat_annotator import annotate_txt_folder
from maccrobat_split import build_split, load_split

# ----- paths (anchored on this file, robust to cwd) -----
_HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_DIR = os.path.join(_HERE, "..", "..", "MACCROBAT2020")
PRED_DIR = os.path.join(_HERE, "MACCROBAT2020_pred_rules")

# Labels the regex annotator actually emits.
LABELS = (
    "Age", "Sex", "Date", "Duration", "Frequency", "Time",
    "Dosage", "Administration",
    "Volume", "Area", "Distance", "Weight",
    "Severity", "Color", "Shape", "Texture", "Lab_value",
)


def _list_anns(folder):
    paths = {}
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".ann"):
                paths[os.path.basename(f)] = os.path.join(root, f)
    return paths


def _read_T_spans(path):
    """Counter keyed by (label, start, end) for T-lines whose label is in LABELS."""
    c = Counter()
    if not path or not os.path.exists(path):
        return c
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("T"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            m = re.match(r"^(\S+)\s+(\d+)\s+(\d+)", parts[1])
            if not m:
                continue
            label, s, e = m.group(1), int(m.group(2)), int(m.group(3))
            if label in LABELS:
                c[(label, s, e)] += 1
    return c


def _overlap_len(a, b):
    (s1, e1), (s2, e2) = a, b
    return max(0, min(e1, e2) - max(s1, s2))


def _score_span(g, p, metric):
    if metric == "exact":
        return 1.0 if g == p else 0.0
    inter = _overlap_len(g, p)
    glen, plen = g[1] - g[0], p[1] - p[0]
    if glen <= 0 or plen <= 0:
        return 0.0
    if metric == "iou":
        denom = glen + plen - inter
        return inter / denom if denom else 0.0
    if metric == "min":
        return inter / min(glen, plen)
    if metric == "gold":
        return inter / glen
    if metric == "pred":
        return inter / plen
    raise ValueError(metric)


def _match(g, p, metric, thr):
    if metric == "exact" or thr >= 1.0:
        tp = sum(min(g[k], p[k]) for k in set(g) | set(p))
        fp = sum(p[k] - min(p[k], g.get(k, 0)) for k in p)
        fn = sum(g[k] - min(g[k], p.get(k, 0)) for k in g)
        return tp, fp, fn
    pairs = []
    for gs in g:
        for ps in p:
            sc = _score_span(gs, ps, metric)
            if sc >= thr:
                pairs.append((sc, gs, ps))
    pairs.sort(key=lambda t: (-t[0], t[1][1] - t[1][0], t[2][1] - t[2][0]))
    rem_g, rem_p = dict(g), dict(p)
    tp = 0
    for sc, gs, ps in pairs:
        if rem_g.get(gs, 0) <= 0 or rem_p.get(ps, 0) <= 0:
            continue
        n = min(rem_g[gs], rem_p[ps])
        tp += n
        rem_g[gs] -= n
        rem_p[ps] -= n
    fp = sum(v for v in rem_p.values() if v > 0)
    fn = sum(v for v in rem_g.values() if v > 0)
    return tp, fp, fn


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate(gold_dir, pred_dir, *, metric="min", thr=0.80, doc_ids=None):
    """doc_ids: optional set of doc stems to restrict evaluation to (the split)."""
    gold_files = _list_anns(gold_dir)
    pred_files = _list_anns(pred_dir)
    keep = set(f"{d}.ann" for d in doc_ids) if doc_ids is not None else None
    counts = {l: {"tp": 0, "fp": 0, "fn": 0} for l in LABELS}

    for name, gpath in gold_files.items():
        if keep is not None and name not in keep:
            continue
        gold = _read_T_spans(gpath)
        pred = _read_T_spans(pred_files.get(name))
        g_by, p_by = defaultdict(Counter), defaultdict(Counter)
        for (l, s, e), n in gold.items():
            g_by[l][(s, e)] += n
        for (l, s, e), n in pred.items():
            p_by[l][(s, e)] += n
        for l in LABELS:
            tp, fp, fn = _match(g_by.get(l, Counter()), p_by.get(l, Counter()),
                                 metric, thr)
            counts[l]["tp"] += tp
            counts[l]["fp"] += fp
            counts[l]["fn"] += fn

    per_label, mtp, mfp, mfn = {}, 0, 0, 0
    for l, d in counts.items():
        p, r, f1 = _prf(d["tp"], d["fp"], d["fn"])
        per_label[l] = {**d, "precision": p, "recall": r, "f1": f1}
        mtp += d["tp"]; mfp += d["fp"]; mfn += d["fn"]
    P, R, F1 = _prf(mtp, mfp, mfn)
    micro = {"tp": mtp, "fp": mfp, "fn": mfn, "precision": P, "recall": R, "f1": F1}
    return {"per_label": per_label, "micro": micro}


def _print_res(title, res):
    print(f"\n===== {title} =====")
    print(f"{'label':<16}{'P':>7}{'R':>7}{'F1':>7}{'TP':>6}{'FP':>6}{'FN':>6}")
    for l in LABELS:
        d = res["per_label"][l]
        print(f"{l:<16}{d['precision']:>7.2f}{d['recall']:>7.2f}{d['f1']:>7.2f}"
              f"{d['tp']:>6}{d['fp']:>6}{d['fn']:>6}")
    m = res["micro"]
    print(f"MICRO  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
          f"  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")


def main():
    build_split()                       # frozen 75/25 split (created once)
    train_ids, test_ids = load_split()
    print(f"Split: {len(train_ids)} train / {len(test_ids)} test")

    print("Annotating MACCROBAT2020 .txt -> predicted .ann ...")
    written = annotate_txt_folder(GOLD_DIR, out_dir=PRED_DIR, recursive=True)
    print(f"  wrote {len(written)} files")

    res_train = evaluate(GOLD_DIR, PRED_DIR, metric="min", thr=0.80,
                         doc_ids=train_ids)
    res_test = evaluate(GOLD_DIR, PRED_DIR, metric="min", thr=0.80,
                        doc_ids=test_ids)

    _print_res("TRAIN (75%) — used for tuning", res_train)
    _print_res("TEST (25%) — held out", res_test)


if __name__ == "__main__":
    main()
