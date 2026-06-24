"""
maccrobat_export_test_csv.py
----------------------------
Exporte les metriques du JEU DE TEST (25% tenu a l'ecart) par entite :
VP (TP), FP, FN, Precision, Recall, F1. Plus une ligne MICRO globale.

Sortie : ./Results/maccrobat_test_metrics.csv
"""

import os
import csv

from maccrobat_brat_annotator import annotate_txt_folder
from maccrobat_run_eval import evaluate, LABELS, GOLD_DIR, PRED_DIR, _list_anns, _read_T_spans
from maccrobat_split import build_split, load_split

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Results", "maccrobat_test_metrics.csv")


def _count_gold(gold_dir, doc_ids):
    """Count gold T-spans per label, read directly from .ann files (test split only)."""
    counts = {l: 0 for l in LABELS}
    for fname, path in _list_anns(gold_dir).items():
        doc_id = os.path.splitext(fname)[0]
        if doc_ids is not None and doc_id not in doc_ids:
            continue
        for (label, _s, _e), n in _read_T_spans(path).items():
            counts[label] += n
    return counts


def main():
    build_split()
    _train, test_ids = load_split()
    annotate_txt_folder(GOLD_DIR, out_dir=PRED_DIR, recursive=True)

    res = evaluate(GOLD_DIR, PRED_DIR, metric="min", thr=0.80, doc_ids=test_ids)
    gold_counts = _count_gold(GOLD_DIR, test_ids)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["entity", "count", "TP", "FP", "FN", "precision", "recall", "f1"])
        for l in LABELS:
            d = res["per_label"][l]
            w.writerow([l, gold_counts[l], d["tp"], d["fp"], d["fn"],
                        round(d["precision"], 3), round(d["recall"], 3),
                        round(d["f1"], 3)])
        m = res["micro"]
        w.writerow(["MICRO", sum(gold_counts.values()), m["tp"], m["fp"], m["fn"],
                    round(m["precision"], 3), round(m["recall"], 3),
                    round(m["f1"], 3)])

    print(f"Ecrit : {OUT_CSV}  ({len(test_ids)} docs test)")


if __name__ == "__main__":
    main()
