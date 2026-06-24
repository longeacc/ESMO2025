"""
quaero_export_test_csv.py
-------------------------
Exporte les métriques du JEU DE TEST (tenu à l'écart) par entité :
TP, FP, FN, Precision, Recall, F1. Plus une ligne MICRO globale.

Sortie : ./Results/quaero_test_metrics.csv
"""

import os
import csv

from quaero_brat_annotator import annotate_txt_folder
from quaero_run_eval import (
    evaluate, LABELS, GOLD_TEST_DIR, PRED_TEST_DIR, _list_anns, _read_T_spans,
)

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Results", "quaero_test_metrics.csv")


def _count_gold(gold_dir):
    """Count gold T-spans per label, read directly from the .ann files."""
    counts = {l: 0 for l in LABELS}
    for path in _list_anns(gold_dir).values():
        for (label, _s, _e), n in _read_T_spans(path).items():
            counts[label] += n
    return counts


def main():
    annotate_txt_folder(GOLD_TEST_DIR, out_dir=PRED_TEST_DIR, recursive=True)

    res = evaluate(GOLD_TEST_DIR, PRED_TEST_DIR, metric="min", thr=0.80)
    gold_counts = _count_gold(GOLD_TEST_DIR)

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

    print(f"Ecrit : {OUT_CSV}")


if __name__ == "__main__":
    main()
