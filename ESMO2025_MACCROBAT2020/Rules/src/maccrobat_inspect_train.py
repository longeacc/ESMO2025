"""
maccrobat_inspect_train.py
--------------------------
Dump top false-positive / false-negative surface forms per label,
RESTRICTED TO THE TRAIN SPLIT ONLY. Use this (never the test set) to tune regex.

    python maccrobat_inspect_train.py [Label1 Label2 ...]
"""

import os
import sys
from collections import defaultdict, Counter

from maccrobat_run_eval import (
    _list_anns, _read_T_spans, _score_span, LABELS, GOLD_DIR, PRED_DIR,
)
from maccrobat_split import load_split


def _txt(name):
    p = os.path.join(GOLD_DIR, name.replace(".ann", ".txt"))
    return open(p, encoding="utf-8", errors="ignore").read() if os.path.exists(p) else ""


def inspect(labels=None, top=15):
    train_ids, _test = load_split()
    keep = set(f"{d}.ann" for d in train_ids)
    gf, pf = _list_anns(GOLD_DIR), _list_anns(PRED_DIR)
    labels = labels or list(LABELS)

    FP, FN = defaultdict(Counter), defaultdict(Counter)
    for name, gp in gf.items():
        if name not in keep:
            continue
        g, p = _read_T_spans(gp), _read_T_spans(pf.get(name))
        t = _txt(name)
        gby, pby = defaultdict(list), defaultdict(list)
        for (l, s, e), _n in g.items():
            gby[l].append((s, e))
        for (l, s, e), _n in p.items():
            pby[l].append((s, e))
        for l in labels:
            gl, pl = gby[l][:], pby[l][:]
            mg, mp = set(), set()
            for i, ps in enumerate(pl):
                for j, gs in enumerate(gl):
                    if j in mg:
                        continue
                    if _score_span(gs, ps, "min") >= 0.8:
                        mg.add(j); mp.add(i); break
            for i, ps in enumerate(pl):
                if i not in mp:
                    FP[l][t[ps[0]:ps[1]].replace("\n", " ")] += 1
            for j, gs in enumerate(gl):
                if j not in mg:
                    FN[l][t[gs[0]:gs[1]].replace("\n", " ")] += 1

    for l in labels:
        print(f"\n### {l}")
        print("  FP:", dict(FP[l].most_common(top)))
        print("  FN:", dict(FN[l].most_common(top)))


if __name__ == "__main__":
    inspect(sys.argv[1:] or None)
