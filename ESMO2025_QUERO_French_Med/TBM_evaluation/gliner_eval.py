"""
GLiNER fine-tuning + evaluation on QUAERO FrenchMed.

Pipeline:
  1. Load train_gliner.json / test_gliner.json (produced by prepare_gliner_data.py)
  2. Fine-tune each model with model.train_model()
  3. Evaluate on test set at multiple thresholds
  4. Save results to gliner_results.csv

Labels passed at inference = exact strings used during training ("DISO", "PROC", ...)
No label hints — follows https://urchade.github.io/GLiNER/training.html

Usage:
    pip install "gliner[training]"
    python gliner_eval.py                  # fine-tune all models + eval
    python gliner_eval.py eval             # eval only (load saved fine-tuned models)
    python gliner_eval.py inspect 0        # inspect doc index 0 from test set
"""

import sys
import csv
import json
from pathlib import Path
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
from gliner import GLiNER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATA_DIR   = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "fine_tuned"

MODELS = [
    # "urchade/gliner_large-v2.1",
    "almanach/camembert-bio-gliner",
]

QUAERO_LABELS = ["DISO", "PROC", "ANAT", "CHEM", "DEVI", "LIVB", "PHYS", "PHEN", "GEOG", "OBJC"]

THRESHOLDS = [0.3, 0.4, 0.5, 0.6]

TRAIN_PARAMS = dict(
    max_steps=1000,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    others_lr=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    others_weight_decay=0.01,
    max_grad_norm=1.0,
    focal_loss_alpha=-1,
    focal_loss_gamma=0,
    loss_reduction="sum",
    save_steps=500,
    logging_steps=50,
    save_total_limit=1,
)

RESULTS_CSV    = DATA_DIR / "gliner_results.csv"
PARAMS_CSV     = DATA_DIR / "gliner_train_params.csv"
PLOTS_DIR      = DATA_DIR / "plots"


def extract_metrics_from_trainer(trainer):
    train_loss, steps_train = [], []
    eval_loss, steps_eval = [], []
    for entry in trainer.state.log_history:
        step = entry.get("step", 0)
        if "loss" in entry:
            train_loss.append(float(entry["loss"]))
            steps_train.append(step)
        if "eval_loss" in entry:
            eval_loss.append(float(entry["eval_loss"]))
            steps_eval.append(step)
    return {
        "train_loss": train_loss, "steps_train": steps_train,
        "eval_loss": eval_loss, "steps_eval": steps_eval,
    }


def plot_training(metrics, model_name, train_f1, test_f1):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "__")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss curve ---
    ax = axes[0]
    ax.plot(metrics["steps_train"], metrics["train_loss"], label="Train loss", linewidth=1.5)
    if metrics["eval_loss"]:
        ax.plot(metrics["steps_eval"], metrics["eval_loss"], label="Eval loss", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- F1 train vs test (bar chart) ---
    ax = axes[1]
    labels_bar = ["Train F1", "Test F1"]
    values = [train_f1, test_f1]
    colors = ["#2196F3", "#FF9800"]
    bars = ax.bar(labels_bar, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title(f"Overfitting check — {model_name}")
    gap = train_f1 - test_f1
    verdict = "OVERFITTING" if gap > 0.05 else "OK" if gap > 0.02 else "Good fit"
    ax.text(0.5, 0.92, f"Gap = {gap:+.3f} → {verdict}",
            transform=ax.transAxes, ha="center", fontsize=12,
            color="red" if gap > 0.05 else "orange" if gap > 0.02 else "green")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = PLOTS_DIR / f"{safe_name}_training.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}_gliner.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run prepare_gliner_data.py first")
    return json.loads(path.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Token offset helpers (tokenized_text → char offsets for predict_entities)
# ---------------------------------------------------------------------------

def build_char_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join tokens with single spaces; return (text, per-token char offsets)."""
    offsets, pos = [], 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + 1
    return " ".join(tokens), offsets


def char_to_token(char_start: int, char_end: int, offsets: list[tuple[int, int]]):
    """Convert a char span to inclusive token indices, or None."""
    ts = te = None
    for i, (cs, ce) in enumerate(offsets):
        if ts is None and cs <= char_start < ce:
            ts = i
        if cs < char_end <= ce:
            te = i
    return (ts, te) if ts is not None and te is not None and ts <= te else None

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(gold: set, pred: set):
    tp = len(gold & pred)
    return tp, len(pred) - tp, len(gold) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


def eval_model(model, docs: list[dict], threshold: float) -> tuple:
    global_tp = global_fp = global_fn = 0
    per_label = defaultdict(lambda: [0, 0, 0])

    for doc in docs:
        tokens = doc["tokenized_text"]
        text, offsets = build_char_offsets(tokens)
        gold = {(e[0], e[1], e[2]) for e in doc["ner"]}

        raw = model.predict_entities(text, QUAERO_LABELS, threshold=threshold)

        pred = set()
        for e in raw:
            result = char_to_token(e["start"], e["end"], offsets)
            if result:
                pred.add((result[0], result[1], e["label"]))

        tp, fp, fn = evaluate(gold, pred)
        global_tp += tp; global_fp += fp; global_fn += fn

        for label in QUAERO_LABELS:
            g = {s for s in gold if s[2] == label}
            p = {s for s in pred if s[2] == label}
            ltp, lfp, lfn = evaluate(g, p)
            per_label[label][0] += ltp
            per_label[label][1] += lfp
            per_label[label][2] += lfn

    return global_tp, global_fp, global_fn, per_label

# ---------------------------------------------------------------------------
# Fine-tune
# ---------------------------------------------------------------------------

def fine_tune(model_id: str, train_docs: list[dict], test_docs: list[dict]) -> tuple[Path, dict]:
    out_dir = OUTPUT_DIR / model_id.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Fine-tuning {model_id} → {out_dir}")
    model = GLiNER.from_pretrained(model_id).to(DEVICE)

    trainer = model.train_model(
        train_dataset=train_docs,
        eval_dataset=test_docs,
        output_dir=str(out_dir),
        eval_strategy="steps",
        eval_steps=100,
        **TRAIN_PARAMS,
    )
    metrics = extract_metrics_from_trainer(trainer)
    model.save_pretrained(str(out_dir))
    print(f"  Saved → {out_dir}")
    return out_dir, metrics

# ---------------------------------------------------------------------------
# Main: fine-tune all models then evaluate at multiple thresholds
# ---------------------------------------------------------------------------

def main(skip_training=False):
    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    train_docs = load_split("train")
    test_docs  = load_split("test")
    print(f"Train: {len(train_docs)} docs  |  Test: {len(test_docs)} docs")

    rows = []

    for model_id in MODELS:
        print(f"\n{'='*60}\nModel: {model_id}")
        out_dir = OUTPUT_DIR / model_id.replace("/", "__")

        metrics = None
        if skip_training and out_dir.exists():
            print(f"  Loading fine-tuned model from {out_dir}")
            model = GLiNER.from_pretrained(str(out_dir)).to(DEVICE)
        else:
            out_dir, metrics = fine_tune(model_id, train_docs, test_docs)
            model = GLiNER.from_pretrained(str(out_dir)).to(DEVICE)

        print(f"\n  Evaluating on train set (F1 @ 0.5 for overfitting check) ...")
        tp_tr, fp_tr, fn_tr, _ = eval_model(model, train_docs, 0.5)
        _, _, train_f1 = prf(tp_tr, fp_tr, fn_tr)
        print(f"    Train F1={train_f1:.3f}")

        print(f"\n  Evaluating on test set at thresholds {THRESHOLDS} ...")
        best_test_f1 = 0.0
        for threshold in THRESHOLDS:
            tp, fp, fn, per_label = eval_model(model, test_docs, threshold)
            p, r, f1 = prf(tp, fp, fn)
            print(f"    threshold={threshold}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

            best_test_f1 = max(best_test_f1, f1)

            row = {
                "model": model_id,
                "threshold": threshold,
                "n_docs": len(test_docs),
                "P": p, "R": r, "F1": f1,
                "TP": tp, "FP": fp, "FN": fn,
            }
            for label in QUAERO_LABELS:
                ltp, lfp, lfn = per_label[label]
                _, _, lf1 = prf(ltp, lfp, lfn)
                row[f"F1_{label}"] = lf1
            rows.append(row)

        if metrics is None:
            metrics = {"train_loss": [], "steps_train": [], "eval_loss": [], "steps_eval": []}
        plot_training(metrics, model_id, train_f1, best_test_f1)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved → {RESULTS_CSV}")

    # Save training params
    from datetime import datetime
    params_row = {"run_date": datetime.now().isoformat(timespec="seconds"),
                  "models": ";".join(MODELS), "device": DEVICE,
                  "n_train": len(train_docs), "n_test": len(test_docs),
                  **TRAIN_PARAMS}
    write_header = not PARAMS_CSV.exists()
    with open(PARAMS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(params_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(params_row)
    print(f"Params saved → {PARAMS_CSV}")

    print("\n=== Best threshold per model ===")
    for model_id in MODELS:
        best = max((r for r in rows if r["model"] == model_id), key=lambda x: x["F1"], default=None)
        if best:
            print(f"  {model_id.split('/')[-1]:<30} F1={best['F1']:.3f}  threshold={best['threshold']}")

# ---------------------------------------------------------------------------
# Inspect: show predictions vs gold for one test document
# ---------------------------------------------------------------------------

def inspect(doc_idx: int = 0):
    test_docs = load_split("test")
    model_id  = MODELS[0]
    threshold = 0.4

    out_dir = OUTPUT_DIR / model_id.replace("/", "__")
    src = str(out_dir) if out_dir.exists() else model_id
    print(f"Loading model from {src} ...")
    model = GLiNER.from_pretrained(src).to(DEVICE)

    doc    = test_docs[doc_idx]
    tokens = doc["tokenized_text"]
    text, offsets = build_char_offsets(tokens)
    print(f"\nText: {text}\n")

    gold = {(e[0], e[1], e[2]) for e in doc["ner"]}
    raw  = model.predict_entities(text, QUAERO_LABELS, threshold=threshold)

    pred = {}
    for e in raw:
        r = char_to_token(e["start"], e["end"], offsets)
        if r:
            pred[(r[0], r[1], e["label"])] = round(e["score"], 3)

    print("=== GOLD ===")
    for ts, te, label in sorted(gold):
        marker = "OK" if (ts, te, label) in pred else "MISSED"
        print(f"  [{label}] {ts}-{te}  '{' '.join(tokens[ts:te+1])}'  {marker}")

    print("\n=== PREDICTIONS ===")
    for (ts, te, label), score in sorted(pred.items()):
        marker = "OK" if (ts, te, label) in gold else "FALSE POS"
        print(f"  [{label}] {ts}-{te}  '{' '.join(tokens[ts:te+1])}'  score={score}  {marker}")

    tp, fp, fn = evaluate(gold, set(pred.keys()))
    p, r, f1 = prf(tp, fp, fn)
    print(f"\nDoc P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "eval":
        main(skip_training=True)
    elif arg == "inspect":
        inspect(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        main()
