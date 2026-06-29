"""
CamemBERT-base + LoRA fine-tuning + evaluation for NER on CANTEMIST35.

Pipeline:
  1. Load train_gliner.json / test_gliner.json (span format)
  2. Convert span annotations to BIO tags + export to {train,test}_bio.json
  3. Fine-tune CamemBERT-base with LoRA (PEFT) via HuggingFace Trainer
  4. Evaluate on test set (token-level seqeval + span-level for GLiNER comparison)
  5. Save results + training plots

Usage:
    pip install transformers datasets seqeval accelerate matplotlib peft
    python camembert-base_ner_eval.py                  # fine-tune + eval
    python camembert-base_ner_eval.py eval             # eval only
    python camembert-base_ner_eval.py inspect 0        # inspect doc 0
"""

import sys
import csv
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "almanach/camembert-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "camembert_finetuned"
RESULTS_CSV = DATA_DIR / "camembert_results.csv"
PARAMS_CSV = DATA_DIR / "camembert_train_params.csv"
PLOTS_DIR = DATA_DIR / "plots"

CANTEMIST_LABELS = [
    "ATCD_geriatriques_et_medicaux_significatifs_pour_la_prise_en_charge",
    "Biomarqueurs_therapeutiques",
    "Evolutivite_en_lien_avec_le_cancer",
    "Histologie_tumorale",
    "Reponse_a_la_chimiotherapie",
    "Signes_physiques",
    "Stade_OMS_ECOG_Karnofsky",
    "Stade_metastatique_avec_localisations",
    "Statut_tabagique",
    "Symptomes",
    "Topographie_du_primitif",
    "Traitement_specifique_du_cancer",
]

BIO_LABELS = ["O"]
for label in CANTEMIST_LABELS:
    BIO_LABELS.append(f"B-{label}")
    BIO_LABELS.append(f"I-{label}")

LABEL2ID = {l: i for i, l in enumerate(BIO_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(BIO_LABELS)}

MAX_LENGTH = 512
SEED = 42


def load_tokenizer(model_path: str):
    """Load tokenizer -- CamemBERT slow tokenizer is broken in transformers >=5.x,
    so fall back to RobertaTokenizerFast which reads tokenizer.json."""
    try:
        return AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    except (ValueError, Exception):
        return PreTrainedTokenizerFast.from_pretrained(model_path, add_prefix_space=True)

LORA_CONFIG = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=16,
    lora_alpha=16,
    lora_dropout=0.25,
    target_modules=["query", "key", "value"],
)

TRAIN_PARAMS = dict(
    num_train_epochs=25,
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    weight_decay=0.05,
    warmup_steps=20,
    lr_scheduler_type="linear",
    max_grad_norm=1.0,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=SEED,
    report_to="none",
    fp16=torch.cuda.is_available(),
)


# ---------------------------------------------------------------------------
# Data conversion: GLiNER span format -> BIO tags
# ---------------------------------------------------------------------------

def spans_to_bio(tokens: list[str], ner_spans: list[list]) -> list[str]:
    """Convert span annotations [start, end, label] to BIO tag sequence."""
    tags = ["O"] * len(tokens)
    for span in ner_spans:
        start, end, label = int(span[0]), int(span[1]), span[2]
        if start < 0 or end >= len(tokens) or start > end:
            continue
        tags[start] = f"B-{label}"
        for i in range(start + 1, end + 1):
            tags[i] = f"I-{label}"
    return tags


def load_json_as_bio(split: str) -> dict:
    """Load a GLiNER-format JSON and return {tokens, bio_tags} lists."""
    path = DATA_DIR / f"{split}_gliner.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    docs = json.loads(path.read_text(encoding="utf-8"))

    all_tokens, all_tags = [], []
    bio_docs = []
    for doc in docs:
        tokens = doc["tokenized_text"]
        bio = spans_to_bio(tokens, doc["ner"])
        all_tokens.append(tokens)
        all_tags.append(bio)
        bio_docs.append({"tokens": tokens, "bio_tags": bio})

    out_path = DATA_DIR / f"{split}_bio.json"
    out_path.write_text(json.dumps(bio_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  BIO export saved -> {out_path}  ({len(bio_docs)} docs)")

    return {"tokens": all_tokens, "bio_tags": all_tags}


# ---------------------------------------------------------------------------
# HuggingFace dataset preparation
# ---------------------------------------------------------------------------

def build_datasets(tokenizer):
    """Load train/test, tokenize with subword alignment, return DatasetDict."""
    train_data = load_json_as_bio("train")
    test_data = load_json_as_bio("test")

    ds = DatasetDict({
        "train": Dataset.from_dict(train_data),
        "test": Dataset.from_dict(test_data),
    })

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
        )
        all_labels = []
        for i, tags in enumerate(examples["bio_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            labels = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    labels.append(-100)
                elif word_id != prev_word_id:
                    labels.append(LABEL2ID.get(tags[word_id], 0))
                else:
                    label = tags[word_id]
                    if label.startswith("B-"):
                        labels.append(LABEL2ID[label.replace("B-", "I-")])
                    else:
                        labels.append(LABEL2ID.get(label, 0))
                prev_word_id = word_id
            all_labels.append(labels)
        tokenized["labels"] = all_labels
        return tokenized

    encoded = ds.map(
        tokenize_and_align,
        batched=True,
        batch_size=64,
        remove_columns=["tokens", "bio_tags"],
    )
    return encoded, train_data, test_data


# ---------------------------------------------------------------------------
# Metrics for Trainer (token-level seqeval)
# ---------------------------------------------------------------------------

def make_compute_metrics():
    def compute_metrics(eval_preds):
        logits, label_ids = eval_preds
        preds = np.argmax(logits, axis=-1)
        y_true, y_pred = [], []
        for pred_seq, label_seq in zip(preds, label_ids):
            true_tags, pred_tags = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                true_tags.append(ID2LABEL[l])
                pred_tags.append(ID2LABEL[p])
            y_true.append(true_tags)
            y_pred.append(pred_tags)
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }
    return compute_metrics


# ---------------------------------------------------------------------------
# Span-level evaluation (comparable to GLiNER eval)
# ---------------------------------------------------------------------------

def bio_to_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    """Convert BIO tags to set of (start, end, label) spans."""
    spans = set()
    start, label = None, None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                spans.add((start, i - 1, label))
            label = tag[2:]
            start = i
        elif tag.startswith("I-"):
            if start is None or tag[2:] != label:
                if start is not None:
                    spans.add((start, i - 1, label))
                label = tag[2:]
                start = i
        else:
            if start is not None:
                spans.add((start, i - 1, label))
                start, label = None, None
    if start is not None:
        spans.add((start, len(tags) - 1, label))
    return spans


def span_evaluate(model, tokenizer, raw_data: dict) -> dict:
    """Run span-level evaluation matching the GLiNER eval format."""
    model.eval()
    all_tokens = raw_data["tokens"]
    all_tags = raw_data["bio_tags"]

    global_tp = global_fp = global_fn = 0
    per_label = defaultdict(lambda: [0, 0, 0])

    for tokens, gold_bio in zip(all_tokens, all_tags):
        gold_spans = bio_to_spans(gold_bio)

        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        word_ids = inputs.word_ids(batch_index=0)
        pred_bio = ["O"] * len(tokens)
        prev_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                pred_bio[word_id] = ID2LABEL[pred_ids[idx]]
            prev_word_id = word_id

        pred_spans = bio_to_spans(pred_bio)

        tp = len(gold_spans & pred_spans)
        fp = len(pred_spans - gold_spans)
        fn = len(gold_spans - pred_spans)
        global_tp += tp
        global_fp += fp
        global_fn += fn

        for label in CANTEMIST_LABELS:
            g = {s for s in gold_spans if s[2] == label}
            p = {s for s in pred_spans if s[2] == label}
            ltp = len(g & p)
            per_label[label][0] += ltp
            per_label[label][1] += len(p - g)
            per_label[label][2] += len(g - p)

    return {
        "tp": global_tp, "fp": global_fp, "fn": global_fn,
        "per_label": per_label,
    }


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f1, 4)


# ---------------------------------------------------------------------------
# Training plots
# ---------------------------------------------------------------------------

def plot_training(log_history, train_f1, test_f1):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loss_by_epoch, eval_loss_by_epoch = [], []
    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        if "loss" in entry and "eval_loss" not in entry:
            train_loss_by_epoch.append((epoch, float(entry["loss"])))
        if "eval_loss" in entry:
            eval_loss_by_epoch.append((epoch, float(entry["eval_loss"])))

    train_epochs = [e for e, _ in train_loss_by_epoch]
    train_losses = [l for _, l in train_loss_by_epoch]
    eval_epochs = [e for e, _ in eval_loss_by_epoch]
    eval_losses = [l for _, l in eval_loss_by_epoch]

    # --- Diagnosis based on loss curves ---
    loss_diagnosis = ""
    if train_losses and eval_losses:
        min_n = min(len(train_losses), len(eval_losses))
        final_train = train_losses[min_n - 1]
        final_eval = eval_losses[min_n - 1]
        best_eval = min(eval_losses)
        best_eval_epoch = eval_epochs[eval_losses.index(best_eval)]

        if final_eval > eval_losses[0] and final_train < train_losses[0] * 0.5:
            loss_diagnosis = f"OVERFITTING (eval loss rises from epoch {best_eval_epoch:.0f})"
            diag_color = "red"
        elif final_train > train_losses[0] * 0.8 and final_eval > eval_losses[0] * 0.8:
            loss_diagnosis = "UNDERFITTING (both losses remain high)"
            diag_color = "red"
        elif final_eval > best_eval * 1.1:
            loss_diagnosis = f"MILD OVERFITTING (best eval @ epoch {best_eval_epoch:.0f})"
            diag_color = "orange"
        else:
            loss_diagnosis = "GOOD FIT (both losses converge)"
            diag_color = "green"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Loss curve (epoch axis) ---
    ax = axes[0]
    if train_losses:
        ax.plot(train_epochs, train_losses, "o-", label="Train loss", linewidth=1.5, markersize=4)
    if eval_losses:
        ax.plot(eval_epochs, eval_losses, "s--", label="Test loss", linewidth=1.5, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss -- CamemBERT (CANTEMIST35)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if loss_diagnosis:
        ax.text(
            0.5, 0.02, loss_diagnosis,
            transform=ax.transAxes, ha="center", fontsize=11, fontweight="bold",
            color=diag_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=diag_color, alpha=0.8),
        )

    # --- F1 bar chart ---
    ax = axes[1]
    labels_bar = ["Train F1", "Test F1"]
    values = [train_f1, test_f1]
    colors = ["#2196F3", "#FF9800"]
    bars = ax.bar(labels_bar, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", fontweight="bold",
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("Overfitting check -- CamemBERT (CANTEMIST35)")
    gap = train_f1 - test_f1
    verdict = "OVERFITTING" if gap > 0.05 else "OK" if gap > 0.02 else "Good fit"
    ax.text(
        0.5, 0.92, f"F1 gap = {gap:+.3f} -> {verdict}",
        transform=ax.transAxes, ha="center", fontsize=12,
        color="red" if gap > 0.05 else "orange" if gap > 0.02 else "green",
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    # Auto-increment run number
    existing = sorted(PLOTS_DIR.glob("CamemBERT__training_run*.png"))
    run_num = len(existing) + 1
    out_path = PLOTS_DIR / f"CamemBERT__training_run{run_num}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Fine-tune
# ---------------------------------------------------------------------------

def fine_tune():
    print(f"Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))

    tokenizer = load_tokenizer(MODEL_ID)
    encoded_ds, train_data, test_data = build_datasets(tokenizer)

    print(f"Train: {len(train_data['tokens'])} docs  |  Test: {len(test_data['tokens'])} docs")
    print(f"BIO labels ({len(BIO_LABELS)}): {BIO_LABELS}")

    base_model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(BIO_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = get_peft_model(base_model, LORA_CONFIG).to(DEVICE)
    model.print_trainable_parameters()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        **TRAIN_PARAMS,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_ds["train"],
        eval_dataset=encoded_ds["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n{'='*60}")
    print(f"Fine-tuning {MODEL_ID} on CANTEMIST35")
    print(f"{'='*60}")
    trainer.train()

    best_dir = OUTPUT_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    merged = model.merge_and_unload()
    print(f"  Best LoRA adapter saved -> {best_dir}")

    return merged, tokenizer, trainer.state.log_history, train_data, test_data


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_model(model, tokenizer, train_data, test_data, log_history=None):
    print(f"\n  Span-level evaluation on train set (overfitting check) ...")
    train_result = span_evaluate(model, tokenizer, train_data)
    _, _, train_f1 = prf(train_result["tp"], train_result["fp"], train_result["fn"])
    print(f"    Train F1 = {train_f1:.3f}")

    print(f"\n  Span-level evaluation on test set ...")
    test_result = span_evaluate(model, tokenizer, test_data)
    tp, fp, fn = test_result["tp"], test_result["fp"], test_result["fn"]
    p, r, f1 = prf(tp, fp, fn)
    print(f"    Test  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    print(f"\n  Per-label F1:")
    for label in CANTEMIST_LABELS:
        ltp, lfp, lfn = test_result["per_label"][label]
        _, _, lf1 = prf(ltp, lfp, lfn)
        print(f"    {label:<65}  F1={lf1:.3f}  (TP={ltp} FP={lfp} FN={lfn})")

    row = {
        "model": MODEL_ID,
        "threshold": "n/a",
        "n_docs": len(test_data["tokens"]),
        "P": p, "R": r, "F1": f1,
        "TP": tp, "FP": fp, "FN": fn,
    }
    for label in CANTEMIST_LABELS:
        ltp, lfp, lfn = test_result["per_label"][label]
        _, _, lf1 = prf(ltp, lfp, lfn)
        row[f"F1_{label}"] = lf1

    fieldnames = list(row.keys())
    existing_csv = sorted(RESULTS_CSV.parent.glob(RESULTS_CSV.stem + "_run*.csv"))
    run_num = len(existing_csv) + 1
    results_path = RESULTS_CSV.parent / f"{RESULTS_CSV.stem}_run{run_num}.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"\n  Results saved -> {results_path}")

    if log_history:
        plot_training(log_history, train_f1, f1)

    params_row = {
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_ID,
        "device": DEVICE,
        "n_train": len(train_data["tokens"]),
        "n_test": len(test_data["tokens"]),
        "n_labels": len(BIO_LABELS),
        **TRAIN_PARAMS,
    }
    write_header = not PARAMS_CSV.exists()
    with open(PARAMS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(params_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(params_row)
    print(f"  Params saved -> {PARAMS_CSV}")

    # seqeval detailed report on test set
    print(f"\n  seqeval classification report (token-level):")
    model.eval()
    y_true_all, y_pred_all = [], []
    for tokens, gold_bio in zip(test_data["tokens"], test_data["bio_tags"]):
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        word_ids = inputs.word_ids(batch_index=0)
        true_seq, pred_seq = [], []
        prev_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != prev_word_id:
                true_seq.append(gold_bio[word_id])
                pred_seq.append(ID2LABEL[pred_ids[idx]])
            prev_word_id = word_id
        y_true_all.append(true_seq)
        y_pred_all.append(pred_seq)
    print(classification_report(y_true_all, y_pred_all, digits=4))

    return f1


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

def inspect(doc_idx: int = 0):
    best_dir = OUTPUT_DIR / "best"
    print(f"Loading LoRA model from {best_dir} ...")
    tokenizer = load_tokenizer(str(best_dir))
    base_model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID, num_labels=len(BIO_LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model = PeftModel.from_pretrained(base_model, str(best_dir)).merge_and_unload().to(DEVICE)
    model.eval()

    test_data = load_json_as_bio("test")
    tokens = test_data["tokens"][doc_idx]
    gold_bio = test_data["bio_tags"][doc_idx]
    gold_spans = bio_to_spans(gold_bio)

    text = " ".join(tokens)
    print(f"\nText: {text}\n")

    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    word_ids = inputs.word_ids(batch_index=0)
    pred_bio = ["O"] * len(tokens)
    prev_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            pred_bio[word_id] = ID2LABEL[pred_ids[idx]]
        prev_word_id = word_id

    pred_spans = bio_to_spans(pred_bio)

    print("=== GOLD ===")
    for ts, te, label in sorted(gold_spans):
        marker = "OK" if (ts, te, label) in pred_spans else "MISSED"
        print(f"  [{label}] {ts}-{te}  '{' '.join(tokens[ts:te+1])}'  {marker}")

    print("\n=== PREDICTIONS ===")
    for ts, te, label in sorted(pred_spans):
        marker = "OK" if (ts, te, label) in gold_spans else "FALSE POS"
        print(f"  [{label}] {ts}-{te}  '{' '.join(tokens[ts:te+1])}'  {marker}")

    tp = len(gold_spans & pred_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)
    p, r, f1 = prf(tp, fp, fn)
    print(f"\nDoc P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(skip_training=False):
    if skip_training:
        best_dir = OUTPUT_DIR / "best"
        if not best_dir.exists():
            print(f"No saved model found at {best_dir}, run training first.")
            return
        print(f"Loading LoRA model from {best_dir}")
        tokenizer = load_tokenizer(str(best_dir))
        base_model = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID, num_labels=len(BIO_LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        )
        model = PeftModel.from_pretrained(base_model, str(best_dir)).merge_and_unload().to(DEVICE)
        train_data = load_json_as_bio("train")
        test_data = load_json_as_bio("test")
        evaluate_model(model, tokenizer, train_data, test_data)
    else:
        model, tokenizer, log_history, train_data, test_data = fine_tune()
        evaluate_model(model, tokenizer, train_data, test_data, log_history)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "eval":
        main(skip_training=True)
    elif arg == "inspect":
        inspect(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    else:
        main()
