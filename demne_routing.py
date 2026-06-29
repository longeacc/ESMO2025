"""
DEMNE Statistical Routing Pipeline
===================================
Compare Rules, TBM, LLM NER methods per entity via Friedman + post-hoc tests.
Determines optimal method per entity following DEMNE frugality principle.

Usage:
    python demne_routing.py --corpus maccrobat
    python demne_routing.py --corpus quaero
    python demne_routing.py --corpus maccrobat --skip-llm
"""

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demne")

# =========================================================================
#  Labels
# =========================================================================

MACCROBAT_LABELS = [
    "Activity", "Administration", "Age", "Area",
    "Biological_attribute", "Biological_structure",
    "Clinical_event", "Color", "Coreference",
    "Date", "Detailed_description", "Diagnostic_procedure",
    "Disease_disorder", "Distance", "Dosage", "Duration",
    "Family_history", "Frequency",
    "Height", "History",
    "Lab_value",
    "Mass", "Medication",
    "Nonbiological_location",
    "Occupation", "Other_entity", "Other_event", "Outcome",
    "Personal_background",
    "Qualitative_concept", "Quantitative_concept",
    "Severity", "Sex", "Shape", "Sign_symptom", "Subject",
    "Texture", "Therapeutic_procedure", "Time",
    "Volume", "Weight",
]

QUAERO_LABELS = [
    "DISO", "PROC", "ANAT", "CHEM", "DEVI",
    "LIVB", "PHYS", "PHEN", "GEOG", "OBJC",
]

# =========================================================================
#  Corpus configuration
# =========================================================================

CORPUS_CFG = {
    "maccrobat": dict(
        gold_dir=ROOT / "ESMO2025_MACCROBAT2020" / "MACCROBAT2020_test",
        rules_dir=ROOT / "ESMO2025_MACCROBAT2020" / "Rules" / "src" / "MACCROBAT2020_pred_rules",
        tbm_model_dir=ROOT / "ESMO2025_MACCROBAT2020" / "TBM_evaluation" / "pubmedbert_finetuned" / "best",
        tbm_base_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        tbm_cache=ROOT / "ESMO2025_MACCROBAT2020" / "TBM_evaluation" / "per_doc_scores_tbm.json",
        llm_cache=ROOT / "ESMO2025_MACCROBAT2020" / "LLM_evaluation" / "per_doc_scores_llm.json",
        llm_model="mistral-medium-latest",
        llm_data_dir=ROOT / "ESMO2025_MACCROBAT2020" / "TBM_evaluation",
        labels=MACCROBAT_LABELS,
        best_tbm="PubMedBERT",
        best_llm="mistral-medium-latest",
        spacy_model="en_core_web_sm",
        spacy_disable=["ner", "lemmatizer"],
        output_dir=ROOT / "ESMO2025_MACCROBAT2020",
        add_prefix_space=False,
    ),
    "quaero": dict(
        gold_dir=ROOT / "ESMO2025_QUERO_French_Med" / "QUAERO_FrenchMed" / "corpus" / "test" / "MEDLINE",
        rules_dir=ROOT / "ESMO2025_QUERO_French_Med" / "Rules" / "src" / "quaero_pred_rules" / "test",
        tbm_model_dir=ROOT / "ESMO2025_QUERO_French_Med" / "TBM_evaluation" / "drbert_finetuned" / "best",
        tbm_base_model="almanach/camembert-base",
        tbm_cache=ROOT / "ESMO2025_QUERO_French_Med" / "TBM_evaluation" / "per_doc_scores_tbm.json",
        llm_cache=ROOT / "ESMO2025_QUERO_French_Med" / "LLM_evaluation" / "per_doc_scores_llm.json",
        llm_model="mistral-large-latest",
        llm_data_dir=ROOT / "ESMO2025_QUERO_French_Med" / "TBM_evaluation",
        labels=QUAERO_LABELS,
        best_tbm="DrBERT-7GB",
        best_llm="mistral-large-latest",
        spacy_model="fr_core_news_sm",
        spacy_disable=["parser", "ner", "lemmatizer"],
        output_dir=ROOT / "ESMO2025_QUERO_French_Med",
        add_prefix_space=True,
        quaero_train_dir=ROOT / "ESMO2025_QUERO_French_Med" / "QUAERO_FrenchMed" / "corpus" / "train" / "MEDLINE",
        quaero_annotator=ROOT / "ESMO2025_QUERO_French_Med" / "Rules" / "src" / "quaero_brat_annotator.py",
    ),
}

# =========================================================================
#  BRAT parsing
# =========================================================================

def parse_brat_ann(ann_path: Path, valid_labels: set | None = None) -> set:
    """Parse BRAT .ann → set of (begin, end, label). Ignores non-T lines."""
    spans = set()
    if not ann_path.exists():
        return spans
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        meta = parts[1].split()
        label = meta[0]
        if valid_labels and label not in valid_labels:
            continue
        nums = []
        for tok in meta[1:]:
            for n in tok.split(";"):
                try:
                    nums.append(int(n))
                except ValueError:
                    pass
        if len(nums) >= 2:
            spans.add((nums[0], nums[-1], label))
    return spans


def get_doc_ids(directory: Path) -> list[str]:
    """Return sorted list of doc IDs (stem of .ann files) in directory."""
    return sorted(p.stem for p in directory.glob("*.ann"))

# =========================================================================
#  F1 computation helpers
# =========================================================================

def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def per_entity_tp_fp_fn(
    gold: set, pred: set, labels: list[str]
) -> dict[str, dict[str, int]]:
    """Compute per-entity TP/FP/FN from two sets of (begin, end, label)."""
    result = {}
    for label in labels:
        g = {(b, e) for b, e, l in gold if l == label}
        p = {(b, e) for b, e, l in pred if l == label}
        tp = len(g & p)
        result[label] = {"TP": tp, "FP": len(p) - tp, "FN": len(g) - tp}
    return result

# =========================================================================
#  BIO ↔ spans conversion
# =========================================================================

def bio_to_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    """Convert BIO tag sequence → set of (token_start, token_end, label)."""
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

# =========================================================================
#  Rules: compute per-doc scores from BRAT prediction files
# =========================================================================

def compute_rules_scores(
    gold_dir: Path,
    rules_dir: Path,
    doc_ids: list[str],
    labels: list[str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Compare gold vs Rules BRAT .ann files → {doc_id: {entity: {TP,FP,FN}}}."""
    valid = set(labels)
    scores = {}
    for doc_id in doc_ids:
        gold = parse_brat_ann(gold_dir / f"{doc_id}.ann", valid)
        pred = parse_brat_ann(rules_dir / f"{doc_id}.ann", valid)
        scores[doc_id] = per_entity_tp_fp_fn(gold, pred, labels)
    return scores


def compute_rules_scores_from_annotator(
    gold_dir: Path,
    doc_ids: list[str],
    doc_txt_paths: dict[str, Path],
    labels: list[str],
    annotator_path: Path,
) -> dict[str, dict[str, dict[str, int]]]:
    """Run Rules annotator on raw text and compare with gold BRAT."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("annotator", str(annotator_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    annotate_fn = mod.annotate_text

    valid = set(labels)
    scores = {}
    for doc_id in doc_ids:
        gold = parse_brat_ann(gold_dir / f"{doc_id}.ann", valid)
        txt_path = doc_txt_paths.get(doc_id) or (gold_dir / f"{doc_id}.txt")
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        raw_spans = annotate_fn(text)
        pred = set()
        for span_text, start, end, lbl in raw_spans:
            if lbl in valid:
                pred.add((start, end, lbl))
        scores[doc_id] = per_entity_tp_fp_fn(gold, pred, labels)
    return scores

# =========================================================================
#  TBM: generate per-doc scores via model inference
# =========================================================================

def generate_tbm_scores(
    cfg: dict,
    doc_ids: list[str],
    doc_txt_paths: dict[str, Path] | None = None,
) -> dict[str, dict[str, dict[str, int]]]:
    """Load saved TBM model, run inference on test docs, compare with gold."""
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        PreTrainedTokenizerFast = None
    from peft import PeftModel
    import spacy

    model_dir = cfg["tbm_model_dir"]
    base_model_id = cfg["tbm_base_model"]
    gold_dir = cfg["gold_dir"]
    valid = set(cfg["labels"])

    config_data = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in config_data["id2label"].items()}
    label2id = {v: int(k) for k, v in config_data["id2label"].items()}
    num_labels = len(id2label)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("TBM device: %s", device)

    if cfg.get("add_prefix_space"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir), add_prefix_space=True
            )
        except Exception:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                str(model_dir), add_prefix_space=True
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    base_model = AutoModelForTokenClassification.from_pretrained(
        base_model_id, num_labels=num_labels, id2label=id2label, label2id=label2id,
    )
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    model = model.merge_and_unload().to(device)
    model.eval()
    log.info("TBM model loaded (%s + LoRA)", base_model_id)

    nlp = spacy.load(cfg["spacy_model"], disable=cfg["spacy_disable"])
    has_sents = "parser" not in cfg["spacy_disable"]
    if not has_sents:
        nlp.add_pipe("sentencizer")

    scores = {}
    total = len(doc_ids)
    for idx, doc_id in enumerate(doc_ids):
        txt_path = (doc_txt_paths or {}).get(doc_id) or (gold_dir / f"{doc_id}.txt")
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        doc = nlp(text)

        pred_char_spans = set()
        sents = list(doc.sents) if has_sents else [doc[:]]
        for sent in sents:
            tokens = [tok.text for tok in sent]
            if not tokens:
                continue

            inputs = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = model(**inputs).logits
            pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()

            word_ids = inputs.word_ids(0)
            pred_bio = ["O"] * len(tokens)
            prev_wid = None
            for i, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid != prev_wid:
                    pred_bio[wid] = id2label.get(pred_ids[i], "O")
                prev_wid = wid

            token_spans = bio_to_spans(pred_bio)
            for ts, te, lbl in token_spans:
                if lbl not in valid:
                    continue
                char_start = sent[ts].idx
                char_end = sent[te].idx + len(sent[te].text)
                pred_char_spans.add((char_start, char_end, lbl))

        gold = parse_brat_ann(gold_dir / f"{doc_id}.ann", valid)
        scores[doc_id] = per_entity_tp_fp_fn(gold, pred_char_spans, cfg["labels"])

        if (idx + 1) % 50 == 0 or idx == 0 or idx == total - 1:
            log.info("  TBM inference: %d/%d docs", idx + 1, total)

    return scores

# =========================================================================
#  QUAERO: reproduce the GLiNER re-split to find test doc IDs
# =========================================================================

def _iterative_stratified_split_assignment(
    label_matrix: np.ndarray, train_ratio: float, seed: int
) -> np.ndarray:
    """Return assignment array (0=train, 1=test) for each doc."""
    rng = np.random.RandomState(seed)
    n, n_labels = label_matrix.shape

    desired = np.zeros((2, n_labels), dtype=float)
    totals = label_matrix.sum(axis=0).astype(float)
    desired[0] = totals * train_ratio
    desired[1] = totals * (1 - train_ratio)

    assignment = np.full(n, -1, dtype=int)
    label_order = np.argsort(totals)

    for l_idx in label_order:
        candidates = np.where((label_matrix[:, l_idx] == 1) & (assignment == -1))[0]
        rng.shuffle(candidates)
        for doc_idx in candidates:
            fold = 0 if desired[0, l_idx] >= desired[1, l_idx] else 1
            assignment[doc_idx] = fold
            desired[fold] -= label_matrix[doc_idx]

    unassigned = np.where(assignment == -1)[0]
    for doc_idx in unassigned:
        assignment[doc_idx] = 0 if rng.random() < train_ratio else 1

    return assignment


def reproduce_quaero_resplit(cfg: dict) -> tuple[list[str], dict[str, Path]]:
    """
    Reproduce the QUAERO GLiNER 75/25 re-split (seed=42) to identify
    test doc IDs and their source .txt paths.

    Returns (test_doc_ids, {doc_id: txt_path}).
    """
    all_labels = sorted(cfg["labels"])
    valid = set(cfg["labels"])

    corpus_dirs = [cfg["quaero_train_dir"], cfg["gold_dir"]]
    doc_paths = []
    label_vectors = []

    for corpus_dir in corpus_dirs:
        if not corpus_dir.exists():
            log.warning("QUAERO dir not found: %s", corpus_dir)
            continue
        for txt_path in sorted(corpus_dir.glob("*.txt")):
            ann_path = txt_path.with_suffix(".ann")
            if not ann_path.exists():
                continue
            entities = parse_brat_ann(ann_path, valid)
            present = {l for _, _, l in entities}
            vec = np.array([1 if l in present else 0 for l in all_labels])
            doc_paths.append(txt_path)
            label_vectors.append(vec)

    label_matrix = np.array(label_vectors)
    assignment = _iterative_stratified_split_assignment(label_matrix, 0.75, 42)

    test_ids = []
    test_txt_paths = {}
    for i, path in enumerate(doc_paths):
        if assignment[i] == 1:
            doc_id = path.stem
            test_ids.append(doc_id)
            test_txt_paths[doc_id] = path

    log.info(
        "QUAERO re-split reproduced: %d total → %d train, %d test",
        len(doc_paths), int((assignment == 0).sum()), len(test_ids),
    )
    return test_ids, test_txt_paths

# =========================================================================
#  Load or generate per-doc scores for each method
# =========================================================================

def load_cached_scores(path: Path) -> dict | None:
    if path.exists():
        log.info("Loading cached scores: %s", path.name)
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cached_scores(scores: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Cached scores saved: %s", path)


def get_all_scores(
    cfg: dict,
    corpus: str,
    skip_llm: bool = False,
) -> tuple[dict, dict | None, dict | None, list[str]]:
    """
    Return (rules_scores, tbm_scores, llm_scores, doc_ids).
    Each *_scores: {doc_id: {entity: {TP, FP, FN}}}.
    """
    labels = cfg["labels"]

    # --- Determine test doc IDs and paths ---
    if corpus == "quaero" and "quaero_train_dir" in cfg:
        doc_ids, doc_txt_paths = reproduce_quaero_resplit(cfg)
        gold_ann_dir = None
        for doc_id in doc_ids:
            txt_path = doc_txt_paths[doc_id]
            ann_path = txt_path.with_suffix(".ann")
            if ann_path.exists():
                gold_ann_dir = ann_path.parent
                break
    else:
        doc_ids = get_doc_ids(cfg["gold_dir"])
        doc_txt_paths = None
        gold_ann_dir = cfg["gold_dir"]

    log.info("Test documents: %d", len(doc_ids))

    # --- Rules ---
    if corpus == "quaero":
        existing_rules = set(get_doc_ids(cfg["rules_dir"]))
        docs_with_rules = [d for d in doc_ids if d in existing_rules]
        docs_without_rules = [d for d in doc_ids if d not in existing_rules]

        rules_scores = {}
        if docs_with_rules:
            r = compute_rules_scores(
                gold_dir=cfg["gold_dir"] if not doc_txt_paths
                else Path(doc_txt_paths[docs_with_rules[0]]).parent,
                rules_dir=cfg["rules_dir"],
                doc_ids=docs_with_rules,
                labels=labels,
            )
            rules_scores.update(r)

        if docs_without_rules and cfg.get("quaero_annotator"):
            log.info(
                "Generating Rules predictions for %d docs not in original test...",
                len(docs_without_rules),
            )
            for doc_id in docs_without_rules:
                txt_p = doc_txt_paths[doc_id]
                gold_ann = txt_p.with_suffix(".ann")
                gold_spans = parse_brat_ann(gold_ann, set(labels))
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "q_ann", str(cfg["quaero_annotator"])
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                text = txt_p.read_text(encoding="utf-8", errors="ignore")
                raw = mod.annotate_text(text)
                pred = {(s, e, l) for _, s, e, l in raw if l in set(labels)}
                rules_scores[doc_id] = per_entity_tp_fp_fn(
                    gold_spans, pred, labels
                )
        elif docs_without_rules:
            log.warning(
                "%d docs missing Rules predictions (no annotator found)",
                len(docs_without_rules),
            )
    else:
        rules_scores = compute_rules_scores(
            cfg["gold_dir"], cfg["rules_dir"], doc_ids, labels
        )

    log.info("Rules scores: %d docs", len(rules_scores))

    # --- TBM ---
    tbm_scores = load_cached_scores(cfg["tbm_cache"])
    if tbm_scores is None:
        if cfg["tbm_model_dir"].exists():
            log.info("Generating TBM scores via model inference...")
            tbm_scores = generate_tbm_scores(cfg, doc_ids, doc_txt_paths)
            save_cached_scores(tbm_scores, cfg["tbm_cache"])
        else:
            log.warning(
                "TBM model not found at %s — skipping TBM", cfg["tbm_model_dir"]
            )

    if tbm_scores:
        log.info("TBM scores: %d docs", len(tbm_scores))

    # --- LLM ---
    llm_scores = None
    if not skip_llm:
        llm_scores = load_cached_scores(cfg["llm_cache"])
        if llm_scores is None:
            log.warning(
                "LLM per-doc scores not found at %s. "
                "Generate them by modifying llm_ner_zero_shot.py to export "
                "per_doc_scores_llm.json, then re-run the LLM evaluation.",
                cfg["llm_cache"],
            )
    else:
        log.info("LLM skipped (--skip-llm)")

    # --- Align doc IDs across methods ---
    available_ids = set(rules_scores.keys())
    if tbm_scores:
        available_ids &= set(tbm_scores.keys())
    if llm_scores:
        available_ids &= set(llm_scores.keys())

    aligned_ids = sorted(available_ids)
    if len(aligned_ids) < len(doc_ids):
        log.info(
            "Aligned doc IDs: %d (from %d total)", len(aligned_ids), len(doc_ids)
        )

    return rules_scores, tbm_scores, llm_scores, aligned_ids

# =========================================================================
#  Build per-(entity, doc) F1 score matrix
# =========================================================================

def build_score_matrix(
    rules_scores: dict,
    tbm_scores: dict | None,
    llm_scores: dict | None,
    labels: list[str],
    doc_ids: list[str],
) -> pd.DataFrame:
    """
    Build DataFrame with columns: entity, doc_id, F1_rules, F1_TBM, F1_LLM.
    Excludes (entity, doc) pairs where entity is absent from BOTH gold AND all preds.
    """
    rows = []
    excluded_counts = defaultdict(int)

    methods = {"rules": rules_scores}
    if tbm_scores:
        methods["TBM"] = tbm_scores
    if llm_scores:
        methods["LLM"] = llm_scores

    for entity in labels:
        for doc_id in doc_ids:
            all_zero = True
            row = {"entity": entity, "doc_id": doc_id}

            for method_name, method_scores in methods.items():
                s = method_scores.get(doc_id, {}).get(entity, {"TP": 0, "FP": 0, "FN": 0})
                tp, fp, fn = s["TP"], s["FP"], s["FN"]
                _, _, f1 = compute_prf(tp, fp, fn)
                col = f"F1_{method_name}"
                row[col] = f1
                if tp + fp + fn > 0:
                    all_zero = False

            if all_zero:
                excluded_counts[entity] += 1
                continue

            rows.append(row)

    df = pd.DataFrame(rows)

    for entity, count in sorted(excluded_counts.items(), key=lambda x: -x[1]):
        log.info(
            "  %s: %d double-absent (entity, doc) pairs excluded",
            entity, count,
        )

    return df

# =========================================================================
#  Statistical analysis: Friedman + post-hoc
# =========================================================================

FRUGALITY_ORDER = {"rules": 0, "TBM": 1, "LLM": 2}


def _frugality_tiebreak(methods: list[str]) -> str:
    return min(methods, key=lambda m: FRUGALITY_ORDER.get(m, 99))


def run_entity_analysis(
    df_entity: pd.DataFrame,
    entity: str,
    method_cols: list[str],
    alpha: float = 0.05,
) -> dict:
    """Run Friedman (3 methods) or Wilcoxon (2 methods) for one entity."""
    n_docs = len(df_entity)
    result = {
        "entity": entity,
        "n_docs": n_docs,
        "n_excluded": 0,
    }

    method_names = [c.replace("F1_", "") for c in method_cols]
    for c in method_cols:
        mn = c.replace("F1_", "")
        result[f"F1_{mn}_mean"] = df_entity[c].mean()

    if n_docs < 10:
        result["p_friedman"] = np.nan
        result["decision"] = _frugality_tiebreak(method_names)
        result["test_post_hoc"] = "n/a"
        result["p_posthoc_best"] = np.nan
        result["notes"] = f"Excluded: n_docs={n_docs} < 10"
        return result

    sparsity = (df_entity[method_cols].sum(axis=1) == 0).mean()
    result["sparsity"] = round(sparsity, 4)

    arrays = [df_entity[c].values for c in method_cols]

    # --- Friedman (3 methods) or Wilcoxon (2 methods) ---
    if len(method_cols) == 3:
        try:
            stat, p_fried = stats.friedmanchisquare(*arrays)
        except Exception:
            p_fried = 1.0
        result["p_friedman"] = round(p_fried, 6)

        if p_fried > alpha:
            result["decision"] = _frugality_tiebreak(method_names)
            result["test_post_hoc"] = "n/a"
            result["p_posthoc_best"] = np.nan
            result["notes"] = "Non significatif — méthode la plus frugale retenue"
            return result

        # Post-hoc pairwise
        alpha_corrected = alpha / 3
        pairs = [(0, 1), (0, 2), (1, 2)]
        pair_results = []

        for i, j in pairs:
            diff = arrays[i] - arrays[j]
            try:
                _, p_shapiro = stats.shapiro(diff)
            except Exception:
                p_shapiro = 0.0

            if p_shapiro > alpha:
                test_name = "t-test"
                try:
                    _, p_pair = stats.ttest_rel(arrays[i], arrays[j])
                except Exception:
                    p_pair = 1.0
            else:
                test_name = "wilcoxon"
                try:
                    _, p_pair = stats.wilcoxon(arrays[i], arrays[j])
                except Exception:
                    p_pair = 1.0

            mean_i = arrays[i].mean()
            mean_j = arrays[j].mean()
            winner = method_names[i] if mean_i > mean_j else method_names[j]

            pair_results.append({
                "pair": f"{method_names[i]}↔{method_names[j]}",
                "test": test_name,
                "p": round(p_pair, 6),
                "significant": p_pair <= alpha_corrected,
                "winner": winner,
                "p_shapiro": round(p_shapiro, 6),
            })

        sig_wins = defaultdict(int)
        for pr in pair_results:
            if pr["significant"]:
                sig_wins[pr["winner"]] += 1

        if sig_wins:
            best = max(sig_wins, key=sig_wins.get)
            if sig_wins[best] == max(sig_wins.values()) and len(
                [m for m, c in sig_wins.items() if c == sig_wins[best]]
            ) > 1:
                candidates = [m for m, c in sig_wins.items() if c == sig_wins[best]]
                best = _frugality_tiebreak(candidates)
            result["decision"] = best
        else:
            result["decision"] = _frugality_tiebreak(method_names)

        result["test_post_hoc"] = "; ".join(
            f"{pr['pair']}:{pr['test']}(p={pr['p']:.4f})"
            for pr in pair_results
        )
        best_pair = min(pair_results, key=lambda pr: pr["p"])
        result["p_posthoc_best"] = best_pair["p"]
        result["notes"] = "; ".join(
            f"Shapiro {pr['pair']}: p={pr['p_shapiro']:.4f}"
            for pr in pair_results
        )

    elif len(method_cols) == 2:
        diff = arrays[0] - arrays[1]
        try:
            _, p_shapiro = stats.shapiro(diff)
        except Exception:
            p_shapiro = 0.0

        if p_shapiro > alpha:
            test_name = "t-test"
            try:
                _, p_val = stats.ttest_rel(arrays[0], arrays[1])
            except Exception:
                p_val = 1.0
        else:
            test_name = "wilcoxon"
            try:
                _, p_val = stats.wilcoxon(arrays[0], arrays[1])
            except Exception:
                p_val = 1.0

        result["p_friedman"] = round(p_val, 6)
        mean_0, mean_1 = arrays[0].mean(), arrays[1].mean()
        winner = method_names[0] if mean_0 > mean_1 else method_names[1]

        if p_val <= alpha:
            result["decision"] = winner
        else:
            result["decision"] = _frugality_tiebreak(method_names)

        result["test_post_hoc"] = f"{test_name}(p={p_val:.4f})"
        result["p_posthoc_best"] = round(p_val, 6)
        result["notes"] = f"Shapiro p={p_shapiro:.4f}; 2-method comparison"

    return result

# =========================================================================
#  Full statistical pipeline
# =========================================================================

def run_full_analysis(
    score_matrix: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    """Run per-entity statistical analysis, return summary DataFrame."""
    method_cols = [c for c in score_matrix.columns if c.startswith("F1_")]
    results = []

    for entity in labels:
        df_e = score_matrix[score_matrix["entity"] == entity].copy()
        if df_e.empty:
            results.append({
                "entity": entity, "n_docs": 0,
                "decision": "excluded", "notes": "No data",
            })
            continue
        res = run_entity_analysis(df_e, entity, method_cols)
        results.append(res)

    return pd.DataFrame(results)

# =========================================================================
#  Output generation
# =========================================================================

def save_outputs(
    score_matrix: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: dict,
    corpus: str,
):
    out_dir = cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_path = out_dir / "scores_by_entity_doc.csv"
    score_matrix.to_csv(scores_path, index=False)
    log.info("Saved: %s", scores_path)

    routing_path = out_dir / "demne_statistical_routing.csv"
    summary.insert(0, "Corpus", corpus)
    summary.to_csv(routing_path, index=False)
    log.info("Saved: %s", routing_path)

    decisions = {}
    for _, row in summary.iterrows():
        ent = row["entity"]
        decisions[ent] = {
            "decision": row.get("decision", "excluded"),
            "p_friedman": row.get("p_friedman"),
            "n_docs": row.get("n_docs", 0),
        }

    routing_json = out_dir / "routing_decisions.json"
    routing_json.write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("Saved: %s", routing_json)

# =========================================================================
#  Visualization
# =========================================================================

METHOD_COLORS = {"rules": "#2ecc71", "TBM": "#e67e22", "LLM": "#e74c3c"}


def plot_boxplots(
    score_matrix: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: dict,
):
    fig_dir = cfg["output_dir"] / "Results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    method_cols = [c for c in score_matrix.columns if c.startswith("F1_")]
    method_names = [c.replace("F1_", "") for c in method_cols]

    sig_entities = summary[
        (summary["p_friedman"].notna())
        & (summary["p_friedman"] <= 0.05)
        & (summary["n_docs"] >= 10)
    ]["entity"].tolist()

    for entity in sig_entities:
        df_e = score_matrix[score_matrix["entity"] == entity]
        if df_e.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        data = [df_e[c].values for c in method_cols]
        colors = [METHOD_COLORS.get(m, "#999") for m in method_names]

        bp = ax.boxplot(
            data, patch_artist=True,
            widths=0.5, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="black", markersize=5),
        )
        ax.set_xticks(range(1, len(method_names) + 1))
        ax.set_xticklabels(method_names)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        row = summary[summary["entity"] == entity].iloc[0]
        p_val = row.get("p_friedman", 1.0)
        stars = "***" if p_val <= 0.001 else "**" if p_val <= 0.01 else "*"
        ax.set_title(f"{entity}  (p={p_val:.4f} {stars})", fontsize=11)
        ax.set_ylabel("F1")
        ax.grid(True, alpha=0.3, axis="y")

        decision = row.get("decision", "")
        ax.text(
            0.98, 0.02, f"→ {decision}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        safe_name = entity.replace("/", "_")
        plt.savefig(fig_dir / f"boxplot_{safe_name}.png", dpi=150)
        plt.close()

    log.info("Boxplots saved: %d entities → %s", len(sig_entities), fig_dir)


def plot_heatmap(
    summary: pd.DataFrame,
    cfg: dict,
):
    fig_dir = cfg["output_dir"] / "Results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    mean_cols = [c for c in summary.columns if c.startswith("F1_") and "_mean" in c]
    if not mean_cols:
        return

    entities = summary[summary["n_docs"] >= 10]["entity"].tolist()
    if not entities:
        return

    sub = summary[summary["entity"].isin(entities)].set_index("entity")[mean_cols]
    sub.columns = [c.replace("F1_", "").replace("_mean", "") for c in sub.columns]

    fig, ax = plt.subplots(figsize=(max(6, len(sub.columns) * 2), max(6, len(entities) * 0.35)))
    data = sub.values.astype(float)

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(sub.columns)))
    ax.set_xticklabels(sub.columns, fontsize=10)
    ax.set_yticks(range(len(entities)))
    ax.set_yticklabels(entities, fontsize=9)

    decisions = dict(zip(summary["entity"], summary.get("decision", [""] * len(summary))))

    for i in range(len(entities)):
        for j in range(len(sub.columns)):
            val = data[i, j]
            ent = entities[i]
            method = sub.columns[j]
            is_decision = decisions.get(ent, "").lower() == method.lower()
            weight = "bold" if is_decision else "normal"
            color = "white" if val < 0.4 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=8, fontweight=weight, color=color,
            )

    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean F1")
    ax.set_title(f"DEMNE Heatmap — {cfg['output_dir'].name}", fontsize=12)

    legend_elements = [
        Patch(facecolor=METHOD_COLORS.get(m, "#999"), label=m)
        for m in sub.columns
    ]
    ax.legend(
        handles=legend_elements, loc="upper left",
        bbox_to_anchor=(1.15, 1), fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(fig_dir / "heatmap_global.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Heatmap saved: %s", fig_dir / "heatmap_global.png")

# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="DEMNE Statistical Routing")
    parser.add_argument(
        "--corpus", required=True, choices=["maccrobat", "quaero"],
        help="Corpus to process",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM (run Rules vs TBM only)",
    )
    args = parser.parse_args()

    cfg = CORPUS_CFG[args.corpus]
    log.info("=" * 60)
    log.info("DEMNE Statistical Routing — %s", args.corpus.upper())
    log.info("=" * 60)

    # Step 0: Load / generate per-doc scores
    rules_scores, tbm_scores, llm_scores, doc_ids = get_all_scores(
        cfg, args.corpus, skip_llm=args.skip_llm,
    )

    if not doc_ids:
        log.error("No aligned documents found. Check data paths.")
        sys.exit(1)

    available_methods = ["rules"]
    method_scores = {"rules": rules_scores}
    if tbm_scores:
        available_methods.append("TBM")
        method_scores["TBM"] = tbm_scores
    if llm_scores:
        available_methods.append("LLM")
        method_scores["LLM"] = llm_scores

    log.info("Methods available: %s", ", ".join(available_methods))
    log.info("Aligned docs: %d", len(doc_ids))

    if len(available_methods) < 2:
        log.error("Need at least 2 methods for statistical comparison.")
        sys.exit(1)

    # Step 1: Build score matrix
    score_matrix = build_score_matrix(
        rules_scores, tbm_scores, llm_scores, cfg["labels"], doc_ids,
    )
    log.info(
        "Score matrix: %d rows (%d entities × %d docs max)",
        len(score_matrix),
        score_matrix["entity"].nunique(),
        len(doc_ids),
    )

    # Steps 2-3: Statistical analysis
    summary = run_full_analysis(score_matrix, cfg["labels"])

    n_sig = (summary["p_friedman"].notna() & (summary["p_friedman"] <= 0.05)).sum()
    n_tested = (summary["n_docs"] >= 10).sum()
    log.info(
        "Statistical results: %d/%d entities tested, %d significant",
        n_tested,
        len(cfg["labels"]),
        n_sig,
    )

    # Decision summary
    for _, row in summary.iterrows():
        if row.get("n_docs", 0) >= 10:
            p = row.get("p_friedman", np.nan)
            dec = row.get("decision", "?")
            flag = "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else "ns"
            log.info(
                "  %-30s  p=%-8s  → %s  (%s)",
                row["entity"],
                f"{p:.4f}" if not np.isnan(p) else "n/a",
                dec,
                flag,
            )

    # Step 4: Save outputs
    save_outputs(score_matrix, summary, cfg, args.corpus)

    # Step 5: Visualization
    plot_boxplots(score_matrix, summary, cfg)
    plot_heatmap(summary, cfg)

    log.info("=" * 60)
    log.info("Done. Outputs in: %s", cfg["output_dir"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
