"""
DEMNE Cochran Q Routing Pipeline
=================================
Binary mention-level comparison of Rules / TBM / LLM NER methods.
For each entity, a binary table records whether each method detected
each gold mention (exact span match). Cochran Q tests overall
significance; McNemar tests with Bonferroni compare pairwise.

Usage:
    python demne_cochran_routing.py --corpus maccrobat
    python demne_cochran_routing.py --corpus quaero
    python demne_cochran_routing.py --corpus maccrobat --skip-llm
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

from scipy.stats import chi2 as chi2_dist
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar

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

# =========================================================================
#  Corpus configuration
# =========================================================================

CORPUS_CFG = {
    "maccrobat": dict(
        gold_dir=ROOT / "ESMO2025_MACCROBAT2020" / "MACCROBAT2020_test",
        rules_dir=ROOT / "ESMO2025_MACCROBAT2020" / "Rules" / "src" / "MACCROBAT2020_pred_rules",
        tbm_model_dir=ROOT / "ESMO2025_MACCROBAT2020" / "TBM_evaluation" / "pubmedbert_finetuned" / "best",
        tbm_base_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        tbm_cache=ROOT / "ESMO2025_MACCROBAT2020" / "TBM_evaluation" / "pred_spans_tbm.json",
        llm_cache=ROOT / "ESMO2025_MACCROBAT2020" / "LLM_evaluation" / "pred_spans_llm.json",
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
        tbm_cache=ROOT / "ESMO2025_QUERO_French_Med" / "TBM_evaluation" / "pred_spans_tbm.json",
        llm_cache=ROOT / "ESMO2025_QUERO_French_Med" / "LLM_evaluation" / "pred_spans_llm.json",
        labels=QUAERO_LABELS,
        best_tbm="DrBERT-7GB",
        best_llm="mistral-large-latest",
        spacy_model="fr_core_news_sm",
        spacy_disable=["parser", "ner", "lemmatizer"],
        output_dir=ROOT / "ESMO2025_QUERO_French_Med",
        add_prefix_space=True,
        quaero_train_dir=ROOT / "ESMO2025_QUERO_French_Med" / "QUAERO_FrenchMed" / "corpus" / "train" / "MEDLINE",
        quaero_annotator=ROOT / "ESMO2025_QUERO_French_Med" / "Rules" / "src" / "quaero_brat_annotator.py",
        split_test_file=ROOT / "ESMO2025_QUERO_French_Med" / "Rules" / "src" / "split_test.txt",
        split_doc_paths_file=ROOT / "ESMO2025_QUERO_French_Med" / "Rules" / "src" / "split_doc_paths.json",
    ),
    "cantemist": dict(
        gold_dir=ROOT / "ESMO2025_CANTESMIST35" / "CANTEMIST35_test",
        rules_dir=ROOT / "ESMO2025_CANTESMIST35" / "Rules" / "src" / "cantemist_pred_rules",
        tbm_model_dir=ROOT / "ESMO2025_CANTESMIST35" / "TBM_evaluation" / "camembert_finetuned" / "best",
        tbm_base_model="almanach/camembert-base",
        tbm_cache=ROOT / "ESMO2025_CANTESMIST35" / "TBM_evaluation" / "pred_spans_tbm.json",
        llm_cache=ROOT / "ESMO2025_CANTESMIST35" / "LLM_evaluation" / "pred_spans_llm.json",
        labels=CANTEMIST_LABELS,
        best_tbm="CamemBERT",
        best_llm="mistral-large-latest",
        spacy_model="fr_core_news_sm",
        spacy_disable=["ner", "lemmatizer"],
        output_dir=ROOT / "ESMO2025_CANTESMIST35",
        add_prefix_space=True,
    ),
}

METHOD_COLORS = {"Rules": "#2ecc71", "TBM": "#e67e22", "LLM": "#e74c3c"}

# =========================================================================
#  BRAT parsing
# =========================================================================

def parse_brat_ann(ann_path: Path, valid_labels: set | None = None) -> set:
    """Parse BRAT .ann -> set of (begin, end, label). Ignores non-T lines."""
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
    return sorted(p.stem for p in directory.glob("*.ann"))

# =========================================================================
#  BIO -> spans
# =========================================================================

def bio_to_spans(tags: list[str]) -> set[tuple[int, int, str]]:
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
#  Collect predictions: Rules (from BRAT)
# =========================================================================

def collect_rules_predictions(
    rules_dir: Path, doc_ids: list[str], valid_labels: set,
) -> dict[str, set]:
    """Return {doc_id: set of (begin, end, label)}."""
    preds = {}
    for doc_id in doc_ids:
        preds[doc_id] = parse_brat_ann(rules_dir / f"{doc_id}.ann", valid_labels)
    return preds


def collect_rules_via_annotator(
    doc_txt_paths: dict[str, Path],
    doc_ids: list[str],
    valid_labels: set,
    annotator_path: Path,
    existing_rules_dir: Path,
) -> dict[str, set]:
    """Run Rules annotator on docs, falling back to existing .ann if available."""
    existing_ids = set(get_doc_ids(existing_rules_dir)) if existing_rules_dir.exists() else set()
    preds = {}

    annotator_mod = None
    need_annotator = [d for d in doc_ids if d not in existing_ids]
    if need_annotator and annotator_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("annotator", str(annotator_path))
        annotator_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(annotator_mod)

    for doc_id in doc_ids:
        if doc_id in existing_ids:
            preds[doc_id] = parse_brat_ann(
                existing_rules_dir / f"{doc_id}.ann", valid_labels
            )
        elif annotator_mod:
            txt_path = doc_txt_paths[doc_id]
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            raw = annotator_mod.annotate_text(text)
            preds[doc_id] = {(s, e, l) for _, s, e, l in raw if l in valid_labels}
        else:
            preds[doc_id] = set()

    if need_annotator:
        log.info("  Rules: generated %d predictions via annotator", len(need_annotator))
    return preds

# =========================================================================
#  Collect predictions: TBM (model inference)
# =========================================================================

def generate_tbm_predictions(
    cfg: dict,
    doc_ids: list[str],
    doc_txt_paths: dict[str, Path] | None = None,
) -> dict[str, set]:
    """Load TBM model, run inference -> {doc_id: set of (begin, end, label)}."""
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
    log.info("  TBM device: %s", device)

    if cfg.get("add_prefix_space"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), add_prefix_space=True)
        except Exception:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir), add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    base_model = AutoModelForTokenClassification.from_pretrained(
        base_model_id, num_labels=num_labels, id2label=id2label, label2id=label2id,
    )
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    model = model.merge_and_unload().to(device)
    model.eval()
    log.info("  TBM model loaded (%s + LoRA)", base_model_id)

    nlp = spacy.load(cfg["spacy_model"], disable=cfg["spacy_disable"])
    has_sents = "parser" not in cfg["spacy_disable"]
    if not has_sents:
        nlp.add_pipe("sentencizer")

    predictions = {}
    total = len(doc_ids)
    for idx, doc_id in enumerate(doc_ids):
        txt_path = (doc_txt_paths or {}).get(doc_id) or (gold_dir / f"{doc_id}.txt")
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        doc = nlp(text)

        pred_spans = set()
        sents = list(doc.sents) if has_sents else [doc[:]]
        for sent in sents:
            tokens = [tok.text for tok in sent]
            if not tokens:
                continue
            inputs = tokenizer(
                tokens, is_split_into_words=True,
                truncation=True, max_length=512, return_tensors="pt",
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

            for ts, te, lbl in bio_to_spans(pred_bio):
                if lbl in valid:
                    pred_spans.add((sent[ts].idx, sent[te].idx + len(sent[te].text), lbl))

        predictions[doc_id] = pred_spans
        if (idx + 1) % 50 == 0 or idx == 0 or idx == total - 1:
            log.info("    TBM inference: %d/%d", idx + 1, total)

    return predictions

# =========================================================================
#  Prediction cache (spans as JSON)
# =========================================================================

def save_pred_cache(preds: dict[str, set], path: Path):
    serializable = {
        doc_id: sorted([list(s) for s in spans], key=lambda x: (x[0], x[1]))
        for doc_id, spans in preds.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("  Cache saved: %s", path.name)


def load_pred_cache(path: Path) -> dict[str, set] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        doc_id: {(s[0], s[1], s[2]) for s in spans}
        for doc_id, spans in raw.items()
    }

# =========================================================================
#  QUAERO re-split reproduction
# =========================================================================

def _iterative_stratified_split_assignment(
    label_matrix: np.ndarray, train_ratio: float, seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n, n_labels = label_matrix.shape
    desired = np.zeros((2, n_labels), dtype=float)
    totals = label_matrix.sum(axis=0).astype(float)
    desired[0] = totals * train_ratio
    desired[1] = totals * (1 - train_ratio)
    assignment = np.full(n, -1, dtype=int)
    for l_idx in np.argsort(totals):
        candidates = np.where((label_matrix[:, l_idx] == 1) & (assignment == -1))[0]
        rng.shuffle(candidates)
        for doc_idx in candidates:
            fold = 0 if desired[0, l_idx] >= desired[1, l_idx] else 1
            assignment[doc_idx] = fold
            desired[fold] -= label_matrix[doc_idx]
    for doc_idx in np.where(assignment == -1)[0]:
        assignment[doc_idx] = 0 if rng.random() < train_ratio else 1
    return assignment


def reproduce_quaero_resplit(cfg: dict) -> tuple[list[str], dict[str, Path]]:
    # If split_test.txt and split_doc_paths.json exist, use them directly
    split_test_file = cfg.get("split_test_file")
    split_doc_paths_file = cfg.get("split_doc_paths_file")

    if split_test_file and split_test_file.exists():
        test_ids = [
            line.strip()
            for line in split_test_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # Load doc_id -> txt_path mapping
        test_paths = {}
        if split_doc_paths_file and split_doc_paths_file.exists():
            all_paths = json.loads(split_doc_paths_file.read_text(encoding="utf-8"))
            for doc_id in test_ids:
                if doc_id in all_paths:
                    test_paths[doc_id] = Path(all_paths[doc_id])
        else:
            # Fallback: search in both corpus dirs
            corpus_dirs = [cfg["quaero_train_dir"], cfg["gold_dir"]]
            for doc_id in test_ids:
                for corpus_dir in corpus_dirs:
                    txt_path = corpus_dir / f"{doc_id}.txt"
                    if txt_path.exists():
                        test_paths[doc_id] = txt_path
                        break

        log.info("QUAERO split from file: %d test docs", len(test_ids))
        return test_ids, test_paths

    # Fallback: algorithmic re-split (legacy)
    all_labels = sorted(cfg["labels"])
    valid = set(cfg["labels"])
    corpus_dirs = [cfg["quaero_train_dir"], cfg["gold_dir"]]
    doc_paths, label_vectors = [], []
    for corpus_dir in corpus_dirs:
        if not corpus_dir.exists():
            continue
        for txt_path in sorted(corpus_dir.glob("*.txt")):
            ann_path = txt_path.with_suffix(".ann")
            if not ann_path.exists():
                continue
            entities = parse_brat_ann(ann_path, valid)
            present = {l for _, _, l in entities}
            label_vectors.append(np.array([1 if l in present else 0 for l in all_labels]))
            doc_paths.append(txt_path)

    assignment = _iterative_stratified_split_assignment(
        np.array(label_vectors), 0.75, 42,
    )
    test_ids, test_paths = [], {}
    for i, p in enumerate(doc_paths):
        if assignment[i] == 1:
            test_ids.append(p.stem)
            test_paths[p.stem] = p
    log.info("QUAERO re-split: %d total -> %d test", len(doc_paths), len(test_ids))
    return test_ids, test_paths

# =========================================================================
#  Build binary table for one entity
# =========================================================================

def extract_binary_table(
    entity: str,
    gold_mentions: list[tuple[str, int, int]],
    pred_rules: dict[str, set],
    pred_tbm: dict[str, set] | None,
    pred_llm: dict[str, set] | None,
) -> pd.DataFrame:
    """
    Build mention-level binary table.
    Rows = gold mentions for `entity`.
    Columns = 1 if method predicted exact (doc_id, begin, end, label).

    Returns DataFrame with columns [doc_id, begin, end, Rules, TBM?, LLM?].
    """
    rows = []
    for doc_id, begin, end in gold_mentions:
        row = {"doc_id": doc_id, "begin": begin, "end": end}
        row["Rules"] = int((begin, end, entity) in pred_rules.get(doc_id, set()))
        if pred_tbm is not None:
            row["TBM"] = int((begin, end, entity) in pred_tbm.get(doc_id, set()))
        if pred_llm is not None:
            row["LLM"] = int((begin, end, entity) in pred_llm.get(doc_id, set()))
        rows.append(row)

    df = pd.DataFrame(rows)
    assert len(df) == len(gold_mentions), (
        f"{entity}: expected {len(gold_mentions)} rows, got {len(df)}"
    )
    return df

# =========================================================================
#  Collect gold mentions per entity
# =========================================================================

def collect_gold_mentions(
    gold_dir: Path | None,
    doc_ids: list[str],
    labels: list[str],
    doc_txt_paths: dict[str, Path] | None = None,
) -> dict[str, list[tuple[str, int, int]]]:
    """Return {entity: [(doc_id, begin, end), ...]} from gold BRAT .ann files."""
    valid = set(labels)
    mentions = defaultdict(list)
    for doc_id in doc_ids:
        if doc_txt_paths and doc_id in doc_txt_paths:
            ann_path = doc_txt_paths[doc_id].with_suffix(".ann")
        else:
            ann_path = gold_dir / f"{doc_id}.ann"
        for begin, end, label in parse_brat_ann(ann_path, valid):
            mentions[label].append((doc_id, begin, end))
    for label in mentions:
        mentions[label].sort()
    return dict(mentions)

# =========================================================================
#  Statistical tests
# =========================================================================

def run_cochran_mcnemar(
    table: pd.DataFrame,
    entity: str,
    method_cols: list[str],
    alpha: float = 0.05,
) -> dict:
    """Run Cochran Q (or McNemar if 2 methods). Return result dict."""
    n = len(table)
    result = {"entity": entity, "n_mentions": n}

    for col in method_cols:
        result[f"L_{col}"] = int(table[col].sum())
        result[f"recall_{col}"] = round(table[col].mean(), 4)

    # Check degeneracy: all columns identical -> no variance
    arr = table[method_cols].values
    if arr.std() == 0:
        result["Q_stat"] = np.nan
        result["p_cochran"] = np.nan
        result["decision"] = "Rules"
        result["motif"] = "Dégénéré: colonnes identiques, pas de variance"
        return result

    # Remove rows where all methods = 0 (no method detects -> no discrimination)
    row_sums = arr.sum(axis=1)
    mask = row_sums > 0
    arr_filtered = arr[mask]
    n_filtered = mask.sum()

    if n_filtered < 10:
        result["Q_stat"] = np.nan
        result["p_cochran"] = np.nan
        result["decision"] = "Rules"
        result["motif"] = f"Exclu: n_effective={n_filtered} < 10 (après retrait lignes tout-0)"
        return result

    result["n_effective"] = int(n_filtered)

    # --- Cochran Q (>= 3 methods) or McNemar (2 methods) ---
    if len(method_cols) >= 3:
        res = cochrans_q(arr_filtered)
        Q_stat, p_val = res.statistic, res.pvalue
        result["Q_stat"] = round(float(Q_stat), 4)
        result["p_cochran"] = round(float(p_val), 6)

        if p_val > alpha:
            result["decision"] = "Rules"
            result["motif"] = "Cochran Q non significatif"
            return result

        # Post-hoc McNemar 2×2 with Bonferroni
        n_pairs = len(method_cols) * (len(method_cols) - 1) // 2
        alpha_corr = alpha / n_pairs
        pair_results = []

        for i in range(len(method_cols)):
            for j in range(i + 1, len(method_cols)):
                a_col, b_col = method_cols[i], method_cols[j]
                a_vals = arr_filtered[:, i]
                b_vals = arr_filtered[:, j]

                n11 = int(((a_vals == 1) & (b_vals == 1)).sum())
                n10 = int(((a_vals == 1) & (b_vals == 0)).sum())
                n01 = int(((a_vals == 0) & (b_vals == 1)).sum())
                n00 = int(((a_vals == 0) & (b_vals == 0)).sum())

                contingency = np.array([[n00, n01], [n10, n11]])
                discordant = n10 + n01

                if discordant == 0:
                    p_mc = 1.0
                elif discordant < 25:
                    res_mc = mcnemar(contingency, exact=True)
                    p_mc = float(res_mc.pvalue)
                else:
                    res_mc = mcnemar(contingency, exact=False, correction=True)
                    p_mc = float(res_mc.pvalue)

                L_a = int(a_vals.sum())
                L_b = int(b_vals.sum())
                winner = a_col if L_a >= L_b else b_col
                sig = p_mc <= alpha_corr

                pair_results.append({
                    "pair": f"{a_col} vs{b_col}",
                    "c01": n10, "c10": n01,
                    "p": round(p_mc, 6),
                    "significant": sig,
                    "winner": winner,
                })
                result[f"p_mcnemar_{a_col.lower()}_{b_col.lower()}"] = round(p_mc, 6)

        FRUGALITY = {"Rules": 0, "TBM": 1, "LLM": 2}
        sig_winners = [pr["winner"] for pr in pair_results if pr["significant"]]
        if sig_winners:
            from collections import Counter
            counts = Counter(sig_winners)
            max_wins = counts.most_common(1)[0][1]
            candidates = [m for m, c in counts.items() if c == max_wins]
            if len(candidates) == 1:
                best = candidates[0]
                result["motif"] = f"McNemar significatif pour {best} ({max_wins} paire(s))"
            else:
                best = min(candidates, key=lambda m: FRUGALITY.get(m, 99))
                result["motif"] = (
                    f"Egalite McNemar {candidates} -> "
                    f"{best} retenu par frugalite"
                )
            result["decision"] = best
        else:
            result["decision"] = "Rules"
            result["motif"] = "Aucun McNemar significatif apres Bonferroni"

        result["_pair_results"] = pair_results

    elif len(method_cols) == 2:
        a_col, b_col = method_cols
        a_vals = arr_filtered[:, 0]
        b_vals = arr_filtered[:, 1]

        n11 = int(((a_vals == 1) & (b_vals == 1)).sum())
        n10 = int(((a_vals == 1) & (b_vals == 0)).sum())
        n01 = int(((a_vals == 0) & (b_vals == 1)).sum())
        n00 = int(((a_vals == 0) & (b_vals == 0)).sum())

        contingency = np.array([[n00, n01], [n10, n11]])
        discordant = n10 + n01

        if discordant == 0:
            p_mc = 1.0
        elif discordant < 25:
            res_mc = mcnemar(contingency, exact=True)
            p_mc = float(res_mc.pvalue)
        else:
            res_mc = mcnemar(contingency, exact=False, correction=True)
            p_mc = float(res_mc.pvalue)

        result["Q_stat"] = np.nan
        result["p_cochran"] = round(p_mc, 6)
        result[f"p_mcnemar_{a_col.lower()}_{b_col.lower()}"] = round(p_mc, 6)

        L_a, L_b = int(a_vals.sum()), int(b_vals.sum())
        winner = a_col if L_a >= L_b else b_col

        if p_mc <= alpha:
            result["decision"] = winner
            result["motif"] = f"McNemar significatif, {winner} supérieur"
        else:
            result["decision"] = "Rules"
            result["motif"] = "McNemar non significatif"

        result["_pair_results"] = [{
            "pair": f"{a_col} vs{b_col}",
            "c01": n10, "c10": n01,
            "p": round(p_mc, 6),
            "significant": p_mc <= alpha,
            "winner": winner,
        }]

    return result

# =========================================================================
#  Terminal display
# =========================================================================

def display_entity_result(result: dict, method_cols: list[str]):
    entity = result["entity"]
    n = result["n_mentions"]
    line = f"-- {entity} | n={n} "
    print(f"\n{line}{'-' * max(1, 55 - len(line))}")

    p_cochran = result.get("p_cochran")
    Q = result.get("Q_stat")

    if pd.isna(p_cochran) if isinstance(p_cochran, float) else p_cochran is None:
        print(f"  {result.get('motif', 'Exclu')}")
        print(f"  Decision : {result.get('decision', '?')}")
        print("-" * 55)
        return

    if Q is not None and not (isinstance(Q, float) and np.isnan(Q)):
        sig_label = "SIGNIFICATIF" if p_cochran <= 0.05 else "non significatif"
        print(f"  Cochran Q = {Q:.2f} | p = {p_cochran:.4f} -> {sig_label}")
    else:
        print(f"  McNemar direct | p = {p_cochran:.4f}")

    for pr in result.get("_pair_results", []):
        bonf = "[sig]" if pr["significant"] else "(ns apres Bonf.)"
        winner_info = f" {pr['winner']} superieur" if pr["significant"] else ""
        print(
            f"    McNemar {pr['pair']:<15s} : "
            f"c01={pr['c01']} c10={pr['c10']} | "
            f"p = {pr['p']:.4f}  {bonf}{winner_info}"
        )

    print(f"  Decision : {result['decision']}")
    print("-" * 55)

# =========================================================================
#  Visualization: grouped barplot
# =========================================================================

def plot_barplot(summary: pd.DataFrame, cfg: dict):
    fig_dir = cfg["output_dir"] / "Results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    recall_cols = [c for c in summary.columns if c.startswith("recall_")]
    method_names = [c.replace("recall_", "") for c in recall_cols]

    df = summary[summary["n_mentions"] >= 10].copy()
    if df.empty:
        return

    entities = df["entity"].tolist()
    n_ent = len(entities)
    n_methods = len(method_names)
    x = np.arange(n_ent)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(10, n_ent * 0.6), 6))

    for i, method in enumerate(method_names):
        col = f"recall_{method}"
        vals = df[col].values
        color = METHOD_COLORS.get(method, "#999")
        bars = ax.bar(x + i * width, vals, width, label=method, color=color, alpha=0.85)

        for j, (bar, val) in enumerate(zip(bars, vals)):
            decision = df.iloc[j].get("decision", "")
            if decision == method:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    "★", ha="center", va="bottom", fontsize=12, color="black",
                )

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(entities, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Recall (L / n_mentions)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"DEMNE Recall per Entity — {cfg['output_dir'].name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = fig_dir / "barplot_recall.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("Barplot saved: %s", out)

# =========================================================================
#  Outputs
# =========================================================================

def save_outputs(
    summary: pd.DataFrame,
    excluded: list[dict],
    all_decisions: dict,
    cfg: dict,
    corpus: str,
):
    out_dir = cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    export_cols = [c for c in summary.columns if not c.startswith("_")]
    export = summary[export_cols].copy()
    export.insert(0, "Corpus", corpus)
    routing_path = out_dir / "demne_cochran_routing.csv"
    export.to_csv(routing_path, index=False)
    log.info("Saved: %s", routing_path)

    routing_json = out_dir / "routing_decisions.json"
    routing_json.write_text(
        json.dumps(all_decisions, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("Saved: %s", routing_json)

    if excluded:
        excl_path = out_dir / "excluded.json"
        excl_path.write_text(
            json.dumps(excluded, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Saved: %s (%d entities)", excl_path, len(excluded))

# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="DEMNE Cochran Q Routing")
    parser.add_argument("--corpus", required=True, choices=["maccrobat", "quaero", "cantemist"])
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    cfg = CORPUS_CFG[args.corpus]
    log.info("=" * 60)
    log.info("DEMNE Cochran Q Routing — %s", args.corpus.upper())
    log.info("=" * 60)

    # --- Determine test docs ---
    if args.corpus == "quaero" and "quaero_train_dir" in cfg:
        doc_ids, doc_txt_paths = reproduce_quaero_resplit(cfg)
    else:
        doc_ids = get_doc_ids(cfg["gold_dir"])
        doc_txt_paths = None

    log.info("Test documents: %d", len(doc_ids))
    valid = set(cfg["labels"])

    # --- Collect gold mentions ---
    gold_mentions = collect_gold_mentions(
        cfg["gold_dir"], doc_ids, cfg["labels"], doc_txt_paths,
    )
    total_mentions = sum(len(v) for v in gold_mentions.values())
    log.info("Gold mentions: %d total across %d entity types",
             total_mentions, len(gold_mentions))

    # --- Collect predictions: Rules ---
    if args.corpus == "quaero" and doc_txt_paths:
        pred_rules = collect_rules_via_annotator(
            doc_txt_paths, doc_ids, valid,
            cfg.get("quaero_annotator", Path("none")),
            cfg["rules_dir"],
        )
    else:
        pred_rules = collect_rules_predictions(cfg["rules_dir"], doc_ids, valid)

    rules_total = sum(len(v) for v in pred_rules.values())
    log.info("Rules predictions: %d spans", rules_total)

    # --- Collect predictions: TBM ---
    pred_tbm = load_pred_cache(cfg["tbm_cache"])
    if pred_tbm is None and cfg["tbm_model_dir"].exists():
        log.info("Generating TBM predictions...")
        pred_tbm = generate_tbm_predictions(cfg, doc_ids, doc_txt_paths)
        save_pred_cache(pred_tbm, cfg["tbm_cache"])
    elif pred_tbm is None:
        log.warning("TBM model not found, skipping TBM")

    if pred_tbm:
        tbm_total = sum(len(v) for v in pred_tbm.values())
        log.info("TBM predictions: %d spans", tbm_total)

    # --- Collect predictions: LLM ---
    pred_llm = None
    if not args.skip_llm:
        pred_llm = load_pred_cache(cfg["llm_cache"])
        if pred_llm is None:
            log.warning(
                "LLM predictions not found at %s. Use --skip-llm or generate cache.",
                cfg["llm_cache"].name,
            )
    else:
        log.info("LLM skipped (--skip-llm)")

    # --- Determine method columns ---
    method_cols = ["Rules"]
    if pred_tbm is not None:
        method_cols.append("TBM")
    if pred_llm is not None:
        method_cols.append("LLM")

    if len(method_cols) < 2:
        log.error("Need at least 2 methods. Available: %s", method_cols)
        sys.exit(1)

    log.info("Methods: %s", ", ".join(method_cols))

    # --- Per-entity analysis ---
    tables_dir = cfg["output_dir"] / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    excluded = []
    all_decisions = {}

    print("\n" + "=" * 60)
    print(f"  DEMNE Cochran Q Analysis — {args.corpus.upper()}")
    print("=" * 60)

    for entity in cfg["labels"]:
        mentions = gold_mentions.get(entity, [])

        if len(mentions) < 10:
            exc = {
                "entity": entity,
                "n_mentions": len(mentions),
                "motif": f"n_mentions={len(mentions)} < 10",
            }
            excluded.append(exc)
            all_decisions[entity] = {"decision": "excluded", "n_mentions": len(mentions)}
            log.info("  %s: excluded (n=%d < 10)", entity, len(mentions))
            continue

        table = extract_binary_table(
            entity, mentions, pred_rules, pred_tbm, pred_llm,
        )

        table.to_csv(tables_dir / f"entity_{entity}.csv", index=False)

        result = run_cochran_mcnemar(table, entity, method_cols)

        display_entity_result(result, method_cols)

        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        summary_rows.append(clean)

        all_decisions[entity] = {
            "decision": result["decision"],
            "n_mentions": result["n_mentions"],
            "p_cochran": result.get("p_cochran"),
        }

    summary = pd.DataFrame(summary_rows)

    # --- Print decision summary ---
    print("\n" + "=" * 60)
    print("  RÉSUMÉ DES DÉCISIONS")
    print("=" * 60)
    if not summary.empty:
        for _, row in summary.iterrows():
            flag = ""
            p = row.get("p_cochran")
            if p is not None and not (isinstance(p, float) and np.isnan(p)):
                flag = "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else "ns"
            print(f"  {row['entity']:<30s}  n={row['n_mentions']:<5d}  -> {row['decision']:<8s}  {flag}")

    if excluded:
        print(f"\n  Exclus ({len(excluded)} entités) :")
        for exc in excluded:
            print(f"    {exc['entity']}: {exc['motif']}")
    print("=" * 60)

    # --- Save outputs ---
    save_outputs(summary, excluded, all_decisions, cfg, args.corpus)

    # --- Visualization ---
    if not summary.empty:
        plot_barplot(summary, cfg)

    log.info("Done. Outputs in: %s", cfg["output_dir"])


if __name__ == "__main__":
    main()
