"""
Few-shot NER evaluation on QUAERO FrenchMed using an LLM.

Supports two backends:
  - Ollama (local)
  - Mistral API (cloud, requires API key)

Pipeline:
  1. Load test_gliner.json (produced by ../TBM_evaluation/prepare_gliner_data.py)
  2. For each document, prompt the LLM with few-shot examples (.txt/.ann)
  3. Parse LLM output, align to token spans
  4. Evaluate with exact span match (same metrics as GLiNER eval)
  5. Save results to llm_results.csv, plot to plots/

Few-shot examples (from train corpus):
  - 1232256: cross-label nesting (cancers pulmonaires/DISO, pulmonaires/ANAT)
  - 1087947: same-label nesting (acide osmique/CHEM, acide/CHEM, osmique/CHEM)
  - 6390647: 5 diverse labels without nesting (PROC, CHEM, DISO, ANAT, LIVB)
  - 6032395: rare labels PHEN/DEVI + cross-label nesting
  - 14706904: GEOG + multi-level nesting (cancer du sein/DISO, sein/ANAT)
  - 5757903: PHYS/OBJC + deep nesting (volume expiratoire.../PROC, expiratoire/PHYS)

Usage:
    # Ollama (local)
    ollama pull qwen2.5:14b
    python llm_ner_zero_shot.py --backend ollama --model qwen2.5:14b

    # Mistral API
    python llm_ner_zero_shot.py --backend mistral --model mistral-large-latest --api-key YOUR_KEY

    # Common options
    python llm_ner_zero_shot.py --split train
    python llm_ner_zero_shot.py --inspect 0
    python llm_ner_zero_shot.py --limit 50
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import requests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR   = Path(__file__).parent.parent / "TBM_evaluation"
OUT_DIR    = Path(__file__).parent
PLOTS_DIR  = OUT_DIR / "plots"

OLLAMA_URL = "http://localhost:11434/api/generate"

QUAERO_LABELS = ["DISO", "PROC", "ANAT", "CHEM", "DEVI", "LIVB", "PHYS", "PHEN", "GEOG", "OBJC"]

RESULTS_CSV = OUT_DIR / "llm_results.csv"
PARAMS_CSV  = OUT_DIR / "llm_params.csv"

# ---------------------------------------------------------------------------
# Prompt template — Few-shot with .txt/.ann examples from TRAIN corpus
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = SYSTEM_PROMPT = """\
Tu es un annotateur expert en entités nommées biomédicales selon le schéma QUAERO/UMLS.
Ton rôle : identifier les concepts UMLS présents dans le texte.

=== PRINCIPE FONDAMENTAL — SPAN MAXIMAL ===
Pour chaque région du texte contenant un concept médical, annote UNIQUEMENT
le span le plus large qui forme un concept UMLS cohérent.
NE JAMAIS décomposer une expression multi-mots en ses parties constituantes.

  ✓ "cancers pulmonaires"   (DISO)   — 1 annotation
  ✗ "cancers" + "pulmonaires"        — INTERDIT

  ✓ "acide osmique"         (CHEM)   — 1 annotation
  ✗ "acide" + "osmique"              — INTERDIT

  ✓ "cancer du sein"        (DISO)   — 1 annotation
  ✗ "cancer" + "sein"                — INTERDIT

  ✓ "protheses de l' oreille moyenne" (DEVI) — 1 annotation
  ✗ "protheses" + "oreille moyenne" + "oreille" — INTERDIT

=== RÈGLES ===

R1 - SPAN MAXIMAL OBLIGATOIRE :
Annoter l'expression complète qui forme un concept UMLS.
Ne jamais annoter séparément les mots constitutifs d'une expression déjà annotée.
Le label est celui du concept composé pris dans son ensemble :
  "arteriographie bronchique"       → PROC  (non ANAT)
  "cancers pulmonaires"             → DISO  (non ANAT)
  "infection a VIH"                 → DISO  (non LIVB)
  "protheses de l' oreille moyenne" → DEVI  (non ANAT)
  "maladie rheumatoide"             → DISO  (non décomposé)
  "maladie de Crohn"                → DISO  (pas "maladie" seul)
  "grossesses extra-uterines"       → DISO  (non PHYS)

R2 - MOTS SIMPLES MÉDICAUX :
Un mot isolé qui n'est pas contenu dans une expression plus large est annoté normalement.
  ✓ "traitement" (PROC) | "HBV" (DISO) | "acetylcholine" (CHEM) | "VEMS" (PROC)
  ✓ "bronchique" (ANAT) si isolé dans le texte — pas s'il est dans "arteriographie bronchique"
  ✓ "lymphocytaires" (ANAT) | "asthmatique" (DISO) | "polytoxicomane" (LIVB)

R3 - NON-ANNOTÉS :
Articles, prépositions, conjonctions, chiffres, années, ponctuation.
Termes génériques non médicaux :
"resultats", "donnees", "valeurs", "niveaux", "cas", "parametres",
"formes", "aspects", "bilan", "utilisation", "remarques", "cout", "mise au point".

R4 - FILTRE PRÉCISION :
    "reponse"      → PROC SEULEMENT si réponse thérapeutique/immunitaire explicite
    "risque"       → JAMAIS seul
    "selective", "significatif", "severe" → adjectifs génériques, NON annotés seuls
    "adulte", "jeune" → LIVB si population de l'étude clairement désignée
    "marqueurs"    → CHEM uniquement en contexte de dosage/étude biologique
    "populations"  → JAMAIS seul
    "Etude"        → PROC si acte de recherche ; sinon ignorer

CALIBRAGE : 3–7 entités par titre médical typique.
Si tu hésites sur un span, préfère l'expression la plus large ou n'annote pas.

=== CATÉGORIES ===
DISO : Maladie, trouble, symptôme, syndrome
PROC : Procédure, examen, traitement, diagnostic, dosage, étude
ANAT : Structure anatomique, adjectif anatomique isolé (hepatique, pulmonaire...)
CHEM : Substance chimique, médicament, molécule, marqueur biologique
DEVI : Dispositif médical, prothèse, implant
LIVB : Être vivant, organisme, patient (homme, rat, souris, enfant, virus...)
PHYS : Processus physiologique (croissance, absorption, expiratoire...)
PHEN : Phénomène physique (acoustique, radioactivite...)
GEOG : Lieu géographique
OBJC : Objet physique (aerosol, appareil, ciment...)

=== 7 EXEMPLES ANNOTÉS ===

Texte 1 : L' arteriographie bronchique selective dans le diagnositc des cancers pulmonaires
[Span maximal : "arteriographie bronchique" → PROC. Ni "bronchique", ni "cancers", ni "pulmonaires" seuls.]
[{"text": "arteriographie bronchique", "label": "PROC"},
 {"text": "diagnositc", "label": "PROC"},
 {"text": "cancers pulmonaires", "label": "DISO"}]

Texte 2 : Les synoviortheses a l' acide osmique dans le traitement de la maladie rheumatoide . Remarques et bilan de 5 annees d' utilisation
[Span maximal : "acide osmique" → CHEM. "maladie rheumatoide" → DISO. Exclus : "Remarques", "bilan", "utilisation".]
[{"text": "synoviortheses", "label": "PROC"},
 {"text": "acide osmique", "label": "CHEM"},
 {"text": "traitement", "label": "PROC"},
 {"text": "maladie rheumatoide", "label": "DISO"}]

Texte 3 : Etude des marqueurs de l' HBV et des populations lymphocytaires chez le polytoxicomane asymptomatique .
[Exclus : "populations" seul. "lymphocytaires" isolé → ANAT.]
[{"text": "Etude", "label": "PROC"},
 {"text": "marqueurs", "label": "CHEM"},
 {"text": "HBV", "label": "DISO"},
 {"text": "lymphocytaires", "label": "ANAT"},
 {"text": "polytoxicomane", "label": "LIVB"},
 {"text": "asymptomatique", "label": "DISO"}]

Texte 4 : L' impedance acoustique des protheses de l' oreille moyenne .
[Span maximal : "protheses de l' oreille moyenne" → DEVI. Ni "protheses", ni "oreille moyenne", ni "oreille" seuls.]
[{"text": "acoustique", "label": "PHEN"},
 {"text": "protheses de l' oreille moyenne", "label": "DEVI"}]

Texte 5 : Le cout du depistage du cancer du sein et des cancers gynecologiques en France .
[Span maximal : "cancer du sein", "cancers gynecologiques". Exclus : "cout", "cancer" seul, "sein" seul.]
[{"text": "depistage", "label": "PROC"},
 {"text": "cancer du sein", "label": "DISO"},
 {"text": "cancers gynecologiques", "label": "DISO"},
 {"text": "France", "label": "GEOG"}]

Texte 6 : Interpretation des variations du volume expiratoire maximal seconde ( VEMS ) apres aerosol d' acetylcholine et d' allergenes chez l' asthmatique jeune .
[Span maximal : "volume expiratoire maximal seconde" → PROC. VEMS = acronyme isolé (hors span), annoté séparément. Exclus : "volume" seul, "expiratoire" seul, "variations".]
[{"text": "Interpretation", "label": "PROC"},
 {"text": "volume expiratoire maximal seconde", "label": "PROC"},
 {"text": "VEMS", "label": "PROC"},
 {"text": "aerosol", "label": "OBJC"},
 {"text": "acetylcholine", "label": "CHEM"},
 {"text": "allergenes", "label": "CHEM"},
 {"text": "asthmatique", "label": "DISO"},
 {"text": "jeune", "label": "LIVB"}]

Texte 7 : Mise au point sur le traitement des formes severes de la maladie de Crohn par immunosuppresseurs .
[Span maximal : "maladie de Crohn" → DISO. Exclus : "Mise au point", "formes", "severes", "maladie" seul.]
[{"text": "traitement", "label": "PROC"},
 {"text": "maladie de Crohn", "label": "DISO"},
 {"text": "immunosuppresseurs", "label": "CHEM"}]

=== FORMAT DE RÉPONSE ===
Tableau JSON uniquement. Aucune explication.
Chaque element : {"text": "...", "label": "..."}
Si aucune entité : []"""


# ---------------------------------------------------------------------------
# Backend: Ollama
# ---------------------------------------------------------------------------

def query_ollama(text: str, model: str, timeout: int = 120) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nTexte :\n{text}"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 2048},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        print("ERREUR: Ollama n'est pas lance. Demarrez-le avec: ollama serve")
        sys.exit(1)
    except Exception as e:
        print(f"  Erreur Ollama: {e}")
        return "[]"


# ---------------------------------------------------------------------------
# Backend: Mistral API (direct REST, no SDK needed)
# ---------------------------------------------------------------------------

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


def query_mistral(text: str, model: str, api_key: str,
                  max_retries: int = 5, retry_backoff: float = 5.0) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Texte :\n{text}"},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(MISTRAL_API_URL, headers=headers,
                                 json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == max_retries:
                print(f"  API Mistral: echec apres {max_retries} tentatives: {e}")
                return "[]"
            time.sleep(retry_backoff * attempt)


# ---------------------------------------------------------------------------
# Unified query function
# ---------------------------------------------------------------------------

def make_query_fn(backend: str, model: str, api_key: str = None):
    if backend == "mistral":
        if not api_key:
            api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            print("ERREUR: --api-key requis ou variable MISTRAL_API_KEY")
            sys.exit(1)
        print(f"Backend: Mistral API ({model})")
        return lambda text: query_mistral(text, model, api_key)
    else:
        print(f"Backend: Ollama local ({model})")
        return lambda text: query_ollama(text, model)


# ---------------------------------------------------------------------------
# Parse LLM output
# ---------------------------------------------------------------------------

def parse_llm_response(response: str) -> list[dict]:
    response = response.strip()
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if not match:
        return []
    try:
        entities = json.loads(match.group())
        if not isinstance(entities, list):
            return []
        valid = []
        for e in entities:
            if isinstance(e, dict) and "text" in e and "label" in e:
                if e["label"] in QUAERO_LABELS:
                    valid.append({"text": e["text"], "label": e["label"]})
        return valid
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# Token alignment
# ---------------------------------------------------------------------------

def build_char_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    offsets, pos = [], 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + 1
    return " ".join(tokens), offsets


def find_entity_in_tokens(entity_text: str, label: str, text: str,
                          offsets: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    results = []
    entity_lower = entity_text.lower()
    text_lower = text.lower()
    start = 0
    while True:
        idx = text_lower.find(entity_lower, start)
        if idx == -1:
            break
        end_char = idx + len(entity_text)
        ts = te = None
        for i, (cs, ce) in enumerate(offsets):
            if ts is None and cs <= idx < ce:
                ts = i
            if cs < end_char <= ce:
                te = i
        if ts is not None and te is not None and ts <= te:
            results.append((ts, te, label))
        start = idx + 1
    return results


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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}_gliner.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run ../TBM_evaluation/prepare_gliner_data.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def eval_llm(query_fn, model_name: str, docs: list[dict], split_name: str):
    global_tp = global_fp = global_fn = 0
    per_label = defaultdict(lambda: [0, 0, 0])

    total = len(docs)
    start_time = time.time()
    errors = 0

    print(f"\n  Evaluating {model_name} on {split_name} ({total} docs) ...")

    for i, doc in enumerate(docs):
        tokens = doc["tokenized_text"]
        text, offsets = build_char_offsets(tokens)
        gold = {(e[0], e[1], e[2]) for e in doc["ner"]}

        response = query_fn(text)
        entities = parse_llm_response(response)

        pred = set()
        for ent in entities:
            spans = find_entity_in_tokens(ent["text"], ent["label"], text, offsets)
            for span in spans:
                pred.add(span)

        tp, fp, fn = evaluate(gold, pred)
        global_tp += tp
        global_fp += fp
        global_fn += fn

        for label in QUAERO_LABELS:
            g = {s for s in gold if s[2] == label}
            p = {s for s in pred if s[2] == label}
            ltp, lfp, lfn = evaluate(g, p)
            per_label[label][0] += ltp
            per_label[label][1] += lfp
            per_label[label][2] += lfn

        if not entities and gold:
            errors += 1

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            eta = (total - i - 1) / speed if speed > 0 else 0
            p_cur, r_cur, f1_cur = prf(global_tp, global_fp, global_fn)
            print(f"    [{i+1}/{total}] F1={f1_cur:.3f} P={p_cur:.3f} R={r_cur:.3f}"
                  f"  ({speed:.1f} doc/s, ETA {eta/60:.0f}min)")

    elapsed = time.time() - start_time
    p_all, r_all, f1_all = prf(global_tp, global_fp, global_fn)
    print(f"\n  Done in {elapsed/60:.1f} min -- F1={f1_all:.3f} P={p_all:.3f} R={r_all:.3f}")
    print(f"  Parse errors (empty response on annotated docs): {errors}/{total}")

    row = {
        "model": model_name,
        "split": split_name,
        "threshold": 0.0,
        "n_docs": total,
        "P": p_all, "R": r_all, "F1": f1_all,
        "TP": global_tp, "FP": global_fp, "FN": global_fn,
        "time_min": round(elapsed / 60, 1),
    }
    for label in QUAERO_LABELS:
        ltp, lfp, lfn = per_label[label]
        _, _, lf1 = prf(ltp, lfp, lfn)
        row[f"F1_{label}"] = lf1

    return [row], per_label


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(rows: list[dict], per_label: dict, model_name: str):
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "__").replace(":", "_")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    labels_sorted = sorted(QUAERO_LABELS, key=lambda l: -prf(*per_label[l])[2])
    f1s = [prf(*per_label[l])[2] for l in labels_sorted]
    colors = ["#4CAF50" if f > 0.5 else "#FF9800" if f > 0.2 else "#F44336" for f in f1s]
    bars = ax.barh(labels_sorted, f1s, color=colors)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(f"F1 per label -- {model_name} (few-shot)")
    ax.grid(True, alpha=0.3, axis="x")

    ax = axes[1]
    row = rows[0]
    metrics = ["P", "R", "F1"]
    vals = [row["P"], row["R"], row["F1"]]
    colors_m = ["#2196F3", "#FF9800", "#4CAF50"]
    bars = ax.bar(metrics, vals, color=colors_m, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontweight="bold", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title(f"Global -- {model_name} (few-shot)\n{row['n_docs']} docs, {row['time_min']} min")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = PLOTS_DIR / f"{safe_name}_few_shot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Inspect one document
# ---------------------------------------------------------------------------

def inspect(doc_idx: int, query_fn, model_name: str, split: str):
    docs = load_split(split)
    doc = docs[doc_idx]
    tokens = doc["tokenized_text"]
    text, offsets = build_char_offsets(tokens)

    print(f"\nText: {text}\n")

    gold = {(e[0], e[1], e[2]) for e in doc["ner"]}

    print("Querying LLM...")
    response = query_fn(text)
    print(f"\nRaw response:\n{response}\n")

    entities = parse_llm_response(response)
    pred = set()
    for ent in entities:
        spans = find_entity_in_tokens(ent["text"], ent["label"], text, offsets)
        for span in spans:
            pred.add(span)

    print("=== GOLD ===")
    for ts, te, label in sorted(gold):
        marker = "OK" if (ts, te, label) in pred else "MISSED"
        print(f"  [{label}] {ts}-{te}  '{' '.join(tokens[ts:te+1])}'  {marker}")

    print("\n=== PREDICTIONS ===")
    for ts, te, label in sorted(pred):
        marker = "OK" if (ts, te, label) in gold else "FALSE POS"
        print(f"  [{label}] {ts}-{te}  '{' '.join(tokens[ts:te+1])}'  {marker}")

    tp, fp, fn = evaluate(gold, pred)
    p, r, f1 = prf(tp, fp, fn)
    print(f"\nDoc P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")


# ---------------------------------------------------------------------------
# Export predicted spans on BRAT test docs → pred_spans_llm.json
# ---------------------------------------------------------------------------

def _get_quaero_test_doc_paths() -> list:
    """Read test doc paths from split_test.txt + split_doc_paths.json."""
    split_dir = Path(__file__).parent.parent / "Rules" / "src"
    split_test_file = split_dir / "split_test.txt"
    split_doc_paths_file = split_dir / "split_doc_paths.json"

    if not split_test_file.exists():
        raise FileNotFoundError(
            f"split_test.txt not found at {split_test_file}. "
            "Run TBM_evaluation/prepare_gliner_data.py first."
        )

    test_ids = [
        line.strip()
        for line in split_test_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Load doc_id -> txt_path mapping
    if split_doc_paths_file.exists():
        all_paths = json.loads(split_doc_paths_file.read_text(encoding="utf-8"))
        test_paths = []
        for doc_id in test_ids:
            if doc_id in all_paths:
                test_paths.append(Path(all_paths[doc_id]))
            else:
                print(f"  WARNING: {doc_id} not found in split_doc_paths.json")
    else:
        # Fallback: search in both corpus dirs
        quaero_root = Path(__file__).parent.parent / "QUAERO_FrenchMed" / "corpus"
        corpus_dirs = [
            quaero_root / "train" / "MEDLINE",
            quaero_root / "test" / "MEDLINE",
        ]
        test_paths = []
        for doc_id in test_ids:
            found = False
            for corpus_dir in corpus_dirs:
                txt_path = corpus_dir / f"{doc_id}.txt"
                if txt_path.exists():
                    test_paths.append(txt_path)
                    found = True
                    break
            if not found:
                print(f"  WARNING: {doc_id}.txt not found in any corpus dir")

    print(f"  QUAERO split from file: {len(test_paths)} test docs")
    return test_paths


def export_spans_brat(query_fn, model_name: str):
    """Run LLM on QUAERO test docs, save character-offset spans."""
    test_paths = _get_quaero_test_doc_paths()
    out_path = OUT_DIR / "pred_spans_llm.json"

    predictions = {}
    total = len(test_paths)
    start_time = time.time()

    print(f"\n  Export spans: {model_name} on {total} QUAERO test docs...")

    for file_idx, txt_path in enumerate(test_paths):
        doc_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8").strip()

        response = query_fn(text)
        entities = parse_llm_response(response)

        pred_spans = set()
        for ent in entities:
            ent_lower = ent["text"].lower()
            text_lower = text.lower()
            pos = 0
            while True:
                idx = text_lower.find(ent_lower, pos)
                if idx == -1:
                    break
                pred_spans.add((idx, idx + len(ent["text"]), ent["label"]))
                pos = idx + 1

        predictions[doc_id] = sorted([list(s) for s in pred_spans])

        if (file_idx + 1) % 50 == 0 or file_idx == 0 or file_idx == total - 1:
            elapsed = time.time() - start_time
            speed = (file_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - file_idx - 1) / speed if speed > 0 else 0
            n_spans = len(pred_spans)
            print(f"    [{file_idx+1}/{total}] {doc_id}: {n_spans} spans "
                  f"({speed:.1f} doc/s, ETA {eta/60:.0f}min)")

    out_path.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    elapsed = time.time() - start_time
    total_spans = sum(len(v) for v in predictions.values())
    print(f"\n  Done in {elapsed/60:.1f} min")
    print(f"  Exported: {out_path}")
    print(f"  {total} docs, {total_spans} total spans")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Few-shot NER eval on QUAERO FrenchMed")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "mistral"])
    parser.add_argument("--model", default=None, help="Model name (default depends on backend)")
    parser.add_argument("--api-key", default=None, help="API key for Mistral backend")
    parser.add_argument("--split", default="test", choices=["train", "test", "both"])
    parser.add_argument("--inspect", type=int, default=None, help="Inspect doc index")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of docs")
    parser.add_argument("--export-spans", action="store_true",
                        help="Export pred_spans_llm.json from BRAT test docs")
    args = parser.parse_args()

    if args.model is None:
        args.model = "mistral-large-latest" if args.backend == "mistral" else "qwen2.5:14b"

    query_fn = make_query_fn(args.backend, args.model, args.api_key)

    if args.export_spans:
        export_spans_brat(query_fn, args.model)
        return

    if args.inspect is not None:
        inspect(args.inspect, query_fn, args.model, args.split if args.split != "both" else "test")
        return

    splits = ["train", "test"] if args.split == "both" else [args.split]
    all_rows = []

    for split_name in splits:
        docs = load_split(split_name)
        if args.limit:
            docs = docs[:args.limit]

        print(f"Model: {args.model}")
        print(f"Split: {split_name} ({len(docs)} docs)")
        print(f"Method: few-shot (4 examples)")

        rows, per_label = eval_llm(query_fn, args.model, docs, split_name)
        all_rows.extend(rows)

        plot_results(rows, per_label, f"{args.model}_{split_name}")

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        write_header = not RESULTS_CSV.exists()
        with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults saved -> {RESULTS_CSV}")

    from datetime import datetime
    for row in all_rows:
        params_row = {
            "run_date": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "backend": args.backend,
            "split": row["split"],
            "n_docs": row["n_docs"],
            "method": "few-shot",
        }
        write_header = not PARAMS_CSV.exists()
        with open(PARAMS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(params_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(params_row)
    print(f"Params saved -> {PARAMS_CSV}")

    print(f"\n=== Summary ===")
    for row in all_rows:
        print(f"  {row['split']:<6} F1={row['F1']:.3f}  P={row['P']:.3f}  R={row['R']:.3f}")
    if len(all_rows) == 2:
        gap = all_rows[0]["F1"] - all_rows[1]["F1"]
        print(f"  Gap train-test: {gap:+.3f}")


if __name__ == "__main__":
    main()
