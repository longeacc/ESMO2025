"""
Few-shot NER evaluation on CANTEMIST35 using an LLM.

Supports two backends:
  - Ollama (local)
  - Mistral API (cloud, requires API key)

Pipeline:
  1. Load test_gliner.json (produced by ../TBM_evaluation/prepare_gliner_data.py)
  2. For each document, prompt the LLM with few-shot examples
  3. Parse LLM output, align to token spans
  4. Evaluate with exact span match (same metrics as GLiNER eval)
  5. Save results to llm_results.csv, plot to plots/

Usage:
    # Ollama (local)
    python llm_ner_zero_shot.py --backend ollama --model qwen2.5:14b

    # Mistral API
    python llm_ner_zero_shot.py --backend mistral --model mistral-large-latest --api-key YOUR_KEY

    # Export spans for DEMNE routing
    python llm_ner_zero_shot.py --export-spans --backend mistral --model mistral-large-latest
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

RESULTS_CSV = OUT_DIR / "llm_results.csv"
PARAMS_CSV  = OUT_DIR / "llm_params.csv"

# ---------------------------------------------------------------------------
# Prompt template -- Few-shot with annotated examples from CANTEMIST35 corpus
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Tu es un annotateur expert en oncologie clinique. Ton role : identifier les entites \
nommees presentes dans des cas cliniques oncologiques en francais, selon le schema CANTEMIST35.

=== CATEGORIES (12 entites) ===

Traitement_specifique_du_cancer :
  Tout traitement anti-cancereux : chimiotherapie, radiotherapie, chirurgie oncologique, \
immunotherapie, hormonotherapie, therapie ciblee. Inclut les noms de molecules \
(cisplatine, bevacizumab, erlotinib...), les protocoles (FOLFOX, BEP, QT+RT), \
et les descriptions de traitements (\"6 cycles de cisplatine\", \"traitement adjuvant par capecitabine\").

Histologie_tumorale :
  Type histologique de la tumeur : adenocarcinome, carcinome epidermoide, melanome, \
hepatoblastome, glioblastome, sarcome, tumeur neuroendocrine, carcinome a grandes cellules, \
carcinome neuroendocrinien. Inclut les qualificatifs de differenciation.

Reponse_a_la_chimiotherapie :
  Evaluation de la reponse au traitement : reponse partielle/complete, progression de la maladie, \
maladie stable, remission, recidive, stabilisation. Inclut les descriptions d'imagerie de suivi \
(\"scanner de controle revele...\", \"TEP-TDM de reevaluation...\").

Stade_metastatique_avec_localisations :
  Stade TNM, stade chiffre (stade IV, IIIB), PRETEXT, presence de metastases avec localisation \
(metastases pulmonaires, osseuses, hepatiques), adenopathies suspectes, nodules metastatiques.

Topographie_du_primitif :
  Localisation anatomique de la tumeur primitive : \"carcinome epidermoide du col de l'uterus\", \
\"adenocarcinome rectal\", \"melanome acral\", \"tumeur testiculaire\", \"carcinome pulmonaire\".

Biomarqueurs_therapeutiques :
  Marqueurs biologiques orientant le traitement : HER2, EGFR, ALK, BRAF, KRAS, NRAS, PD-L1, \
Ki-67, TTF-1, C-KIT, BRCA1/2, recepteurs hormonaux, synaptophysine, chromogranine.

Stade_OMS_ECOG_Karnofsky :
  Score de performance : ECOG 0/1/2..., PS 1, Karnofsky, indice de performance.

Statut_tabagique :
  Consommation tabagique : fumeur, non-fumeur, ex-fumeur, X paquets-annees, tabagisme.

Symptomes :
  Manifestations cliniques rapportees : douleurs, dyspnee, cephalees, nausees, vomissements, \
asthenie, perte de poids, perte de vision, diarrhee, saignement, alopecie.

Signes_physiques :
  Constatations a l'examen clinique : adenopathie, hepatomegalie, cachexie, oedeme, ascite, \
ictere, masse palpable, etat general.

ATCD_geriatriques_et_medicaux_significatifs_pour_la_prise_en_charge :
  Antecedents medicaux pertinents : \"antecedents de...\", HTA, diabete, BPCO, hypothyroidie, \
\"sans antecedent medical pertinent\".

Evolutivite_en_lien_avec_le_cancer :
  Progression ou rechute liee au cancer (sans detail de localisation) : rechute, evolutivite, \
progression pulmonaire.

=== 3 EXEMPLES ANNOTES ===

Texte 1 (hepatoblastome) :
\"un homme de 22 ans diagnostique avec un hepatoblastome metastatique [...] douleurs abdominales \
[...] 10 paquets-annees [...] douleurs abdominales diffuses\"
[{"text": "hepatoblastome", "label": "Histologie_tumorale"},
 {"text": "hepatoblastome metastatique", "label": "Stade_metastatique_avec_localisations"},
 {"text": "douleurs abdominales", "label": "Symptomes"},
 {"text": "10 paquets-annees", "label": "Statut_tabagique"},
 {"text": "douleurs abdominales diffuses", "label": "Symptomes"}]

Texte 2 (carcinome epidermoide du col) :
\"carcinome epidermoide du col de l'uterus [...] QT + RT concomitants, avec six cycles de \
cisplatine [...] metastases viscerales au niveau pulmonaire et osseux [...] platine-taxane-bevacizumab\"
[{"text": "carcinome epidermoide", "label": "Histologie_tumorale"},
 {"text": "carcinome epidermoide du col de l'uterus", "label": "Topographie_du_primitif"},
 {"text": "QT + RT concomitants, avec six cycles de cisplatine", "label": "Traitement_specifique_du_cancer"},
 {"text": "metastases viscerales au niveau pulmonaire et osseux", "label": "Stade_metastatique_avec_localisations"},
 {"text": "platine-taxane-bevacizumab", "label": "Traitement_specifique_du_cancer"}]

Texte 3 (carcinome pulmonaire) :
\"ex-fumeuse depuis 9 ans (indice cumule 2 paquets/an) [...] sans antecedent medical pertinent \
[...] ECOG 0 [...] TTF-1 intensement positif [...] Carcinome pulmonaire a grandes cellules \
[...] carboplatine-paclitaxel-bevacizumab [...] EGFR positif/mute, mutation L858R-exon 21 [...] erlotinib\"
[{"text": "ex-fumeuse depuis 9 ans (indice cumule 2 paquets/an)", "label": "Statut_tabagique"},
 {"text": "sans antecedent medical pertinent", "label": "ATCD_geriatriques_et_medicaux_significatifs_pour_la_prise_en_charge"},
 {"text": "ECOG 0", "label": "Stade_OMS_ECOG_Karnofsky"},
 {"text": "TTF-1 intensement positif.", "label": "Biomarqueurs_therapeutiques"},
 {"text": "Carcinome pulmonaire a grandes cellules", "label": "Histologie_tumorale"},
 {"text": "carboplatine-paclitaxel-bevacizumab,", "label": "Traitement_specifique_du_cancer"},
 {"text": "EGFR positif/mute, mutation L858R-exon 21, mutation T790M-exon 20", "label": "Biomarqueurs_therapeutiques"},
 {"text": "erlotinib", "label": "Traitement_specifique_du_cancer"}]

=== REGLES ===

R1 - SPAN MAXIMAL : Annoter l'expression complete. Ne pas decomposer.
R2 - CHEVAUCHEMENTS : La meme portion de texte peut etre annotee avec plusieurs labels \
(ex: \"hepatoblastome metastatique\" = Histologie + Stade_metastatique).
R3 - PRECISION : Ne pas annoter les termes generiques non medicaux.

=== FORMAT DE REPONSE ===
Tableau JSON uniquement. Aucune explication.
Chaque element : {"text": "...", "label": "..."}
Si aucune entite : []"""


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
                if e["label"] in CANTEMIST_LABELS:
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

        for label in CANTEMIST_LABELS:
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
    for label in CANTEMIST_LABELS:
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

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    labels_sorted = sorted(CANTEMIST_LABELS, key=lambda l: -prf(*per_label[l])[2])
    f1s = [prf(*per_label[l])[2] for l in labels_sorted]
    # Shorten label names for display
    short_labels = [l.replace("_", " ")[:40] for l in labels_sorted]
    colors = ["#4CAF50" if f > 0.5 else "#FF9800" if f > 0.2 else "#F44336" for f in f1s]
    bars = ax.barh(short_labels, f1s, color=colors)
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
# Export predicted spans on BRAT test docs -> pred_spans_llm.json
# ---------------------------------------------------------------------------

def export_spans_brat(query_fn, model_name: str):
    """Run LLM on CANTEMIST35 test docs, sentence by sentence,
    then map predicted token spans back to document-level character offsets.

    Saves pred_spans_llm.json = {doc_id: [[begin, end, label], ...]}
    """
    import spacy
    nlp = spacy.load("fr_core_news_sm", disable=["ner", "lemmatizer"])

    brat_dir = Path(__file__).parent.parent / "CANTEMIST35_test"
    out_path = OUT_DIR / "pred_spans_llm.json"

    txt_files = sorted(brat_dir.glob("*.txt"))
    if not txt_files:
        print(f"ERROR: no .txt files in {brat_dir}")
        sys.exit(1)

    # Build all segments with doc_id + spaCy token offsets
    segments = []
    for txt_path in txt_files:
        doc_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8")
        doc = nlp(text)
        for sent in doc.sents:
            tokens = [tok.text for tok in sent]
            if not tokens:
                continue
            segments.append({
                "doc_id": doc_id,
                "tokens": tokens,
                "spacy_sent": sent,
            })

    total = len(segments)
    print(f"\n  Export spans: {model_name} on {total} segments "
          f"({len(txt_files)} docs)...")

    predictions = {}
    start_time = time.time()

    for i, seg in enumerate(segments):
        doc_id = seg["doc_id"]
        tokens = seg["tokens"]
        sent = seg["spacy_sent"]

        text, offsets = build_char_offsets(tokens)
        response = query_fn(text)
        entities = parse_llm_response(response)

        for ent in entities:
            tok_spans = find_entity_in_tokens(
                ent["text"], ent["label"], text, offsets,
            )
            for ts, te, label in tok_spans:
                char_start = sent[ts].idx
                char_end = sent[te].idx + len(sent[te].text)
                predictions.setdefault(doc_id, set()).add(
                    (char_start, char_end, label)
                )

        if (i + 1) % 50 == 0 or i == 0 or i == total - 1:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            print(f"    [{i+1}/{total}] ({speed:.1f} seg/s, ETA {eta/60:.0f}min)")

    serializable = {
        doc_id: sorted([list(s) for s in spans])
        for doc_id, spans in predictions.items()
    }
    # Include docs with 0 predictions
    for txt_path in txt_files:
        serializable.setdefault(txt_path.stem, [])

    out_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    elapsed = time.time() - start_time
    total_spans = sum(len(v) for v in serializable.values())
    print(f"\n  Done in {elapsed/60:.1f} min")
    print(f"  Exported: {out_path}")
    print(f"  {len(serializable)} docs, {total_spans} total spans, "
          f"{total} segments queried")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Few-shot NER eval on CANTEMIST35")
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
        print(f"Method: few-shot (3 examples)")

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
