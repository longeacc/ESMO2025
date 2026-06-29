"""
cantemist_brat_annotator.py
---------------------------
Rule-based (hard-coded regex) BRAT annotator for the CANTEMIST35 dataset.

CANTEMIST35 has 12 entity types from French oncology clinical case reports.
Regex patterns are built in French (re.IGNORECASE|re.UNICODE) covering:

    ATCD_geriatriques_et_medicaux_significatifs_pour_la_prise_en_charge
    Biomarqueurs_therapeutiques
    Evolutivite_en_lien_avec_le_cancer
    Histologie_tumorale
    Reponse_a_la_chimiotherapie
    Signes_physiques
    Stade_OMS_ECOG_Karnofsky
    Stade_metastatique_avec_localisations
    Statut_tabagique
    Symptomes
    Topographie_du_primitif
    Traitement_specifique_du_cancer

Output: BRAT `.ann` files with T-lines only:
    Tn<TAB><label> start end<TAB><span_text>

Offsets are absolute character indices over the whole .txt (multi-line safe).

Usage:
    python cantemist_brat_annotator.py ../../Emmanuelle_35_cantemist -o cantemist_pred_rules
"""

from pathlib import Path
import re
import os
import argparse
from typing import List, Tuple

# eco2ai is optional
try:
    from eco2ai import set_params, Tracker
    set_params(
        project_name="Consumption_of_cantemist_brat_annotator.py",
        experiment_description="Rule-based CANTEMIST35 annotation",
        file_name="Consumption_of_cantemist.csv",
    )
    _tracker = Tracker()
    _tracker.start()
except Exception:
    _tracker = None


# ==================================================================
#  Regex patterns  (label, compiled later)
#  French clinical case reports -> IGNORECASE | UNICODE.
#  Each entry: (raw_pattern, label, priority)
#  priority is used to resolve overlaps. Higher wins.
# ==================================================================

# ---- Stade_OMS_ECOG_Karnofsky (prio=10) ----
STADE_OMS = (
    r"\bECOG\s*(?:PS)?\s*(?:de\s+)?\d"
    r"|\bPS\s*(?:=\s*)?\d"
    r"|\bKarnofsky\s*>?\s*\d+\s*%?"
    r"|\bindice\s+de\s+performance\s+(?:de\s+)?\d"
    r"|\b[ÉE]tat\s+g[ée]n[ée]ral\s+\d",
    "Stade_OMS_ECOG_Karnofsky", 10,
)

# ---- Statut_tabagique (prio=10) ----
STATUT_TABAGIQUE = (
    r"\b\d+\s*paquets?[-/\s]?an(?:n[ée]es?)?\b"
    r"|\b(?:ex[-\s]?fumeu[rs]e?|non[-\s]?fumeu[rs]e?"
    r"|fumeu[rs]e?\s+acti[fv]e?"
    r"|fumeu[rs]e?)\b"
    r"|\btabagi[sq]\w*\b"
    r"|\bindice\s+cumul[ée]\s+\d+\s+paquets?",
    "Statut_tabagique", 10,
)

# ---- Biomarqueurs_therapeutiques (prio=9) ----
BIOMARQUEURS = (
    r"\bHER[-\s]?2(?:\s+(?:positif|n[ée]gatif|surexprim[ée]))?\b"
    r"|\bEGFR\s*(?:positif|n[ée]gatif|mut[ée]|mutat\w*)?\b"
    r"|\b(?:ALK|ROS[-\s]?1|BRAF|KRAS|NRAS|C[-\s]?KIT|BRCA[12])\s*"
    r"(?:(?:non\s+)?mut[ée]s?|positif|n[ée]gatif|natif|nativ[ae]|mutat\w*)?\b"
    r"|\bPD[-\s]?L1\s*(?:positif|n[ée]gatif|>\s*\d+)?\b"
    r"|\bKi[-\s]?67\s*(?:de\s+)?\d*%?\b"
    r"|\bTTF[-\s]?1\s*(?:positif|n[ée]gatif|intens[ée]ment\s+positif)?\b"
    r"|\br[ée]cepteurs?\s+(?:hormonaux|[oœe]strog[èe]nes?|prog[eé]st[eé]rone)\b"
    r"|\b(?:ER|PR)\s*[+-]\b"
    r"|\bimmunohistochimie\s+a?\s*montr[ée]\w*\s+.{5,120}(?:positivi|n[ée]gativ|marqueur)\w*"
    r"|\b(?:synaptophysine|chromogranine|CD56|CK7|b[êe]ta[-\s]?cat[ée]nine)\b",
    "Biomarqueurs_therapeutiques", 9,
)

# ---- Histologie_tumorale (prio=9) ----
HISTOLOGIE = (
    r"\b(?:m[ée]lanome|h[ée]patoblastome|glioblastome|sarcome|lymphome|"
    r"carcinome\s+[ée]pidermo[ïi]de|carcinome\s+canalaire\s+infiltrant|"
    r"carcinome\s+neuroendocrinien(?:\s+(?:m[ée]tastatique\s+)?[àa]\s+grandes\s+cellules)?|"
    r"carcinome\s+(?:pulmonaire\s+)?[àa]\s+grandes\s+cellules|"
    r"ad[ée]nocarcinome(?:\s+rectal)?|"
    r"tumeur\s+neuroendocrine|tumeur\s+[àa]\s+cellules\s+germinales|"
    r"n[ée]oplasie\s+germinale|"
    r"m[ée]soth[ée]liome(?:\s+pleural(?:\s+malin)?)?|"
    r"tumeur\s+maligne|"
    r"(?:bien|moyennement|peu)\s+diff[ée]renci[ée]\w*)\b",
    "Histologie_tumorale", 9,
)

# ---- Stade_metastatique_avec_localisations (prio=8) ----
STADE_META = (
    r"\bm[ée]tastas(?:e|es)\s+\w[\w\s]{2,60}(?:pulmonaire|osseu[sx]|h[ée]patique|"
    r"c[ée]r[ée]bral|surr[ée]nal|p[ée]riton[ée]al|m[ée]lanome|colorectal)\w*\b"
    r"|\bm[ée]tastas(?:e|es)\s+(?:visc[ée]rales?|de\s+\w+)\b"
    r"|\bm[ée]tastatique\b"
    r"|\bstade\s+(?:IV|IIIB|III[ABC]?)\b"
    r"|\b[cp]?T\d[a-d]?\s*N\d[a-c]?\s*M[01]\b"
    r"|\bPRETEXT[-\s]?(?:IV|III|II|I)\b"
    r"|\bad[ée]nopathies?\s+(?:m[ée]diastin\w+|supraclav\w+|"
    r"cervical\w+|inguinal\w+|r[ée]trop[ée]riton[ée]al\w*)\b"
    r"|\bad[ée]nopathies?\s+\w[\w\s]{2,40}suspect\w*\s+de\s+maligni"
    r"|\bnodules?\s+pulmonaires?\b"
    r"|\borigine\s+m[ée]tastatique\b"
    r"|\bconglom[ée]rat\s+ad[ée]nopathique\b"
    r"|\blymphad[ée]nopathie\s+r[ée]trop[ée]riton[ée]al\w*\b",
    "Stade_metastatique_avec_localisations", 8,
)

# ---- Traitement_specifique_du_cancer (prio=8) ----
TRAITEMENT = (
    # Specific drug names (French oncology)
    r"\b(?:pazopanib|cap[ée]citabine|[ée]v[ée]rolimus|topotécan|topot[ée]can|"
    r"sunitinib|doxorubicine|adriamycine|cisplat(?:ine|in)|CDDP|"
    r"carboplatine?|paclitaxel|taxol|bev[aà]cizuma?b?|erlotinib|"
    r"vinblastine|gemcitabine|t[ée]mozol[oa]mide|ipilimumab|fot[ée]mustine|"
    r"dacarbazine|c[ée]tuximab|irinotécan|irinotécan|irinot[ée]can|"
    r"oxaliplatine?|bl[ée]omycine|[ée]toposide|5[-\s]?FU|"
    r"interf[ée]ron(?:\s+alfa?\s*2?b?)?|leucovorine|"
    r"nivolumab|pembrolizumab|sorafenib|imatinib)\b"
    # Treatment classes
    r"|\b(?:chimioth[ée]rapie|radioth[ée]rapie|immunoth[ée]rapie|"
    r"hormonoth[ée]rapie|chimioradioth[ée]rapie)\b"
    r"|\btraitement\s+(?:de\s+)?chimioth[ée]rapie\b"
    r"|\btraitement\s+(?:adjuvant|n[ée]oadjuvant|d['’]entretien)"
    r"(?:\s+par\s+\S+(?:\s+\S+)?)?\b"
    r"|\b(?:QT|RT)\s*(?:\+\s*(?:QT|RT))?\s*(?:concomitant\w*)?\b"
    r"|\b(?:FOLFOX|FOLFIRI|XELOX|BEP)\b"
    r"|\b\d+\s+cycles?\s+(?:de\s+)?\w+"
    # Surgical treatments
    r"|\b(?:chirurgie(?:\s+radicale)?|r[ée]section(?:\s+chirurgicale)?|"
    r"orchid(?:ectomie|ecto)|excision|amputation\s+abdomino[-\s]?p[ée]rin[ée]ale|"
    r"omentectomie|r[ée]section\s+(?:transur[ée]trale|des?\s+m[ée]tastas))\b"
    r"|\bsch[ée]ma\s+th[ée]rapeutique\b"
    r"|\bplatine[-\s]?taxane[-\s]?bev\w*\b"
    r"|\bcarboplatine[-\s]?(?:paclitaxel|[ée]toposide)\w*\b"
    r"|\bcarboplatine\s+AUC\d+[-\s]?[ée]toposide\b",
    "Traitement_specifique_du_cancer", 8,
)

# ---- Reponse_a_la_chimiotherapie (prio=7) ----
REPONSE_CHIMIO = (
    r"\br[ée]ponse\s+(?:partielle|compl[èe]te|radiologique(?:\s+(?:partielle|compl[èe]te))?|"
    r"pathologique\s+compl[èe]te|m[ée]tabolique\s+compl[èe]te|"
    r"dissoci[ée]e)\b"
    r"|\br[ée]mission\s+(?:partielle|compl[èe]te)\b"
    r"|\bprogression\s+(?:de\s+la\s+maladie|tumorale|radiologique|"
    r"h[ée]patique|pulmonaire|au\s+niveau)\b"
    r"|\bmaladie\s+(?:radiologiquement\s+)?stable\b"
    r"|\bstabilisation\s+de\s+la\s+r[ée]ponse\b"
    r"|\bRECIST\b"
    r"|\bnette\s+(?:am[ée]lioration|progression)\b"
    r"|\bbonne\s+r[ée]ponse\b"
    r"|\br[ée]cidive\s+(?:endobronchique|locale)\b"
    r"|\bapparition\s+de\s+(?:l[ée]sions?|nodules?|polypes?)\b",
    "Reponse_a_la_chimiotherapie", 7,
)

# ---- Topographie_du_primitif (prio=7) ----
TOPOGRAPHIE = (
    r"\b(?:carcinome|ad[ée]nocarcinome|tumeur|m[ée]lanome|m[ée]soth[ée]liome)"
    r"\s+\w[\w\s]{0,40}(?:du\s+(?:col\s+de\s+l['’]ut[ée]rus|"
    r"poumon|sein|c[ôo]lon|rectum|pancr[ée]as|testicule|rein|foie|"
    r"cerveau|estomac|vessie|rectum)|"
    r"pleural\w*|rectal\w*|pulmonaire\w*|testiculaire\w*|acral\w*)\b"
    r"|\btumeur\s+(?:de\s+la\s+m[êe]me\s+)?(?:marge\s+anale|testiculaire)\b"
    r"|\bn[ée]oformation\s+.{5,40}marge\s+anale\b"
    r"|\bm[ée]lanome\s+(?:lentigineux\s+)?acral\b"
    r"|\bcarcinome\s+pulmonaire\b"
    r"|\borigine\s+(?:pulmonaire|colorectal\w*|h[ée]patique)\b",
    "Topographie_du_primitif", 7,
)

# ---- Symptomes (prio=6) ----
SYMPTOMES = (
    r"\b(?:douleurs?\s+(?:abdominale|thoracique|pleur[ée]tique)\w*"
    r"(?:\s+(?:diffus\w*|bilateral\w*|mod[ée]r[ée]\w*))?)\b"
    r"|\b(?:crampes|alop[ée]cie|diarrh[ée]e|mucite|h[ée]mat[ée]m[èe]se|"
    r"h[ée]matoch[ée]zie|dyspn[ée]e|c[ée]phal[ée]es?|"
    r"fi[èe]vre|naus[ée]es?|vomissements?|asth[ée]nie|fatigue|"
    r"anorexie|saignement(?:\s+\w+)?|lombalgies?|asymptomatique|"
    r"constipation|neurop[ée]nie\s+f[ée]brile|neuropathie|"
    r"thrombop[ée]nie|h[ée]matome)\b"
    r"|\bc[ée]phal[ée]es?\s+\w[\w\s]{2,60}(?:oppressiv|intens)\w*\b"
    r"|\bperte\s+de\s+(?:poids|vision)(?:\s+\w+)*\b"
    r"|\bsyndrome\s+toxique\b"
    r"|\bcrise\s+tonico[-\s]?clonique\b"
    r"|\bmaux\s+de\s+t[êe]te\b"
    r"|\bhypertension\s+art[ée]rielle\b",
    "Symptomes", 6,
)

# ---- Signes_physiques (prio=6) ----
SIGNES_PHYSIQUES = (
    r"\b(?:ad[ée]nopathie|h[ée]patomégalie|cachexie|[oœ]?[ée]d[èe]me|"
    r"p[âa]leur|ascite|ict[èe]re|masse\s+palpable|"
    r"hypophon[èe]se|tachycarde)\b"
    r"(?:\s+\w+)*"
    r"|\blymphad[ée]nopathie\s+palpable\b"
    r"|\b(?:bon\s+)?[ée]tat\s+g[ée]n[ée]ral\b"
    r"|\bperte\s+de\s+poids\s+d['’]environ\s+\d+\s+kg\b"
    r"|\btoucher\s+rectal\b"
    r"|\babdomen\s+mou\b"
    r"|\bzone\s+de\s+plus\s+grande\s+consistance\b"
    r"|\btesticule\s+\w+\s+plus\s+petit\b",
    "Signes_physiques", 6,
)

# ---- ATCD_geriatriques_et_medicaux (prio=4) ----
ATCD = (
    r"\b(?:ant[ée]c[ée]dents?\s+(?:de\s+|personnels?\s*:?\s*|m[ée]dicaux?\s*|"
    r"m[ée]dico[-\s]?chirurgicaux?\s*|familiaux?\s*)?)\w[\w\s,\-’']{3,80}\b"
    r"|\bsans\s+ant[ée]c[ée]dent\w*(?:\s+m[ée]dica\w*)?\s*(?:pertinent\w*)?\b"
    r"|\b(?:HTA|hypertension(?:\s+art[ée]rielle)?|diab[èe]te|BPCO|"
    r"hypothyro[ïi]die|cryptorchidie|bronchite\s+emphys[ée]mateus\w*)\b"
    r"|\ballergie\s+m[ée]dicamenteus\w*\b"
    r"|\bd[ée]ficit\s+(?:mod[ée]r[ée]\s+)?en\s+facteur\b",
    "ATCD_geriatriques_et_medicaux_significatifs_pour_la_prise_en_charge", 4,
)

# ---- Evolutivite_en_lien_avec_le_cancer (prio=3) ----
EVOLUTIVITE = (
    r"\bprogression\s+pulmonaire\b"
    r"|\brechute\b"
    r"|\b[ée]volutivit[ée]\b"
    r"|\bintervalle\s+sans\s+progression\b",
    "Evolutivite_en_lien_avec_le_cancer", 3,
)


# Order matters only for tie-breaking display; overlap resolution uses priority.
ALL_PATTERNS: List[Tuple[str, str, int]] = [
    STADE_OMS,
    STATUT_TABAGIQUE,
    BIOMARQUEURS,
    HISTOLOGIE,
    STADE_META,
    TRAITEMENT,
    REPONSE_CHIMIO,
    TOPOGRAPHIE,
    SYMPTOMES,
    SIGNES_PHYSIQUES,
    ATCD,
    EVOLUTIVITE,
]

_COMPILED = [
    (re.compile(pat, re.IGNORECASE | re.UNICODE), label, prio)
    for (pat, label, prio) in ALL_PATTERNS
]


# ==================================================================
#  Extraction
# ==================================================================
def extract_entities(text: str) -> List[Tuple[str, int, int, str, int]]:
    """
    Run every pattern over `text`.
    Returns list of (span_text, start, end, label, priority).
    """
    out = []
    for regex, label, prio in _COMPILED:
        for m in regex.finditer(text):
            s, e = m.start(), m.end()
            if e <= s:
                continue
            out.append((m.group(), s, e, label, prio))
    return out


def _overlaps(a, b) -> bool:
    return a[1] < b[2] and b[1] < a[2]


def resolve_overlaps(
    spans: List[Tuple[str, int, int, str, int]]
) -> List[Tuple[str, int, int, str]]:
    """
    Global overlap resolution. When two spans overlap, keep the higher-priority
    one (tie -> longer span -> earlier start). Returns (span_text, start, end, label).
    """
    # de-dup exact
    seen = set()
    uniq = []
    for sp in spans:
        key = (sp[3], sp[1], sp[2])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(sp)

    # greedy: best first, drop anything overlapping an already-kept span
    uniq.sort(key=lambda sp: (-sp[4], -(sp[2] - sp[1]), sp[1]))
    kept = []
    for sp in uniq:
        if any(_overlaps(sp, k) for k in kept):
            continue
        kept.append(sp)

    kept.sort(key=lambda sp: (sp[1], sp[2]))
    return [(sp[0], sp[1], sp[2], sp[3]) for sp in kept]


# ==================================================================
#  BRAT writer (T-lines only, matches gold scheme)
# ==================================================================
def annotate_text(text: str) -> List[Tuple[str, int, int, str]]:
    return resolve_overlaps(extract_entities(text))


def write_brat_ann(text: str, ann_path: Path) -> List[str]:
    spans = annotate_text(text)
    lines = []
    for i, (span_text, start, end, label) in enumerate(spans, 1):
        safe = span_text.replace("\n", " ").replace("\r", " ")
        lines.append(f"T{i}\t{label} {start} {end}\t{safe}")
    ann_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return lines


def annotate_txt_folder(
    in_dir: str,
    out_dir: str | None = None,
    *,
    recursive: bool = False,
    overwrite: bool = True,
) -> List[str]:
    """
    Annotate every .txt under `in_dir`. Writes <stem>.ann to `out_dir` (mirroring
    subfolders) or next to the .txt when out_dir is None.
    """
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    processed: List[str] = []
    for root, _dirs, files in os.walk(in_dir):
        for f in files:
            if not f.lower().endswith(".txt"):
                continue
            txt_path = Path(root) / f
            if out_dir:
                rel = Path(root).relative_to(in_dir)
                target_dir = Path(out_dir) / rel
                target_dir.mkdir(parents=True, exist_ok=True)
                ann_path = target_dir / (txt_path.stem + ".ann")
            else:
                ann_path = txt_path.with_suffix(".ann")

            if ann_path.exists() and not overwrite:
                processed.append(str(ann_path))
                continue

            with open(txt_path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
                text = fh.read()
            write_brat_ann(text, ann_path)
            processed.append(str(ann_path))

        if not recursive:
            break
    return processed


# ==================================================================
#  CLI
# ==================================================================
def _cli():
    p = argparse.ArgumentParser(
        description="Rule-based BRAT annotator for CANTEMIST35 (regex)."
    )
    p.add_argument("input", help="Input .txt file OR a folder of .txt files")
    p.add_argument("-o", "--output", help="Output .ann file or output folder")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="Recurse into subfolders (folder mode)")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    if in_path.is_dir():
        written = annotate_txt_folder(str(in_path), args.output,
                                      recursive=args.recursive)
        print(f"Wrote {len(written)} .ann files")
    else:
        out_path = Path(args.output) if args.output else in_path.with_suffix(".ann")
        with open(in_path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
            text = fh.read()
        write_brat_ann(text, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    _cli()

if _tracker is not None:
    try:
        _tracker.stop()
    except Exception as e:
        print(f"\nWarning: eco2ai tracking error: {e}")
