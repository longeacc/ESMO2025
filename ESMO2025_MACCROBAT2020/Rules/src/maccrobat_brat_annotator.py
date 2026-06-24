"""
maccrobat_brat_annotator.py
---------------------------
Rule-based (hard-coded regex) BRAT annotator for the MACCROBAT2020 dataset,
built in the same spirit as `biomarker_brat_annotator.py` (Breast).

MACCROBAT2020 has ~41 entity types. Only a subset is reliably extractable by
regex: the *structured* entities (demographics, temporal, measurements,
dose, and closed-vocabulary descriptors). The *open-vocabulary* clinical
entities (Sign_symptom, Disease_disorder, Biological_structure,
Diagnostic_procedure, Medication, Therapeutic_procedure, Detailed_description,
History, Clinical_event, Coreference, ...) are NOT covered here: they require a
NER model or large terminologies, not patterns.

Regex-covered labels (matching the gold BRAT label names exactly):
    Age, Sex, Date, Duration, Frequency, Dosage, Administration,
    Distance, Area, Volume, Weight,
    Severity, Color, Shape, Texture, Time, Lab_value

Output: BRAT `.ann` files with T-lines only:
    Tn<TAB><label> start end<TAB><span_text>
which is directly comparable to the gold `.ann` with the provided
`evaluate_ann_folders()` harness (label + character-offset overlap).

Offsets are absolute character indices over the whole .txt (multi-line safe),
because all matching is done with re.finditer on the full document text.
"""

from pathlib import Path
import re
import os
import argparse
from typing import List, Tuple

# eco2ai is optional: keep parity with the Breast script but never crash if absent
try:
    from eco2ai import set_params, Tracker
    set_params(
        project_name="Consumption_of_maccrobat_brat_annotator.py",
        experiment_description="Rule-based MACCROBAT2020 annotation",
        file_name="Consumption_of_maccrobat.csv",
    )
    _tracker = Tracker()
    _tracker.start()
except Exception:  # pragma: no cover - eco2ai not installed / env issue
    _tracker = None


# ==================================================================
#  Regex patterns  (label, compiled later)
#  English clinical case reports -> ASCII-centric, IGNORECASE.
#  Each entry: (raw_pattern, label, priority)
#  priority is used only to resolve overlaps inside the dimensional
#  measurement family (Volume > Area > Distance/Weight). Higher wins.
# ==================================================================

# ---- Demographics ----
AGE = (
    r"\b\d{1,3}\s*[-–]?\s*(?:year|yr)s?\s*[-–]?\s*old\b"
    r"|\b\d{1,3}\s+years?\s+of\s+age\b"
    r"|\baged\s+\d{1,3}\b",
    "Age", 9,
)

SEX = (
    r"\b(?:wom[ae]n|m[ae]n|males?|females?|boys?|girls?|"
    r"lad(?:y|ies)|gentlem[ae]n|"
    r"primigravida|primipara|multipara|nullipara|primiparous)\b",
    "Sex", 9,
)

# ---- Temporal ----
# Date = a quantity tied to a temporal RELATION (later/after/ago/...), an
# explicit month/year, ordinal-day, or age-of expressions.
DATE = (
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b"
    r"|\b(?:19|20)\d{2}\b"
    r"|\b(?:after|within|following|since)\s+"
    r"(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:day|week|month|year)s?\b"
    r"|\b(?:\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:day|week|month|year)s?\s+"
    r"(?:later|after|ago|prior|previously|before|earlier|post|on)\b"
    r"|\b(?:next|following|previous|same|second|third|first|fourth|fifth)\s+day\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)\s+day\b"
    r"|\bday\s+\d{1,2}\b"
    r"|\b(?:age|at\s+age)\s+(?:of\s+)?\d{1,3}\b"
    r"|\b\d{1,3}\s+(?:years?|months?)\s+of\s+age\b"
    r"|\bpast\s+(?:year|month|week|few\s+(?:years|months|weeks))\b",
    "Date", 7,
)

# Duration = bare span quantity (no relation word).
DURATION = (
    r"\b(?:\d{1,3}|several|few|a\s+couple\s+of|a)\s*[-–]?\s*"
    r"(?:second|minute|hour|day|week|month|year|decade)s?\b",
    "Duration", 4,
)

FREQUENCY = (
    r"\b(?:once|twice|thrice)\s+(?:a\s+|per\s+|each\s+)?"
    r"(?:day|daily|week|weekly|month|monthly|year)\b"
    r"|\bevery\s+(?:other\s+)?\d*\s*(?:hour|day|week|month|year)s?\b"
    r"|\b\d+\s+times?\s+(?:per|a|each)\s+(?:day|week|month|year)\b"
    r"|\bseveral\s+times\b|\b\d+\s+times\b"
    r"|\b(?:daily|weekly|monthly|yearly|nightly|hourly|"
    r"intermittent(?:ly)?|frequent(?:ly)?|occasional(?:ly)?|"
    r"recurrent|sporadic|sometimes|rarely|"
    r"regularly|periodically|continuously|constantly)\b"
    r"|\b(?:q\.?d|b\.?i\.?d|t\.?i\.?d|q\.?h\.?s|p\.?r\.?n)\.?\b",
    "Frequency", 6,
)

TIME = (
    r"\bwithin\s+(?:the\s+)?(?:first\s+)?\d*\s*"
    r"(?:second|minute|hour|h|day)s?\b"
    r"|\bwithin\s+(?:hours|minutes|seconds)\b"
    r"|\b(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|few)\s*"
    r"(?:h|hours?|minutes?|seconds?)\s+(?:later|after|post|postpartum)\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|few|several)\s+"
    r"hours?\s+postpartum\b"
    r"|\b(?:the\s+)?(?:following|next|same)\s+"
    r"(?:morning|night|evening|afternoon|day)\b"
    r"|\bthat\s+(?:night|morning|evening|afternoon)\b"
    r"|\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?\b"
    r"|\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b",
    "Time", 7,
)

# ---- Dose / administration ----
DOSAGE = (
    r"\b(?:high|low|standard|maximum|maintenance|full|reduced)\s*[-\s]\s*doses?\b"
    r"|\b\d+(?:\.\d+)?\s*"
    r"(?:mg|g|µg|mcg|ug)(?:\s*/\s*(?:kg|m2|day|d|dose))?"
    r"(?:\s*(?:once|twice|three\s+times)?\s*"
    r"(?:daily|per\s+day|/day|/d|bid|tid|qd))?\b"
    r"|\b\d+(?:\.\d+)?\s*(?:IU|units?)\b"
    r"|\b\d+(?:\.\d+)?\s*Gy\b"
    r"|\b\d+\s+million\s+units?\b"
    r"|\b\d+\s+(?:cycles?|fractions?|sessions?)\b",
    "Dosage", 6,
)

ADMINISTRATION = (
    r"\b(?:intravenous(?:ly)?|oral(?:ly)?|per\s+os|"
    r"subcutaneous(?:ly)?|intramuscular(?:ly)?|topical(?:ly)?|"
    r"parenteral(?:ly)?|sublingual(?:ly)?|transdermal(?:ly)?|"
    r"intraperitoneal(?:ly)?|intrathecal(?:ly)?|intradermal(?:ly)?|"
    r"inhaled|inhalation|enteral|nasogastric|"
    r"infusions?|injections?|tablets?|capsules?|suppositor(?:y|ies))\b"
    r"|\b(?:i\.?v\.?|p\.?o\.?|s\.?c\.?|s\.?q\.?|i\.?m\.?|i\.?p\.?)\b",
    "Administration", 6,
)

# ---- Measurements (dimensional family resolved by priority) ----
# Allow unit after each operand and 'by' as separator (e.g. "4 cm × 9 cm",
# "3.0 by 2.6 cm", "2.5 cm × 2.4 cm in diameter").
_NUM = r"\d+(?:\.\d+)?"
_UNIT = r"(?:cm|mm)"
VOLUME = (
    rf"\b{_NUM}\s*{_UNIT}?\s*(?:[x×✕]|by)\s*{_NUM}\s*{_UNIT}?\s*"
    rf"(?:[x×✕]|by)\s*{_NUM}\s*{_UNIT}\b"
    r"|\b\d+(?:\.\d+)?\s*(?:ml|mL|cc|l|L|litres?|liters?|milli?litres?|milliliters?)"
    r"(?:\s*/\s*(?:day|d))?\b",
    "Volume", 9,
)

AREA = (
    rf"\b{_NUM}\s*{_UNIT}?\s*(?:[x×✕]|by)\s*{_NUM}\s*{_UNIT}"
    r"(?:\s+in\s+diameter)?\b",
    "Area", 8,
)

DISTANCE = (
    r"\b\d+(?:\.\d+)?\s*[-\s]?\s*(?:cm|mm|centimet(?:er|re)s?|millimet(?:er|re)s?)\b",
    "Distance", 5,
)

WEIGHT = (
    r"\b\d+(?:\.\d+)?\s*(?:kg|kilograms?)\b",
    "Weight", 7,
)

# ---- Closed-vocabulary descriptors (trimmed to high-precision cores) ----
SEVERITY = (
    r"\b(?:severe(?:ly)?|mild(?:ly)?|moderate(?:ly)?|massive(?:ly)?|"
    r"extensive(?:ly)?|marked(?:ly)?|slight(?:ly)?|profound(?:ly)?|"
    r"high\s*[-\s]\s*grade|low\s*[-\s]\s*grade)\b",
    "Severity", 4,
)

COLOR = (
    r"\b(?:whit(?:e|ish)|black(?:ish)?|red(?:dish)?|blu(?:e|ish)|"
    r"yellow(?:ish)?|green(?:ish)?|brown(?:ish)?|gr[ae]y(?:ish)?|"
    r"pink(?:ish)?|purpl(?:e|ish)|violet|tan|"
    r"salmon|hyperpigmented|depigmented)\b"
    r"|\b[a-z]+\s*[-\s]\s*colou?red\b",
    "Color", 5,
)

SHAPE = (
    r"\b(?:round(?:ed)?|oval|ovoid|spherical|circular|elliptical|"
    r"nodular|lobulated|polygonal|globular|crystalline|elongated|"
    r"cylindrical|tubular|linear|stellate|annular|dome[-\s]shaped|"
    r"spindle\s*[-\s]?\s*shaped)\b"
    r"|\b[a-z]+\s*[-\s]\s*shaped\b",
    "Shape", 5,
)

TEXTURE = (
    r"\b(?:smooth|soft|firm|hard|rough|rigid|fibrous|flat|dry|dense|"
    r"granular|spongy|rubbery|gelatinous|fluctuant|indurated|"
    r"vesicular|friable|boggy|ground\s*[-\s]?\s*glass)\b",
    "Texture", 4,
)

# ---- Lab values (numeric + units, grades, qualitative cues) ----
LAB_VALUE = (
    r"\bgrade\s+\d(?:\s*/\s*\d+)?\b"
    r"|\b\d+(?:\.\d+)?\s*"
    r"(?:mmHg|bpm|beats?\s*/\s*min|/\s*min|mg\s*/\s*dL|g\s*/\s*dL|"
    r"mg\s*/\s*L|mmol\s*/\s*L|µmol\s*/\s*L|umol\s*/\s*L|mEq\s*/\s*L|"
    r"U\s*/\s*L|IU\s*/\s*L|ng\s*/\s*mL|µg\s*/\s*L|/\s*µL|/\s*mm3|"
    r"cells?\s*/\s*[µu]?L|%|°\s*C|°\s*F)\b"
    r"|\b(?:within\s+normal\s+limits|wnl|unremarkable|negative|positive|"
    r"normal|abnormal|elevated|increased?|decreased?|reduced|raised|"
    r"stable|improved|improvement|uneventful|resolved|worsened|unchanged|"
    r"good|poor)\b",
    "Lab_value", 3,
)


# Order matters only for tie-breaking display; overlap resolution uses priority.
ALL_PATTERNS: List[Tuple[str, str, int]] = [
    AGE, SEX, DATE, DURATION, FREQUENCY, TIME,
    DOSAGE, ADMINISTRATION,
    VOLUME, AREA, DISTANCE, WEIGHT,
    SEVERITY, COLOR, SHAPE, TEXTURE, LAB_VALUE,
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
    one (tie -> longer span -> earlier start). This makes the temporal family
    (Date > Time > Frequency > Duration) and the measurement/dose family
    (Volume > Area > Weight > Dosage/Distance) mutually exclusive, killing the
    double-fire false positives (e.g. "6 months later" no longer yields both
    Date and Duration; "60 mg daily" no longer yields both Dosage and Frequency).
    Returns (span_text, start, end, label).
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

            text = txt_path.read_text(encoding="utf-8", errors="ignore")
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
        description="Rule-based BRAT annotator for MACCROBAT2020 (regex)."
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
        text = in_path.read_text(encoding="utf-8", errors="ignore")
        write_brat_ann(text, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    _cli()

if _tracker is not None:
    try:
        _tracker.stop()
    except Exception as e:  # pragma: no cover
        print(f"\nWarning: eco2ai tracking error: {e}")
