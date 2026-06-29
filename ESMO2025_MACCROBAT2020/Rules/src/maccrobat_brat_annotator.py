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
    r"(?<=weight was )\d+(?:\.\d+)?\s*(?:kg|kilograms?)\b"
    r"|(?<=weight of )\d+(?:\.\d+)?\s*(?:kg|kilograms?)\b"
    r"|(?<=weight is )\d+(?:\.\d+)?\s*(?:kg|kilograms?)\b"
    r"|(?<=weighed )\d+(?:\.\d+)?\s*(?:kg|kilograms?)\b"
    r"|(?<=weighing )\d+(?:\.\d+)?\s*(?:kg|kilograms?)\b",
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


# ==================================================================
#  NEW: 24 additional entity types
# ==================================================================

# ---- Outcome ----
OUTCOME = (
    r"\b(?:died|death|passed\s+away|expired|survived|recovered|"
    r"full\s+recovery|lethal|fatal|alive\s+and\s+well|"
    r"doing\s+(?:very\s+)?well|resolution|asymptomatic|"
    r"continues\s+to\s+do\s+(?:very\s+)?well|felt\s+well)\b",
    "Outcome", 11,
)

# ---- Personal_background (ethnicity / nationality / origin) ----
PERSONAL_BACKGROUND = (
    r"\b(?:caucasian|white|asian|japanese|chinese|korean|"
    r"african[\s-]?american|african|afro-brazilian|"
    r"hispanic|latin[oa]?|cuban|brazilian|portuguese|"
    r"jewish|ashkenazi(?:\s+jewish)?|arab|muslim|palestinian|"
    r"british|italian|spanish|french|german|dutch|turkish|tunisian|"
    r"tibetan|han|malay|sri\s+lankan|azeri|pakistani|indian)\b",
    "Personal_background", 10,
)

# ---- Clinical_event ----
CLINICAL_EVENT = (
    r"\b(?:presented|present(?:ing|s)?|presentation|"
    r"admitted|admission|readmitted|"
    r"discharged|discharge|"
    r"referred|referral|"
    r"transferred|transfer|"
    r"follow[\s-]?up|followed[\s-]?up|"
    r"hospitali[sz](?:ed|ation)|"
    r"consultation|"
    r"diagnosed|diagnosis|"
    r"delivered|born|birth|"
    r"visited|visit)\b",
    "Clinical_event", 5,
)

# ---- Subject (family members, medical personnel as subjects) ----
SUBJECT = (
    r"\b(?:mother|father|parents|sister|brother|siblings?|"
    r"daughter|son|husband|wife|"
    r"neonate|infant|baby|fetus|donor|"
    r"first-degree\s+relatives|family\s+members)\b",
    "Subject", 10,
)

# ---- Nonbiological_location ----
NONBIOLOGICAL_LOCATION = (
    r"\b(?:hospital|emergency\s+(?:department|room)|"
    r"intensive\s+care\s+unit|icu|"
    r"coronary\s+care\s+unit|ccu|"
    r"operating\s+(?:room|theatre)|"
    r"oncology\s+department|"
    r"outpatient\s+clinic|clinic|"
    r"primary\s+care|"
    r"(?:local|university|our)\s+hospital)\b"
    r"|\b(?:ed|er|ward|nicu)\b",
    "Nonbiological_location", 5,
)

# ---- Medication (drug names, drug classes) ----
MEDICATION = (
    # Chemotherapy & targeted
    r"\b(?:chemotherapy|gemcitabine|cisplatin|carboplatin|"
    r"paclitaxel|docetaxel|doxorubicin|cyclophosphamide|"
    r"methotrexate|5-fluorouracil|fluorouracil|etoposide|"
    r"vincristine|vinblastine|irinotecan|oxaliplatin|"
    r"rituximab|nivolumab|pembrolizumab|erlotinib|apatinib|"
    r"bevacizumab|cetuximab|sorafenib|sunitinib|imatinib|"
    r"tamoxifen|letrozole|bortezomib|thalidomide|lenalidomide|"
    r"azathioprine|mycophenolate(?:\s+mofetil)?|tacrolimus|"
    r"infliximab|adalimumab|6-mp)\b"
    # Steroids & anti-inflammatory
    r"|\b(?:prednisolone|prednisone|methylprednisolone|"
    r"dexamethasone|hydrocortisone|"
    r"corticosteroids?|steroids?|"
    r"ibuprofen|naproxen|diclofenac)\b"
    # Cardiovascular
    r"|\b(?:furosemide|spironolactone|digoxin|amiodarone|"
    r"warfarin|heparin|rivaroxaban|acenocumarol|"
    r"aspirin|clopidogrel|"
    r"ramipril|captopril|enalapril|losartan|olmesartan|"
    r"amlodipine|nifedipine|diltiazem|verapamil|"
    r"metoprolol(?:\s+succinate)?|carvedilol|atenolol|bisoprolol|"
    r"landiolol(?:\s+hydrochloride)?|"
    r"dobutamine|dopamine|noradrenaline|norepinephrine|"
    r"nitroprusside|nitroglycerin)\b"
    # Anti-infective
    r"|\b(?:antibiotics?|penicillin|amoxicillin|ampicillin|"
    r"ceftriaxone|cefazolin|ceftazidime|"
    r"levofloxacin|ciprofloxacin|moxifloxacin|"
    r"meropenem|vancomycin|amikacin|gentamicin|"
    r"trimethoprim[-\s]?sulfamethoxazole|sulfamethoxazole/trimethoprim|"
    r"isoniazid|ethambutol|pyrazinamide|rifampicin|"
    r"fluconazole|voriconazole|posaconazole|micafungin|"
    r"ribavirin|acyclovir|ganciclovir)\b"
    # CNS & psych
    r"|\b(?:levodopa|phenytoin|levetiracetam|valproate|"
    r"sodium\s+valproate|quetiapine(?:\s+fumarate)?|methylphenidate|"
    r"midazolam|propofol|fentanyl|morphine|remifentanyl|"
    r"cisatracurium)\b"
    # Endocrine & metabolic
    r"|\b(?:insulin|levothyroxine|methimazole|metyrapone|"
    r"desmopressin|febuxostat|allopurinol|miglustat|"
    r"carnitine(?:\s+supplements)?)\b"
    # Other drug classes & misc
    r"|\b(?:diuretics?|beta[\s-]?blockers?|"
    r"ace\s+inhibitors?|angiotensin[\s-]converting[\s-]enzyme\s+inhibitors?|"
    r"calcium\s+channel\s+blockers?|"
    r"inotropic\s+agents?|"
    r"anticoagulant|anticoagulation|"
    r"anti[\s-]?fungal|immunosuppressants?|"
    r"granulocyte\s+colony[-\s]stimulating\s+factor|g-csf|"
    r"ivig|edta|tham|"
    r"omeprazole|esomeprazole|pantoprazole|"
    r"medications?|sedation|"
    r"oxygen|bicarbonate|sodium\s+bicarbonate|"
    r"crystalloids?|packed\s+red\s+blood\s+cells|"
    r"platelets?|fluids|zinc\s+supplementation|"
    r"calcium\s+supplements|calcium\s+lactate|chelation\s+therapy)\b",
    "Medication", 8,
)

# ---- Sign_symptom ----
SIGN_SYMPTOM = (
    r"\b(?:mass(?:es)?|pain(?:ful|s)?|lesions?|tumou?rs?|"
    r"dyspn[oe]ea|nausea|fever|rash|bleeding|[eo]?dema|oedema|"
    r"tachycardi[ac]|enlarged|enlargement|"
    r"vomiting|emesis|symptoms?|"
    r"recurrence|nodules?|dilat(?:ed|ion|ation)|"
    r"fatigue|metastas[ie]s|hemoptysis|diarrh[oe]ea|"
    r"cough|jaundice|icterus|epistaxis|"
    r"invasion|seizures?|swelling|tenderness|"
    r"neutropenia|an[ae]?mia|"
    r"shortness\s+of\s+breath|dysphagia|ascites|"
    r"necrosis|pleural\s+effusion|aneurysm|varices|"
    r"splenomegaly|hepatomegaly|hepatosplenomegaly|"
    r"murmur|headaches?|syncope|crackles|crepitations|"
    r"lymphadenopathy|hypotension|hypotensive|"
    r"dizziness|cardiomegaly|pericardial\s+effusion|"
    r"thicken(?:ed|ing)|hemorrhage|fistula|fibrosis|"
    r"hematoma|ischemia|weakness|proteinuria|"
    r"hypertrophy|hypertrophied|lactic\s+acidosis|"
    r"infiltrat(?:es?|ion)|leukocytosis|atrophy|"
    r"palpitations?|hypoxemia|"
    r"stricture|stenosis|thrombus|calcifications?|"
    r"dysarthria|consolidations?|rales|nystagmus|"
    r"thrombocytopenia|drowsiness|numbness|"
    r"obstruction|petechiae|hematemesis|hematochezia|"
    r"loss\s+of\s+appetite|decreased\s+appetite|"
    r"weight\s+loss|malaise|"
    r"cyanosis|pallor|pale|orthopn[oe]ea|"
    r"distension|distention|distended|"
    r"hypokinesi[as]|akinesis|"
    r"compression|diverticula|ulcers?|ulcerat(?:ed|ions)|"
    r"confusion|rigidity|tremor|spasms|"
    r"sweating|constipation|"
    r"respiratory\s+distress|"
    r"tachypnea|breathlessness|"
    r"abscess|eruption|lethargy|lethargic|"
    r"arthralgia|myalgia|dysuria|hematuria|"
    r"pneumothorax|"
    r"purpura|papules|pustules|vesicles|"
    r"leukopenia|inflammation|"
    r"melena|sputum|lump)\b",
    "Sign_symptom", 6,
)

# ---- Disease_disorder ----
DISEASE_DISORDER = (
    r"\b(?:pleural\s+effusion|adenocarcinoma|cardiomyopath(?:y|ies)|"
    r"heart\s+failure|congestive\s+heart\s+failure|chf|"
    r"renal\s+failure|respiratory\s+failure|"
    r"mitral\s+regurgitation|"
    r"sebaceous\s+carcinoma|squamous\s+cell\s+carcinoma(?:\s+in\s+situ)?|scc|"
    r"hepatocellular\s+carcinoma|hcc|basal\s+cell\s+carcinoma|bcc|"
    r"myocarditis|pneumonia|"
    r"coronary\s+artery\s+disease|cad|"
    r"atrial\s+fibrillation|atrial\s+flutter|"
    r"ventricular\s+tachycardia|vt|"
    r"diabetes(?:\s+mellitus)?(?:\s+type\s+[12])?|"
    r"diabetes\s+insipidus|"
    r"melanoma|cellulitis|"
    r"pulmonary\s+(?:edema|oedema|hypertension|congestion)|"
    r"osteoporosis|arthritis|"
    r"metabolic\s+acidosis|dic|"
    r"hyperparathyroidism|hypothyroidism|hyperthyroidism|thyroid\s+storm|"
    r"pancreatitis|pheochromocytoma|"
    r"deep\s+vein\s+thrombosis|thrombosis|thromboembolism|"
    r"myocardial\s+infarction|stemi|"
    r"carcinoma|malignancy|"
    r"nsclc|lymphoma|liposarcoma|meningioma|prolactinoma|"
    r"amyloidosis|al\s+amyloidosis|"
    r"hepatitis\s+[a-e]|cirrhosis|liver\s+cirrhosis|"
    r"copd|bronchiectasis|atelectasis|"
    r"hydrocephalus|encephalopathy|epilepsy|"
    r"stroke|cerebral\s+infarction|"
    r"vasculitis|gastritis|"
    r"cardiogenic\s+shock|cardiac\s+arrest|"
    r"fractures?|goiter|"
    r"gvhd|pres|dka)\b"
    r"|\b(?:infections?|cancer|embolism|infarct|"
    r"dissection|hemorrhage|"
    r"coagulopathy|sepsis|peritonitis|"
    r"alopecia|granulomas?|"
    r"developmental\s+delay)\b",
    "Disease_disorder", 7,
)

# ---- Therapeutic_procedure ----
THERAPEUTIC_PROCEDURE = (
    r"\b(?:resect(?:ed|ion)|surgical\s+(?:resection|excision|intervention)|"
    r"surgery|operation|"
    r"intubat(?:ed|ion)|extubat(?:ed|ion)|"
    r"ablations?|"
    r"cholecystectomy|splenectomy|thyroidectomy|nephrectomy|"
    r"esophagectomy|gastrectomy|pancreatectomy|"
    r"pancreaticoduodenectomy|parathyroidectomy|"
    r"lobectomy|lumpectomy|mastectomy|"
    r"laparotomy|thoracotomy|mini[-\s]thoracotomy|"
    r"median\s+sternotomy|craniotomy|rhinotomy|"
    r"lymph(?:aden)?ectomy|lymph\s+node\s+dissection|"
    r"radiotherapy|radiation(?:\s+therapy)?|chemoradiotherapy|"
    r"hemodialysis|dialysis|"
    r"ventilat(?:ed|ion|or)|mechanical\s+ventilation|"
    r"cardiopulmonary\s+bypass|"
    r"anastomosis|"
    r"caesarean\s+(?:section|operation)|cesarean\s+section|"
    r"transfusions?|blood\s+transfusions?|"
    r"transplant(?:ation)?|"
    r"supportive\s+care|"
    r"lumbar\s+puncture|"
    r"incision|sutures?|"
    r"excis(?:ed|ion)|mass\s+excision|"
    r"remov(?:ed|al)|"
    r"cannulat(?:ed|ion)|catheter(?:isation)?|"
    r"resuscitation|"
    r"total\s+parenteral\s+nutrition|tpn|"
    r"nasogastric\s+tube|peg\s+tube(?:\s+insertion)?|"
    r"tracheostomy|"
    r"plasma\s+exchange|"
    r"reconstruction|repair|grafting|"
    r"drainage(?:\s+tube)?|"
    r"pci|opcab|dbs|"
    r"stereotactic\s+frame|"
    r"symptomatic\s+treatment|"
    r"immunosuppressive\s+therapy|"
    r"decongestive\s+therapy|"
    r"septodermoplasty|"
    r"prophylaxis|fasting|diet|physiotherapy)\b",
    "Therapeutic_procedure", 7,
)

# ---- Diagnostic_procedure ----
DIAGNOSTIC_PROCEDURE = (
    r"\b(?:ct|computed\s+tomography|"
    r"mri|magnetic\s+resonance\s+imaging|"
    r"pet(?:\s*/\s*ct)?|"
    r"ultrasound|ultrasonography|"
    r"x[-\s]?ray|radiograph|"
    r"echocardiograph(?:y|am)|tte|tee|"
    r"ecg|electrocardiogram|"
    r"eeg|electroencephalogra(?:m|phy)|"
    r"endoscopy|colonoscopy|bronchoscopy|gastroscopy|"
    r"angiograph(?:y|am)|coronary\s+angiography|"
    r"catheterization|cardiac\s+catheterization|"
    r"biops(?:y|ies)|"
    r"physical\s+examination|neurological\s+examination|"
    r"pathological\s+examination|"
    r"examination|auscultation|palpation)\b"
    r"|\b(?:blood\s+pressure|heart\s+rate|pulse\s+rate|"
    r"respiratory\s+rate|body\s+temperature|"
    r"oxygen\s+saturation|vital\s+signs|"
    r"ejection\s+fraction|lvef)\b"
    r"|\b(?:hemoglobin|hematocrit|"
    r"platelet\s+count|white\s+blood\s+cell(?:\s+count)?|wbc|"
    r"red\s+blood\s+cell(?:\s+count)?|rbc|"
    r"creatinine|bilirubin|albumin|"
    r"c[-\s]reactive\s+protein|crp|"
    r"erythrocyte\s+sedimentation\s+rate|esr|"
    r"procalcitonin|"
    r"calcium|potassium|sodium|magnesium|phosphorus|"
    r"glucose|lactate|"
    r"troponin|bnp|nt[-\s]?pro[-\s]?bnp|"
    r"afp|alpha[-\s]?fetoprotein|cea|ca\s*(?:19[-\s]?9|125|15[-\s]?3)|"
    r"pth|tsh|t3|t4|"
    r"psa|ldh|alt|ast|alp|ggt|"
    r"inr|aptt|pt|d[-\s]?dimer|fibrinogen|"
    r"renal\s+function|liver\s+function(?:\s+tests?)?|"
    r"laboratory\s+(?:tests?|findings?|results?)|"
    r"urin(?:e\s+)?analysis|urinalysis)\b",
    "Diagnostic_procedure", 8,
)

# ---- Biological_structure (anatomy) ----
BIOLOGICAL_STRUCTURE = (
    r"\b(?:chest|abdominal|abdomen|pelvis|pelvic|"
    r"liver|hepatic|spleen|splenic|"
    r"brain|cerebral|cerebellar|cerebellum|"
    r"pulmonary|lung|lungs|bronch(?:us|i|ial)|"
    r"skin|subcutaneous|"
    r"lymph\s+nodes?|"
    r"cardiac|heart|myocardi(?:um|al)|pericardi(?:um|al)|endocardi(?:um|al)|"
    r"left\s+ventricl(?:e|ular)|right\s+ventricl(?:e|ular)|lv|rv|"
    r"mitral\s+valve|aortic\s+valve|tricuspid\s+valve|"
    r"aort(?:a|ic)|"
    r"neck|cervical|"
    r"gallbladder|bile\s+duct|pancrea(?:s|tic)|"
    r"esophag(?:us|eal)|gastric|stomach|"
    r"duoden(?:um|al)|jejun(?:um|al)|ile(?:um|al)|"
    r"colon(?:ic)?|rect(?:um|al)|sigmoid|cecum|appendix|"
    r"peritoneum|peritoneal|retroperitoneal|mesenteri?c|oment(?:um|al)|"
    r"renal|kidney|kidneys|adrenal|"
    r"bladder|ureter|urethra|prostate|"
    r"uterus|uterine|ovari(?:an|es|y)|cervix|"
    r"thyroid|parathyroid|pituitary|"
    r"cranial|intracranial|"
    r"bone|bones|rib|ribs|spine|spinal|vertebra(?:e|l)?|"
    r"femur|femoral|tibia(?:l)?|humerus|"
    r"muscle|muscular|diaphragm|"
    r"blood|serum|plasma|"
    r"arteri(?:al|es|y)|ven(?:ous|a\s+cava)|portal\s+vein|"
    r"carotid|jugular|iliac|mesenteric\s+arter(?:y|ies)|"
    r"left\s+eye|right\s+eye|orbit(?:al)?|retin(?:a|al)|cornea(?:l)?|"
    r"face|facial|oral\s+cavity|tongue|pharyn(?:x|geal)|"
    r"laryn(?:x|geal)|trachea(?:l)?|"
    r"pleura(?:l)?|mediastin(?:um|al)|"
    r"upper\s+(?:limbs?|extremit(?:y|ies))|"
    r"lower\s+(?:limbs?|extremit(?:y|ies))|"
    r"right\s+(?:arm|leg|lobe|atrium|side)|"
    r"left\s+(?:arm|leg|lobe|atrium|side)|"
    r"thorac(?:ic|x)|sternum|sternal|"
    r"axill(?:a|ary)|inguinal|groin|"
    r"transthoracic)\b",
    "Biological_structure", 5,
)

# ---- Detailed_description (clinical adjectives/modifiers) ----
DETAILED_DESCRIPTION = (
    r"\b(?:bilateral(?:ly)?|unilateral(?:ly)?|"
    r"multifocal|multiple|diffuse|focal|"
    r"acute|chronic|subacute|"
    r"progressive|persistent|intermittent|"
    r"recurrent|metastatic|"
    r"contrast[-\s]enhanced|"
    r"left[-\s]sided|right[-\s]sided|"
    r"systolic|diastolic|"
    r"primary|secondary|"
    r"solitary|scattered|"
    r"concentric|eccentric|"
    r"spontaneous|"
    r"endoscopic|intraoperative|laparoscopic|"
    r"doppler|"
    r"emergency|elective|"
    r"congenital|acquired|"
    r"benign|malignant|"
    r"proximal|distal|anterior|posterior|lateral|medial|"
    r"superficial|deep|peripheral|central|"
    r"homogeneous|heterogeneous|"
    r"well[-\s]?differentiated|poorly\s+differentiated|"
    r"non[-\s]?specific|idiopathic)\b",
    "Detailed_description", 2,
)

# ---- History (past medical history patterns) ----
HISTORY = (
    r"\b(?:(?:no\s+)?(?:significant\s+)?(?:past\s+)?medical\s+history|"
    r"previously\s+healthy|"
    r"(?:no\s+)?(?:known\s+)?(?:cardiac|pulmonary|medical)\s+history|"
    r"(?:history\s+of|h/o)\s+\w[\w\s]{2,30}|"
    r"smoking\s+history|"
    r"non[-\s]?smok(?:er|ing))\b",
    "History", 4,
)

# ---- Activity ----
ACTIVITY = (
    r"\b(?:smoking|smoked|smoker|"
    r"(?:cigarette|tobacco)\s+smoking|"
    r"(?:illicit\s+)?drugs?(?:\s+use)?|"
    r"(?:physical\s+)?exercise|exertion|"
    r"sun\s+exposure|"
    r"climbing|swimming|walking|pedaling|"
    r"sleeping|chewing|peristalsis)\b",
    "Activity", 4,
)

# ---- Height ----
HEIGHT = (
    r"(?<=height was )\d+(?:\.\d+)?\s*(?:cm|m)\b"
    r"|(?<=height of )\d+(?:\.\d+)?\s*(?:cm|m)\b"
    r"|(?<=height is )\d+(?:\.\d+)?\s*(?:cm|m)\b"
    r"|\b\d+(?:\.\d+)?\s*cm\s+(?:tall|in\s+height)\b"
    r"|\b\d+(?:\.\d+)?\s*(?:cm|m)\s+tall\b",
    "Height", 10,
)

# ---- Mass (body mass) ----
MASS = (
    r"\b\d+(?:\.\d+)?\s*kg\s+(?:at\s+birth|body\s+weight)\b"
    r"|\bbody\s+weight\s+(?:of\s+)?\d+(?:\.\d+)?\s*kg\b"
    r"|\bweigh(?:ed|ing|s)\s+\d+(?:\.\d+)?\s*kg\b",
    "Mass", 10,
)

# ---- Occupation ----
OCCUPATION = (
    r"\b(?:farmer|student|college\s+student|office\s+worker|"
    r"worker|athlete|breeder)\b",
    "Occupation", 10,
)

# ---- Coreference (anaphoric references to clinical entities) ----
# Only the most frequent and unambiguous forms
COREFERENCE = (
    r"\b(?:the\s+(?:tumou?r|mass|lesion|procedure|disease|surgery|"
    r"resection|condition))\b",
    "Coreference", 2,
)

# ---- Family_history ----
FAMILY_HISTORY = (
    r"\b(?:(?:no\s+)?family\s+history\s+of\s+\w[\w\s]{2,30}|"
    r"non[-\s]?consanguineous\s+parents|"
    r"consanguineous\s+(?:parents|marriage))\b",
    "Family_history", 11,
)

# ---- Biological_attribute (rarely annotated — minimal patterns) ----
BIOLOGICAL_ATTRIBUTE = (
    r"\b(?:cytologic\s+features|nucleoli|eczematous)\b",
    "Biological_attribute", 3,
)

# ---- Qualitative_concept (already partially in Lab_value — distinct forms) ----
QUALITATIVE_CONCEPT = (
    r"\b(?:within\s+(?:normal\s+limits|the\s+therapeutic\s+range)|"
    r"above\s+normal\s+ranges?|"
    r"infracentimetric|deteriorating|declining|ineffective)\b",
    "Qualitative_concept", 2,
)

# ---- Quantitative_concept ----
QUANTITATIVE_CONCEPT = (
    r"\b\d+\s+pack[-\s]?years?\b",
    "Quantitative_concept", 10,
)

# ---- Other_event & Other_entity: too heterogeneous, skip ----


# Order matters only for tie-breaking display; overlap resolution uses priority.
ALL_PATTERNS: List[Tuple[str, str, int]] = [
    AGE, SEX, DATE, DURATION, FREQUENCY, TIME,
    DOSAGE, ADMINISTRATION,
    VOLUME, AREA, DISTANCE, WEIGHT, HEIGHT, MASS,
    SEVERITY, COLOR, SHAPE, TEXTURE, LAB_VALUE,
    OUTCOME, PERSONAL_BACKGROUND, CLINICAL_EVENT, SUBJECT,
    NONBIOLOGICAL_LOCATION, MEDICATION,
    SIGN_SYMPTOM, DISEASE_DISORDER,
    THERAPEUTIC_PROCEDURE, DIAGNOSTIC_PROCEDURE,
    BIOLOGICAL_STRUCTURE, DETAILED_DESCRIPTION,
    HISTORY, ACTIVITY, OCCUPATION,
    COREFERENCE, FAMILY_HISTORY,
    BIOLOGICAL_ATTRIBUTE, QUALITATIVE_CONCEPT, QUANTITATIVE_CONCEPT,
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
