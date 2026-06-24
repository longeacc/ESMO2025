"""
quaero_brat_annotator.py
------------------------
Rule-based (hard-coded regex) BRAT annotator for the QUAERO FrenchMed dataset
(MEDLINE subset), built in the same spirit as `maccrobat_brat_annotator.py`.

QUAERO FrenchMed uses 10 semantic categories (UMLS semantic groups):
    ANAT  - Anatomy
    CHEM  - Chemicals & Drugs
    DEVI  - Devices
    DISO  - Disorders
    GEOG  - Geographic Areas
    LIVB  - Living Beings
    OBJC  - Objects
    PHEN  - Phenomena
    PHYS  - Physiology
    PROC  - Procedures

Text is French medical text (MEDLINE abstracts). Patterns use re.IGNORECASE +
re.UNICODE and are accent-aware where needed.

Output: BRAT `.ann` files with T-lines only:
    Tn<TAB><label> start end<TAB><span_text>

NOTE: regex patterns are stubs — fill them in using quaero_inspect_train.py
to guide tuning on the TRAIN split only.
"""

from pathlib import Path
import re
import os
import argparse
from typing import List, Tuple

try:
    from eco2ai import set_params, Tracker
    set_params(
        project_name="Consumption_of_quaero_brat_annotator.py",
        experiment_description="Rule-based QUAERO FrenchMed annotation",
        file_name="Consumption_of_maccrobat.csv",
    )
    _tracker = Tracker()
    _tracker.start()
except Exception:
    _tracker = None


# ==================================================================
#  Regex patterns  (label, compiled later)
#  French MEDLINE abstracts -> IGNORECASE | UNICODE.
#  Each entry: (raw_pattern, label, priority)
#  priority resolves overlaps; higher wins.
#  TODO: replace stub patterns with tuned ones via quaero_inspect_train.py
# ==================================================================

# ---- ANAT : Anatomy ----
# Strategy: organ keyword list + common French anatomical adjectives.
# Adjectives (hépatique, pulmonaire…) dominate in the corpus (avg 1.3 words/span).
# Open vocabulary → no suffix pattern; keyword recall ~50 %.
ANAT = (
    # Organs / structures
    r"\b(?:foie|rein|reins|c[oœ]ur|poumon|poumons|cerveau|cervelet|"
    r"moelle|moelle[s]?\s+(?:[eé]pini[eè]re[s]?|osseuse[s]?)|"
    r"os|muscle|muscles|art[eè]re|art[eè]res|veine|veines|nerf|nerfs|"
    r"ganglion|ganglions|pancr[eé]as|rate|v[eé]sicule|v[eé]sicules|"
    r"intestin|intestins|c[oô]lon|rectum|[oœ]sophage|estomac|duod[eé]num|"
    r"j[eé]junum|il[eé]on|thyro[iï]de|surr[eé]nale|surr[eé]nales|"
    r"hypophyse|[eé]piphyse|thymus|prostate|ut[eé]rus|ovaire|ovaires|"
    r"testicule|testicules|sein|seins|peau|r[eé]tine|corn[eé]e|cristallin|"
    r"tympan|cochl[eé]e|pharynx|larynx|trach[eé]e|bronche|bronches|"
    r"alv[eé]ole|alv[eé]oles|p[eé]ritoine|pl[eè]vre|p[eé]ricarde|"
    r"m[eé]ninges|aorte|carotide|jugulaire|ligament|tendon|cartilage|"
    r"cortex|hippocampe|amygdale|noyau|noyaux|colonne\s+vert[eé]brale|"
    r"moelle[s]?\s+[eé]pini[eè]re[s]?|disque\s+intervert[eé]bral|"
    r"glande|glandes|canal|canaux|voie[s]?\s+biliaire[s]?|"
    r"cellule|cellules|membre|membres|membres?\s+(?:inf[eé]rieur[s]?|sup[eé]rieur[s]?)|"
    r"art[eè]re|biliaire|biliaires|myocarde|f[oœ]tus|col\s+ut[eé]rin|"
    r"appareil\s+(?:digestif|respiratoire|cardio[-\s]vasculaire|g[eé]nital|urinaire))\b"
    # French anatomical adjectives (frequent form in corpus)
    r"|\b(?:h[eé]patique|h[eé]patiques|pulmonaire|pulmonaires|"
    r"cardiaque|cardiaques|r[eé]nal|r[eé]nale|r[eé]naux|r[eé]nales|"
    r"c[eé]r[eé]bral|c[eé]r[eé]brale|c[eé]r[eé]braux|"
    r"gastrique|gastriques|intestinal|intestinale|intestinaux|"
    r"colique|coliques|rectal|rectale|rectaux|"
    r"bronchique|bronchiques|alv[eé]olaire|alv[eé]olaires|"
    r"pleural|pleurale|pleuraux|vasculaire|vasculaires|"
    r"lymphatique|lymphatiques|osseux|osseuse|musculaire|musculaires|"
    r"nerveux|nerveuse|art[eé]riel|art[eé]rielle|veineux|veineuse|"
    r"vestibulaire|vestibulaires|cochle[eé]aire|r[eé]tinien|r[eé]tinienne|"
    r"thyro[iï]dien|thyro[iï]dienne|surr[eé]nalien|surr[eé]nalienne|"
    r"hypophysaire|pancr[eé]atique|splénique|spl[eé]nique|"
    r"m[eé]dullaire|m[eé]ning[eé]|m[eé]ning[eé]e|coronal|coronale|"
    r"vertébral|vertébrale|épidural|épidurale|sous-cutané|sous-cutanée|"
    r"cervical|cervicale|cervicaux|cutan[eé]|cutan[eé]e|cutan[eé]s|cutan[eé]es|"
    r"ovarien|ovarienne)\b"
    # Standalone organ terms missed above (oesophage/coeur with elision, etc.)
    r"|\boesophage\b|\bc[oœ]ur\b",
    "ANAT", 6,
)

# ---- CHEM : Chemicals & Drugs ----
# Strategy: (1) biomolecule abbreviations, (2) pharma suffixes specific to French
# drug names, (3) chemical/drug keyword anchors.
# Suffix guards: min total length ≥8 to avoid short false positives.
CHEM = (
    # Biomolecule abbreviations
    r"\b(?:ADN|ARN|ATP|ADP|AMP|GTP|GDP|NAD|NADH|FAD|CoA|"
    r"IgG|IgA|IgE|IgM|IgD|HLA|PCR|LDH|ASAT|ALAT|CRP|"
    r"TSH|FSH|LH|HbA1c|PSA|CEA|AFP|CA[-\s]?125|CA[-\s]?19[-\s]?9|"
    r"IL[-\s]?\d+|TNF|INF|IFN|VEGF|EGF|TGF|FGF|NGF|"
    r"LPS|ATP|cAMP|cGMP|mRNA|siRNA|miRNA|tRNA|rRNA)\b"
    # Vitamins (space optional between letter and number: "vitamine B 12")
    r"|\bvitamine\s+[A-Z]\s*\d*\b"
    # Acids: bare noun + qualified compound forms
    r"|\bacide[s]?\b"
    r"|\bacide[s]?\s+\w+(?:ique|aminé|gras|nucl[eé]ique|amin[eé])\b"
    r"|\bacides?\s+biliaires?\b"
    # Pharma suffixes – French drug naming conventions
    # Pharma suffixes: stem ≥3 for most (short drug names still have 7+ total chars)
    r"|\b\w{3,}cillines?\b"   # pénicilline (stem=péni=4✓), ampicilline, amoxicilline
    r"|\b\w{3,}mycines?\b"    # érythromycine (stem=éryth=5✓), streptomycine, gentamicine
    r"|\b\w{3,}cyclines?\b"   # tétracycline (stem=tétra=5✓), doxycycline
    r"|\b\w{3,}azoles?\b"     # métronidazole (stem=métroni=7✓), fluconazole
    r"|\b\w{3,}statines?\b"   # simvastatine (stem=simva=5✓), atorvastatine
    r"|\b\w{3,}prils?\b"      # captopril (stem=capto=5✓), énalapril, lisinopril
    r"|\b\w{3,}sartans?\b"    # losartan (stem=lo=2 → needs stem≥2)
    r"|\b\w{2,}sartans?\b"    # losartan (stem=lo=2✓), valsartan
    r"|\b\w{4,}mabs?\b"       # trastuzumab (stem=trastuzuma=9✓), rituximab
    r"|\b\w{4,}nibs?\b"       # imatinib (stem=imatini=7✓), erlotinib
    r"|\b\w{4,}dr[eé]nergiques?\b"  # adrénergique (stem=adré=4✓)
    # Chemical/drug keyword anchors
    r"|\b(?:m[eé]dicament|m[eé]dicaments|antibiotique|antibiotiques|"
    r"vaccin|vaccins|hormone|hormones|enzyme|enzymes|anticorps|"
    r"st[eé]ro[iï]de|st[eé]ro[iï]des|cortico[iï]de|cortico[iï]des|"
    r"analg[eé]sique|analg[eé]siques|anticoagulant|anticoagulants|"
    r"antid[eé]presseur|antid[eé]presseurs|"
    r"anti[eé]pileptique|anti[eé]pileptiques|hypnotique|s[eé]datif|"
    r"immunosuppresseur|immunosuppresseurs|cytostatique|cytostatiques|"
    r"antiparasitaire|antifongique|antiviral|antiviraux|"
    r"amidon|alcool|alcoolique|insuline|cortisone|morphine|h[eé]parine|"
    r"glucose|fructose|saccharose|lactose|glycog[eè]ne|"
    r"prot[eé]ine|prot[eé]ines|lipide|lipides|glucide|glucides|"
    r"immunoglobuline|immunoglobulines|albumine|globuline|fibrine|"
    r"dopamine|s[eé]rotonine|adr[eé]naline|noradr[eé]naline|"
    r"progest[eé]rone|oestrog[eè]ne|testost[eé]rone|thyroxine|"
    r"r[eé]cepteur|r[eé]cepteurs|m[eé]dicamenteux|m[eé]dicamenteuse|"
    r"h[eé]moglobine|aminoglycosides?)\b",
    "CHEM", 6,
)

# ---- DEVI : Devices ----
# Strategy: keyword list. Corpus has only 39 DEVI; closed-ish vocabulary.
# Covers ~80 % of corpus DEVI instances.
DEVI = (
    r"\b(?:proth[eè]se|proth[eè]ses|prothesses|implant|implants|"
    r"stimulateur|stimulateurs|stimulateurs?\s+cardiaques?|pacemaker|"
    r"[eé]lectrode|[eé]lectrodes|valve|valves|respirateur|respirateurs|"
    r"microscope|microscopes|pompe|pompes|pompe\s+[àa]\s+insuline|"
    r"laser|lasers|sonde|sondes|endoproth[eè]se|endoproth[eè]ses|"
    r"suture|sutures|fixateur|fixateurs|fixateur\s+externe|"
    r"plaque|plaques|marqueur|marqueurs|dialyseur|dialyseurs|"
    r"d[eé]fibrillateur|d[eé]fibrillateurs|cath[eé]ter|cath[eé]ters|"
    r"masque|canule|canules|speculum|otoscope|stent|stents|"
    r"microsonde|microsondes|analyseur|analyseurs|"
    r"dispositif[s]?\s+(?:intra[-\s]ut[eé]rin[s]?|orthodontique[s]?)|"
    r"implant[s]?\s+en\s+silastic|double\s+chambre|"
    r"instrument|instruments|syst[eè]me|syst[eè]mes|cr[eè]me|cr[eè]mes|cryoth[eé]rapie|"
    r"proth[eè]se[s]?\s+(?:dentaire[s]?|de\s+l\'oreille\s+moyenne|conjointe[s]?))\b",
    "DEVI", 5,
)

# ---- DISO : Disorders ----
# Strategy: (1) high-frequency disease keywords, (2) morphological suffixes
# specific to French pathological nouns.
# Suffix guards:
#   -ite  → \w{8,}ites? (≥8 chars total avoids "petite"=6, "suite"=5, "droite"=6)
#   -ose  → \w{5,}oses? (≥5 chars: "dose"=4 excluded; fibrose=7 ✓)
#   -ome  → \w{6,}omes? (≥6: "atome"=5 excluded; gliome=6 ✓)
#   -émie → \w{7,}[eé]mies? (≥7: leucémie=8 ✓)
#   -pathie / -algie: long enough by nature
DISO = (
    # High-frequency disease/disorder keywords
    r"\b(?:cancer|cancers|tumeur|tumeurs|m[eé]tastase|m[eé]tastases|"
    r"carcinome|carcinomes|lymphome|lymphomes|sarcome|sarcomes|"
    r"ad[eé]nome|ad[eé]nomes|m[eé]lanome|m[eé]lanomes|"
    r"fibrome|fibromes|gliome|gliomes|h[eé]patome|"
    r"maladie|maladies|syndrome|syndromes|infection|infections|"
    r"insuffisance|insuffisances|d[eé]ficit|d[eé]ficits|"
    r"trouble|troubles|l[eé]sion|l[eé]sions|"
    r"st[eé]nose|st[eé]noses|thrombose|thromboses|"
    r"embolie|embolies|infarctus|isch[eé]mie|isch[eé]mies|"
    r"n[eé]crose|n[eé]croses|[oœ]d[eè]me|[oœ]d[eè]mes|"
    r"h[eé]morragie|h[eé]morragies|h[eé]morrhagie|h[eé]morrhagies|"
    r"allergie|allergies|fracture|fractures|plaie|plaies|"
    r"ulc[eè]re|ulc[eè]res|absc[eè]s|kyste|kystes|"
    r"ad[eé]nopathie|ad[eé]nopathies|douleur|douleurs|"
    r"d[eé]pression|psychose|n[eé]vrose|[eé]pilepsie|"
    r"zona|acrom[eé]galie|alcoolisme|diab[eè]te|"
    r"hypertension|hypotension|asthme|cirrhose|pneumonie|"
    r"sep(?:tic[eé]mie|sis)|choc\s+(?:septique|anaphylactique|cardiog[eé]nique)|"
    r"coma|d[eé]mence|alzheimer|parkinson|scl[eé]rose\s+en\s+plaques|"
    r"polyarthrite|spondylarthrite|goutte|ob[eé]sit[eé]|"
    r"malnutrition|d[eé]shydratation|intoxication|empoisonnement|"
    r"paludisme|tuberculose|sida|VIH|[eé]pilepsie|psoriasis|eczema|eczéma|"
    r"diagnostic|diagnostics|complication|complications|"
    r"tol[eé]rance|polyarthrite|spondylarthrite)\b"
    # Suffix guards: {N,} = stem min chars (NOT total); backtracking handles it.
    # -ite: stem ≥5 avoids "petites" (stem=petit=5 → 'e' ≠ 'i' → fail), "suites" (4)
    r"|\b\w{5,}ites?\b"
    # -ose: stem ≥3 catches "fibrose" (stem=fibr=4 ✓), avoids "dose" (stem=d=1)
    r"|\b\w{3,}oses?\b"
    # -ome: stem ≥3 catches "gliome" (stem=gli=3 ✓), avoids "homme" (stem=ho=2)
    r"|\b\w{3,}omes?\b"
    # -émie: stem ≥2 catches "anémie" (stem=an=2 ✓)
    r"|\b\w{2,}[eé]mies?\b"
    # -pathie: stem ≥3 catches "neuropathie" (stem=neuro=5 ✓)
    r"|\b\w{3,}pathies?\b"
    # -algie: stem ≥3 catches "névralgie" (stem=névr=4 ✓)
    r"|\b\w{3,}algies?\b",
    "DISO", 7,
)

# ---- GEOG : Geographic Areas ----
# Strategy: explicit list (corpus has only 34 GEOG). Covers corpus + common neighbours.
# Also catches derived adjectives (française, belge, tunisienne…).
GEOG = (
    r"\b(?:France|Paris|Europe|Afrique|Asie|Am[eé]rique|Oc[eé]anie|"
    r"Alg[eé]rie|Maroc|Tunisie|Libye|Mauritanie|Mali|Niger|Tchad|Soudan|"
    r"S[eé]n[eé]gal|Guinée|Guadeloupe|Martinique|R[eé]union|Mayotte|"
    r"Bénin|B[eé]nin|Burkina|Togo|Cameroun|Gabon|Congo|Rwanda|Burundi|"
    r"Djibouti|[Éé]thiopie|Somalie|Kenya|Tanzanie|Madagascar|Mozambique|"
    r"Liban|Syrie|Jordanie|Irak|Iran|Turquie|Égypte|Arabie|"
    r"Canada|Qu[eé]bec|[ÉE]tats-Unis|Br[eé]sil|Argentine|"
    r"Suisse|Belgique|Allemagne|Espagne|Italie|Portugal|"
    r"Royaume-Uni|Grande-Bretagne|Pays-Bas|Sude|Norv[eè]ge|Danemark|Finlande|"
    r"Russie|Chine|Japon|Inde|Cor[eé]e|Vietnam|Tha[ïi]lande|Indon[eé]sie|"
    r"Australie|Tahiti|Polyn[eé]sie|Nouvelle-Cal[eé]donie|Antilles|"
    r"Méditerran[eé]e|M[eé]diterran[eé]e|Atlantique|Pacifique)\b"
    # Derived adjectives present in corpus
    r"|\b(?:fran[cç]aise?|belge|europ[eé]enn[e]?s?|africaine?s?|asiatiques?|"
    r"californiens?|tunisiens?|tunisienne?s?|alg[eé]riens?|marocains?|s[eé]n[eé]galais[e]?|"
    r"sub-saharien|sud-est\s+asiatique|antillais[e]?s?|qu[eé]b[eé]cois[e]?s?)\b"
    r"|\bmars\b|\b[iî]le[s]?\b",
    "GEOG", 4,
)

# ---- LIVB : Living Beings ----
# Strategy: (1) human subjects, (2) lab animals, (3) microorganism keywords,
# (4) binomial species name pattern (Genus species).
# LIVB is open vocab but many entities are high-frequency human/animal terms.
LIVB = (
    # Binomial species names FIRST (before keyword list) so alternation picks
    # the longer match "Candida albicans" over the shorter keyword "candida"
    r"\b(?:Candida|Aspergillus|Bacillus|Staphylococcus|Streptococcus|"
    r"Escherichia|Salmonella|Listeria|Saccharomyces|Caenorhabditis|"
    r"Mycobacterium|Plasmodium|Leishmania|Toxoplasma|Trypanosoma|"
    r"Boophilus|Amblyomma|Centrorhynchus|Aerobacter|Arizona|Papio|"
    r"Alytes|Averrhoa)\s+[a-z][a-z]{2,}\b"
    # Human subjects
    r"|\b(?:patient|patients|malade|malades|enfant|enfants|"
    r"nourrisson|nourrissons|adulte|adultes|homme|hommes|femme|femmes|"
    r"sujet|sujets|individu|individus|nouveau-n[eé]|nouveau-n[eé]s|"
    r"pr[eé]matur[eé]|pr[eé]matur[eé]s|f[oœ]tus|vieillard|vieillards|"
    r"[aâ]g[eé]|jeune|jeunes|adolescent|adolescents|"
    r"m[eé]decin|m[eé]decins|chirurgien|chirurgiens|infirmier|infirmiers|"
    r"[eé]colier|[eé]coliers|volontaire|volontaires|donneur|donneurs|receveur|receveurs|"
    r"humaine|humaines|personne\s+[aâ]g[eé]e|p[eé]diatrique|p[eé]diatriques)\b"
    # Lab animals
    r"|\b(?:souris|rat|rats|ratte|rattes|lapin|lapins|singe|singes|chien|chiens|"
    r"chat|chats|porc|porcs|cheval|chevaux|vache|vaches|cobaye|cobayes|"
    r"primate|primates|rongeur|rongeurs|hamster|hamsters|gerbille|"
    r"grenouille|grenouilles|t[eé]l[eé]ost[eé]en|t[eé]l[eé]ost[eé]ens|"
    r"murin|murine|murins|murines)\b"
    # Microorganisms
    r"|\b(?:virus|bact[eé]rie|bact[eé]ries|micro-organisme|micro-organismes|"
    r"champignon|champignons|levure|levures|parasite|parasites|"
    r"protozoaire|protozoaires|amibe|amibes|trypanosome|trypanosomes|"
    r"mycobact[eé]rie|mycobact[eé]ries|staphylocoque|streptocoque|"
    r"entrobact[eé]rie|salmonelle|listeria|candida|aspergillus)\b",
    "LIVB", 5,
)

# ---- OBJC : Objects ----
# Strategy: keyword list only. Corpus has 27 OBJC; very heterogeneous.
# Low expected recall (~40 %); precision-oriented.
OBJC = (
    r"\b(?:h[oô]pital|h[oô]pitaux|service\s+de\s+soins|service\s+(?:m[eé]dical|hospitalier)|"
    r"institution\s+hospitali[eè]re|entreprise|usine|machines?|"
    r"substance|substances|m[eé]lange|m[eé]langes|"
    r"a[eé]rosol|a[eé]rosols|appareillage|appareillages|"
    r"pr[eé]l[eè]vement|pr[eé]l[eè]vements|miniordinateur|"
    r"verre\s+ionomère|caf[eé]|Coffea|"
    r"association\s+m[eé]dicamenteuse|associations?|scolaire|appareil|"
    r"di[eé]t[eé]tique|dispositifs?|mol[eé]culaire|service)\b",
    "OBJC", 3,
)

# ---- PHEN : Phenomena ----
# Strategy: physical/observable phenomena keywords.
# Only 60 PHEN in corpus → keyword list sufficient.
PHEN = (
    r"\b(?:r[eé]sonance\s+magn[eé]tique|rayonnement|radiation|"
    r"radiations?\s+ionisantes?|radioactivit[eé]|"
    r"absorption|diffusion|fluorescence|bioluminescence|luminescence|"
    r"[eé]lectrophor[eè]se|[eé]lectrophor[eé]tique|"
    r"ultrasonographie|ultrason|ultrasons|acoustique|"
    r"chromatographie|spectrometrie|spectrom[eé]trie|"
    r"centrifugation|osmose|dialyse|[eé]lectrolyse|"
    r"pharmacodynamie|pharmacocin[eé]tique|cin[eé]tique|"
    r"s[eé]dimentation|agr[eé]gation|pr[eé]cipitation|"
    r"pression\s+(?:art[eé]rielle|veineuse|atmosph[eé]rique)|"
    r"tension\s+(?:art[eé]rielle|superficielle)|"
    r"rayons?\s+(?:gamma|X|ultraviolets?|infrarouges?)|"
    r"ondes?\s+(?:[eé]lectromagn[eé]tiques?|sonores?)|"
    r"transmission\s+(?:virale?|bact[eé]rienne?|g[eé]n[eé]tique)|"
    r"r[eé]action\s+(?:immunologique|inflammatoire|allergique)|"
    r"r[eé]sultats?|[eé]lectrique|pression|ph[eé]nom[eè]nes?|"
    r"[eé]pid[eé]mique|applications?\s+th[eé]rapeutiques?|rejet|"
    r"processus|magn[eé]tique)\b",
    "PHEN", 4,
)

# ---- PHYS : Physiology ----
# Strategy: physiological process keywords.
# 160 PHYS — semi-open; keyword list covers major processes.
PHYS = (
    r"\b(?:grossesse|allaitement|lactation|sommeil|respiration|"
    r"circulation|m[eé]tabolisme|croissance|d[eé]veloppement|"
    r"vieillissement|reproduction|digestion|absorption|s[eé]cr[eé]tion|"
    r"excr[eé]tion|ovulation|menstruation|menstruel|f[eé]condation|"
    r"innervation|contraction|relaxation|inflammation|coagulation|"
    r"h[eé]mostase|immunit[eé]|angiog[eè]ne[s]e|mitose|apoptose|"
    r"anoïkis|anoikis|[eé]volution|h[eé]r[eé]dit[eé]|"
    r"transition\s+[eé]pith[eé]lio[-\s]m[eé]senchymateuse|"
    r"synth[eè]se\s+des?\s+(?:prot[eé]ines?|ADN|ARN|acides?|lipides?)|"
    r"[eé]tat\s+nutritionnel|travail\s+(?:obst[eé]trical)?|"
    r"accouchement|parturition|pu(?:bert[eé]|bert[eé])|"
    r"m[eé]nopause|andropause|s[eé]nescence|proliferation|"
    r"prol[eé]f[eé]ration|diff[eé]renciation|activation|inhibition|"
    r"m[eé]taboliques?|gestation|r[eé]sistance|sexe|vision|nutrition|"
    r"respiratoires?|n[eé]crotiques?|h[eé]modynamique|naissance)\b",
    "PHYS", 5,
)

# ---- PROC : Procedures ----
# Strategy: (1) medical procedure suffixes (-graphie, -scopie, -ectomie,
# -plastie, -thérapie), (2) procedure keyword anchors.
# Suffix guards: min total length ≥9 to avoid common non-medical words.
PROC = (
    # Procedure suffixes (highly diagnostic in French medical text)
    # stem ≥3 catches all medical terms; false positives minimal in MEDLINE
    r"\b\w{3,}graphies?\b"    # échographie, radiographie, angiographie
    r"|\b\w{3,}scopies?\b"    # endoscopie, bronchoscopie, laparoscopie
    r"|\b\w{3,}ectomies?\b"   # appendicectomie, thyroïdectomie, cholécystectomie
    r"|\b\w{3,}plasties?\b"   # rhinoplastie, arthroplastie, mammoplastie
    r"|\b\w{3,}th[eé]rapies?\b"  # chimiothérapie, radiothérapie, antibiothérapie
    r"|\b\w{4,}tomies?\b"     # laparotomie, craniotomie; stem≥4 avoids "anatomie" (stem=ana=3)
    r"|\b\w{3,}stomies?\b"    # colostomie, trachéostomie, iléostomie
    # Procedure keyword anchors
    r"|\b(?:traitement|traitements|chirurgie|chirurgies|op[eé]ration|op[eé]rations|"
    r"intervention|interventions|anesth[eé]sie|"
    r"bilan|bilans|analyse|analyses|[eé]valuation|[eé]valuations|"
    r"surveillance|[eé]chographie|radiographie|scanner|IRM|"
    r"ponction|biopsie|biopsies|incision|dissection|r[eé]section|"
    r"cath[eé]t[eé]risme|perfusion|perfusions|transfusion|transfusions|"
    r"transplantation|greffe|greffes|dialyse|radioth[eé]rapie|"
    r"chimioth[eé]rapie|immunoth[eé]rapie|phototh[eé]rapie|"
    r"[eé]lectroenc[eé]phalogramme|[eé]lectromyogramme|"
    r"[eé]lectronystagmographie|[eé]lectrophor[eè]se|"
    r"angiographie|artériographie|veinographie|"
    r"anastomose|anastomoses|suture|ligature|drainage|"
    r"d[eé]bridement|curetage|lavage|irriguation|"
    r"immunofl[eé]orescence|immunohistochimie|cytologie|histologie|"
    r"autopsie|autopsies|n[eé]cropsie|"
    r"[eé]tude|[eé]tudes|[ÉE]TUDE|observation|observations|"
    r"d[eé]pistage|soins|th[eé]rapeutique|trait[eé]|tests?|"
    r"pr[eé]vention|recherche|recherches|prise\s+en\s+charge)\b",
    "PROC", 6,
)


ALL_PATTERNS: List[Tuple[str, str, int]] = [
    ANAT, CHEM, DEVI, DISO, GEOG, LIVB, OBJC, PHEN, PHYS, PROC,
]

_COMPILED = [
    (re.compile(pat, re.IGNORECASE | re.UNICODE), label, prio)
    for (pat, label, prio) in ALL_PATTERNS
]


# ==================================================================
#  Extraction
# ==================================================================
def extract_entities(text: str) -> List[Tuple[str, int, int, str, int]]:
    """Run every pattern over `text`. Returns (span_text, start, end, label, priority)."""
    out = []
    for regex, label, prio in _COMPILED:
        for m in regex.finditer(text):
            s, e = m.start(), m.end()
            if e <= s:
                continue
            out.append((m.group(), s, e, label, prio))
    return out


def resolve_overlaps(
    spans: List[Tuple[str, int, int, str, int]]
) -> List[Tuple[str, int, int, str]]:
    """
    Dedup exact duplicates only — DO NOT suppress overlapping spans.
    QUAERO gold annotates nested entities by design (e.g. DISO "métastases
    hépatiques" containing ANAT "hépatiques", or DISO "cancers colorectaux"
    containing DISO "cancers"). Mutual-exclusion overlap suppression caps
    recall far below what the regex actually covers, because every nested
    gold entity inside an already-kept span becomes an unreachable FN.
    """
    seen = set()
    uniq = []
    for sp in spans:
        key = (sp[3], sp[1], sp[2])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(sp)

    uniq.sort(key=lambda sp: (sp[1], sp[2]))
    return [(sp[0], sp[1], sp[2], sp[3]) for sp in uniq]


# ==================================================================
#  BRAT writer
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
    """Annotate every .txt under `in_dir`, writing <stem>.ann to `out_dir`."""
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
        description="Rule-based BRAT annotator for QUAERO FrenchMed MEDLINE (regex)."
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
    except Exception as e:
        print(f"\nWarning: eco2ai tracking error: {e}")
