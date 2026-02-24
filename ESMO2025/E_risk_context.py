"""
Calculate 'Risk Context' (Probability of Ambiguity) for biomarkers and named entities.

Risk Context is the probability that an extracted value is ambiguous and requires
deep contextual analysis to resolve its meaning.

Definitions:
- Low Risk (Score ~ 0.0): Unambiguous entities. Self-contained values. 
  Example: "Grade 3", "HER2 positive", "T2N0M0".
- High Risk (Score ~ 1.0): Polysemic terms or raw numbers requiring context.
  Example: "positif" (could refer to ER, PR, HER2), "10%" (what is 10%?), "++".
"""

import os
import re
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any, Tuple
from eco2ai import set_params, Tracker

set_params(
    project_name="Consumtion_of_E_risk_context.py",
    experiment_description="We Calculate...",
    file_name="Consumtion_of_Duraxell.csv"
)

tracker = Tracker()
tracker.start()

class RiskContextAnalyzer:
    """Analyze the ambiguity risk of extracted entities."""

    def __init__(self, data_dirs: List[str]):
        """
        Initialize with data directories.
        """
        self.data_dirs = [Path(d) for d in data_dirs]
        self.entities_values = defaultdict(list)
        self.total_files_count = 0
        
        # Polysemic terms that are highly ambiguous on their own
        self.ambiguous_terms = {
            'positif', 'negatif', 'positive', 'negative', 'pos', 'neg', 
            'detected', 'detecte', 'non detecte', 'not detected',
            'amplifié', 'amplifie', 'non amplifié', 'non amplifie',
            'muté', 'mute', 'wild type', 'sauvage',
            '+', '-', '++', '+++', '++++', '+/-', '?', '1+', '2+', '3+',
            'faible', 'modéré', 'fort', 'low', 'high', 'intermediate',
            'absent', 'presence', 'indetermine', 'equivoque'
        }

    def parse_ann_file(self, ann_path: Path) -> None:
        """Parse .ann file and collect entity values."""
        if not ann_path.exists():
            return

        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): 
                    continue
                
                parts = line.split('\t')
                # We typically care about Text-bound annotations (T...)
                if line.startswith('T') and len(parts) >= 3:
                    # parts[0] = ID (e.g., T1)
                    # parts[1] = Type Start End (e.g., Ki67 10 14)
                    # parts[2] = Text (e.g., 10%)
                    
                    type_info = parts[1].split(' ', 1)
                    entity_type = type_info[0]
                    text_value = parts[2]
                    
                    self.entities_values[entity_type].append(text_value)

    def extract_data(self):
        """Walk through directories and extract entity data."""
        processed_files = set()
        
        for data_dir in self.data_dirs:
            if not data_dir.exists():
                print(f"Warning: Directory {data_dir} does not exist.")
                continue
                
            for ann_file in data_dir.glob('**/*.ann'):
                if ann_file.name in processed_files:
                    continue
                    
                self.parse_ann_file(ann_file)
                processed_files.add(ann_file.name)
        
        self.total_files_count = len(processed_files)
        print(f"Total files processed: {self.total_files_count}")

    def calculate_risk_score(self, values: List[str], entity_type: str) -> Dict[str, Any]:
        """
        Calculate risk score for a specific entity type based on its values.
        
        Returns a dictionary with score and analysis details.
        """
        if not values:
            return {'risk_score': 0.0, 'reason': 'No values', 'details': {}}

        total_count = len(values)
        high_risk_count = 0
        low_risk_count = 0
        
        # Details counters
        polysemic_count = 0
        numeric_only_count = 0
        explicit_count = 0
        structured_count = 0
        
        normalized_entity_type = entity_type.lower()
        
        # Common entity mapping synonyms to check explicitly
        # e.g. if entity is "ER", "Estrogen" is also explicit.
        type_synonyms = {
            'estrogen_receptor': ['estrogen', 'oestrogene', 're', 'ro', 'rh', 'recepteur', 'luminal'],
            'progesterone_receptor': ['progesterone', 'rp', 'rh', 'recepteur', 'pgr'],
            'her2_status': ['her2', 'erbb2', 'cerb', 'cerbb2', 'cerb-b2', 'cerb2', 'neu', 'c-erb'],
            'her2_ihc': ['her2', 'erbb2', 'cerb', 'cerbb2', 'cerb-b2', 'cerb2', 'neu', 'c-erb'],
            'her2_fish': ['her2', 'erbb2', 'cerb', 'cerbb2', 'cerb-b2', 'cerb2', 'neu', 'c-erb', 'fish', 'ddish', 'sish'],
            'ki67': ['mib1', 'mib-1', 'ki-67', 'ki'],
            'genetic_mutation': ['mutation', 'gene', 'brca', 'pik3ca']
        }
        
        # Normalize entity type key (some might use underscores or different casing)
        # Try to find the best match in keys
        matches = [k for k in type_synonyms.keys() if k in normalized_entity_type]
        synonyms = []
        if matches:
            synonyms = type_synonyms[matches[0]]
        
        # Always include the entity type itself (split by parts)
        parts = normalized_entity_type.split('_')
        synonyms.extend(parts)
        
        # Filter synonyms to avoid very short generic ones if needed, but 're' is valid for ER.
        # Ensure we don't duplicate
        synonyms = list(set(synonyms))

        for v in values:
            v_norm = v.strip().lower()
            v_clean = re.sub(r'[^a-z0-9\+\-%]', '', v_norm) # Simplified for checking
            
            is_ambiguous = False
            
            # Check 1: Is it a generic polysemic term? (High Risk)
            if v_norm in self.ambiguous_terms or v_clean in self.ambiguous_terms:
                polysemic_count += 1
                is_ambiguous = True
                
            # Check 2: Is it purely numeric/percentage without context? (High Risk)
            # Matches: "10", "10%", "10.5", "<10", ">10", "3+"
            # Also catch "Score 3" if "Score" is not in synonyms (for HER2, Score is not synonym)
            # But "Score" + Number is structured. Be careful.
            
            # If matches simple number/percent pattern
            if not is_ambiguous and re.match(r'^(<|>|<=|>=)?\s*\d+([.,]\d+)?\s*%?$', v_norm):
                numeric_only_count += 1
                is_ambiguous = True

            # Special case: "Score X" where X is number, but no entity name.
            # If entity is HER2, "Score 3" is somewhat ambiguous (could be SBR).
            # If entity is Ki67, "Score" is rare.
            # We assume "Score X" is Medium/High risk unless it has the entity name.
            if not is_ambiguous and re.match(r'^score\s*:?\s*\d+\+?$', v_norm):
                 # "Score 3", "Score 3+"
                 # Check if "score" is a synonym (it shouldn't be for ER/HER2 usually)
                 if 'score' not in synonyms:
                     polysemic_count += 1
                     is_ambiguous = True

            # Check 3: Does it contain the entity Name/Explicit descriptor? (Low Risk)
            # If the value says "Grade 3", it's explicit for "Grade".
            # If it says "ER Positive", it's explicit for "ER".
            is_explicit = False
            for syn in synonyms:
                # Use regex word boundary check for short synonyms (<= 3 chars) to avoid false matches
                # e.g. avoid 're' matching 'score' or 'premiere'
                if len(syn) <= 3:
                     if re.search(r'\b' + re.escape(syn) + r'\b', v_norm):
                         is_explicit = True
                         break
                else:
                    if syn in v_norm:
                        is_explicit = True
                        break
            
            if is_explicit:
                explicit_count += 1
                # Being explicit overrides ambiguity (e.g. "ER Positive" contains "Positive" but is explicit)
                is_ambiguous = False 
            
            # Check 4: Structured/Complex specific formats (Low Risk)
            # e.g. "pT2N0" - unlikely to be ambiguous
            if not is_ambiguous and not is_explicit:
                # If it has mixed letters and numbers and length > 3, assume some structure
                # BUT exclude "Score 3" pattern we just caught (unless logic flow handles it)
                # If it was marked ambiguous above, it stays ambiguous.
                
                has_alpha = bool(re.search(r'[a-z]', v_norm))
                has_digit = bool(re.search(r'[0-9]', v_norm))
                
                if has_alpha and has_digit and len(v_norm) > 3:
                    # Additional check: exclude if the ONLY text is "score", "grade" etc + number
                    # and not in synonyms.
                    # e.g. "grade 3" for Ki67 (wrong context maybe, but generic)
                    # "pT2N0" is specific.
                    
                    is_generic_score = re.match(r'^(score|grade|stade)\s*:?\s*\d', v_norm)
                    if not is_generic_score:
                        structured_count += 1
                        # Treat as lower risk
                    
            if is_ambiguous:
                high_risk_count += 1
            else:
                low_risk_count += 1

        # Calculate final score
        # Risk Score = Fraction of values that are ambiguous
        risk_score = high_risk_count / total_count if total_count > 0 else 0.0
        
        return {
            'risk_score': round(risk_score, 4),
            'total_mentions': total_count,
            'high_risk_mentions': high_risk_count,
            'metrics': {
                'polysemic_terms_ratio': polysemic_count / total_count,
                'numeric_only_ratio': numeric_only_count / total_count,
                'explicit_naming_ratio': explicit_count / total_count,
                'structured_format_ratio': structured_count / total_count
            }
        }

    def run_analysis(self, output_dir: str):
        """Run full analysis and save reports."""
        self.extract_data()
        
        results = []
        out_path = Path(output_dir)
        out_path.mkdir(exist_ok=True, parents=True)
        
        for entity_type, values in self.entities_values.items():
            analysis = self.calculate_risk_score(values, entity_type)
            
            # Determine Risk Class
            score = analysis['risk_score']
            if score < 0.3:
                risk_class = "Low (Faible)"
            elif score < 0.7:
                risk_class = "Medium (Modéré)"
            else:
                risk_class = "High (Elevé)"
                
            results.append({
                'Entity': entity_type,
                'Risk_Score': score,
                'Risk_Class': risk_class,
                'Total_Mentions': analysis['total_mentions'],
                'Ambiguity_Rate': f"{score*100:.1f}%",
                'Polysemy_Rate': f"{analysis['metrics']['polysemic_terms_ratio']*100:.1f}%",
                'Numeric_Ambiguity_Rate': f"{analysis['metrics']['numeric_only_ratio']*100:.1f}%",
                'Self_Described_Rate': f"{analysis['metrics']['explicit_naming_ratio']*100:.1f}%"
            })
            
        # Sort by Risk Score descending (High Risk first)
        results.sort(key=lambda x: x['Risk_Score'], reverse=True)
        
        # Save CSV
        csv_file = out_path / 'risk_context_summary.csv'
        if results:
            keys = results[0].keys()
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
                
        # Save JSON
        json_file = out_path / 'risk_context_full.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
            
        print(f"Analysis complete. Results saved to {output_dir}")
        return results

if __name__ == "__main__":
    # Standardized paths and priority order to match templeability.py
    # Ideally pointing to: d:\CLEM\ESIEE SCHOOL\PARCOURS RECHERCHE\Le juste usage des LLM et méthode NLP en cancélorlogie\ESMO2025_Clement\ESMO2025\Rules\src
    base_src_dir = Path(__file__).parent / "Rules/src"
    rcp_dir = base_src_dir / "Breast/RCP"
    chir_dir = base_src_dir / "Breast/CHIR"
    
    # Priority Order: GS > Pred
    # This list allows us to process Gold Standard files first.
    # Duplicates (same filename) in later folders are skipped by the analyzer.
    data_dirs = [
        rcp_dir / "training_set_breast_cancer",
        rcp_dir / "evaluation_set_breast_cancer_GS",
        chir_dir / "evaluation_set_breast_cancer_chir_GS",
        
        rcp_dir / "training_set_breast_cancer_pred",
        rcp_dir / "evaluation_set_breast_cancer_pred_rules",
        rcp_dir / "evaluation_set_breast_cancer_pred_ner",
        chir_dir / "evaluation_set_breast_cancer_chir_pred_rules"
    ]
    
    # Filter only existing directories
    valid_dirs = [str(d) for d in data_dirs if d.exists()]
    
    # Output to Rules/src/Results to be consistent
    output_dir = base_src_dir / "Results"
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists
    
    if valid_dirs:
        print(f"Processing {len(valid_dirs)} directories in priority order.")
        analyzer = RiskContextAnalyzer(valid_dirs)
        analyzer.run_analysis(str(output_dir))
    else:
        print("No valid data directories found.")

try:
    tracker.stop()
except Exception as e:
    print(f"\nWarning: Generalized error in Eco2AI tracking (likely 'N/A' vs float dtype issue): {e}")
    print("Carbon emission tracking data could not be saved, but analysis results are preserved.")
