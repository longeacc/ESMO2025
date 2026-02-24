"""
Calculate Linguistic Homogeneity Scores (He) based on Word-level repetition.

Formula (corrected interpretation):
He = (Te - Ue) / Te
Where:
    Te = Total number of words for entity e
    Ue = Number of unique words for entity e

The score is then transformed using a Sigmoid function:
Score = 1 / (1 + e^(-k * (x - 0.5)))

Note: Using standard libraries only (no pandas) due to environment issues.
"""

import json
import math
import os
import re
import csv
from pathlib import Path

from collections import defaultdict

from eco2ai import set_params, Tracker

set_params(
    project_name="Consumtion_of_E_homogeneity.py",
    experiment_description="We Calculate...",
    file_name="Consumtion_of_Duraxell.csv"
)

tracker = Tracker()
tracker.start()

# Configuration
# Calculate paths relative to this script (ESMO2025/homogeneity.py)
BASE_DIR = Path(__file__).parent
DATA_REL_PATH = Path("Rules/src/Results")
INPUT_JSON_PATH = BASE_DIR / DATA_REL_PATH / "templeability_analysis.json"
OUTPUT_DIR = BASE_DIR / DATA_REL_PATH
SIGMOID_K = 10 
SIGMOID_CENTER = 0.5 

def parse_ann_file(ann_path):
    """Parse a BRAT .ann annotation file to extract entity texts."""
    entities = defaultdict(list)
    try:
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('A'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 3 and parts[0].startswith('T'):
                    entity_info = parts[1].split()
                    if not entity_info: continue
                    entity_type = entity_info[0]
                    text = parts[2]
                    entities[entity_type].append(text)
    except Exception as e:
        print(f"Error parsing {ann_path}: {e}")
    return entities

def load_all_entity_values(data_dirs):
    """Load all entity text values from .ann files in directories."""
    entities_values = defaultdict(list)
    for d in data_dirs:
        path = Path(d)
        if not path.exists():
            print(f"Warning: Directory not found: {path}")
            continue
        ann_files = list(path.glob('*.ann'))
        print(f"Found {len(ann_files)} annotation files in {path.name}")
        for ann in ann_files:
            file_entities = parse_ann_file(ann)
            for etype, texts in file_entities.items():
                entities_values[etype].extend(texts)
    return entities_values

def sigmoid(x, k=10, x0=0.5):
    """Apply sigmoid transformation."""
    try:
        return 1 / (1 + math.exp(-k * (x - x0)))
    except OverflowError:
        return 0.0 if (-k * (x - x0)) > 0 else 1.0

def tokenize(text):
    """Simple word tokenization."""
    # Split by non-alphanumeric chars, keep cleaner
    return [w.lower() for w in re.split(r'[^a-zA-Z0-9%]+', text) if w.strip()]

def calculate_word_stats(entities_values):
    """Calculate word-level statistics for each entity type across all loaded docs."""
    stats = {}
    for entity_type, values in entities_values.items():
        all_tokens = []
        for val in values:
            all_tokens.extend(tokenize(val))
            
        N_total = len(all_tokens)
        N_unique = len(set(all_tokens))
        
        stats[entity_type] = {
            'Te_words': N_total,
            'Ue_words': N_unique
        }
    return stats

def run_homogeneity_analysis():
    # 1. Load Templeability Scores (for reference/comparison)
    if not INPUT_JSON_PATH.exists():
        print(f"Error: {INPUT_JSON_PATH} not found. Run templeability.py first.")
        return

    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        te_data = json.load(f)

    # 2. Re-Load Documents to get Text Content
    # Use BASE_DIR defined at top of script
    base_path = BASE_DIR / "Rules/src/Breast/RCP"
    chir_base_path = BASE_DIR / "Rules/src/Breast/CHIR"
    
    # Fallback paths if Rules/src structure isn't populated but root Breast is
    if not base_path.exists():
        base_path = BASE_DIR / "Breast/RCP"
    if not chir_base_path.exists():
        chir_base_path = BASE_DIR / "Breast/CHIR"

    data_dirs = [
        base_path / "training_set_breast_cancer",
        base_path / "evaluation_set_breast_cancer_pred_rules",
        base_path / "evaluation_set_breast_cancer_pred_ner",
        base_path / "evaluation_set_breast_cancer_GS",
        chir_base_path / "evaluation_set_breast_cancer_chir_GS",
        chir_base_path / "evaluation_set_breast_cancer_chir_pred_rules"
    ]
    
    print("Loading documents for linguistic analysis...")
    entities_values = load_all_entity_values(data_dirs)
    
    # 3. Calculate Word Statistics
    word_stats = calculate_word_stats(entities_values)
    
    results = []
    print("\nHomogeneity Analysis (He = (TotalWords - UniqueWords) / TotalWords):")
    print(f"{'Entity':<25} | {'Struc(Te)':<9} | {'Homo(He)':<9} | {'Sigmoid':<9}")
    print("-" * 75)

    for entity_type, metrics in te_data.items():
        if entity_type not in word_stats:
            continue
            
        # Structural Templeability (Reference)
        Struc_Te = metrics.get('templeability_score', 0.0)
        
        # Word Stats
        w_stat = word_stats[entity_type]
        Te_words = w_stat['Te_words']
        Ue_words = w_stat['Ue_words']
        
        # He Calculation (Redundancy Ratio)
        if Te_words > 0:
            He_raw = (Te_words - Ue_words) / Te_words
        else:
            He_raw = 0.0
            
        # Sigmoid Transformation
        He_final = sigmoid(He_raw, k=SIGMOID_K, x0=SIGMOID_CENTER)
        
        results.append({
            'Entity_Type': entity_type,
            'Structural_Templeability': Struc_Te,
            'Total_Words_Te': Te_words,
            'Unique_Words_Ue': Ue_words,
            'Homogeneity_Raw_He': round(He_raw, 4),
            'Homogeneity_Score': round(He_final, 4),
            'Prevalence': f"{metrics.get('docs_count', 0)}/{metrics.get('total_docs_in_set', 0)}"
        })
        
        print(f"{entity_type:<25} | {Struc_Te:<9.4f} | {He_raw:<9.4f} | {He_final:<9.4f}")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sort results using lambda (since no pandas)
    results.sort(key=lambda x: x['Homogeneity_Score'], reverse=True)
    
    # Save CSV
    csv_path = OUTPUT_DIR / "homogeneity_analysis.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Entity_Type', 'Homogeneity_Score', 'Homogeneity_Raw_He', 'Structural_Templeability', 
                      'Total_Words_Te', 'Unique_Words_Ue', 'Prevalence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nCSV Saved: {csv_path}")
    
    generate_html(results, OUTPUT_DIR / "homogeneity_report.html")

def generate_html(data, output_path):
    html = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Analyse d'Homogénéité Linguistique</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #f1f5f9; padding: 40px; color: #333; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #334155; text-align: center; margin-bottom: 20px; }
            
            .method-box { background: #e0f2fe; border-left: 5px solid #0ea5e9; padding: 20px; margin-bottom: 30px; }
            .formula { font-family: monospace; font-weight: bold; font-size: 1.1em; color: #0369a1; background: rgba(255,255,255,0.5); padding: 5px; border-radius: 4px; display: inline-block; margin: 5px 0; }
            
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th { background: #475569; color: white; padding: 12px; text-align: left; }
            td { padding: 12px; border-bottom: 1px solid #e2e8f0; }
            tr:hover { background: #f8fafc; }
            
            .badge { padding: 4px 10px; border-radius: 15px; font-weight: bold; color: white; font-size: 0.9em; }
            .high { background: #22c55e; }
            .medium { background: #f59e0b; }
            .low { background: #b91c1c; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Homogénéité Linguistique (He) vs Templeabilité (Te)</h1>
            
            <div class="method-box">
                <h3>Méthodologie & Définitions</h3>
                <p><strong>Homogénéité Linguistique (He):</strong> Mesure la redondance lexicale, c'est-à-dire la fréquence à laquelle le même vocabulaire est réutilisé pour une entité donnée.</p>
                <p>Une homogénéité élevée indique un vocabulaire très contrôlé et répétitif (peu de variation dans les mots utilisés).</p>
                
                <p><strong>Formules:</strong></p>
                <div class="formula">He = (TotalMots - MotsUniques) / TotalMots</div>
                <br>
                <div class="formula">Score = Sigmoïde(He)</div>
                <p style="font-size: 0.9em; margin-top: 10px;"><em>Le score est normalisé par une fonction sigmoïde pour accentuer les différences proches de 0.5.</em></p>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Type d'Entité</th>
                        <th>Homogénéité (Score)</th>
                        <th>He Brut (Redondance)</th>
                        <th>Templeabilité (Structure)</th>
                        <th>Stats Mots (Uniques/Total)</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for row in data:
        s = row['Homogeneity_Score']
        cls = 'high' if s > 0.8 else 'medium' if s > 0.5 else 'low'
        
        html += f"""
                <tr>
                    <td><strong>{row['Entity_Type']}</strong></td>
                    <td><span class="badge {cls}">{s:.4f}</span></td>
                    <td>{row['Homogeneity_Raw_He']:.4f}</td>
                    <td>{row['Structural_Templeability']:.4f}</td>
                    <td>{int(row['Unique_Words_Ue'])} / {int(row['Total_Words_Te'])}</td>
                </tr>
        """
        
    html += """
                </tbody>
            </table>
            
            <div style="margin-top: 30px; font-size: 0.9em; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 20px;">
                <p><strong>Note d'Interprétation:</strong></p>
                <ul style="margin-top: 5px;">
                    <li><strong>Haute Homogénéité:</strong> Vocabulaire très restreint et répétitif.</li>
                    <li><strong>Haute Templeabilité:</strong> Respect strict de formats structurels (types de caractères, ponctuation).</li>
                    <li>Ces deux métriques corrèlent souvent mais mesurent deux aspects distincts de la qualité des données (Lexical vs Structurel).</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML Saved: {output_path}")

if __name__ == "__main__":
    run_homogeneity_analysis()

try:
    tracker.stop()
except Exception as e:
    print(f"\nWarning: Generalized error in Eco2AI tracking (likely 'N/A' vs float dtype issue): {e}")
    print("Carbon emission tracking data could not be saved, but analysis results are preserved.")