"""
Calculate entity frequency: (Total occurrences of an entity) / (Total words in all patient .txt files).
"""

import os
from pathlib import Path
from collections import defaultdict
import csv

from eco2ai import set_params, Tracker

set_params(
    project_name="Consumtion_of_E_frequency.py",
    experiment_description="We Calculate...",
    file_name="Consumtion_of_Duraxell.csv"
)

tracker = Tracker()
tracker.start()

def count_words_in_file(file_path):
    """Count words in a text file using simple whitespace splitting."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Simple whitespace tokenization
            words = content.split()
            return len(words)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def parse_ann_file(ann_path):
    """
    Parse a BRAT .ann annotation file to count entity types.
    Returns a dictionary of entity_type -> count
    """
    entity_counts = defaultdict(int)
    try:
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): 
                    continue
                
                # Check for Entity lines (T...)
                if line.startswith('T'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        entity_info = parts[1]
                        # The entity info is usually "Type Start End"
                        # We just need the Type
                        raw_type = entity_info.split()[0]
                        # Parse entity type from annotation line
                        entity_counts[raw_type] += 1
                        raw_type = entity_info.split()[0]
                        entity_counts[raw_type] += 1
                        
            
                # ATTRIBUTE PARSING
          
                if line.startswith('A'):
                    # Format: A1    Type T1
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        attr_info = parts[1].split()
                        attr_type = attr_info[0]
                        
                        # Capture base attribute type
                        entity_counts[attr_type] += 1
                        
                        # If attribute has a value (e.g. RO_values T1 positive)
                        # attr_info would be ['RO_values', 'T1', 'positive']
                        if len(attr_info) >= 3:
                            attr_value = attr_info[2]
                            detailed_type = f"{attr_type}:{attr_value}"
                            entity_counts[detailed_type] += 1

                # EVENT PARSING
              
                if line.startswith('E'):
                     parts = line.split('\t')
                     if len(parts) >= 2:
                         event_info = parts[1].split()
                         event_type_raw = event_info[0]
                         if ':' in event_type_raw:
                             event_type = event_type_raw.split(':')[0]
                         else:
                             event_type = event_type_raw
                         entity_counts[f"Event_{event_type}"] += 1

                
                # RELATION PARSING
                
                if line.startswith('R'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        rel_info = parts[1].split()
                        rel_type = rel_info[0]
                        entity_counts[f"Relation_{rel_type}"] += 1

                # ----------------
                # NORMALIZATION PARSING
                # ----------------
                if line.startswith('N'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                         norm_info = parts[1].split()
                         norm_type = "Normalization_" + norm_info[0] 
                         entity_counts[norm_type] += 1

    except Exception as e:
        print(f"Error reading {ann_path}: {e}")
    
    return entity_counts

def main():
    """
    Analyze entity frequencies across breast cancer datasets.
    Processes .txt and .ann files from multiple RCP and CHIR directories,
    calculating total word counts and entity occurrence frequencies.
    Results are sorted by entity count and exported to CSV.
    """
    
    script_dir = Path(__file__).parent
    base_path = script_dir / "Rules/src/Breast/RCP"
    chir_base_path = script_dir / "Rules/src/Breast/CHIR"
    
    print(f"Base Path (RCP): {base_path}")
    print(f"Base Path (CHIR): {chir_base_path}")

    # List of directories to process as requested
    data_dirs_paths = [
        base_path / "training_set_breast_cancer",
        base_path / "evaluation_set_breast_cancer_pred_rules",
        base_path / "evaluation_set_breast_cancer_pred_ner",
        base_path / "evaluation_set_breast_cancer_GS",
        chir_base_path / "evaluation_set_breast_cancer_chir_GS",
        chir_base_path / "evaluation_set_breast_cancer_chir_pred_rules"
    ]
    
    # Track statistics
    total_word_count = 0
    entity_global_counts = defaultdict(int)
    processed_files = set() # To avoid duplicates if any
    
    print("\nStarting analysis...")
    
    file_count = 0
    
    for folder_path in data_dirs_paths:
        if not folder_path.exists():
            print(f"Warning: Directory not found: {folder_path}")
            continue
            
        print(f"Processing folder: {folder_path.name}")
    
        txt_files = list(folder_path.glob("*.txt"))
        
        for txt_file in txt_files:
            if txt_file.name.startswith('.'):
                continue
                
            # Track file to avoid duplicate word counts if same file appears in multiple folders
            file_stem = txt_file.stem
            if file_stem in processed_files:
                continue
            processed_files.add(file_stem)
            w_count = count_words_in_file(txt_file)
            total_word_count += w_count
            file_count += 1
            
            # Entity Count (.ann with same stem)
            ann_file = txt_file.with_suffix('.ann')
            if ann_file.exists():
                counts = parse_ann_file(ann_file)
                for ent, cnt in counts.items():
                    entity_global_counts[ent] += cnt
                    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total Text Files Analyzed: {file_count}")
    print(f"Total Word Count: {total_word_count}")
    print(f"Total Unique Entity Types: {len(entity_global_counts)}")
    print("-" * 60)
    
    # Calculate Frequency
    frequency_results = []
    
    if total_word_count > 0:
        for ent, count in entity_global_counts.items():
            freq = count / total_word_count
            frequency_results.append({
                'Entity': ent,
                'Count': count,
                'Frequency': freq
            })
            
        # Sort by Count desc
        frequency_results.sort(key=lambda x: x['Count'], reverse=True)
        
        print(f"{'Entity':<30} | {'Count':<10} | {'Frequency':<15}")
        print("-" * 65)
        for item in frequency_results:
            print(f"{item['Entity']:<30} | {item['Count']:<10} | {item['Frequency']:.6f}")
            
        # Optional: Save to CSV
        results_dir = script_dir.parent / "ESMO2025/Rules/src/Results/"
        results_dir.mkdir(exist_ok=True)
        output_csv = results_dir / "entity_frequencies.csv"
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Entity', 'Count', 'Frequency'])
            writer.writeheader()
            writer.writerows(frequency_results)
            
        print(f"\nResults saved to {output_csv}")
        
    else:
        print("No words found. Cannot calculate frequency.")

if __name__ == "__main__":
    main()

try:
    tracker.stop()
except Exception as e:
    print(f"\nWarning: Generalized error in Eco2AI tracking (likely 'N/A' vs float dtype issue): {e}")
    print("Carbon emission tracking data could not be saved, but analysis results are preserved.")