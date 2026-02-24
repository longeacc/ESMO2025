import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from eco2ai import set_params, Tracker

set_params(
    project_name="Consumtion_of_E_fesability_NER.py",
    experiment_description="We Calculate...",
    file_name="Consumtion_of_Duraxell.csv"
)

tracker = Tracker()
tracker.start()

def analyze_patient_files():
    """
    This function scans the specified directories for patient records in .txt and .ann 
    formats, counts the total number of files, and identifies 
    how many are unique across all directories. It also provides 
    a summary of the findings and checks if the total number of 
    files exceeds 500 for fine-tuning purposes.
    """
    root_dir = Path.cwd()
    
    print(f"Scanning for patient records (.txt files) in workspace: {root_dir}")
    
    # Liste des répertoires à analyser
    liste = [
        "Breast/CHIR/evaluation_set_breast_cancer_chir_GS",
        "Breast/CHIR/evaluation_set_breast_cancer_chir_pred_rules",
        "Breast/CHIR/evaluation_set_breast_cancer_chir_pred_ner",
        "Breast/RCP/evaluation_set_breast_cancer_GS",
        "Breast/RCP/evaluation_set_breast_cancer_pred_ner",
        "Breast/RCP/evaluation_set_breast_cancer_pred_rules",
        "Breast/RCP/training_set_breast_cancer",
        "Breast/RCP/training_set_breast_cancer_pred",
        "ESMO2025/Rules/src/Breast/RCP/training_set_breast_cancer_pred",
        "ESMO2025/NER/data/raw/brat",
        "ESMO2025/NER/data/raw/test"
    ]
    
    print(f"{'Nouveaux':<8} | {'Total':<8} | {'Dossier'}")
    print("-" * 100)
    
    total_count = 0
    unique_files_registry = set()

    for path_str in liste:
        # Construction du chemin complet
        full_path = root_dir / Path(path_str)
        
        if full_path.exists():
            # Compter les fichiers .txt et .ann
            # On combine les deux recherches
            import itertools
            current_files = [f for f in itertools.chain(full_path.glob("*.txt"), full_path.glob("*.ann")) if not f.name.startswith('.')]
            
            count_total_in_folder = len(current_files)
            total_count += count_total_in_folder
            count_new_unique = 0
            
            for f in current_files:
                if f.name not in unique_files_registry:
                    unique_files_registry.add(f.name)
                    count_new_unique += 1
            
            print(f"{count_new_unique:<8} | {count_total_in_folder:<8} | {path_str}")
        else:
            print(f"{'MISSING':<8} | {'-':<8} | {path_str}")

    print("-" * 100)
    print(f"Total fichiers (somme brute) : {total_count}")
    
    if total_count > 500:
        print(f"The total number of patient files exceeds 500 (Total: {total_count}), we can fine-tune on all models.")
    else : 
        print(f"Warning: The total number of patient files is less than 500 ({total_count}), we may need to consider data augmentation or other strategies for fine-tuning.")

if __name__ == "__main__":
    analyze_patient_files()

try:
    tracker.stop()
except Exception as e:
    print(f"\nWarning: Generalized error in Eco2AI tracking (likely 'N/A' vs float dtype issue): {e}")
    print("Carbon emission tracking data could not be saved, but analysis results are preserved.")