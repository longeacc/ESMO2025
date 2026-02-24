import os 
import pandas 
import numpy as np
import matplotlib.pyplot as plt

from eco2ai import set_params, Tracker
set_params(
    project_name="Consumtion_of_E_creation_arbre_decision.py",
    experiment_description="We Calculate...",
    file_name="Consumtion_of_Duraxell.csv"
)
tracker = Tracker()
tracker.start()

"""
   ┌───────────────────────┐
   │ DÉBUT Entité e        │
   └──────────┬────────────┘
              │ 
              ▼
   ┌───────────────────────┐
   │ Calcul des métriques  │
   │ SUR JEU D'ENTRAINEMENT│
   └──────────┬────────────┘
              │
              ▼
    ◊ Templateabilité ◊ ────────── (Oui) ─────────┐
    ◊     élevée      ◊                           ▼
    └─────────┬───────┘                   ◊ Variabilité  ◊ ────── (Oui) ──────┐
              │ (Non)                     ◊ lexicale     ◊                    ▼
              ▼                           ◊   faible     ◊                ◊ Risque    ◊ ───(Oui)──▶ 🟩 FEUILLE NER À 
     ◊ Templateabilité ◊ ────── (Non) ────▶   (Non)      │                ◊ contextuel◊             BASE DE RÈGLES
     ◊    moyenne      ◊ ◀───────────────────────────────┘                ◊  faible   ◊
     └────────┬────────┘                                                      │ (Non)
              │ (Oui)                                                         ▼
              ▼                                                       ┌────────────────┐
     ◊    Fréquence    ◊ ────── (Non) ───────────────────────────────▶│ Faisabilité    │
     ◊   suffisante    ◊                                              │   NER élevée   │◀─────────────────────────┐
     └────────┬────────┘                                              └───────┬────────┘                          │
              │ (Oui)                                                         │ (Oui)                             │
              ▼                                                               ▼                                   │
     ◊    Rendement    ◊ ────── (Non) ────────────────────────────────────────┘                           ◊ Décalage de ◊ ── (Oui) ──▶ 🟪 FEUILLE NER TRANSFORMER
     ◊   d'annotation  ◊                                                                                  ◊   domaine   ◊                 BIDIRECTIONNEL
     ◊    suffisant    ◊                                                                                  ◊   faible    ◊ 
     └────────┬────────┘                                                                                      │ (Non)
              │ (Oui)                                                                                         ▼
              ▼                                                                                      ◊ Nécessité LLM ◊ ── (Oui) ──▶ 🩷 FEUILLE NER LLM
     🔵 FEUILLE ML                                                                                   ◊    élevée     ◊
        LÉGER NER                                                                                    └────────┬──────┘
                                                                                                              │ (Non)
                                                                                                              ▼
                                                                                                     ◊    Fréquence  ◊ ── (Oui) ──▶ 🟦 FEUILLE ML LÉGER PAR DÉFAUT
                                                                                                     ◊   suffisante  ◊
                                                                                                     └────────┬──────┘
                                                                                                              │ (Non)
                                                                                                              ▼
                                                                                                     🟩 FEUILLE RÈGLES PAR DÉFAUT



                                                                                                        

# Liste des méthodes possibles (pour référence)
METHODES_POSSIBLES = [
    "ML Léger (CRF)",
    "Règles (Regex)",
    "NER Transformer (Fine-tuning)",
    "LLM (Few-shot/Prompting)",
    "Règles par défaut",
    "ML par défaut"
]

Exemple de tableau de decison pour une entité donnée :

HER2 || Méthode recommandée| Templateabilité | Variabilité lexicale | Fréquence | Rendement d'annotation | Décalage de domaine | Nécessité LLM 
        
"""

import json
import logging
import pandas as pd
import os

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- DEFINITIONS DES SEUILS (Calibration) ---
# Ces valeurs doivent être ajustées selon vos analyses statistiques
THRESHOLDS = {
    "TE_HIGH": 0.5,       # Templateabilité élevée
     "TE_LOW": 0.2,        # Templateabilité basse
    "HE_HIGH": 0.5,       # Homogénéité (Variabilité lexicale FAIBLE si He > ce seuil)
    "RISK_LOW": 0.3,      # Risque contextuel faible
    "FREQ_SUFFICIENT": 0.001, # Fréquence suffisante
    "ANNOTATION_YIELD": 0.5, # Rendement d'annotation 
    "NER_FEASIBILITY": 0.6, 
    "DOMAIN_SHIFT_LOW": 0.8, 
    "LLM_NECESSITY": 0.7   
}

def load_data():
    """Charge les données JSON/CSV produites par les scripts précédents."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "Rules/src/Results")
    
    # 1. TEMPLATEABILITE (Te)
    te_data = {}
    te_path = os.path.join(base_path, "templeability_analysis.json")
    try:
        if os.path.exists(te_path):
            with open(te_path, "r", encoding="utf-8") as f:
                raw_te = json.load(f)
                if isinstance(raw_te, dict):
                    te_data = raw_te
                elif isinstance(raw_te, list):
                    for item in raw_te:
                        ent = item.get("entity_type", "Unknown")
                        score = item.get("templeability_score", 0.0)
                        te_data[ent] = {"templeability_score": score}
        else:
             logging.warning(f"{te_path} introuvable.")

    except Exception as e:
        logging.error(f"Erreur chargement Te: {e}")

    # 2. RISQUE CONTEXTUEL (R)
    risk_data = {}
    risk_path = os.path.join(base_path, "risk_context_full.json")
    try:
        if os.path.exists(risk_path):
            with open(risk_path, "r", encoding="utf-8") as f:
                raw_risk = json.load(f)
                if isinstance(raw_risk, dict):
                    risk_data = raw_risk
                elif isinstance(raw_risk, list):
                    for item in raw_risk:
                        ent = item.get("entity_type", "Unknown")
                        score = item.get("risk_score", 1.0)
                        risk_data[ent] = {"risk_score": score}
        else:
             logging.warning(f"{risk_path} introuvable. R=1.0 par défaut.")
             
    except Exception as e:
        logging.error(f"Erreur chargement Risk: {e}")

    # 3. HOMOGENEITE (He)
    df_he = pd.DataFrame()
    he_path = os.path.join(base_path, "homogeneity_analysis.csv")
    try:
        if os.path.exists(he_path):
            df_he = pd.read_csv(he_path)
        else:
            logging.warning(f"{he_path} introuvable. He=0.0 par défaut.")
    except Exception as e:
        logging.error(f"Erreur chargement He: {e}")

    # 4. FREQUENCE (Freq)
    df_freq = pd.DataFrame()
    freq_path = os.path.join(base_path, "entity_frequencies.csv")
    try:
        if os.path.exists(freq_path):
            df_freq = pd.read_csv(freq_path)
        else:
            logging.warning(f"{freq_path} introuvable. Freq=0.0.")
    except Exception as e:
         logging.error(f"Erreur chargement Freq: {e}")

    return te_data, risk_data, df_he, df_freq

def get_metrics_for_entity(entity_name, te_data, risk_data, df_he, df_freq):
    """Extrait et nettoie les métriques pour une entité donnée."""
    
    # Te
    te_val = 0.0
    if entity_name in te_data:
         te_val = te_data[entity_name].get("templeability_score", 0.0)

    # R (Risque)
    r_val = 1.0 # High risk by default
    if entity_name in risk_data:
        r_val = risk_data[entity_name].get("risk_score", 1.0)
    
    # He (Homogénéité)
    he_val = 0.0
    if not df_he.empty and "Entity" in df_he.columns:
        row = df_he[df_he["Entity"] == entity_name]
        if not row.empty:
            he_val = row["Homogeneity_Score"].values[0]

    # Freq
    freq_val = 0.0
    total_count = 0
    if not df_freq.empty and "Entity" in df_freq.columns:
        row = df_freq[df_freq["Entity"] == entity_name]
        if not row.empty:
            freq_val = row["Frequency"].values[0]
            total_count = row["Count"].values[0]

    return {
        "Te": te_val,
        "He": he_val,
        "R": r_val,
        "Freq": freq_val,
        "Count": total_count
    }

def decide_nlp_method(metrics):
    """
    Implémente l'arbre de décision.
    Retourne (Méthode, Justification)
    """
    Te = metrics["Te"]
    He = metrics["He"]
    R = metrics["R"]
    Freq = metrics["Freq"]
    Count = metrics["Count"]
    
    trace = [] # Pour stocker le chemin emprunté
    
    # 1. Templateabilité Élevée ?
    if Te > THRESHOLDS["TE_HIGH"]:
        trace.append("Te > High")
        
        # 2. Variabilité Lexicale Faible ? (He Élevée = Variabilité Faible)
        if He > THRESHOLDS["HE_HIGH"]:
            trace.append("He > High")
            
            # 3. Risque Contextuel Faible ?
            if R < THRESHOLDS["RISK_LOW"]:
                return "🟩 NER À BASE DE RÈGLES", " -> ".join(trace) + " -> R < Low"
            else:
                trace.append("R >= Low")
        else:
            trace.append("He <= High")
    else:
        trace.append("Te <= High")

    # Si on arrive ici, la branche "Règles" a échoué.
    
    # 4. Templateabilité Moyenne ?
    is_te_medium = (THRESHOLDS["TE_LOW"] < Te <= THRESHOLDS["TE_HIGH"])
    
    if is_te_medium:
        trace.append("Te Medium")
        
        # 5. Fréquence Suffisante ?
        if Freq > THRESHOLDS["FREQ_SUFFICIENT"]:
            trace.append("Freq > Suff")
            
            # 6. Rendement d'annotation (Proxy: Count > 50)
            if Count > 50: 
                return "🔵 ML LÉGER NER (CRF)", " -> ".join(trace) + " -> Yield OK"
            else:
                 trace.append("Yield Low")
        else:
             trace.append("Freq Low")
    else:
         trace.append("Te Low")

    # Si on arrive ici, on a échoué "Règles" et "ML Léger".
    
    # 7. Faisabilité NER Élevée ? (Proxy combiné)
    Faisabilite_NER_High = (Freq > THRESHOLDS["FREQ_SUFFICIENT"]) or (R < 0.8)
    
    if Faisabilite_NER_High:
        trace.append("NER Feasibe")
        # 8. Décalage de Domaine Faible ? (Suppose Oui par défaut)
        Decalage_Domaine_Faible = True 
        if Decalage_Domaine_Faible:
             return "🟣 NER TRANSFORMER", " -> ".join(trace) + " -> Domain OK"
    else:
        trace.append("NER Not Feasible")
    
    # 9. Nécessité LLM Élevée ? (Si Tâche complexe: R élevé, Te faible)
    Necessite_LLM_High = (R > 0.6 and Te < 0.2)
    
    if Necessite_LLM_High:
         return "🩷 NER LLM", " -> ".join(trace) + " -> LLM Needed"
         
    # 10. Fréquence Suffisante (Finale) ?
    if Freq > THRESHOLDS["FREQ_SUFFICIENT"]:
        return "🟦 ML LÉGER PAR DÉFAUT", " -> ".join(trace) + " -> Fallback ML"
    else:
        return "🟩 RÈGLES PAR DÉFAUT", " -> ".join(trace) + " -> Fallback Règles"

def main():
    te_data, risk_data, df_he, df_freq = load_data()
    
    entities = list(te_data.keys()) if te_data else []
    # Fusionner avec les entités de frequentcy si non trouvé
    if not entities and not df_freq.empty:
        entities = df_freq["Entity"].unique().tolist()
    # Ou les clés de risk_data
    if not entities and risk_data:
        entities = list(risk_data.keys())

    results = []
    
    for entity in sorted(list(set(entities))):
        if not entity: continue
        metrics = get_metrics_for_entity(entity, te_data, risk_data, df_he, df_freq)
        decision, justification = decide_nlp_method(metrics)
        
        results.append({
            "Entité": entity,
            "Méthode Recommandée": decision,
            "Te": metrics['Te'],
            "He": metrics['He'],
            "R": metrics['R'],
            "Freq": metrics['Freq'],
            "Total": metrics['Count'],
            "Justification": justification
        })
        
    df_res = pd.DataFrame(results)
    
    # Affichage MARKDOWN du tableau
    print("\n" + "="*80)
    print("🌳 RÉSULTATS DE L'ARBRE DE DÉCISION NLP 🌳")
    print("="*80 + "\n")
    print(df_res.to_markdown(index=False, floatfmt=".3f"))
    
    # Sauvegarde CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "Rules/src/Results/decision_tree_results.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_res.to_csv(output_path, index=False)
    print(f"\nRésultats sauvegardés dans : {output_path}")

if __name__ == "__main__":
    main()

try:
    tracker.stop()
except:
    pass







