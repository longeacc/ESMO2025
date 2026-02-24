"""
Calculate templeability of biomarkers and named entities.

Templeability is the capacity of an entity to follow predictable structured patterns 
(formats, constant prefixes/suffixes).

Example: TNM staging always follows the pattern T[0-4]N[0-3]M[0-1].
"""

import os
import re
import csv
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple
import json
from eco2ai import set_params, Tracker

set_params(
    project_name="Consumtion_of_E_templeability.py",
    experiment_description="We Calculate...",
    file_name="Consumtion_of_Duraxell.csv"
)

tracker = Tracker()
tracker.start()

class TempleabilityAnalyzer:
    """Analyze templeability of biomarkers extracted from medical text."""
    
    def __init__(self, data_dirs: List[str]):
        """
        Initialize the analyzer with the path to the dataset.
        
        Args:
            data_dirs: List of paths to the directories containing .txt and .ann files
        """
        self.data_dirs = [Path(d) for d in data_dirs]
        self.entities_values = defaultdict(list)  # entity_type -> [values]
        self.entity_doc_presence = defaultdict(set) # entity_type -> set(file_ids)
        self.total_files_count = 0
        
    def parse_ann_file(self, ann_path: str) -> Dict[str, list]:
        """
        Parse a BRAT .ann annotation file.
        
        Args:
            ann_path: Path to .ann file
            
        Returns:
            Dictionary mapping entity types to list of (start, end, text) tuples
        """
        entities = defaultdict(list)
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('A'):
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 3 and parts[0].startswith('T'):
                    # Format: T{id}\t{entity_type} {start} {end}\t{text}
                    # Note: {start} {end} can have multiple spans like "1702 1705;1706 1710"
                    entity_info_parts = parts[1].split()
                    entity_type = entity_info_parts[0]
                    text = parts[2]
                    
                    # Extract first position for reference
                    try:
                        start_pos = int(entity_info_parts[1])
                        # Handle non-contiguous spans
                        end_info = entity_info_parts[2]
                        if ';' in end_info:
                            # Take first span's end
                            end_pos = int(end_info.split(';')[0])
                        else:
                            end_pos = int(end_info)
                        
                        entities[entity_type].append({
                            'text': text,
                            'start': start_pos,
                            'end': end_pos
                        })
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
        
        return entities
    
    def load_all_documents(self):
        """Load all annotations from the dataset."""
        processed_files = set()
        
        for data_dir in self.data_dirs:
            if not data_dir.exists():
                print(f"Warning: Directory not found: {data_dir}")
                continue

            ann_files = list(data_dir.glob('*.ann'))
            print(f"Found {len(ann_files)} annotation files in {data_dir.name}")
            
            for ann_file in ann_files:
                # Use filename as unique identifier to avoid duplicates across datasets (GS vs predictions)
                file_id = ann_file.name
                if file_id in processed_files:
                    continue
                    
                processed_files.add(file_id)
                entities = self.parse_ann_file(str(ann_file))
                
                for entity_type, mentions in entities.items():
                    # Record document presence
                    self.entity_doc_presence[entity_type].add(file_id)
                    
                    for mention in mentions:
                         # Extract the value (e.g., "95%" from "RO+(95%)")
                        self.entities_values[entity_type].append(mention['text'])
        
        self.total_files_count = len(processed_files)
        print(f"Total unique patient records analyzed: {self.total_files_count}")
    
    def analyze_pattern_complexity(self, values: List[str], entity_type: str) -> Dict:
        """
        Analyze the structural patterns of a set of values with advanced heuristics.
        
        Args:
            values: List of text values for an entity type
            entity_type: Name of the entity
            
        Returns:
            Dictionary with pattern analysis results
        """
        if not values:
            return {'count': 0, 'patterns': [], 'templeability': 0.0}
        
        total_count = len(values)
        unique_values = sorted(list(set(values)))  # Sort for easier reading
        
        # Calculate document coverage
        docs_with_entity = len(self.entity_doc_presence[entity_type])
        presence_ratio = docs_with_entity / self.total_files_count if self.total_files_count > 0 else 0
        
        # 1. Normalization & Abstract Pattern Generation
        # Convert "Ki67 90%" -> "XXXX DD%" (X=Upper, D=Digit)
        abstract_patterns = []
        normalized_values = []
        pattern_to_examples = defaultdict(list)
        
        for v in values:
            # Basic normalization (case, spaces)
            norm = v.strip()
            normalized_values.append(norm)
            
            # Create abstract pattern
            # Replace Digits with D, Uppercase with X, Lowercase with x
            pattern = re.sub(r'[0-9]', 'D', norm)
            pattern = re.sub(r'[A-Z]', 'X', pattern)
            pattern = re.sub(r'[a-z]', 'x', pattern)
            abstract_patterns.append(pattern)
            pattern_to_examples[pattern].append(norm)
            
        # Analyze Abstract Patterns
        pattern_counts = Counter(abstract_patterns)
        
        # Create detailed top patterns list with examples
        top_patterns_details = []
        # Get top 5 patterns
        for pat, count in pattern_counts.most_common(5):
            pct = (count / total_count) * 100
            # Get most common real example for this pattern
            examples = pattern_to_examples[pat]
            representative = ""
            if examples:
                representative = Counter(examples).most_common(1)[0][0]
            
            top_patterns_details.append(f"{pat}  (Ex: {representative})")
            
        top_patterns = top_patterns_details
        
        # Calculate coverage of top 3 patterns
        top_3_count = sum(c for _, c in pattern_counts.most_common(3))
        top_3_coverage = top_3_count / total_count
        
        # 2. Heuristic Pattern Detection
        patterns_found = []
        
        # Percentage check (robust)
        has_percentage = sum(1 for v in normalized_values if '%' in v) / total_count > 0.7
        if has_percentage:
            patterns_found.append('percentage_format')
            
        # Plus/Minus check
        has_plus_minus = sum(1 for v in normalized_values if any(c in v for c in ['+', '-'])) / total_count > 0.5
        if has_plus_minus:
            patterns_found.append('plus_minus_format')
            
        # Score pattern (e.g. "Score 3", "Score2")
        score_pattern = sum(1 for v in normalized_values if re.search(r'score\s*:?\s*\d', v, re.IGNORECASE)) / total_count
        if score_pattern > 0.3:
            patterns_found.append('score_keyword_pattern')

        # Status keywords (Positif, Negatif, Amplifié)
        status_keywords = r'(positif|négatif|amplifié|amplifie|negatif|positive|negative|detected|mutation)'
        status_match = sum(1 for v in normalized_values if re.search(status_keywords, v, re.IGNORECASE)) / total_count
        if status_match > 0.4:
            patterns_found.append('status_keyword_pattern')
            
        # Range pattern (10-20, <10, >5)
        range_match = sum(1 for v in normalized_values if re.search(r'(>|<|\d+\s*-\s*\d+)', v)) / total_count
        if range_match > 0.2:
            patterns_found.append('range_operator_pattern')
            
        # 3. Structure Consistency Analysis
        prefixes = [v[:2] for v in normalized_values if len(v) >= 2]
        suffixes = [v[-1] for v in normalized_values if len(v) >= 1]
        
        prefix_consistency = 0.0
        if prefixes:
            top_prefix_count = Counter(prefixes).most_common(1)[0][1]
            prefix_consistency = top_prefix_count / len(prefixes)
            if prefix_consistency > 0.5:
                patterns_found.append('consistent_start')

        suffix_consistency = 0.0
        if suffixes:
            top_suffix_count = Counter(suffixes).most_common(1)[0][1]
            suffix_consistency = top_suffix_count / len(suffixes)
            if suffix_consistency > 0.5:
                patterns_found.append('consistent_end')
        
        # 4. Templeability Score Calculation (Refined)
        # Base: Pattern coverage (how much of the data follows the top abstract patterns?)
        # Bonus: Detected specific semantic patterns (percentage, score, etc.)
        # Penalty: High number of unique abstract patterns (high logical diversity)
        
        unique_abstract_ratio = len(pattern_counts) / total_count
        structure_score = top_3_coverage  # If 90% of data fits into 3 shapes, it's very templeable
        
        semantic_score = min(len(patterns_found) * 0.15, 0.4) # Max 0.4 bonus for recognized types
        
        templeability = (structure_score * 0.7) + (semantic_score) + (0.1 if prefix_consistency > 0.8 else 0)
        templeability = min(templeability, 1.0) # Cap at 1.0
        
        return {
            'count': total_count,
            'docs_count': docs_with_entity,
            'total_docs_in_set': self.total_files_count,
            'presence_ratio': round(presence_ratio, 3),
            'unique_values_count': len(unique_values),
            'diversity_ratio': round(len(unique_values) / total_count, 3),
            'top_abstract_patterns': top_patterns,
            'top_3_pattern_coverage': round(top_3_coverage, 3),
            'patterns': patterns_found,
            'prefixes': list(set(prefixes[:10])) if prefixes else [], # Sample prefixes
            'suffixes': list(set(suffixes[:10])) if suffixes else [],
            'has_percentage': has_percentage,
            'has_plus_minus': has_plus_minus,
            'templeability_score': round(templeability, 4),
            'sample_values': unique_values[:20]  # Take 20 samples
        }

    
    def generate_report(self) -> Dict[str, Dict]:
        """
        Generate templeability report for all entities.
        
        Returns:
            Dictionary with analysis for each entity type
        """
        report = {}
        
        for entity_type in sorted(self.entities_values.keys()):
            values = self.entities_values[entity_type]
            analysis = self.analyze_pattern_complexity(values, entity_type)
            report[entity_type] = analysis
        
        return report
    
    def print_report(self, report: Dict[str, Dict]):
        """Print a formatted report of templeability analysis."""
        print("\n" + "="*80)
        print("TEMPLEABILITY ANALYSIS REPORT (ADVANCED)")
        print("="*80 + "\n")
        
        # Sort by templeability score (descending)
        sorted_entities = sorted(
            report.items(),
            key=lambda x: x[1]['templeability_score'],
            reverse=True
        )
        
        for entity_type, analysis in sorted_entities:
            print(f"Entity Type: {entity_type}")
            print(f"  Total Occurrences: {analysis['count']}")
            print(f"  Unique Values: {analysis['unique_values_count']}")
            print(f"  Diversity Ratio: {analysis['diversity_ratio']} (Lower is better)")
            
            print(f"  Top Abstract Patterns ({analysis['top_3_pattern_coverage']*100:.1f}% coverage):")
            for pat in analysis['top_abstract_patterns'][:3]:
                print(f"    - {pat}")
            
            if analysis['patterns']:
                print(f"  Detected Semantics: {', '.join(analysis['patterns'])}")
            
            # Templeability interpretation
            score = analysis['templeability_score']
            if score >= 0.8:
                interpretation = "HIGHLY TEMPLEABLE "
            elif score >= 0.5:
                interpretation = "MODERATELY TEMPLEABLE "
            else:
                interpretation = "LOW TEMPLEABILITY "
            
            print(f" TEMPLEABILITY SCORE: {score}/1.0 - {interpretation}")
            
            # Sample values
            print(f"  Sample values (first 5 of {len(analysis['sample_values'])}): {analysis['sample_values'][:5]}")
            
            print("-" * 40)

    
    def save_report_to_csv(self, report: Dict[str, Dict], output_path: str):
        """
        Save templeability report to CSV file.
        
        Args:
            report: Report dictionary
            output_path: Path to save CSV file
        """
        rows = []
        
        for entity_type, analysis in sorted(report.items()):
            rows.append({
                'Entity_Type': entity_type,
                'Total_Occurrences': analysis['count'],
                'Documents_With_Entity': analysis['docs_count'],
                'Total_Documents_Analyzed': analysis['total_docs_in_set'],
                'Presence_Ratio': analysis['presence_ratio'],
                'Unique_Values': analysis['unique_values_count'],
                'Diversity_Ratio': analysis['diversity_ratio'],
                'Templeability_Score': analysis['templeability_score'],
                'Top_Pattern_Coverage': analysis['top_3_pattern_coverage'],
                'Top_Abstract_Patterns': '|'.join(analysis['top_abstract_patterns']),
                'Detected_Patterns_List': '|'.join(analysis['patterns']),
                'Sample_Values': '|'.join(str(v) for v in analysis['sample_values'])
            })
        
        # Sort rows by score desc
        rows.sort(key=lambda x: x['Templeability_Score'], reverse=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Entity_Type', 'Total_Occurrences', 'Documents_With_Entity', 
                          'Total_Documents_Analyzed', 'Presence_Ratio', 'Unique_Values',
                          'Diversity_Ratio', 'Templeability_Score', 'Top_Pattern_Coverage',
                          'Top_Abstract_Patterns', 'Detected_Patterns_List', 'Sample_Values']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"\nReport saved to: {output_path}")

    
    def save_report_to_json(self, report: Dict[str, Dict], output_path: str):
        """
        Save templeability report to JSON file.
        
        Args:
            report: Report dictionary
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"JSON report saved to: {output_path}")


def main():
    """Main execution function."""
    base_path = Path(__file__).parent / "Rules/src/Breast/RCP"
    chir_base_path = Path(__file__).parent / "Rules/src/Breast/CHIR"
    
    # Define directories in strict priority order (GS > Pred) to ensure consistency
    # This list must be identical to the one in risk_context.py
    data_dirs_paths = [
        # Gold Standards (Highest Quality)
        base_path / "training_set_breast_cancer",
        base_path / "evaluation_set_breast_cancer_GS",
        chir_base_path / "evaluation_set_breast_cancer_chir_GS",
        
        # Predictions (fallback for missing files, though duplicates are skipped)
        base_path / "training_set_breast_cancer_pred",
        base_path / "evaluation_set_breast_cancer_pred_rules",
        base_path / "evaluation_set_breast_cancer_pred_ner",
        chir_base_path / "evaluation_set_breast_cancer_chir_pred_rules"
    ]
    
    # Convert to strings and filter existing
    data_dirs = [str(p) for p in data_dirs_paths]
    
    # Initialize analyzer
    analyzer = TempleabilityAnalyzer(data_dirs)
    
    # Load all documents
    print(f"Loading data from {len(data_dirs)} directories...")
    analyzer.load_all_documents()
    
    print(f"Total entity types found: {len(analyzer.entities_values)}")
    
    # Generate report
    report = analyzer.generate_report()
    
    # Print report
    analyzer.print_report(report)
    
    # Save reports
    results_dir = base_path.parent.parent / "Results"
    output_csv = results_dir / "templeability_analysis.csv"
    output_json = results_dir / "templeability_analysis.json"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer.save_report_to_csv(report, str(output_csv))
    analyzer.save_report_to_json(report, str(output_json))
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    scores = [analysis['templeability_score'] for analysis in report.values()]
    if scores:
        print(f"Average Templeability: {sum(scores) / len(scores):.3f}")
        print(f"Highest Templeability: {max(scores):.3f} ({[k for k, v in report.items() if v['templeability_score'] == max(scores)][0]})")
        print(f"Lowest Templeability: {min(scores):.3f} ({[k for k, v in report.items() if v['templeability_score'] == min(scores)][0]})")


if __name__ == "__main__":
    main()

try:
    tracker.stop()
except Exception as e:
    print(f"\nWarning: Generalized error in Eco2AI tracking (likely 'N/A' vs float dtype issue): {e}")
    print("Carbon emission tracking data could not be saved, but analysis results are preserved.")
