"""
Evaluation utilities for TransMLLM

Helper functions for evaluation, metric computation, and result formatting.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import pandas as pd


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple samples.
    
    Args:
        results: List of result dictionaries with metrics
    
    Returns:
        Aggregated metrics (mean values)
    """
    if not results:
        return {}
    
    # Collect all metric keys
    metric_keys = set()
    for result in results:
        metric_keys.update(k for k in result.keys() if isinstance(result[k], (int, float)))
    
    aggregated = {}
    for key in metric_keys:
        values = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
        if values:
            aggregated[f"{key}_mean"] = sum(values) / len(values)
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)
    
    return aggregated


def save_evaluation_report(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    format: str = "json"
) -> None:
    """
    Save evaluation report.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Output directory
        format: Output format ("json", "csv", or "both")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format in ["json", "both"]:
        json_path = output_dir / "evaluation_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    if format in ["csv", "both"]:
        # Flatten nested dictionaries for CSV
        flattened = _flatten_dict(results)
        df = pd.DataFrame([flattened])
        csv_path = output_dir / "evaluation_report.csv"
        df.to_csv(csv_path, index=False)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compare_results(
    results1: Dict[str, Any],
    results2: Dict[str, Any],
    metric_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare two evaluation results.
    
    Args:
        results1: First results dictionary
        results2: Second results dictionary
        metric_names: List of metric names to compare (None = all)
    
    Returns:
        Comparison dictionary with differences
    """
    if metric_names is None:
        # Find common metrics
        metric_names = list(set(results1.keys()) & set(results2.keys()))
        metric_names = [m for m in metric_names if isinstance(results1[m], (int, float))]
    
    comparison = {}
    for metric in metric_names:
        if metric in results1 and metric in results2:
            val1 = results1[metric]
            val2 = results2[metric]
            
            if isinstance(val1, dict) and isinstance(val2, dict):
                # Handle nested metrics (e.g., rouge)
                comparison[metric] = {}
                for sub_key in set(val1.keys()) & set(val2.keys()):
                    diff = val2[sub_key] - val1[sub_key]
                    comparison[metric][sub_key] = {
                        "result1": val1[sub_key],
                        "result2": val2[sub_key],
                        "difference": diff,
                        "improvement": diff > 0
                    }
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val2 - val1
                comparison[metric] = {
                    "result1": val1,
                    "result2": val2,
                    "difference": diff,
                    "improvement": diff > 0
                }
    
    return comparison

