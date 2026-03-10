"""
Report generation orchestrators for ChamberCheck.

Handles visualization and summary report generation from processed LLM analysis results.
Separates business logic from CLI handling.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt


def load_processed_files(subreddit: str, processed_dir: str = "data/processed") -> List[Dict[str, Any]]:
    """Load all analysis results for a subreddit from processed folder."""
    all_results = []
    processed_path = Path(processed_dir)
    
    for file in processed_path.glob(f"{subreddit}_analysis_*.json"):
        if "_metadata" not in file.name:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    all_results.extend(results)
            except Exception as e:
                print(f"⚠️  Error loading {file.name}: {e}")
    
    return all_results


def load_all_subreddit_data(processed_dir: str = "data/processed") -> Dict[str, Dict[str, Any]]:
    """Load analysis results for all subreddits."""
    processed_path = Path(processed_dir)
    subreddit_data = {}
    
    for file in processed_path.glob("*_analysis_*.json"):
        if "_metadata" not in file.name:
            subreddit = file.name.split("_analysis_")[0]
            
            if subreddit not in subreddit_data:
                subreddit_data[subreddit] = {
                    "argument_narrowness": [],
                    "hostility": [],
                    "suppression": [],
                    "epistemic_closure": [],
                    "comment_types": []
                }
            
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    
                for result in results:
                    if 'error' in result:
                        continue
                    
                    for metric in ["argument_narrowness", "hostility", "suppression", "epistemic_closure"]:
                        value = result.get(metric)
                        if value is not None and not (isinstance(value, str) and value.upper() == "N/A"):
                            try:
                                subreddit_data[subreddit][metric].append(float(value))
                            except (ValueError, TypeError):
                                pass
                    
                    comment_types = result.get("comment_types", [])
                    if isinstance(comment_types, list):
                        subreddit_data[subreddit]["comment_types"].extend(comment_types)
            except Exception as e:
                print(f"⚠️  Error loading {file.name}: {e}")
    
    return subreddit_data


def extract_metric_data(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Extract metric scores and compute reporting statistics."""
    metrics = [
        "argument_narrowness",
        "hostility",
        "suppression",
        "epistemic_closure",
        "argument_avoidance"
    ]
    
    metric_data = {}
    
    for metric in metrics:
        scores = []
        na_count = 0
        
        for result in results:
            if 'error' in result:
                continue
            
            value = result.get(metric)
            if value is not None:
                if isinstance(value, str) and value.upper() == "N/A":
                    na_count += 1
                else:
                    try:
                        score = float(value)
                        scores.append(score)
                    except (ValueError, TypeError):
                        pass
        
        total = len(scores) + na_count
        reported_count = len(scores)
        percent_reported = (reported_count / total * 100) if total > 0 else 0
        
        metric_data[metric] = {
            "scores": scores,
            "reported_count": reported_count,
            "total_count": total,
            "percent_reported": percent_reported,
            "na_count": na_count
        }
    
    return metric_data


def compute_means(subreddit_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """Compute mean scores for each metric per subreddit."""
    means = {}
    
    for subreddit, metrics in subreddit_data.items():
        means[subreddit] = {}
        
        for metric, scores in metrics.items():
            if metric == "comment_types":
                continue
            
            if scores:
                means[subreddit][metric] = np.mean(scores)
            else:
                means[subreddit][metric] = np.nan
        
        valid_means = [v for v in means[subreddit].values() if not np.isnan(v)]
        if valid_means:
            means[subreddit]["overall"] = np.mean(valid_means)
        else:
            means[subreddit]["overall"] = np.nan
    
    return means


def create_metric_plots(metric_data: Dict[str, Dict[str, Any]], subreddit: str, output_dir: str = "reports") -> str:
    """Create combined plot with all metrics and return output path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pretty_names = {
        "argument_narrowness": "Argument Narrowness",
        "hostility": "Hostility",
        "suppression": "Suppression",
        "epistemic_closure": "Epistemic Closure"
    }
    
    fig = plt.figure(figsize=(12, 16))
    fig.suptitle(f'r/{subreddit} - Echo Chamber Metrics', fontsize=16, fontweight='bold', y=0.995)
    
    metrics_order = ["argument_narrowness", "hostility", "suppression", "epistemic_closure"]
    
    for idx, metric in enumerate(metrics_order):
        data = metric_data[metric]
        scores = data["scores"]
        reported_count = data["reported_count"]
        total_count = data["total_count"]
        pretty_name = pretty_names.get(metric, metric)
        
        gs = fig.add_gridspec(4, 3, width_ratios=[0.2, 0.1, 2], wspace=0.05, hspace=0.3)
        
        # Bar plot showing percent reported
        ax_bar = fig.add_subplot(gs[idx, 0])
        percent = data["percent_reported"]
        ax_bar.bar(0, percent, color='green', alpha=0.7, width=0.6)
        ax_bar.set_ylim(0, 105)
        ax_bar.text(0, percent + 2, f'{percent:.0f}%', ha='center', fontsize=10, fontweight='bold')
        ax_bar.set_xlabel('% reported', fontsize=9)
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        
        # Spacer
        ax_spacer = fig.add_subplot(gs[idx, 1])
        ax_spacer.axis('off')
        
        # Histogram
        ax_hist = fig.add_subplot(gs[idx, 2])
        if scores:
            ax_hist.hist(scores, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
            ax_hist.set_xlabel('Score', fontsize=11)
            ax_hist.set_ylabel('Frequency', fontsize=11)
            ax_hist.set_title(f'{pretty_name} (n={reported_count}/{total_count})', fontsize=12, fontweight='bold')
            ax_hist.set_xlim(0, 10)
            ax_hist.grid(axis='y', alpha=0.3)
            
            mean_score = np.mean(scores)
            ax_hist.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
            ax_hist.legend(fontsize=9)
        else:
            ax_hist.text(0.5, 0.5, 'No valid scores', ha='center', va='center', fontsize=11)
            ax_hist.set_title(f'{pretty_name} (n={reported_count}/{total_count})', fontsize=12, fontweight='bold')
            ax_hist.set_xlim(0, 1)
            ax_hist.set_ylim(0, 1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = Path(output_dir) / f'{subreddit}_all_metrics_combined.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_file)


def create_comparison_plot(means: Dict[str, Dict[str, float]], output_dir: str = "reports") -> str:
    """Create comparison plot across all subreddits and return output path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pretty_names = {
        "argument_narrowness": "Argument Narrowness",
        "hostility": "Hostility",
        "suppression": "Suppression",
        "epistemic_closure": "Epistemic Closure",
        "overall": "Overall Echo Chamber"
    }
    
    metrics_to_plot = ["argument_narrowness", "hostility", "suppression", "epistemic_closure", "overall"]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    fig.suptitle('Echo Chamber Metrics Comparison Across Subreddits', fontsize=16, fontweight='bold')
    
    subreddits = sorted(means.keys())
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [means[sub].get(metric, np.nan) for sub in subreddits]
        valid_values = [v for v in values if not np.isnan(v)]
        
        bars = ax.bar(range(len(subreddits)), values, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Color invalid bars differently
        for i, v in enumerate(values):
            if np.isnan(v):
                bars[i].set_color('lightgray')
                bars[i].set_alpha(0.3)
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(pretty_names.get(metric, metric), fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(subreddits)))
        ax.set_xticklabels([f'r/{s}' for s in subreddits], rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 10)
        ax.grid(axis='y', alpha=0.3)
        
        if valid_values:
            mean_val = np.mean(valid_values)
            ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.2f}')
            ax.legend(fontsize=9)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'comparison_all_subreddits.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_file)


def create_summary_stats(metric_data: Dict[str, Dict[str, Any]], subreddit: str, output_dir: str = "reports") -> str:
    """Create and save summary statistics table. Return output path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pretty_names = {
        "argument_narrowness": "Argument Narrowness",
        "hostility": "Hostility",
        "suppression": "Suppression",
        "epistemic_closure": "Epistemic Closure",
        "argument_avoidance": "Argument Avoidance"
    }
    
    summary_lines = [
        f"{'='*80}",
        f"ECHO CHAMBER ANALYSIS SUMMARY - r/{subreddit}",
        f"{'='*80}\n"
    ]
    
    for metric, data in metric_data.items():
        scores = data["scores"]
        pretty_name = pretty_names.get(metric, metric)
        
        summary_lines.append(f"{pretty_name}:")
        summary_lines.append(f"  Reported: {data['reported_count']}/{data['total_count']} ({data['percent_reported']:.1f}%)")
        
        if scores:
            summary_lines.append(f"  Mean: {np.mean(scores):.2f}")
            summary_lines.append(f"  Median: {np.median(scores):.2f}")
            summary_lines.append(f"  Std Dev: {np.std(scores):.2f}")
            summary_lines.append(f"  Range: {min(scores):.1f} - {max(scores):.1f}")
        else:
            summary_lines.append(f"  No valid scores")
        
        summary_lines.append("")
    
    summary_text = "\n".join(summary_lines)
    
    output_file = Path(output_dir) / f"{subreddit}_summary.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    return str(output_file), summary_text


def generate_subreddit_report(subreddit: str, processed_dir: str = "data/processed", output_dir: str = "reports") -> Dict[str, str]:
    """
    Generate complete report for a subreddit.
    
    Main orchestrator function that users can call directly.
    
    Args:
        subreddit: Subreddit name (e.g., "samharris")
        processed_dir: Path to processed data directory
        output_dir: Path to output reports directory
    
    Returns:
        Dictionary with paths to generated files
    """
    results = load_processed_files(subreddit, processed_dir)
    
    if not results:
        raise FileNotFoundError(f"No processed results found for r/{subreddit}")
    
    valid_results = [r for r in results if 'error' not in r]
    metric_data = extract_metric_data(valid_results)
    
    plot_file = create_metric_plots(metric_data, subreddit, output_dir)
    summary_file, summary_text = create_summary_stats(metric_data, subreddit, output_dir)
    
    return {
        "plot": plot_file,
        "summary": summary_file,
        "summary_text": summary_text,
        "valid_results": len(valid_results),
        "total_results": len(results)
    }


def generate_comparison_report(processed_dir: str = "data/processed", output_dir: str = "reports") -> Dict[str, Any]:
    """
    Generate comparison report across all subreddits.
    
    Main orchestrator function for cross-subreddit analysis.
    
    Args:
        processed_dir: Path to processed data directory
        output_dir: Path to output reports directory
    
    Returns:
        Dictionary with paths and statistics from generated report
    """
    subreddit_data = load_all_subreddit_data(processed_dir)
    
    if not subreddit_data:
        raise FileNotFoundError(f"No processed results found in {processed_dir}")
    
    means = compute_means(subreddit_data)
    comparison_file = create_comparison_plot(means, output_dir)
    
    return {
        "comparison_plot": comparison_file,
        "subreddits": len(subreddit_data),
        "subreddit_names": sorted(subreddit_data.keys())
    }


def batch_generate_reports(processed_dir: str = "data/processed", output_dir: str = "reports") -> Dict[str, Any]:
    """
    Generate reports for all subreddits.
    
    Orchestrator for batch report generation.
    
    Args:
        processed_dir: Path to processed data directory
        output_dir: Path to output reports directory
    
    Returns:
        Dictionary with summary of all generated reports
    """
    # Find all unique subreddits
    processed_path = Path(processed_dir)
    subreddits = set()
    
    for file in processed_path.glob("*_analysis_*.json"):
        if "_metadata" not in file.name:
            subreddit = file.name.split("_analysis_")[0]
            subreddits.add(subreddit)
    
    if not subreddits:
        raise FileNotFoundError(f"No processed results found in {processed_dir}")
    
    results = {}
    failed = []
    
    for subreddit in sorted(subreddits):
        try:
            results[subreddit] = generate_subreddit_report(subreddit, processed_dir, output_dir)
        except Exception as e:
            failed.append((subreddit, str(e)))
    
    return {
        "successful": len(results),
        "failed": len(failed),
        "subreddits": results,
        "errors": failed
    }
