"""
Generate reports and visualizations from processed LLM analysis results.

Analyzes echo chamber metrics by subreddit and creates box plot + histogram visualizations.

Usage:
  python scripts/report_results.py samharris
  python scripts/report_results.py samharris --output reports/
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse

import matplotlib.pyplot as plt
import numpy as np


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
                print(f"✓ Loaded {len(results)} results from {file.name}")
            except Exception as e:
                print(f"⚠️  Error loading {file.name}: {e}")
    
    return all_results


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


def create_metric_plots(metric_data: Dict[str, Dict[str, Any]], subreddit: str, output_dir: str = "reports") -> None:
    """Create combined plot with all metrics in one figure."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pretty_names = {
        "argument_narrowness": "Argument Narrowness",
        "hostility": "Hostility",
        "suppression": "Suppression",
        "epistemic_closure": "Epistemic Closure"
    }
    
    # Create single figure with 4 rows (one per metric)
    fig = plt.figure(figsize=(12, 16))
    fig.suptitle(f'r/{subreddit} - Echo Chamber Metrics', fontsize=16, fontweight='bold', y=0.995)
    
    metrics_order = ["argument_narrowness", "hostility", "suppression", "epistemic_closure"]
    
    for idx, metric in enumerate(metrics_order):
        data = metric_data[metric]
        scores = data["scores"]
        reported_count = data["reported_count"]
        total_count = data["total_count"]
        pretty_name = pretty_names.get(metric, metric)
        
        # Create gridspec for this row: bar plot (narrow left) + histogram (large right)
        gs = fig.add_gridspec(4, 3, width_ratios=[0.2, 0.1, 2], wspace=0.05, hspace=0.3)
        
        # Left: Bar plot showing percent reported
        ax_bar = fig.add_subplot(gs[idx, 0])
        percent = data["percent_reported"]
        ax_bar.bar(0, percent, color='green', alpha=0.7, width=0.6)
        ax_bar.set_ylim(0, 105)
        ax_bar.text(0, percent + 2, f'{percent:.0f}%', ha='center', fontsize=10, fontweight='bold')
        ax_bar.set_xlabel('% reported', fontsize=9)
        
        # Remove ticks and spines from bar plot
        ax_bar.set_xticks([])
        ax_bar.set_yticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        
        # Spacer (empty)
        ax_spacer = fig.add_subplot(gs[idx, 1])
        ax_spacer.axis('off')
        
        # Right: Histogram (main visualization)
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
    print(f"✓ Saved: {output_file}")
    plt.close()


def create_summary_stats(metric_data: Dict[str, Dict[str, Any]], subreddit: str, output_dir: str = "reports") -> None:
    """Create and save summary statistics table."""
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
    print("\n" + summary_text)
    
    output_file = Path(output_dir) / f"{subreddit}_summary.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"✓ Summary saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate reports from processed LLM analysis")
    parser.add_argument("subreddit", help="Subreddit name (e.g., samharris)")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed data (default: data/processed)")
    parser.add_argument("--output", default="reports", help="Output directory for plots (default: reports)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"ChamberCheck - Report Generator")
    print("=" * 80)
    print(f"\n📊 Analyzing r/{args.subreddit}...\n")
    
    results = load_processed_files(args.subreddit, args.processed_dir)
    
    if not results:
        print(f"❌ No processed results found for r/{args.subreddit}")
        sys.exit(1)
    
    valid_results = [r for r in results if 'error' not in r]
    print(f"✓ Loaded {len(valid_results)} valid results (skipped {len(results) - len(valid_results)} errors)\n")
    
    print("📈 Computing metrics...\n")
    metric_data = extract_metric_data(valid_results)
    
    print(f"🎨 Creating visualizations in {args.output}/...\n")
    create_metric_plots(metric_data, args.subreddit, args.output)
    
    print()
    create_summary_stats(metric_data, args.subreddit, args.output)
    
    print(f"\n{'='*80}")
    print(f"✅ Report complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
