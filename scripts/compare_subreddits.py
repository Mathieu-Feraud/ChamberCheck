"""
Compare echo chamber metrics across all subreddits.

Creates a comparative visualization showing how all subreddits score on the 4 metrics
plus an overall echo chamber score.

Usage:
  python scripts/compare_subreddits.py
  python scripts/compare_subreddits.py --output reports/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt


def load_all_subreddit_data(processed_dir: str = "data/processed") -> Dict[str, Dict[str, Any]]:
    """Load analysis results for all subreddits."""
    processed_path = Path(processed_dir)
    subreddit_data = {}
    
    # Find all unique subreddits
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
            
            # Load results
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
                    
                    # Extract comment types
                    comment_types = result.get("comment_types", [])
                    if isinstance(comment_types, list):
                        subreddit_data[subreddit]["comment_types"].extend(comment_types)
                
                print(f"✓ Loaded r/{subreddit}")
            except Exception as e:
                print(f"⚠️  Error loading {file.name}: {e}")
    
    return subreddit_data


def compute_means(subreddit_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """Compute mean scores for each metric per subreddit."""
    means = {}
    
    for subreddit, metrics in subreddit_data.items():
        means[subreddit] = {}
        
        for metric, scores in metrics.items():
            # Skip comment_types (not a numeric metric)
            if metric == "comment_types":
                continue
            
            if scores:
                means[subreddit][metric] = np.mean(scores)
            else:
                means[subreddit][metric] = np.nan
        
        # Compute overall score (average of all 4 metrics)
        valid_means = [v for v in means[subreddit].values() if not np.isnan(v)]
        if valid_means:
            means[subreddit]["overall"] = np.mean(valid_means)
        else:
            means[subreddit]["overall"] = np.nan
    
    return means


def create_comparison_plot(means: Dict[str, Dict[str, float]], output_dir: str = "reports") -> None:
    """Create comparison plot with 5 panels (4 metrics + overall)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pretty_names = {
        "argument_narrowness": "Argument Narrowness",
        "hostility": "Hostility",
        "suppression": "Suppression",
        "epistemic_closure": "Epistemic Closure",
        "overall": "Overall Echo Chamber Score"
    }
    
    metrics_order = ["argument_narrowness", "hostility", "suppression", "epistemic_closure", "overall"]
    subreddits = sorted(means.keys())
    
    # Create figure with 5 subplots (one per metric + overall)
    fig, axes = plt.subplots(5, 1, figsize=(14, 20))
    fig.suptitle('Echo Chamber Metrics - Subreddit Comparison', fontsize=18, fontweight='bold', y=0.995)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(subreddits)))
    
    for idx, metric in enumerate(metrics_order):
        ax = axes[idx]
        
        # Get scores for this metric
        scores = [means[sub].get(metric, np.nan) for sub in subreddits]
        positions = np.arange(len(subreddits))
        
        # Create bar chart
        bars = ax.bar(positions, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Highlight bars by value
        for bar, score in zip(bars, scores):
            if not np.isnan(score):
                if score >= 7:
                    bar.set_edgecolor('red')
                    bar.set_linewidth(2.5)
        
        ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
        ax.set_title(pretty_names[metric], fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(positions)
        ax.set_xticklabels([f'r/{sub}' for sub in subreddits], rotation=45, ha='right')
        ax.set_ylim(0, 10.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for pos, score in zip(positions, scores):
            if not np.isnan(score):
                ax.text(pos, score + 0.2, f'{score:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = Path(output_dir) / 'subreddit_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()


def print_summary_table(means: Dict[str, Dict[str, float]]) -> None:
    """Print summary table of all scores."""
    print("\n" + "=" * 120)
    print("SUBREDDIT COMPARISON - MEAN SCORES")
    print("=" * 120)
    
    metrics = ["argument_narrowness", "hostility", "suppression", "epistemic_closure", "overall"]
    pretty_names = {
        "argument_narrowness": "Arg. Narrow",
        "hostility": "Hostility",
        "suppression": "Suppression",
        "epistemic_closure": "Epist. Close",
        "overall": "Overall"
    }
    
    # Header
    header = f"{'Subreddit':<25}"
    for metric in metrics:
        header += f"{pretty_names[metric]:>14}"
    print(header)
    print("-" * 120)
    
    # Sort by overall score (descending)
    sorted_subs = sorted(means.items(), key=lambda x: x[1].get('overall', 0), reverse=True)
    
    for subreddit, scores in sorted_subs:
        row = f"r/{subreddit:<23}"
        for metric in metrics:
            score = scores.get(metric, np.nan)
            if np.isnan(score):
                row += f"{'N/A':>14}"
            else:
                row += f"{score:>14.2f}"
        print(row)
    
    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare echo chamber metrics across subreddits")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed data (default: data/processed)")
    parser.add_argument("--output", default="reports", help="Output directory for plots (default: reports)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ChamberCheck - Subreddit Comparison")
    print("=" * 80)
    print("\n📊 Loading data from all subreddits...\n")
    
    subreddit_data = load_all_subreddit_data(args.processed_dir)
    
    if not subreddit_data:
        print("❌ No processed results found")
        return
    
    print(f"\n✓ Loaded {len(subreddit_data)} subreddits\n")
    print("📈 Computing mean scores...")
    
    means = compute_means(subreddit_data)
    
    print_summary_table(means)
    
    print(f"🎨 Creating comparison visualization in {args.output}/...")
    create_comparison_plot(means, args.output)
    
    print(f"🎨 Creating comment type distribution plot...")
    create_comment_type_plot(subreddit_data, args.output)
    
    print("\n" + "=" * 80)
    print("✅ Comparison complete!")
    print("=" * 80)


def create_comment_type_plot(subreddit_data: Dict[str, Dict[str, Any]], output_dir: str = "reports") -> None:
    """Create heatmap showing comment type counts per subreddit."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Valid comment types (filter out metric names that LLM incorrectly added)
    valid_types = {"argumentative", "factual", "anecdotal", "question", "humor_or_irony", "other"}
    
    # Count comment types per subreddit
    type_counts = {}
    all_types = set()
    
    for subreddit, data in subreddit_data.items():
        type_counts[subreddit] = {}
        for comment_type in data["comment_types"]:
            # Filter out invalid types (metric names mistakenly added by LLM)
            if comment_type and comment_type.lower() in valid_types:
                all_types.add(comment_type)
                type_counts[subreddit][comment_type] = type_counts[subreddit].get(comment_type, 0) + 1
    
    # Prepare data for heatmap
    subreddits = sorted(type_counts.keys())
    comment_types = sorted(all_types)
    
    # Build matrix for heatmap
    matrix = []
    for subreddit in subreddits:
        row = [type_counts[subreddit].get(ctype, 0) for ctype in comment_types]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(comment_types)))
    ax.set_yticks(np.arange(len(subreddits)))
    ax.set_xticklabels(comment_types, rotation=45, ha='right')
    ax.set_yticklabels([f'r/{sub}' for sub in subreddits])
    
    # Add count annotations
    for i in range(len(subreddits)):
        for j in range(len(comment_types)):
            count = matrix[i, j]
            if count > 0:
                text = ax.text(j, i, int(count), ha="center", va="center", 
                              color="white" if count > matrix.max() * 0.5 else "black",
                              fontweight='bold', fontsize=10)
    
    ax.set_title('Comment Type Distribution by Subreddit (Counts)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / 'comment_types_by_subreddit.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    main()
