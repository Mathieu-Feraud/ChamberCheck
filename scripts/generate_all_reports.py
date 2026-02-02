"""
Batch generate reports for all subreddits with processed analysis data.

Usage:
  python scripts/generate_all_reports.py
"""

import subprocess
from pathlib import Path


def main():
    processed_dir = Path("data/processed")
    
    # Find all unique subreddit names from analysis files
    subreddits = set()
    for file in processed_dir.glob("*_analysis_*.json"):
        if "_metadata" not in file.name:
            # Extract subreddit name (everything before _analysis_)
            subreddit = file.name.split("_analysis_")[0]
            subreddits.add(subreddit)
    
    print("=" * 80)
    print("ChamberCheck - Batch Report Generator")
    print("=" * 80)
    print(f"\nFound {len(subreddits)} subreddits to process:")
    for sub in sorted(subreddits):
        print(f"  • r/{sub}")
    print()
    
    # Generate report for each subreddit
    success_count = 0
    for subreddit in sorted(subreddits):
        print(f"\n{'='*80}")
        print(f"Processing r/{subreddit}...")
        print(f"{'='*80}")
        
        try:
            result = subprocess.run(
                ["python", "scripts/report_results.py", subreddit],
                check=True,
                capture_output=False
            )
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate report for r/{subreddit}")
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Batch processing complete!")
    print(f"Successfully generated {success_count}/{len(subreddits)} reports")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
