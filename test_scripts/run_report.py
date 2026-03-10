import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ChamberCheck.reporting import generate_subreddit_report

# Generate a test report
result = generate_subreddit_report(
    subreddit="samharris",
    processed_dir="data/processed",
    output_dir="data/output",
)
print(f"\nReport Generated: {result}")
