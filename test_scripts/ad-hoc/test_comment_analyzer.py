import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from ChamberCheck.analysis import batch_analyze_comments
from ChamberCheck.analysis.openai_provider import OpenAIProvider

load_dotenv()

# Analyze comments with same parameters as 001 run
metadata = batch_analyze_comments(
    comment_files=["data/raw/scrape_001/samharris.json"],
    provider=OpenAIProvider(),
    model_name="gpt-4o-mini",
    comment_ids=["fuxlkh2", "fvrueof", "i9cp4qy", "fuxy7el", "ikro9p1"],
)

print(f"\nAnalysis complete. Metadata:\n{metadata}")
