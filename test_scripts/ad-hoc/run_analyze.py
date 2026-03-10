import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from ChamberCheck.analysis import batch_analyze_comments
load_dotenv()

result = batch_analyze_comments(
    comment_files=["data/raw/scrape_001/samharris.json"],
    limit=20,
    mode="top",
)

print(result)
