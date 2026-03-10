import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ChamberCheck.analysis import export_comment_prompts

# Export prompts with same comment IDs
metadata = export_comment_prompts(
    input_file="data/raw/scrape_001/samharris.json",
    comment_ids=["fuxlkh2", "fvrueof", "i9cp4qy", "fuxy7el", "ikro9p1"],
)

print(f"\nPrompts exported. Metadata:\n{metadata}")
