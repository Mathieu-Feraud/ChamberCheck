import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ChamberCheck.preprocessing import process_folder

result = process_folder("data/raw/scrape_001", force_reprocess=True)
print(f"\nResult: {result}")
