import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ChamberCheck.preprocessing import preprocess_posts

# Auto-detect latest scrape directory, or accept one as a CLI argument
raw_dir     = Path("data/raw")
latest      = sorted([p for p in raw_dir.glob("scrape_*") if p.is_dir()])[-1]
scrape_dir  = sys.argv[1] if len(sys.argv) > 1 else str(latest)
config_path = sys.argv[2] if len(sys.argv) > 2 else "config/config.yaml"

preprocess_posts(scrape_dir, config_path=config_path)
