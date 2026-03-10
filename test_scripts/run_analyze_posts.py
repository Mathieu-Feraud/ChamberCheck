import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from ChamberCheck.analysis import analyze_posts

scrape_dir = sys.argv[1] if len(sys.argv) > 1 else None
config_path = sys.argv[2] if len(sys.argv) > 2 else "config/config.yaml"

analyze_posts(scrape_dir=scrape_dir, config_path=config_path)
