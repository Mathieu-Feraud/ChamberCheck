import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ChamberCheck.scrapers import batch_scrape_posts_only

batch_scrape_posts_only("config/config.test.yaml")
