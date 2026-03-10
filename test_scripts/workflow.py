"""
ChamberCheck - Full Pipeline Workflow
Comment out stages you don't need to run.
"""
import sys
import runpy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

CONFIG     = "config/config.test.yaml"
SCRAPE_DIR = None   # set to e.g. "data/raw/scrape_006" to skip scrape_posts

# --- 1. Scrape posts ----------------------------------------------------------
from ChamberCheck.scrapers import batch_scrape_posts_only
batch_scrape_posts_only(CONFIG)

# --- 2. Analyze post titles ---------------------------------------------------
from ChamberCheck.analysis import analyze_posts
analyze_posts(scrape_dir=SCRAPE_DIR, config_path=CONFIG)

# --- 3. Preprocess posts (filter + rank) -------------------------------------
from ChamberCheck.preprocessing import preprocess_posts
preprocess_posts(SCRAPE_DIR, config_path=CONFIG)

# --- 4. Scrape comments -------------------------------------------------------
from ChamberCheck.scrapers import scrape_comments
scrape_comments(scrape_dir=SCRAPE_DIR, config_path=CONFIG)

# --- 5. Preprocess comments ---------------------------------------------------
from ChamberCheck.preprocessing import preprocess_comments
preprocess_comments(scrape_dir=SCRAPE_DIR, config_path=CONFIG)

# --- 6. Analyze comments ------------------------------------------------------
from ChamberCheck.analysis import run_comment_analysis
run_comment_analysis(scrape_dir=SCRAPE_DIR, config_path=CONFIG)

# --- 7. Compute V3 metrics ----------------------------------------------------
runpy.run_path(str(Path(__file__).parent / "run_v3_metrics.py"), run_name="__main__")

# --- 8. Plot metrics ----------------------------------------------------------
runpy.run_path(str(Path(__file__).parent / "ad-hoc" / "plot_v3_metrics.py"), run_name="__main__")
