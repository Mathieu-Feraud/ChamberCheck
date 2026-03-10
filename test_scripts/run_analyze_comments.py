import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from ChamberCheck.analysis import run_comment_analysis

scrape_dir    = sys.argv[1] if len(sys.argv) > 1 else None
config_path   = sys.argv[2] if len(sys.argv) > 2 else "config/config.yaml"
_max          = int(sys.argv[3]) if len(sys.argv) > 3 else 0
max_comments  = _max if _max > 0 else None   # 0 means "no limit"
start_offset  = int(sys.argv[4]) if len(sys.argv) > 4 else 0

run_comment_analysis(
    scrape_dir=scrape_dir,
    config_path=config_path,
    max_comments=max_comments,
    start_offset=start_offset,
)
