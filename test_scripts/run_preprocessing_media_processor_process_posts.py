"""
Enrich posts with media context (images, links, video) via OpenAI.

Writes:
    data/raw/scrape_XXX/comments/posts_context_NNN.json
    data/raw/scrape_XXX/comments/posts_context_NNN_metadata.json

Usage:
    python test_scripts/run_preprocessing_media_processor_process_posts.py
    python test_scripts/run_preprocessing_media_processor_process_posts.py data/raw/scrape_006
    python test_scripts/run_preprocessing_media_processor_process_posts.py data/raw/scrape_006 --force
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from ChamberCheck.preprocessing.media_processor import enrich_posts_context
from ChamberCheck.config import Config
from ChamberCheck.constants import RAW_DATA_DIR


def _resolve_scrape_dir(arg: str = None) -> str:
    if arg:
        p = Path(arg)
        return str(p.parent.parent if p.is_file() else p)
    cfg = Config()
    raw = Path(cfg.get("raw_data_dir") or RAW_DATA_DIR)
    folders = sorted(d for d in raw.glob("scrape_*") if d.is_dir())
    if not folders:
        raise FileNotFoundError(f"No scrape_XXX folders found in {raw}")
    return str(folders[-1])


positional = [a for a in sys.argv[1:] if not a.startswith("--")]
force = "--force" in sys.argv
scrape_dir = _resolve_scrape_dir(positional[0] if positional else None)
print(f"Scrape dir  : {scrape_dir}")
enrich_posts_context(scrape_dir=scrape_dir, force=force)
