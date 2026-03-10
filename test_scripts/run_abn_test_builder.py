import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from ChamberCheck.model_analysis.abn_test import generate_abn_test_set

load_dotenv()

result = generate_abn_test_set(
    raw_folder_path="data/raw/scrape_003",
    num_comments=50,
    random_seed=42,
)

print(result)
