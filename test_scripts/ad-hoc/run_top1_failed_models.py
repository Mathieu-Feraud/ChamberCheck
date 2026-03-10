import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dotenv import load_dotenv
from ChamberCheck.model_analysis.abn_test import run_abn_llm_analysis

PROMPTS_METADATA_FILE = "data/output/scrape_003/abn_test/abn_test_prompts_metadata_024.json"
OUTPUT_DIR = "data/output/scrape_003"

MODEL_RUNS = [
    ("GPT-5.2", "gpt-5.2"),
    ("Claude Haiku 3.5", "claude-3-5-haiku-latest"),
]


def main() -> None:
    load_dotenv()

    for display_name, model_name in MODEL_RUNS:
        print(f"\nRunning top=1 for {display_name} ({model_name})...")
        try:
            result = run_abn_llm_analysis(
                metadata_json_path=PROMPTS_METADATA_FILE,
                top=1,
                model=model_name,
                output_dir=OUTPUT_DIR,
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "model": model_name,
                "error": str(exc),
            }

        print(result)


if __name__ == "__main__":
    main()
