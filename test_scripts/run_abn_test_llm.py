import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from ChamberCheck.model_analysis.abn_test import run_abn_llm_analysis

PROMPTS_METADATA_FILE = "data/output/scrape_003/abn_test/abn_test_prompts_metadata_024.json"
TOP = 20
OUTPUT_DIR = "data/output/scrape_003"

MODEL_RUNS = [
    ("GPT-5.2",            "gpt-5.2"),
    ("Claude Sonnet 4.6",  "claude-sonnet-4-6"),
    ("Claude Haiku 4.5",   "claude-haiku-4-5-20251001"),
]


def main():
    load_dotenv()

    results = []
    for display_name, model_name in MODEL_RUNS:
        print(f"\nRunning ABN LLM analysis for {display_name} ({model_name})...")
        try:
            result = run_abn_llm_analysis(
                metadata_json_path=PROMPTS_METADATA_FILE,
                top=TOP,
                model=model_name,
                output_dir=OUTPUT_DIR,
            )
        except Exception as exc:
            result = {"status": "failed", "model": model_name, "error": str(exc)}
        results.append((display_name, model_name, result))
        print(result)

    print("\nBatch run completed.")
    for display_name, model_name, result in results:
        print(
            f"- {display_name} -> "
            f"{result.get('output_file', 'N/A')} "
            f"(errors: {result.get('entries_parse_errors', 'N/A')})"
        )


if __name__ == "__main__":
    main()

