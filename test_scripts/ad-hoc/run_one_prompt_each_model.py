import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dotenv import load_dotenv

from ChamberCheck.model_analysis.abn_test import run_abn_llm_analysis

PROMPTS_METADATA_FILE = "data/output/scrape_003/abn_test/abn_test_prompts_metadata_024.json"
OUTPUT_DIR = "data/output/scrape_003"
TOP = 1

MODEL_RUNS = [
    ("GPT-5 nano", "gpt-5-nano"),
    ("GPT-5 mini", "gpt-5-mini"),
    ("GPT-5.2", "gpt-5.2"),
    ("Claude Sonnet 4.6", "claude-sonnet-4-6"),
    ("Claude Haiku 4.5", "claude-haiku-4-5-20251001"),
    ("Claude Haiku 3.5", "claude-3-5-haiku-latest"),
]


def main() -> None:
    load_dotenv()
    all_results = []

    for display_name, model_name in MODEL_RUNS:
        print(f"\nRunning top=1 for {display_name} ({model_name})...")
        try:
            result = run_abn_llm_analysis(
                metadata_json_path=PROMPTS_METADATA_FILE,
                top=TOP,
                model=model_name,
                output_dir=OUTPUT_DIR,
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "model": model_name,
                "error": str(exc),
            }

        all_results.append((display_name, model_name, result))
        print(result)

    print("\n=== One-prompt model smoke summary ===")
    for display_name, model_name, result in all_results:
        if result.get("status") == "success":
            print(
                f"[PASS] {display_name} ({model_name}) -> "
                f"{result.get('output_file')} "
                f"parse_errors={result.get('entries_parse_errors')}"
            )
        else:
            print(f"[FAIL] {display_name} ({model_name}) -> {result.get('error')}")


if __name__ == "__main__":
    main()
