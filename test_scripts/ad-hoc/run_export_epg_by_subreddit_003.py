import json
from pathlib import Path
from typing import Any, Dict


INPUT_PATH = Path("data/output/scrape_003/fake_llm_derived_metrics_001.json")
OUTPUT_DIR = Path("data/output/scrape_003/expressive_participation_gap_by_subreddit_001")


def _sanitize_filename(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    name = "".join(keep).strip("_")
    return name or "subreddit"


def _extract_epg_topic_map(payload: Dict[str, Any], subreddit: str) -> Dict[str, Any]:
    subreddit_payload = payload.get(subreddit, {})
    if not isinstance(subreddit_payload, dict):
        return {}

    epg_payload = subreddit_payload.get("Expressive Participation Gap", {})
    if not isinstance(epg_payload, dict):
        return {}

    return epg_payload


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Expected top-level dict in topic metrics file")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index: Dict[str, Dict[str, Any]] = {}

    for subreddit in sorted(payload.keys()):
        epg_topic_map = _extract_epg_topic_map(payload, subreddit)

        output_payload = {
            "subreddit": subreddit,
            "metric": "Expressive Participation Gap",
            "source_file": str(INPUT_PATH).replace("/", "\\"),
            "topic_count": len(epg_topic_map),
            "by_topic": epg_topic_map,
        }

        file_name = f"{_sanitize_filename(subreddit)}_epg_by_topic.json"
        output_path = OUTPUT_DIR / file_name
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output_payload, file, indent=2, ensure_ascii=False)

        index[subreddit] = {
            "file": str(output_path).replace("/", "\\"),
            "topic_count": len(epg_topic_map),
        }

    index_path = OUTPUT_DIR / "index.json"
    with index_path.open("w", encoding="utf-8") as file:
        json.dump(index, file, indent=2, ensure_ascii=False)

    print(f"Saved EPG folder: {OUTPUT_DIR}")
    print(f"Subreddits exported: {len(index)}")


if __name__ == "__main__":
    main()
