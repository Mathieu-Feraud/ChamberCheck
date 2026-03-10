import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _infer_single_topic(entries: List[Dict[str, Any]], explicit: Optional[str]) -> str:
    if explicit and explicit.strip():
        return explicit.strip()

    for item in entries:
        if not isinstance(item, dict):
            continue
        entry = item.get("entry")
        if not isinstance(entry, dict):
            continue

        parent_topic = entry.get("parent_topic")
        if isinstance(parent_topic, str) and parent_topic.strip():
            return parent_topic.strip()

        topic_obj = entry.get("topic")
        if isinstance(topic_obj, dict):
            label = topic_obj.get("label")
            if isinstance(label, str) and label.strip():
                return label.strip()

    return "Unified Parent Post Topic"


def normalize_entries_to_single_topic(entries: List[Dict[str, Any]], topic_label: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    for item in entries:
        if not isinstance(item, dict):
            continue

        cloned = json.loads(json.dumps(item))
        entry = cloned.get("entry")
        if not isinstance(entry, dict):
            normalized.append(cloned)
            continue

        entry["parent_topic"] = topic_label

        topic_obj = entry.get("topic")
        if not isinstance(topic_obj, dict):
            topic_obj = {}
            entry["topic"] = topic_obj
        topic_obj["label"] = topic_label

        normalized.append(cloned)

    return normalized


def build_fake_single_topic_dataset(
    entries_path: Path,
    metadata_path: Optional[Path],
    output_entries_path: Path,
    output_metadata_path: Optional[Path],
    topic_label: Optional[str],
) -> Dict[str, Any]:
    entries_payload = _load_json(entries_path)
    if not isinstance(entries_payload, list):
        raise ValueError(f"Expected list payload in entries file: {entries_path}")

    unified_topic = _infer_single_topic(entries_payload, topic_label)
    normalized_entries = normalize_entries_to_single_topic(entries_payload, unified_topic)
    _save_json(output_entries_path, normalized_entries)

    metadata_written = None
    if metadata_path and output_metadata_path:
        metadata_payload = _load_json(metadata_path)
        if not isinstance(metadata_payload, dict):
            raise ValueError(f"Expected dict payload in metadata file: {metadata_path}")

        cloned_meta = json.loads(json.dumps(metadata_payload))
        cloned_meta["run_timestamp"] = datetime.now().isoformat()
        cloned_meta["function"] = "run_make_fake_abn_single_topic"
        cloned_meta["output_file"] = str(output_entries_path).replace("/", "\\")
        cloned_meta["metadata_file"] = str(output_metadata_path).replace("/", "\\")
        cloned_meta["entries_extracted"] = len(normalized_entries)
        cloned_meta["single_topic_label"] = unified_topic
        cloned_meta["source_entries_file"] = str(entries_path).replace("/", "\\")
        cloned_meta["notes"] = "All entries normalized to a single parent-topic/topic.label for synthetic testing."

        _save_json(output_metadata_path, cloned_meta)
        metadata_written = str(output_metadata_path)

    return {
        "output_entries_file": str(output_entries_path),
        "output_metadata_file": metadata_written,
        "single_topic_label": unified_topic,
        "entries_count": len(normalized_entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a fake ABN entries dataset where all comments share one topic label."
    )
    parser.add_argument(
        "--entries",
        default="data/output/scrape_003/abn_test/abn_test_llm_entries_009.json",
        help="Input ABN entries JSON file.",
    )
    parser.add_argument(
        "--metadata",
        default="data/output/scrape_003/abn_test/abn_test_llm_entries_metadata_009.json",
        help="Optional matching metadata JSON file.",
    )
    parser.add_argument(
        "--output-entries",
        default="data/output/scrape_003/abn_test/abn_test_llm_entries_fake_single_topic_001.json",
        help="Output fake entries JSON file.",
    )
    parser.add_argument(
        "--output-metadata",
        default="data/output/scrape_003/abn_test/abn_test_llm_entries_fake_single_topic_metadata_001.json",
        help="Output metadata JSON file.",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Explicit topic label to apply. If omitted, inferred from first available parent_topic/topic.label.",
    )

    args = parser.parse_args()

    entries_path = Path(args.entries)
    metadata_path = Path(args.metadata) if args.metadata else None
    output_entries_path = Path(args.output_entries)
    output_metadata_path = Path(args.output_metadata) if args.output_metadata else None

    result = build_fake_single_topic_dataset(
        entries_path=entries_path,
        metadata_path=metadata_path,
        output_entries_path=output_entries_path,
        output_metadata_path=output_metadata_path,
        topic_label=args.topic,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
