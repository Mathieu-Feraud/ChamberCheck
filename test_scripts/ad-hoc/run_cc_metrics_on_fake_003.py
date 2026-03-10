import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from ChamberCheck.CC_derived_metrics.derived_metrics import CC_Metrics, MetricResult


def serialize_metric(metric: MetricResult) -> Dict[str, Any]:
    payload = asdict(metric)
    return payload


def main() -> None:
    metadata_path = Path("data/output/scrape_003/fake_llm_entries_metadata_001.json")
    output_path = Path("data/output/scrape_003/fake_llm_derived_metrics_001.json")

    cc = CC_Metrics.from_abn_llm_run_metadata(metadata_path)
    by_subreddit_topic = cc.compute_all_by_subreddit_topic()

    serializable: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for subreddit, metric_map in by_subreddit_topic.items():
        serializable[subreddit] = {}
        for metric_name, topic_map in metric_map.items():
            serializable[subreddit][metric_name] = {
                topic: serialize_metric(result) for topic, result in topic_map.items()
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, ensure_ascii=False)

    print(f"Saved full results to: {output_path}")
    print(f"Subreddits: {len(serializable)}")

    for subreddit in sorted(serializable.keys()):
        metric_map = serializable[subreddit]
        print(f"\n[{subreddit}]")
        for metric_name in sorted(metric_map.keys()):
            topic_map = metric_map[metric_name]
            valid = [v for v in topic_map.values() if v.get('value') is not None]
            print(f"- {metric_name}: topics={len(topic_map)}, valid={len(valid)}")


if __name__ == "__main__":
    main()
