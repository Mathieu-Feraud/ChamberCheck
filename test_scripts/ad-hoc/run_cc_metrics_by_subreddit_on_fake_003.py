import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ChamberCheck.CC_derived_metrics.derived_metrics import CC_Metrics, MetricResult


def _bootstrap_dropout_ci(
    cc: CC_Metrics,
    subreddit: str,
    iterations: int = 2000,
    epsilon: float = 0.01,
) -> tuple[float, float] | None:
    debug = cc.dropout_rate_debug(subreddit=subreddit)
    counter_depths = debug.get("counter_depths")
    aligned_depths = debug.get("aligned_depths")
    min_group_n = int(debug.get("min_group_n", 10))

    if not isinstance(counter_depths, list) or not isinstance(aligned_depths, list):
        return None
    if len(counter_depths) < min_group_n or len(aligned_depths) < min_group_n:
        return None

    counter = np.array(counter_depths, dtype=float)
    aligned = np.array(aligned_depths, dtype=float)
    if counter.size == 0 or aligned.size == 0:
        return None

    seed = abs(hash(subreddit)) % (2**32)
    rng = np.random.default_rng(seed)
    boot_values = np.empty(iterations, dtype=float)

    for idx in range(iterations):
        counter_sample = rng.choice(counter, size=counter.size, replace=True)
        aligned_sample = rng.choice(aligned, size=aligned.size, replace=True)
        mean_counter = float(np.mean(counter_sample))
        mean_aligned = float(np.mean(aligned_sample))
        boot_values[idx] = 1.0 - ((mean_counter + epsilon) / (mean_aligned + epsilon))

    lower = float(np.quantile(boot_values, 0.025))
    upper = float(np.quantile(boot_values, 0.975))
    return lower, upper


def _metric_to_dict(metric: MetricResult) -> Dict[str, Any]:
    payload = asdict(metric)

    value = payload.get("value")
    sample_size = payload.get("sample_size")
    metric_name = str(payload.get("name", ""))
    ci_lower = None
    ci_upper = None
    ci_method = None

    proportion_metrics = {
        "Selective Engagement",
        "Discreditation Rate",
        "Counter-Evidence Exposure Rate (CER)",
        "Constructive Counter-View Engagement (CCVE)",
        "Cross-Stance Interaction Rate (CSIR)",
    }

    if (
        metric_name in proportion_metrics
        and
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isinstance(sample_size, int)
        and sample_size > 0
        and 0.0 <= float(value) <= 1.0
    ):
        p = float(value)
        n = sample_size
        z = 1.96
        denom = 1.0 + (z * z) / n
        center = (p + (z * z) / (2.0 * n)) / denom
        half_width = (
            z
            * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
            / denom
        )
        ci_lower = max(0.0, center - half_width)
        ci_upper = min(1.0, center + half_width)
        ci_method = "wilson_95"

    payload["ci_95_lower"] = ci_lower
    payload["ci_95_upper"] = ci_upper
    payload["ci_method"] = ci_method
    return payload


def main() -> None:
    metadata_path = Path("data/output/scrape_003/fake_llm_entries_metadata_001.json")
    output_path = Path("data/output/scrape_003/fake_llm_derived_metrics_by_subreddit_001.json")

    cc = CC_Metrics.from_abn_llm_run_metadata(metadata_path)
    by_subreddit = cc.compute_all_by_subreddit()

    serializable: Dict[str, Dict[str, Any]] = {}
    for subreddit, metric_map in by_subreddit.items():
        serializable[subreddit] = {}
        for metric_name, payload in metric_map.items():
            if isinstance(payload, MetricResult):
                metric_dict = _metric_to_dict(payload)
                if metric_name == "Dropout Rate":
                    value = metric_dict.get("value")
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        ci = _bootstrap_dropout_ci(cc=cc, subreddit=subreddit)
                        if ci is not None:
                            ci_lower, ci_upper = ci
                            metric_dict["ci_95_lower"] = ci_lower
                            metric_dict["ci_95_upper"] = ci_upper
                            metric_dict["ci_method"] = "bootstrap_95_b2000"
                serializable[subreddit][metric_name] = metric_dict
            elif isinstance(payload, dict):
                topic_payload: Dict[str, Any] = {}
                for topic, result in payload.items():
                    if isinstance(result, MetricResult):
                        topic_payload[topic] = _metric_to_dict(result)
                    else:
                        topic_payload[topic] = result
                serializable[subreddit][metric_name] = topic_payload
            else:
                serializable[subreddit][metric_name] = payload

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")
    print(f"Subreddits: {len(serializable)}")
    sample_sub = sorted(serializable.keys())[0]
    print(f"Sample subreddit: {sample_sub}")
    print(f"Metrics: {list(serializable[sample_sub].keys())[:6]}")


if __name__ == "__main__":
    main()
