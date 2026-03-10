import json
import re
from pathlib import Path

path = Path("data/output/scrape_003/fake_llm_derived_metrics_by_subreddit_001.json")
data = json.loads(path.read_text(encoding="utf-8"))

print("subreddit | dropout | counter_n | aligned_n | counter_zero_rate | aligned_zero_rate")
for subreddit in sorted(data.keys()):
    metric = data[subreddit]["Dropout Rate"]
    notes = metric.get("notes", "")

    def grab(name: str) -> str:
        match = re.search(rf"{name}=([^,;]+)", notes)
        return match.group(1).strip() if match else "NA"

    print(
        f"{subreddit} | {metric.get('value')} | {grab('counter_n')} | {grab('aligned_n')} | "
        f"{grab('counter_zero_rate')} | {grab('aligned_zero_rate')}"
    )
