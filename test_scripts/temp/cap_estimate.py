import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

data = json.loads(Path('data/raw/scrape_006/comments/comments_001.json').read_text(encoding='utf-8'))

by_sub = defaultdict(list)
for post in data:
    by_sub[post['community']].append(post['comment_count'])

print(f"{'Subreddit':<30} {'Posts':>5}  {'Raw':>8}  {'@200':>7}  {'@500':>7}")
print("-" * 62)
t_raw = t_200 = t_500 = 0
for sub, counts in sorted(by_sub.items()):
    raw  = sum(counts)
    c200 = sum(min(c, 200) for c in counts)
    c500 = sum(min(c, 500) for c in counts)
    t_raw += raw; t_200 += c200; t_500 += c500
    print(f"  {sub:<28} {len(counts):>5}  {raw:>8,}  {c200:>7,}  {c500:>7,}")
print("-" * 62)
print(f"  {'TOTAL':<28} {sum(len(v) for v in by_sub.values()):>5}  {t_raw:>8,}  {t_200:>7,}  {t_500:>7,}")
print()

df = pd.read_csv('data/output/scrape_003/abn_test/model_stats/model_cost_summary_usd.csv')
haiku35 = df[df.source_label == 'claude-3-5-haiku-latest'].iloc[0]
haiku45 = df[df.source_label == 'claude-haiku-4-5-20251001'].iloc[0]

print(f"{'Model':<12} {'Cap=200 calls':>14}  {'Cost @200':>10}  {'Cost @500':>10}")
print("-" * 52)
for label, row in [('haiku-3.5', haiku35), ('haiku-4.5', haiku45)]:
    cost200 = row['total_cost_usd_per_1000_calls'] * t_200 / 1000
    cost500 = row['total_cost_usd_per_1000_calls'] * t_500 / 1000
    saving  = cost500 - cost200
    print(f"  {label:<10} {t_200:>14,}  ${cost200:>9.2f}  ${cost500:>9.2f}  (save ${saving:.2f})")
