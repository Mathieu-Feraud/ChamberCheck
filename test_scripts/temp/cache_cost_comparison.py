"""
Cost comparison: with and without prompt caching, for all models, at cap=200 and cap=500.

Cache mechanics:
  Anthropic — explicit cache_control required:
    write: 1.25x input price (one-time per session)
    read:  0.10x input price (90% discount)
  OpenAI — automatic (no code change needed):
    read:  0.50x input price (50% discount, no write cost)
"""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

# ── comment counts at each cap ────────────────────────────────────────────────
data = json.loads(Path('data/raw/scrape_006/comments/comments_001.json').read_text(encoding='utf-8'))
by_sub = defaultdict(list)
for post in data:
    by_sub[post['community']].append(post['comment_count'])

n200 = sum(min(c, 200) for counts in by_sub.values() for c in counts)
n500 = sum(min(c, 500) for counts in by_sub.values() for c in counts)

# ── token profile per call (from ABN test run averages) ─────────────────────
# avg_input from test; comment body ≈ 46 tokens → rest is system/prompt overhead
AVG_COMMENT_TOKENS = 46
token_profile = {
    'claude-3-5-haiku-latest':     {'avg_input': 2281, 'avg_output': 279,  'provider': 'anthropic'},
    'claude-haiku-4-5-20251001':   {'avg_input': 2457, 'avg_output': 288,  'provider': 'anthropic'},
    'claude-sonnet-4-6':           {'avg_input': 2457, 'avg_output': 282,  'provider': 'anthropic'},
    'gpt-5.2':                     {'avg_input': 2141, 'avg_output': 439,  'provider': 'openai'},
    'gpt-5-mini':                  {'avg_input': 1979, 'avg_output': 247,  'provider': 'openai'},
    'gpt-5-nano':                  {'avg_input': 1979, 'avg_output': 244,  'provider': 'openai'},
}

# ── pricing (USD per 1M tokens) ───────────────────────────────────────────────
pricing = {
    'claude-3-5-haiku-latest':   {'input': 0.80,  'output': 4.00},
    'claude-haiku-4-5-20251001': {'input': 1.00,  'output': 5.00},
    'claude-sonnet-4-6':         {'input': 3.00,  'output': 15.00},
    'gpt-5.2':                   {'input': 1.75,  'output': 14.00},
    'gpt-5-mini':                {'input': 0.25,  'output': 2.00},
    'gpt-5-nano':                {'input': 0.05,  'output': 0.40},
}

M = 1_000_000

def cost_no_cache(model, n_calls):
    p = pricing[model]; t = token_profile[model]
    input_cost  = (t['avg_input']  * n_calls / M) * p['input']
    output_cost = (t['avg_output'] * n_calls / M) * p['output']
    return input_cost, output_cost

def cost_with_cache(model, n_calls):
    p = pricing[model]; t = token_profile[model]
    system_toks  = t['avg_input'] - AVG_COMMENT_TOKENS
    comment_toks = AVG_COMMENT_TOKENS
    provider = t['provider']

    if provider == 'anthropic':
        # 1 write at 1.25x, (n-1) reads at 0.10x; comment tokens always full price
        write_cost  = (system_toks * 1   / M) * p['input'] * 1.25
        read_cost   = (system_toks * max(n_calls - 1, 0) / M) * p['input'] * 0.10
        comment_cost= (comment_toks * n_calls / M) * p['input']
        input_cost  = write_cost + read_cost + comment_cost
    else:  # openai — automatic, no write cost, 50% on cached tokens
        input_cost  = (system_toks  * n_calls / M) * p['input'] * 0.50 \
                    + (comment_toks * n_calls / M) * p['input']

    output_cost = (t['avg_output'] * n_calls / M) * p['output']
    return input_cost, output_cost

# ── build table ───────────────────────────────────────────────────────────────
rows = []
for model in token_profile:
    for cap, n in [('@200', n200), ('@500', n500)]:
        i_nc, o_nc = cost_no_cache(model, n)
        i_c,  o_c  = cost_with_cache(model, n)
        rows.append({
            'model':    model,
            'cap':      cap,
            'n_calls':  n,
            'no_cache': round(i_nc + o_nc, 2),
            'cached':   round(i_c  + o_c,  2),
            'saving':   round((i_nc + o_nc) - (i_c + o_c), 2),
            'save_pct': round(((i_nc + o_nc) - (i_c + o_c)) / (i_nc + o_nc) * 100, 1),
        })

df = pd.DataFrame(rows)

for cap in ['@200', '@500']:
    sub = df[df.cap == cap].sort_values('no_cache')
    print(f"\n{'='*72}")
    print(f"  Cap = {cap}  ({sub.iloc[0]['n_calls']:,} calls)")
    print(f"{'='*72}")
    print(f"  {'Model':<30} {'No cache':>10}  {'With cache':>10}  {'Saving':>8}  {'%':>5}")
    print(f"  {'-'*62}")
    for _, r in sub.iterrows():
        print(f"  {r['model']:<30} ${r['no_cache']:>9.2f}  ${r['cached']:>9.2f}  ${r['saving']:>7.2f}  {r['save_pct']:>4.1f}%")
